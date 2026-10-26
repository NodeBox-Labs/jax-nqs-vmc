from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, Optional

import jax
import jax.numpy as jnp
import numpy as np
import optax
from tqdm import tqdm

from .local_energy import make_local_energy, energy_and_variance
from .optimizers import SRUpdater, quantum_geometric_tensor
from .sampler import sample, integrated_autocorr_time


@dataclass
class TrainState:
    params: dict
    opt_state: Optional[optax.OptState] = None
    step: int = 0
    history: Dict[str, list] = field(default_factory=lambda: {
        "energy": [], "variance": [], "stderr": [], "acceptance": [],
        "step_size": [], "tau_int": [], "grad_norm": [],
    })


def _init_chain(dim: int, n_chains: int, key: jnp.ndarray, scale: float = 1.0):
    return scale * jax.random.normal(key, (n_chains, dim), dtype=jnp.float64)


def train(
    logpsi_fn: Callable[[dict, jnp.ndarray], jnp.ndarray],
    params: dict,
    potential_fn: Callable[[jnp.ndarray], jnp.ndarray],
    dim: int,
    *,
    g: float = 0.0,
    n_iter: int = 1000,
    n_chains: int = 64,
    n_sweeps: int = 60,
    burn_in: int = 100,
    step_size_init: float = 0.5,
    target_accept: float = 0.55,
    optimizer: str = "adam",
    lr: float = 1e-3,
    sr_diag_shift: float = 1e-3,
    sr_use_diag: bool = False,
    seed: int = 0,
    orthogonalize_against: Optional[Callable[[jnp.ndarray], jnp.ndarray]] = None,
    ortho_penalty: float = 0.0,
    verbose: bool = True,
):
    """Run VMC training. Returns ``TrainState``."""
    key = jax.random.PRNGKey(seed)
    key, k_chain = jax.random.split(key)
    x = _init_chain(dim, n_chains, k_chain)

    local_energy = make_local_energy(logpsi_fn, potential_fn, g=g, vmap=True)

    if optimizer == "adam":
        opt = optax.adam(lr)
        opt_state = opt.init(params)
        sr = None
    elif optimizer == "sr":
        opt = None
        opt_state = None
        sr = SRUpdater(lr=lr, diag_shift=sr_diag_shift, use_diag=sr_use_diag)
    else:
        raise ValueError(f"unknown optimizer {optimizer!r}")

    state = TrainState(params=params, opt_state=opt_state)
    pbar = tqdm(range(n_iter), disable=not verbose, desc=f"[{optimizer}]")

    for it in pbar:
        key, k_samp = jax.random.split(key)
        samples, s_state = sample(
            logpsi_fn, state.params, x, k_samp,
            n_sweeps=n_sweeps, step_size=float(s_state_step(step_size_init, state.history)),
            target_accept=target_accept, burn_in=burn_in if it == 0 else 0,
        )
        x = samples[-n_chains:]  # carry the tail of the chain forward

        e_loc = local_energy(state.params, samples)
        E, Var, stderr = energy_and_variance(e_loc)

        if sr is None:
            def loss_fn(p):
                el = local_energy(p, samples)
                base = jnp.mean(el)
                if orthogonalize_against is not None and ortho_penalty > 0:
                    log_psi = logpsi_fn(p, samples)
                    log_ref = orthogonalize_against(samples)
                    ratio = jnp.exp(log_ref - log_psi)
                    base = base + ortho_penalty * jnp.abs(jnp.mean(ratio)) ** 2
                return base, jnp.mean(el)

            (loss, _), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
            updates, state.opt_state = opt.update(grads, state.opt_state, state.params)
            state.params = optax.apply_updates(state.params, updates)
            gn = jnp.sqrt(sum(jnp.sum(g * g) for g in jax.tree_util.tree_leaves(grads)))
        else:
            S, F = quantum_geometric_tensor(logpsi_fn, state.params, samples, e_loc)
            if orthogonalize_against is not None and ortho_penalty > 0:
                # add penalty gradient wrt energy: fold into F via finite-diff style
                # (kept simple: SR on the pure energy; penalty handled through F extension)
                log_psi = logpsi_fn(state.params, samples)
                log_ref = orthogonalize_against(samples)
                ratio = jnp.exp(log_ref - log_psi)
                # approximate penalty gradient by autograd on the *loss* form
                def pen_fn(p):
                    lp = logpsi_fn(p, samples)
                    r = jnp.exp(log_ref - lp)
                    return jnp.abs(jnp.mean(r)) ** 2
                pen_grad = jax.grad(pen_fn)(state.params)
                pen_flat, _ = jax.tree_util.tree_flatten(pen_grad)
                F = F + ortho_penalty * jnp.concatenate([l.ravel() for l in pen_flat])
            state.params, _ = sr.step(state.params, None, S, F)
            gn = jnp.linalg.norm(F)

        tau = integrated_autocorr_time(e_loc)
        state.history["energy"].append(float(E))
        state.history["variance"].append(float(Var))
        state.history["stderr"].append(float(stderr))
        state.history["acceptance"].append(float(s_state.acc_rate))
        state.history["step_size"].append(float(s_state.step_size))
        state.history["tau_int"].append(float(tau))
        state.history["grad_norm"].append(float(gn))

        if verbose and it % 20 == 0:
            pbar.set_postfix({
                "E": f"{float(E):.6f}",
                "Var": f"{float(Var):.2e}",
                "Acc": f"{float(s_state.acc_rate):.2f}",
                "τ": f"{float(tau):.1f}",
            })
        state.step = it

    return state


def s_state_step(step_size_init: float, history: dict) -> float:
    """Return a stable step size: last adapted value, or init on the first iter."""
    if not history["step_size"]:
        return step_size_init
    return history["step_size"][-1]