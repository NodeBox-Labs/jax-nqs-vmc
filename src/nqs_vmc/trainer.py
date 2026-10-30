"""VMC training loop."""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import jax
import jax.numpy as jnp
from tqdm import tqdm

from . import jax_config  # noqa: F401
from .checkpoint import load as load_ckpt
from .checkpoint import save as save_ckpt
from .config import RunConfig
from .local_energy import (
    energy_and_variance,
    make_local_energy,
    overlap_squared,
)
from .metrics import JsonlLogger
from .optimizers import build_optimizer, quantum_geometric_tensor
from .sampler import integrated_autocorr_time, sample
from .systems import BaseSystem


@dataclass
class TrainState:
    params: dict
    opt_state: object = None
    history: dict[str, list[float]] = field(
        default_factory=lambda: {
            "energy": [],
            "variance": [],
            "stderr": [],
            "acceptance": [],
            "step_size": [],
            "tau_int": [],
            "grad_norm": [],
            "wall_time": [],
            "iteration": [],
        }
    )


def train(
    logpsi_fn: Callable,
    params: dict,
    system: BaseSystem,
    cfg: RunConfig,
    seed: int,
    ortho_ref: Callable | None = None,
    *,
    outdir: Path | None = None,
    verbose: bool = True,
    checkpoint_every: int = 0,
) -> TrainState:
    cfg.validate()
    key = jax.random.PRNGKey(seed)
    key, k_chain = jax.random.split(key)
    x = jax.random.normal(k_chain, (cfg.sampler.n_chains, system.dim), dtype=jnp.float64)

    local_energy = make_local_energy(logpsi_fn, system.potential, g=system.g)
    opt = build_optimizer(cfg.optimizer)

    state = TrainState(params=params, opt_state=opt.init(params))
    if outdir is not None:
        ckpt_path = outdir / "checkpoint"
        resumed = load_ckpt(ckpt_path)
        if resumed is not None:
            state.params, start_step = resumed
            if verbose:
                print(f"[resume] step {start_step}")

    step_size = cfg.sampler.step_size_init
    n_sweeps = cfg.sampler.n_samples // cfg.sampler.n_chains
    logger = JsonlLogger(outdir / "metrics.jsonl") if outdir is not None else None

    pbar = tqdm(range(cfg.optimizer.n_iter), disable=not verbose, desc=f"[{cfg.optimizer.name}]")

    for it in pbar:
        t0 = time.time()
        key, k_samp = jax.random.split(key)

        samples, s_state = sample(
            logpsi_fn,
            state.params,
            x,
            k_samp,
            n_sweeps=n_sweeps,
            step_size=step_size,
            target_accept=cfg.sampler.target_accept,
            burn_in=cfg.sampler.burn_in if it == 0 else 0,
            adapt_rate=cfg.sampler.adapt_rate,
        )
        x = samples[-cfg.sampler.n_chains :]
        step_size = float(s_state.step_size)

        e_loc = local_energy(state.params, samples)

        # --- DIVERGENCE GUARD ---
        if not bool(jnp.isfinite(e_loc).all()) or jnp.max(jnp.abs(e_loc)) > 1e6:
            pbar.write(f"[divergence] Energy exploded at step {it}. Stopping.")
            break
        # ------------------------

        tau = integrated_autocorr_time(e_loc)
        E, Var, stderr = energy_and_variance(e_loc, tau_int=tau)

        if cfg.optimizer.name == "adam":

            def loss_fn(p, s=samples):
                base = jnp.mean(local_energy(p, s))
                if ortho_ref is not None and cfg.excited.penalty > 0:
                    base = base + cfg.excited.penalty * overlap_squared(logpsi_fn, p, ortho_ref, s)
                return base

            grads = jax.grad(loss_fn)(state.params)
            new_params, state.opt_state = opt.update(state.params, state.opt_state, grads)
            gn = jnp.sqrt(sum(jnp.sum(g * g) for g in jax.tree_util.tree_leaves(grads)))
        else:
            S, F = quantum_geometric_tensor(logpsi_fn, state.params, samples, e_loc)
            new_params, state.opt_state = opt.update(state.params, state.opt_state, (S, F))
            gn = jnp.linalg.norm(F)

        state.params = new_params

        # Save history
        state.history["energy"].append(float(E))
        state.history["variance"].append(float(Var))
        state.history["stderr"].append(float(stderr))
        state.history["acceptance"].append(float(s_state.acc_rate))
        state.history["step_size"].append(float(s_state.step_size))
        state.history["tau_int"].append(float(tau))
        state.history["grad_norm"].append(float(gn))
        state.history["wall_time"].append(time.time() - t0)
        state.history["iteration"].append(it)

        if logger is not None:
            logger.log(
                {
                    "iter": it,
                    "energy": float(E),
                    "variance": float(Var),
                    "stderr": float(stderr),
                    "acceptance": float(s_state.acc_rate),
                    "step_size": float(s_state.step_size),
                    "tau_int": float(tau),
                    "grad_norm": float(gn),
                    "wall_time": time.time() - t0,
                }
            )

        if checkpoint_every > 0 and outdir is not None and (it + 1) % checkpoint_every == 0:
            save_ckpt(outdir / "checkpoint", state.params, it + 1)

        if verbose and it % 20 == 0:
            pbar.set_postfix(
                {
                    "E": f"{float(E):.6f}",
                    "Var": f"{float(Var):.2e}",
                    "Acc": f"{float(s_state.acc_rate):.2f}",
                    "tau": f"{float(tau):.1f}",
                }
            )

    if logger is not None:
        logger.close()
    return state
