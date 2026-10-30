"""Adam and SR behind a common Protocol, with SR solvers, grad clipping,
and learning-rate schedules."""

from __future__ import annotations

from typing import Any, Protocol

import jax
import jax.numpy as jnp
import optax

from . import jax_config  # noqa: F401
from .optimizers_minsr import MatrixFreeSR


class Optimizer(Protocol):
    def init(self, params: dict) -> object: ...
    def update(self, params: dict, state: object, payload: object) -> tuple[dict, object]: ...


def _make_schedule(base_lr: float, warmup_steps: int, schedule: str, n_iter: int):
    if schedule == "constant" and warmup_steps == 0:
        return lambda step: base_lr
    if schedule == "constant":
        return optax.warmup_constant_schedule(
            init_value=0.0,
            peak_value=base_lr,
            warmup_steps=warmup_steps,
        )
    # cosine
    return optax.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=base_lr,
        warmup_steps=warmup_steps,
        decay_steps=max(n_iter, warmup_steps + 1),
        end_value=base_lr * 1e-2,
    )


class AdamOpt:
    def __init__(
        self,
        lr: float,
        clip_norm: float = 0.0,
        schedule: str = "constant",
        warmup_steps: int = 0,
        n_iter: int = 1,
    ):
        tx_chain = [optax.clip_by_global_norm(1.0)]  # <--- Force this
        tx_chain.append(optax.adam(lr))
        self.tx = optax.chain(*tx_chain)
        if clip_norm > 0.0:
            tx_chain.append(optax.clip_by_global_norm(clip_norm))

        # Use a simple constant learning rate if schedule is constant
        if schedule == "constant":
            tx_chain.append(optax.adam(lr))
        else:
            sched = optax.warmup_cosine_decay_schedule(
                init_value=0.0,
                peak_value=lr,
                warmup_steps=warmup_steps,
                decay_steps=max(n_iter, warmup_steps + 1),
                end_value=lr * 1e-2,
            )
            tx_chain.append(optax.adam(sched))

        self.tx = optax.chain(*tx_chain)

    def init(self, params: dict) -> optax.OptState:
        return self.tx.init(params)

    def update(self, params, state, grads):
        updates, state = self.tx.update(grads, state, params)
        return optax.apply_updates(params, updates), state


class SRUpdater:
    def __init__(
        self,
        lr: float,
        diag_shift: float = 1e-3,
        use_diag: bool = False,
        solver: str = "solve",
        clip_norm: float = 0.0,
    ):
        self.lr = lr
        self.diag_shift = diag_shift
        self.use_diag = use_diag
        self.solver = solver
        self.clip_norm = clip_norm

    def init(self, params: dict) -> None:
        return None

    def _solve(self, S: jnp.ndarray, F: jnp.ndarray) -> jnp.ndarray:
        if self.use_diag:
            return F / (jnp.diag(S) + self.diag_shift)
        A = S + self.diag_shift * jnp.eye(S.shape[0])
        if self.solver == "solve":
            return jnp.linalg.solve(A, F)
        if self.solver == "lstsq":
            x, *_ = jnp.linalg.lstsq(A, F, rcond=1e-8)
            return x
        # pinv — safest fallback
        return jnp.linalg.pinv(A, rcond=1e-8) @ F

    def update(self, params, state, S_F):
        S, F = S_F
        delta = -self.lr * self._solve(S, F)
        if self.clip_norm > 0.0:
            n = jnp.linalg.norm(delta)
            delta = delta * jnp.minimum(1.0, self.clip_norm / (n + 1e-30))
        return _apply_flat_update(params, delta), None


def _apply_flat_update(params: dict, delta: jnp.ndarray) -> dict:
    leaves, treedef = jax.tree_util.tree_flatten(params)
    out, i = [], 0
    for leaf in leaves:
        n = leaf.size
        out.append(delta[i : i + n].reshape(leaf.shape))
        i += n
    return jax.tree_util.tree_unflatten(treedef, out)


def quantum_geometric_tensor(logpsi_fn, params, samples, e_loc):
    """Return ``(S, F)`` for the SR update. Memory: ``O(N·P)``."""
    leaves, treedef = jax.tree_util.tree_flatten(params)
    shapes = [leaf.shape for leaf in leaves]
    p0 = jnp.concatenate([leaf.ravel() for leaf in leaves])

    def logpsi_flat(pflat, x):
        ps, i = [], 0
        for s in shapes:
            n = int(jnp.prod(jnp.asarray(s)))
            ps.append(pflat[i : i + n].reshape(s))
            i += n
        return logpsi_fn(jax.tree_util.tree_unflatten(treedef, ps), x)

    def O_fn(pflat, x):
        return jax.jacrev(lambda p: logpsi_flat(p, x.reshape(1, -1))[0])(pflat)

    O = jax.vmap(O_fn, in_axes=(None, 0))(p0, samples)  # noqa: E741
    O_c = O - O.mean(0)
    E_c = e_loc - e_loc.mean()
    S = (O_c.T @ O_c) / samples.shape[0]
    F = (O_c.T @ E_c) / samples.shape[0]
    return S, F


def build_optimizer(cfg: Any) -> Optimizer:
    if cfg.name == "adam":
        return AdamOpt(
            lr=cfg.lr,
            clip_norm=cfg.clip_norm,
            schedule=cfg.schedule,
            warmup_steps=cfg.warmup_steps,
            n_iter=cfg.n_iter,
        )
    if cfg.name == "sr":
        return SRUpdater(
            lr=cfg.lr,
            diag_shift=cfg.diag_shift,
            use_diag=cfg.use_diag,
            solver=cfg.sr_solver,
            clip_norm=cfg.clip_norm,
        )
    if cfg.name == "minsr":
        return MatrixFreeSR(lr=cfg.lr, diag_shift=cfg.diag_shift)
    raise ValueError(f"Unknown optimizer {cfg.name}")
