"""Adam and Stochastic Reconfiguration (natural gradient).

SR: at each step, compute

    S_ij = <O_i O_j> - <O_i><O_j>,   F_i = <E_L O_i> - <E_L><O_i>,
    δθ = -lr * (S + ε I)^{-1} F

where O_i = ∂ log ψ / ∂θ_i. This is the quantum Fisher information metric
(natural gradient / imaginary-time propagation).
"""

from __future__ import annotations

from typing import Callable

import jax
import jax.numpy as jnp
import optax


def _flatten(tree):
    leaves, treedef = jax.tree_util.tree_flatten(tree)
    return jnp.concatenate([l.ravel() for l in leaves]), treedef


def _unflatten(vec, treedef):
    shapes = [l.shape for l in jax.tree_util.tree_leaves(treedef)]
    # rebuild
    leaves = []
    i = 0
    for leaf in jax.tree_util.tree_leaves(jax.tree_util.tree_unflatten(treedef, [None])):
        pass
    raise NotImplementedError  # replaced below


def make_adam(params, lr: float = 1e-3):
    opt = optax.adam(lr)
    state = opt.init(params)
    return opt, state


class SRUpdater:
    """Stochastic Reconfiguration with diagonal-shift regularization."""

    def __init__(self, lr: float = 0.05, diag_shift: float = 1e-3, use_diag: bool = False):
        self.lr = lr
        self.diag_shift = diag_shift
        self.use_diag = use_diag

    def step(self, params, grads, S, F):
        if self.use_diag:
            F = F / (jnp.diag(S) + self.diag_shift)
        else:
            F = jnp.linalg.solve(
                S + self.diag_shift * jnp.eye(S.shape[0]), F
            )
        delta = -self.lr * F
        # walk the tree structure back
        leaves, treedef = jax.tree_util.tree_flatten(params)
        sizes = [l.size for l in leaves]
        out, i = [], 0
        for leaf, s in zip(leaves, sizes):
            out.append(delta[i:i + s].reshape(leaf.shape))
            i += s
        new_params = jax.tree_util.tree_unflatten(treedef, out)
        return new_params, delta


def quantum_geometric_tensor(
    logpsi_fn: Callable[[dict, jnp.ndarray], jnp.ndarray],
    params: dict,
    samples: jnp.ndarray,
    local_energies: jnp.ndarray,
):
    """Return (S, F) as in the SR update.

    Uses ``jax.jacrev`` over the flattened parameter vector for O_i.
    """
    def logpsi_flat(pflat, x):
        params = jax.tree_util.tree_unflatten(treedef, _split(pflat, leaves_shapes))
        return logpsi_fn(params, x)

    leaves, treedef = jax.tree_util.tree_flatten(params)
    leaves_shapes = [l.shape for l in leaves]

    def _split(pflat, shapes):
        out, i = [], 0
        for s in shapes:
            n = 1
            for d in s:
                n *= d
            out.append(pflat[i:i + n].reshape(s))
            i += n
        return out

    p0, _ = jnp.concatenate([l.ravel() for l in leaves]), treedef

    # O_i(x) = ∂ log ψ / ∂θ_i, shape (N, P)
    def O_fn(pflat, x):
        # per-sample O over params: shape (P,)
        def f(p):
            return logpsi_flat(p, x.reshape(1, -1))[0]
        return jax.jacrev(f)(pflat)

    O = jax.vmap(O_fn, in_axes=(None, 0))(p0, samples)  # (N, P)
    O_mean = O.mean(0)
    O_c = O - O_mean
    E_mean = local_energies.mean()
    EL_c = local_energies - E_mean

    S = (O_c.T @ O_c) / samples.shape[0]
    F = (O_c.T @ EL_c) / samples.shape[0]
    return S, F