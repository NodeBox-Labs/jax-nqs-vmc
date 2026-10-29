"""Matrix-Free Stochastic Reconfiguration (CG and MinSR)."""

from __future__ import annotations
from typing import Callable, Tuple

import jax
import jax.numpy as jnp
import optax


def _cg_solve(matvec: Callable, b: jnp.ndarray, x0: jnp.ndarray, tol: float = 1e-5, max_iter: int = 100) -> jnp.ndarray:
    """Conjugate Gradient solver."""
    def body_fun(val):
        x, r, p, rsold, i = val
        Ap = matvec(p)
        alpha = rsold / (jnp.dot(p, Ap) + 1e-30)
        x_new = x + alpha * p
        r_new = r - alpha * Ap
        rsnew = jnp.dot(r_new, r_new)
        beta = rsnew / (rsold + 1e-30)
        p_new = r_new + beta * p
        return x_new, r_new, p_new, rsnew, i + 1

    def cond_fun(val):
        x, r, p, rsold, i = val
        return (jnp.sqrt(rsold) > tol) & (i < max_iter)

    r0 = b - matvec(x0)
    rsold0 = jnp.dot(r0, r0)
    p0 = r0
    x, _, _, _, _ = jax.lax.while_loop(cond_fun, body_fun, (x0, r0, p0, rsold0, 0))
    return x


class MatrixFreeSR:
    def __init__(self, lr: float = 0.05, diag_shift: float = 1e-3, max_iter: int = 100):
        self.lr = lr
        self.diag_shift = diag_shift
        self.max_iter = max_iter

    def init(self, params: dict) -> None:
        return None

    def update(self, params, state, S_F):
        # Here, S_F is not a matrix. It is (O, F) where O is shape (N, P)
        # and F is shape (P,). 
        O, F = S_F
        
        # O is the Jacobian: O_{i, \theta} = \partial \log \psi(x_i) / \partial \theta
        O_centered = O - jnp.mean(O, axis=0)
        F_centered = F - jnp.mean(F)
        
        # S is implicitly defined. S * v = (O^T (O v)) / N + diag_shift * v
        n_samples = O.shape[0]
        def matvec(v):
            Ov = jnp.dot(O_centered, v)
            return jnp.dot(O_centered.T, Ov) / n_samples + self.diag_shift * v

        # Solve S delta = F
        b = jnp.dot(O_centered.T, F_centered) / n_samples
        delta = _cg_solve(matvec, b, jnp.zeros_like(b), max_iter=self.max_iter)
        
        # Apply update
        delta = -self.lr * delta
        return _apply_flat_update(params, delta), None

def _apply_flat_update(params: dict, delta: jnp.ndarray) -> dict:
    leaves, treedef = jax.tree_util.tree_flatten(params)
    out, i = [], 0
    for leaf in leaves:
        n = leaf.size
        out.append(delta[i:i + n].reshape(leaf.shape))
        i += n
    return jax.tree_util.tree_unflatten(treedef, out)