"""1D exact diagonalization on a fine grid."""

from __future__ import annotations
from typing import Callable, Tuple

import numpy as np
from scipy.sparse import diags
from scipy.sparse.linalg import eigsh


def grid_hamiltonian(V: Callable[[np.ndarray], np.ndarray], x: np.ndarray):
    n = x.size
    dx = x[1] - x[0]
    T = diags([np.full(n - 1, -0.5 / dx**2),
               np.full(n, 1.0 / dx**2),
               np.full(n - 1, -0.5 / dx**2)],
              offsets=[-1, 0, 1], format="csr")
    return (T + diags([V(x)], [0], format="csr")).tocsr()


def exact_1d(V: Callable[[np.ndarray], np.ndarray], L: float = 10.0,
             n_grid: int = 2048, k: int = 6) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = np.linspace(-L, L, n_grid)
    vals, vecs = eigsh(grid_hamiltonian(V, x), k=k, which="SM")
    order = np.argsort(vals)
    vals, vecs = vals[order], vecs[:, order]
    for j in range(vecs.shape[1]):
        i = int(np.argmax(np.abs(vecs[:, j])))
        if vecs[i, j] < 0:
            vecs[:, j] *= -1
    return vals, x, vecs