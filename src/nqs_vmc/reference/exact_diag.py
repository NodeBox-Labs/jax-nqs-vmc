"""1D exact diagonalization on a fine grid."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from scipy.sparse import diags
from scipy.sparse.linalg import eigsh


def grid_hamiltonian(V: Callable[[np.ndarray], np.ndarray], x: np.ndarray):
    n = x.size
    dx = x[1] - x[0]
    T = diags(
        [np.full(n - 1, -0.5 / dx**2), np.full(n, 1.0 / dx**2), np.full(n - 1, -0.5 / dx**2)],
        offsets=[-1, 0, 1],
        format="csr",
    )
    return (T + diags([V(x)], [0], format="csr")).tocsr()


def exact_1d(
    V: Callable[[np.ndarray], np.ndarray], L: float = 10.0, n_grid: int = 2048, k: int = 6
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = np.linspace(-L, L, n_grid)
    vals, vecs = eigsh(grid_hamiltonian(V, x), k=k, which="SM")
    order = np.argsort(vals)
    vals, vecs = vals[order], vecs[:, order]
    for j in range(vecs.shape[1]):
        i = int(np.argmax(np.abs(vecs[:, j])))
        if vecs[i, j] < 0:
            vecs[:, j] *= -1
    return vals, x, vecs


def harmonic_exact(
    omega: float = 1.0, n_max: int = 6, L: float = 10.0, n_grid: int = 2048
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Exact diagonalization for 1D harmonic oscillator."""
    return exact_1d(lambda x: 0.5 * omega**2 * x**2, L=L, n_grid=n_grid, k=n_max)


def double_well_exact(
    a: float = -0.5, b: float = 0.25, n_max: int = 6, L: float = 8.0, n_grid: int = 2048
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Exact diagonalization for 1D double well potential V(x) = a*x^2 + b*x^4."""
    return exact_1d(lambda x: a * x**2 + b * x**4, L=L, n_grid=n_grid, k=n_max)
