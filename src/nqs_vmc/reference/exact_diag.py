"""Exact diagonalization on a fine 1D grid with scipy.sparse.linalg.eigsh.

Discretization: 3-point stencil for the kinetic operator; Dirichlet BC at the
box edges (L large enough that the wavefunction is exponentially small).
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from scipy.sparse import diags
from scipy.sparse.linalg import eigsh


def _grid_hamiltonian_1d(V: Callable[[np.ndarray], np.ndarray], x: np.ndarray):
    n = x.size
    dx = x[1] - x[0]
    main = np.full(n, 1.0 / dx**2)
    off = np.full(n - 1, -0.5 / dx**2)
    T = diags([off, main, off], offsets=[-1, 0, 1], format="csr")
    Vd = diags([V(x)], offsets=[0], format="csr")
    return (T + Vd).tocsr()


def diagonalize(
    V: Callable[[np.ndarray], np.ndarray],
    L: float = 10.0,
    n_grid: int = 2048,
    k: int = 6,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (eigenvalues ascending, grid, eigenvectors columns)."""
    x = np.linspace(-L, L, n_grid)
    H = _grid_hamiltonian_1d(V, x)
    vals, vecs = eigsh(H, k=k, which="SM")
    order = np.argsort(vals)
    vals, vecs = vals[order], vecs[:, order]
    # sign convention: positive at max |ψ|
    for j in range(vecs.shape[1]):
        i = np.argmax(np.abs(vecs[:, j]))
        if vecs[i, j] < 0:
            vecs[:, j] = -vecs[:, j]
    return vals, x, vecs


def harmonic_exact(n_max: int = 6, L: float = 10.0, n_grid: int = 2048):
    return diagonalize(lambda x: 0.5 * x**2, L=L, n_grid=n_grid, k=n_max)


def double_well_exact(a: float = -0.5, b: float = 0.25, n_max: int = 6,
                      L: float = 8.0, n_grid: int = 2048):
    return diagonalize(lambda x: a * x**2 + b * x**4, L=L, n_grid=n_grid, k=n_max)