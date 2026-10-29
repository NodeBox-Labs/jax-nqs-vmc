from __future__ import annotations

import numpy as np
from scipy.sparse import diags, identity, kron
from scipy.sparse.linalg import eigsh


def two_boson_energy_grid(
    omega: float = 1.0, g: float = 1.0, L: float = 8.0, n: int = 200, k: int = 2
):
    x = np.linspace(-L, L, n)
    dx = x[1] - x[0]
    main = np.full(n, 1.0 / dx**2)
    off = np.full(n - 1, -0.5 / dx**2)
    T = diags([off, main, off], offsets=[-1, 0, 1], format="csr")
    V = diags([0.5 * omega**2 * x**2], offsets=[0], format="csr")
    H1 = (T + V).tocsr()
    H = kron(H1, identity(n)) + kron(identity(n), H1)
    # Build 2D operator g δ(x1-x2) via mask diag
    X, Y = np.meshgrid(x, x, indexing="ij")
    mask = np.exp(-0.5 * ((X - Y) / dx) ** 2)  # smeared delta (weight ~1)
    mask = mask / (mask.sum(axis=1, keepdims=True) * dx)  # normalize columns
    Hint = diags([mask.ravel()], offsets=[0], format="csr")
    H = H + g * Hint
    vals, vecs = eigsh(H.tocsr(), k=k, which="SM")
    order = np.argsort(vals)
    vals, vecs = vals[order], vecs[:, order]
    return vals, x, vecs
