from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp


@dataclass
class GrossPitaevskii:
    """1D GPE in a harmonic trap, g > 0 (repulsive).

    Local energy: −½ ψ''/ψ + ½ω²x² + g|ψ|².  The eigenvalue is the chemical
    potential μ, and Var(E_L) → 0 at the mean-field solution.
    """

    omega: float = 1.0
    g: float = 1.0
    dim: int = 1

    @property
    def potential(self):
        omega = self.omega

        def V(x):
            return 0.5 * omega**2 * jnp.sum(x**2, axis=-1)

        return V
