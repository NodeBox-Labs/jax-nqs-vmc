from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp


@dataclass
class TwoBosons:
    """Two identical bosons in a 1D harmonic trap with contact interaction.

    Hamiltonian: −½(∂²₁+∂²₂) + ½(x₁²+x₂²) + g δ(x₁−x₂)

    We use the standard regularized contact (finite grid in the reference
    solver; the NQS ansatz includes the exact cusp −g/2·|x₁−x₂| in the Jastrow).
    """
    omega: float = 1.0
    g: float = 1.0
    N: int = 2
    dim: int = 1

    @property
    def potential(self):
        omega = self.omega
        def V(x):
            return 0.5 * omega**2 * jnp.sum(x**2, axis=-1)
        return V