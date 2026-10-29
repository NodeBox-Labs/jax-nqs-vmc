from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp


@dataclass
class HarmonicOscillator:
    omega: float = 1.0
    dim: int = 1
    g: float = 0.0  # unused; kept for common interface

    @property
    def potential(self):
        omega = self.omega

        def V(x):
            return 0.5 * omega**2 * jnp.sum(x**2, axis=-1)

        return V

    @property
    def exact_energies(self):
        # E_n = omega (n + 1/2); levels indexed from 0
        return lambda n: self.omega * (n + 0.5)
