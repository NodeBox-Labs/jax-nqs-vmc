from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp


@dataclass
class DoubleWell:
    a: float = -0.5
    b: float = 0.25
    dim: int = 1
    g: float = 0.0

    @property
    def potential(self):
        a, b = self.a, self.b
        def V(x):
            r2 = jnp.sum(x**2, axis=-1)
            return a * r2 + b * r2**2
        return V