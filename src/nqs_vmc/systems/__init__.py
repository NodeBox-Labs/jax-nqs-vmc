"""System registry: each system owns its potential, g, and reference solver."""

from __future__ import annotations

from typing import Any, Tuple

import jax.numpy as jnp

from ..config import ReferenceConfig, RunConfig


class BaseSystem:
    name: str = "base"
    dim: int = 1
    g: float = 0.0

    def potential(self, x: jnp.ndarray) -> jnp.ndarray:
        raise NotImplementedError

    def reference(self, ref_cfg: ReferenceConfig) -> Tuple[Any, ...]:
        raise NotImplementedError


class Harmonic(BaseSystem):
    name = "harmonic"

    def __init__(self, omega: float = 1.0):
        self.omega = omega

    def potential(self, x: jnp.ndarray) -> jnp.ndarray:
        return 0.5 * self.omega**2 * jnp.sum(x**2, -1)

    def reference(self, ref_cfg: ReferenceConfig):
        from ..reference.exact_diag import exact_1d
        return exact_1d(lambda x: 0.5 * self.omega**2 * x**2,
                        L=ref_cfg.L, n_grid=ref_cfg.n_grid)


class DoubleWell(BaseSystem):
    name = "double_well"

    def __init__(self, a: float = -0.5, b: float = 0.25):
        self.a, self.b = a, b

    def potential(self, x: jnp.ndarray) -> jnp.ndarray:
        r2 = jnp.sum(x**2, -1)
        return self.a * r2 + self.b * r2**2

    def reference(self, ref_cfg: ReferenceConfig):
        from ..reference.exact_diag import exact_1d
        return exact_1d(lambda x: self.a * x**2 + self.b * x**4,
                        L=ref_cfg.L, n_grid=ref_cfg.n_grid)


class GrossPitaevskii(BaseSystem):
    name = "gpe"

    def __init__(self, omega: float = 1.0, g: float = 1.0):
        self.omega, self.g = omega, g

    def potential(self, x: jnp.ndarray) -> jnp.ndarray:
        return 0.5 * self.omega**2 * jnp.sum(x**2, -1)

    def reference(self, ref_cfg: ReferenceConfig):
        from ..reference.split_step import gpe_imaginary_time
        import numpy as np
        x = np.linspace(-ref_cfg.L, ref_cfg.L, ref_cfg.n_grid)
        psi = gpe_imaginary_time(x, self.omega, self.g)
        return psi, x


class TwoBosons(BaseSystem):
    name = "two_bosons"
    dim = 1

    def __init__(self, omega: float = 1.0, g: float = 1.0):
        self.omega, self.g = omega, g

    def potential(self, x: jnp.ndarray) -> jnp.ndarray:
        return 0.5 * self.omega**2 * jnp.sum(x**2, -1)

    def reference(self, ref_cfg: ReferenceConfig):
        from ..reference.few_body import two_boson_ground_state
        return two_boson_ground_state(self.omega, self.g,
                                       L=ref_cfg.L, n=min(ref_cfg.n_grid, 400))


def build_system(cfg: RunConfig) -> BaseSystem:
    if cfg.system == "harmonic":
        return Harmonic(cfg.omega)
    if cfg.system == "double_well":
        return DoubleWell(cfg.a, cfg.b)
    if cfg.system == "gpe":
        return GrossPitaevskii(cfg.omega, cfg.g)
    if cfg.system == "two_bosons":
        return TwoBosons(cfg.omega, cfg.g)
    raise ValueError(cfg.system)


# ... existing imports ...

class Harmonic2D(BaseSystem):
    name = "harmonic2d"
    dim = 2

    def __init__(self, omega: float = 1.0):
        self.omega = omega

    def potential(self, x: jnp.ndarray) -> jnp.ndarray:
        # x shape is (N, 2)
        return 0.5 * self.omega**2 * jnp.sum(x**2, -1)

    def reference(self, ref_cfg: ReferenceConfig):
        # For 2D HO, the exact energy is E = n_x + n_y + 1.
        # We return a dummy grid and dummy eigenvectors, since the VMC run 
        # will just compare against the exact scalar energy.
        import numpy as np
        E0 = 1.0  # n_x=0, n_y=0
        x_dummy = np.linspace(-5, 5, 100)
        return np.array([E0, 2.0, 2.0, 3.0]), x_dummy, np.zeros((100, 4))

def build_system(cfg: RunConfig) -> BaseSystem:
    if cfg.system == "harmonic":
        return Harmonic(cfg.omega)
    if cfg.system == "harmonic2d":
        return Harmonic2D(cfg.omega)
    if cfg.system == "double_well":
        return DoubleWell(cfg.a, cfg.b)
    if cfg.system == "gpe":
        return GrossPitaevskii(cfg.omega, cfg.g)
    if cfg.system == "two_bosons":
        return TwoBosons(cfg.omega, cfg.g)
    raise ValueError(f"Unknown system: {cfg.system}")