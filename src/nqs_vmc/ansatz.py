"""Neural-network wavefunction ansätze.

Convention: all ansätze return ``log ψ(x)`` (complex or real). For real,
parity-symmetric problems we keep ``log|ψ|`` real and handle signs via
explicit symmetry factors.
"""

from __future__ import annotations

from collections.abc import Sequence

import flax.linen as nn
import jax.numpy as jnp


def _act(name: str):
    if name == "tanh":
        return nn.tanh
    if name == "gelu":
        return nn.gelu
    if name == "silu":
        return nn.silu
    raise ValueError(f"unknown activation {name!r}")


class MLP(nn.Module):
    """Plain MLP mapping x (..., 1) -> log|ψ| (...,)."""

    hidden: Sequence[int] = (64, 64)
    activation: str = "tanh"

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        act = _act(self.activation)
        h = x
        for width in self.hidden:
            h = act(nn.Dense(width)(h))
        return nn.Dense(1)(h).squeeze(-1)


class ParityEven(nn.Module):
    """Even-parity ansatz: MLP on |x|. Guarantees ψ(-x)=ψ(x)."""

    hidden: Sequence[int] = (64, 64)
    activation: str = "tanh"

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        return MLP(self.hidden, self.activation)(jnp.abs(x))


class ParityOdd(nn.Module):
    """Odd-parity ansatz: sign(x) * exp(MLP(|x|)). log|ψ| = MLP(|x|) + log|x|."""

    hidden: Sequence[int] = (64, 64)
    activation: str = "tanh"

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        inner = MLP(self.hidden, self.activation)(jnp.abs(x))
        return inner + jnp.log(jnp.abs(x).squeeze(-1) + 1e-12)


class DeepSets(nn.Module):
    """Permutation-symmetric ansatz for N particles in 1D.

    log ψ(x_1..x_N) = ρ( Σ_i φ(x_i), Σ_{i<j} J(|x_i - x_j|) )

    With a learned Jastrow-like pair function J for the contact cusp.
    """

    hidden: Sequence[int] = (64, 64)
    pair_hidden: Sequence[int] = (32, 32)
    activation: str = "tanh"
    cusp: float = 0.0  # contact interaction strength g -> exact cusp -g/2 |x_i-x_j|

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        # x: (..., N, 1)
        act = _act(self.activation)
        n = x.shape[-2]

        # single-particle embedding
        h = x
        for w in self.hidden:
            h = act(nn.Dense(w)(h))
        phi = nn.Dense(1)(h).squeeze(-1)  # (..., N)
        single = jnp.sum(phi, axis=-1)  # (...,)

        # pair embedding (upper triangle)
        iu = jnp.triu_indices(n, k=1)
        xi = x[..., iu[0], 0]
        xj = x[..., iu[1], 0]
        r = jnp.abs(xi - xj)
        p = r[..., None]
        for w in self.pair_hidden:
            p = act(nn.Dense(w)(p))
        pair = nn.Dense(1)(p).squeeze(-1)
        # exact cusp: exp(-g/2 |x_i - x_j|) -> log contribution -g/2 * |x_i-x_j|
        pair = pair - 0.5 * self.cusp * r
        pairs = jnp.sum(pair, axis=-1)

        z = (single + pairs)[..., None]
        for w in self.hidden:
            z = act(nn.Dense(w)(z))
        return nn.Dense(1)(z).squeeze(-1)


class LinearCombination(nn.Module):
    """ψ = ψ_0 + α ψ_1 for excited-state searches; α is a learnable scalar."""

    hidden: Sequence[int] = (64, 64)
    activation: str = "tanh"
    odd: bool = True

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        even = ParityEven(self.hidden, self.activation)(x)
        odd_net = ParityOdd(self.hidden, self.activation)(x)
        alpha = self.param("alpha", nn.initializers.zeros, ())
        # log|ψ| for the combination — used only when we know signs don't flip
        a = jnp.exp(even)
        b = jnp.sign(x.squeeze(-1) + 1e-30) * jnp.exp(odd_net)
        psi = a + jnp.exp(alpha) * b
        return jnp.log(jnp.abs(psi) + 1e-30)


ANSATZ_REGISTRY = {
    "mlp": MLP,
    "even": ParityEven,
    "odd": ParityOdd,
    "deepsets": DeepSets,
    "combo": LinearCombination,
}


def build_ansatz(cfg: dict) -> nn.Module:
    kind = cfg["type"]
    kwargs = {k: v for k, v in cfg.items() if k != "type"}
    if isinstance(kwargs.get("hidden"), list):
        kwargs["hidden"] = tuple(kwargs["hidden"])
    if isinstance(kwargs.get("pair_hidden"), list):
        kwargs["pair_hidden"] = tuple(kwargs["pair_hidden"])
    return ANSATZ_REGISTRY[kind](**kwargs)