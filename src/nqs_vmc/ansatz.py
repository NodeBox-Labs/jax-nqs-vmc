"""Neural-network wavefunction ansätze.

Convention: all ansätze return ``log ψ(x)`` (complex or real). For real,
parity-symmetric problems we keep ``log|ψ|`` real and handle signs via
explicit symmetry factors.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import flax.linen as nn
import jax.numpy as jnp

from .ansatz_autoregressive import AutoregressiveMLP
from .ansatz_transformer import TransformerAnsatz
from .config import AnsatzConfig


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
        # Zero init the output layer
        return nn.Dense(1, kernel_init=nn.initializers.zeros)(h).squeeze(-1)


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
    """Permutation-symmetric ansatz for N particles in 1D."""

    hidden: Sequence[int] = (64, 64)
    pair_hidden: Sequence[int] = (32, 32)
    activation: str = "tanh"
    cusp: float = 0.0

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        act = _act(self.activation)
        n = x.shape[-2]

        h = x
        for w in self.hidden:
            h = act(nn.Dense(w, kernel_init=nn.initializers.zeros)(h))
        phi = nn.Dense(1, kernel_init=nn.initializers.zeros)(h).squeeze(-1)
        single = jnp.sum(phi, axis=-1)

        iu = jnp.triu_indices(n, k=1)
        xi = x[..., iu[0], 0]
        xj = x[..., iu[1], 0]
        r = jnp.abs(xi - xj)
        p = r[..., None]
        for w in self.pair_hidden:
            p = act(nn.Dense(w)(p))
        pair = nn.Dense(1, kernel_init=nn.initializers.zeros)(p).squeeze(-1)
        pair = pair - 0.5 * self.cusp * r
        pairs = jnp.sum(pair, axis=-1)

        z = (single + pairs)[..., None]
        for w in self.hidden:
            z = act(nn.Dense(w)(z))
        return nn.Dense(1, kernel_init=nn.initializers.zeros)(z).squeeze(-1)


class LinearCombination(nn.Module):
    """ψ = ψ_0 + α ψ_1 for excited-state searches."""

    hidden: Sequence[int] = (64, 64)
    activation: str = "tanh"
    odd: bool = True

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        even = ParityEven(self.hidden, self.activation)(x)
        odd_net = ParityOdd(self.hidden, self.activation)(x)
        alpha = self.param("alpha", nn.initializers.zeros, ())
        a = jnp.exp(even)
        b = jnp.sign(x.squeeze(-1) + 1e-30) * jnp.exp(odd_net)
        psi = a + jnp.exp(alpha) * b
        return jnp.log(jnp.abs(psi) + 1e-30)


ANSATZ_REGISTRY: dict[str, type[nn.Module]] = {
    "mlp": MLP,
    "even": ParityEven,
    "odd": ParityOdd,
    "deepsets": DeepSets,
    "combo": LinearCombination,
    "autoregressive": AutoregressiveMLP,
    "transformer": TransformerAnsatz,
}

REGISTRY = ANSATZ_REGISTRY


def build_ansatz(cfg: dict[str, Any] | AnsatzConfig) -> nn.Module:
    if isinstance(cfg, AnsatzConfig):
        kind = cfg.type
        kwargs: dict[str, Any] = {
            "hidden": tuple(cfg.hidden),
            "activation": cfg.activation,
        }
        if cfg.type == "deepsets":
            kwargs["pair_hidden"] = tuple(cfg.pair_hidden)
            kwargs["cusp"] = cfg.cusp
        elif cfg.type == "transformer":
            kwargs["num_layers"] = 2
    else:
        kind = cfg["type"]
        kwargs = {k: v for k, v in cfg.items() if k != "type"}
        if isinstance(kwargs.get("hidden"), list):
            kwargs["hidden"] = tuple(kwargs["hidden"])
        if isinstance(kwargs.get("pair_hidden"), list):
            kwargs["pair_hidden"] = tuple(kwargs["pair_hidden"])
    return ANSATZ_REGISTRY[kind](**kwargs)