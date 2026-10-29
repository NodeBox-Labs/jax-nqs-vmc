"""Transformer (Psiformer-style) ansatz for continuous space."""

from __future__ import annotations

from collections.abc import Sequence

import flax.linen as nn
import jax.numpy as jnp


class TransformerBlock(nn.Module):
    embed_dim: int
    num_heads: int = 4

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        # Self-attention
        attn = nn.MultiHeadDotProductAttention(
            num_heads=self.num_heads, qkv_features=self.embed_dim
        )(x, x)
        x = nn.LayerNorm()(x + attn)
        # Feed-forward
        ff = nn.Dense(self.embed_dim * 2)(x)
        ff = nn.gelu(ff)
        ff = nn.Dense(self.embed_dim)(ff)
        x = nn.LayerNorm()(x + ff)
        return x


class TransformerAnsatz(nn.Module):
    hidden: Sequence[int] = (64, 64)
    embed_dim: int = 64
    num_layers: int = 2

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        # x is shape (N, n_particles, 1) -> flatten to (N, n_particles)
        n_particles = x.shape[-2]
        h = x.reshape(x.shape[0], n_particles, -1)

        # Embed each particle
        h = nn.Dense(self.embed_dim)(h)

        # Apply transformer layers
        for _ in range(self.num_layers):
            h = TransformerBlock(self.embed_dim)(h)

        # Global pooling (sum over particles)
        h = jnp.sum(h, axis=1)

        # Final MLP to scalar log|psi|
        for w in self.hidden:
            h = nn.gelu(nn.Dense(w)(h))
        return nn.Dense(1)(h).squeeze(-1)
