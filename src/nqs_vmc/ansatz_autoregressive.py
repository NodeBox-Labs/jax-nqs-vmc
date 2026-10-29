"""Autoregressive Neural Quantum State (exact sampling, no MCMC)."""

from __future__ import annotations

from collections.abc import Sequence

import flax.linen as nn
import jax
import jax.numpy as jnp
from jax.scipy.special import logsumexp


class MaskedDense(nn.Module):
    features: int
    mask: jnp.ndarray

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        kernel = self.param("kernel", nn.initializers.lecun_normal(), (x.shape[-1], self.features))
        bias = self.param("bias", nn.initializers.zeros, (self.features,))
        # Apply the mask to enforce the autoregressive property
        return jnp.dot(x, kernel * self.mask) + bias


class AutoregressiveMLP(nn.Module):
    hidden: Sequence[int] = (64, 64)
    n_particles: int = 1
    n_bins: int = 16  # Discretize space for autoregressive modelling
    L: float = 10.0

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        # x is shape (N, n_particles), discretized into n_bins
        # Returns log_probabilities of shape (N, n_particles, n_bins)
        n = self.n_particles
        h = x

        # To enforce autoregression, we mask the weights so that the prediction for
        # particle i only sees particles < i.
        for w in self.hidden:
            # Create a strictly lower triangular mask
            mask = jnp.tril(jnp.ones((h.shape[-1], w)), k=-1)
            h = nn.tanh(MaskedDense(w, mask)(h))

        # Output layer: predict bins for each particle
        out_mask = jnp.tril(jnp.ones((h.shape[-1], n * self.n_bins)), k=-1)
        out = MaskedDense(n * self.n_bins, out_mask)(h)

        # Reshape to (N, n_particles, n_bins) and normalize over bins
        logits = out.reshape(-1, n, self.n_bins)
        return logits - logsumexp(logits, axis=-1, keepdims=True)

    def sample(self, params, key, n_samples):
        """Exact autoregressive sampling."""

        def _sample_one(key):
            x = jnp.zeros(self.n_particles, dtype=jnp.int32)

            def _sample_step(i, val):
                x_so_far, k = val
                k, subk = jax.random.split(k)
                # Mask the input to only see previous particles
                x_masked = jnp.where(jnp.arange(self.n_particles) < i, x_so_far, 0)
                logits = self.apply(params, x_masked[None, :])[0, i]
                # Sample from categorical
                new_x = jax.random.categorical(subk, logits)
                x_new = x_so_far.at[i].set(new_x)
                return x_new, k

            x, _ = jax.lax.fori_loop(0, self.n_particles, _sample_step, (x, key))
            return x

        keys = jax.random.split(key, n_samples)
        return jax.vmap(_sample_one)(keys)
