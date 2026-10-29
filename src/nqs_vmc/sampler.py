"""Metropolis–Hastings with adaptive step size."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import jax
import jax.numpy as jnp

from . import jax_config  # noqa: F401


@dataclass
class SamplerState:
    x: jnp.ndarray
    step_size: jnp.ndarray
    acc_rate: jnp.ndarray


def sample(
    logpsi_fn: Callable,
    params: dict,
    x0: jnp.ndarray,
    key: jnp.ndarray,
    n_sweeps: int,
    step_size: float,
    target_accept: float,
    burn_in: int,
    adapt_rate: float,
) -> tuple[jnp.ndarray, SamplerState]:
    """Return (samples, final_state)."""

    n_chains = x0.shape[0]

    # Pre-generate all random numbers
    total_steps = burn_in + n_sweeps
    keys = jax.random.split(key, total_steps)

    # Initial state
    x = x0
    sigma = jnp.asarray(step_size, dtype=jnp.float64)
    acc_rate = jnp.asarray(0.5, dtype=jnp.float64)

    def step_fn(i, carry):
        x, sigma, acc_rate = carry
        k = keys[i]
        k_prop, k_acc = jax.random.split(k)

        # Propose
        noise = jax.random.normal(k_prop, x.shape, dtype=jnp.float64)
        proposal = x + sigma * noise

        # Acceptance probability
        log_ratio = 2.0 * (logpsi_fn(params, proposal) - logpsi_fn(params, x))
        log_u = jnp.log(jax.random.uniform(k_acc, (n_chains,), dtype=jnp.float64))
        accept = log_u < log_ratio

        # Update x
        x_new = jnp.where(accept[:, None], proposal, x)

        # Update step size (only during burn-in)
        current_acc = jnp.where(i < burn_in, jnp.mean(accept), 0.0)
        new_acc_rate = jnp.where(i < burn_in, 0.99 * acc_rate + 0.01 * current_acc, acc_rate)
        new_sigma = jnp.where(
            i < burn_in,
            sigma * jnp.exp(adapt_rate * (new_acc_rate - target_accept)),
            sigma,
        )

        return x_new, new_sigma, new_acc_rate

    # Run burn-in to find good step size
    x_burned, sigma_final, acc_final = jax.lax.fori_loop(0, burn_in, step_fn, (x, sigma, acc_rate))

    # Now run production and record the trajectory
    def record_step(i, carry):
        x, sigma, acc_rate, _ = carry  # unpack 4 elements
        k = keys[burn_in + i]
        k_prop, k_acc = jax.random.split(k)

        noise = jax.random.normal(k_prop, x.shape, dtype=jnp.float64)
        proposal = x + sigma * noise
        log_ratio = 2.0 * (logpsi_fn(params, proposal) - logpsi_fn(params, x))
        log_u = jnp.log(jax.random.uniform(k_acc, (n_chains,), dtype=jnp.float64))
        accept = log_u < log_ratio
        x_new = jnp.where(accept[:, None], proposal, x)

        # We only record x_new, step size and acc_rate stay fixed
        return x_new, sigma, acc_rate, x_new

    # Run production from the burned-in state
    _, _, _, traj = jax.lax.fori_loop(
        0, n_sweeps, record_step, (x_burned, sigma_final, acc_final, x_burned)
    )

    # Extract samples (traj shape is (n_sweeps, n_chains, dim))
    samples = traj.reshape(-1, x0.shape[-1])
    state = SamplerState(x=x_burned, step_size=sigma_final, acc_rate=acc_final)

    return samples, state


def integrated_autocorr_time(values: jnp.ndarray, c: float = 5.0) -> jnp.ndarray:
    """Sokal's automatic-windowing estimator."""
    values = jnp.asarray(values)
    x = values - jnp.mean(values)
    var = jnp.mean(x * x)
    var = jnp.where(var < 1e-30, 1e-30, var)
    n = values.shape[0]
    max_lag = min(n // 2, 2000)

    def acf(lag):
        # JAX vmap cannot handle dynamic slicing (x[:n-lag]).
        # Instead, we pad x with zeros at the front by 'lag' elements,
        # then use a static slice to get the shifted version.
        x_padded = jnp.pad(x, (lag, 0), mode='constant')
        return jnp.sum(x_padded[:n] * x) / ((n - lag) * var)

    acfs = jax.vmap(acf)(jnp.arange(max_lag))
    tau = 1.0 + 2.0 * jnp.cumsum(acfs)
    m = jnp.arange(max_lag) + 1
    idx = jnp.argmax(m >= c * tau)
    return tau[idx]
