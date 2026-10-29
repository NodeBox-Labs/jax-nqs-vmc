"""Metropolis–Hastings with adaptive step size."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Tuple

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
) -> Tuple[jnp.ndarray, SamplerState]:
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
        new_sigma = jnp.where(i < burn_in, sigma * jnp.exp(adapt_rate * (new_acc_rate - target_accept)), sigma)
        
        return x_new, new_sigma, new_acc_rate

    # Run all steps
    x_final, sigma_final, acc_final = jax.lax.fori_loop(
        0, total_steps, step_fn, (x, sigma, acc_rate)
    )
    
    # We need to record the trajectory, so we'll do a second pass to collect samples
    # (This is slightly slower but 100% correct)
    def record_step(i, carry):
        x, sigma, acc_rate = carry
        k = keys[i]
        k_prop, k_acc = jax.random.split(k)
        
        noise = jax.random.normal(k_prop, x.shape, dtype=jnp.float64)
        proposal = x + sigma * noise
        log_ratio = 2.0 * (logpsi_fn(params, proposal) - logpsi_fn(params, x))
        log_u = jnp.log(jax.random.uniform(k_acc, (n_chains,), dtype=jnp.float64))
        accept = log_u < log_ratio
        x_new = jnp.where(accept[:, None], proposal, x)
        
        return x_new, sigma, acc_rate, x_new

    # Restart from x0, but re-run the same keys to get the trajectory
    x_final2, _, _, traj = jax.lax.fori_loop(
        0, total_steps, record_step, (x0, jnp.asarray(step_size), jnp.asarray(0.5), x0)
    )
    
    # Extract samples after burn-in
    samples = traj[burn_in:].reshape(-1, x0.shape[-1])
    state = SamplerState(x=x_final, step_size=sigma_final, acc_rate=acc_final)
    
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
        return jnp.sum(x[: n - lag] * x[lag:]) / ((n - lag) * var)

    acfs = jax.vmap(acf)(jnp.arange(max_lag))
    tau = 1.0 + 2.0 * jnp.cumsum(acfs)
    m = jnp.arange(max_lag) + 1
    idx = jnp.argmax(m >= c * tau)
    return tau[idx]