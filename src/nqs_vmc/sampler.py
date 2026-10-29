"""Metropolis–Hastings with adaptive step size."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Tuple

import jax
import jax.numpy as jnp

from . import jax_config  # noqa: F401


@jax.tree_util.register_dataclass
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
    
    # PRE-GENERATE ALL RANDOM NUMBERS OUTSIDE THE SCAN
    # This prevents lax.scan from corrupting the PRNG key.
    total_steps = burn_in + n_sweeps
    keys = jax.random.split(key, total_steps)
    
    state = SamplerState(
        x=x0,
        step_size=jnp.asarray(step_size, dtype=jnp.float64),
        acc_rate=jnp.asarray(0.5, dtype=jnp.float64),
    )

    def _step(c, k):
        k_prop, k_acc = jax.random.split(k)
        
        # Draw proposal
        noise = jax.random.normal(k_prop, c.x.shape, dtype=jnp.float64)
        proposal = c.x + c.step_size * noise
        
        # Metropolis acceptance ratio
        log_ratio = 2.0 * (logpsi_fn(params, proposal) - logpsi_fn(params, c.x))
        
        # Accept/Reject
        log_u = jnp.log(jax.random.uniform(k_acc, log_ratio.shape, dtype=jnp.float64))
        accept = log_u < log_ratio
        
        x_new = jnp.where(accept[..., None], proposal, c.x)
        
        # Update step size
        acc_rate = 0.99 * c.acc_rate + 0.01 * jnp.mean(accept)
        step_size = c.step_size * jnp.exp(adapt_rate * (acc_rate - target_accept))
        
        new_c = SamplerState(x_new, step_size, acc_rate)
        return new_c, new_c.x

    # Run the full chain
    state, traj = jax.lax.scan(_step, state, keys)
    
    # Discard burn-in, reshape output
    samples = traj[burn_in:].reshape(-1, traj.shape[-1])
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