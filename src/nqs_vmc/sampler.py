"""Metropolis–Hastings with adaptive step size and Sokal autocorrelation.

The sampler is a pure function of (logψ, params, x0, key). All randomness
comes from ``key``; there is no global RNG. The full sampling loop is jitted
once per (logψ, params shape); shape-locked via ``donate_argnums``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Tuple

import jax
import jax.numpy as jnp

from . import jax_config  # noqa: F401 — enables x64


@dataclass
class SamplerState:
    x: jnp.ndarray
    step_size: jnp.ndarray
    acc_rate: jnp.ndarray
    key: jnp.ndarray


def _mh_step(logpsi_fn, params, state: SamplerState,
             target_accept: float, adapt_rate: float) -> SamplerState:
    key, k_prop, k_u = jax.random.split(state.key, 3)
    proposal = state.x + state.step_size * jax.random.normal(k_prop, state.x.shape)
    log_ratio = 2.0 * (logpsi_fn(params, proposal) - logpsi_fn(params, state.x))
    accept = jnp.log(jax.random.uniform(k_u, log_ratio.shape)) < log_ratio
    x_new = jnp.where(accept[..., None], proposal, state.x)
    acc_rate = 0.99 * state.acc_rate + 0.01 * jnp.mean(accept)
    step_size = state.step_size * jnp.exp(adapt_rate * (acc_rate - target_accept))
    return SamplerState(x_new, step_size, acc_rate, key)


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
    """Return ``(samples, final_state)``.

    ``samples`` has shape ``(n_sweeps * n_chains, dim)`` — one row per walker
    per sweep, flattened. ``final_state`` carries the adapted step size and
    running acceptance rate for the next call.
    """
    state = SamplerState(
        x=x0,
        step_size=jnp.asarray(step_size, dtype=jnp.float64),
        acc_rate=jnp.asarray(0.5, dtype=jnp.float64),
        key=key,
    )

    def burn(c, _):
        return _mh_step(logpsi_fn, params, c, target_accept, adapt_rate), None

    def prod(c, _):
        c = _mh_step(logpsi_fn, params, c, target_accept, adapt_rate)
        return c, c.x

    if burn_in > 0:
        state, _ = jax.lax.scan(burn, state, None, length=burn_in)
    state, traj = jax.lax.scan(prod, state, None, length=n_sweeps)
    return traj.reshape(-1, traj.shape[-1]), state


def integrated_autocorr_time(values: jnp.ndarray, c: float = 5.0) -> jnp.ndarray:
    """Sokal's automatic-windowing estimator for the integrated autocorrelation
    time. Returns a scalar jnp array."""
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