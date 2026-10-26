"""Metropolis–Hastings sampler with adaptive step size.

The sampler is a pure function: given a wavefunction, an initial state, and a
PRNG key, it runs ``n_sweeps`` sweeps and returns the collected samples along
with diagnostic information (acceptance rate, step size trajectory).

For efficiency, all chains evolve in lockstep, and we use ``jax.lax.scan`` so
the whole sampling loop is jitted end to end.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import jax
import jax.numpy as jnp


LogPsi = Callable[[dict, jnp.ndarray], jnp.ndarray]  # (params, x) -> log ψ (real)


@dataclass
class SamplerState:
    x: jnp.ndarray          # (n_chains, dim)
    step_size: jnp.ndarray  # scalar
    acc_rate: jnp.ndarray   # scalar (running)
    key: jnp.ndarray


def _metropolis_step(logpsi_fn, params, state: SamplerState, target_accept: float):
    key, k_prop, k_acc, k_u = jax.random.split(state.key, 4)

    noise = jax.random.normal(k_prop, state.x.shape)
    proposal = state.x + state.step_size * noise

    log_old = logpsi_fn(params, state.x)
    log_new = logpsi_fn(params, proposal)
    log_ratio = 2.0 * (log_new - log_old)  # log|ψ|^2

    log_u = jnp.log(jax.random.uniform(k_u, log_ratio.shape))
    accept = log_u < log_ratio

    x_new = jnp.where(accept[..., None], proposal, state.x)
    acc_rate = 0.99 * state.acc_rate + 0.01 * jnp.mean(accept)

    # step-size adaptation (log-domain, kept small per step to avoid instability)
    delta = 0.005 * (acc_rate - target_accept)
    step_size = state.step_size * jnp.exp(delta)

    new_state = SamplerState(
        x=x_new, step_size=step_size, acc_rate=acc_rate, key=key
    )
    return new_state, accept


def sample(
    logpsi_fn: LogPsi,
    params: dict,
    x0: jnp.ndarray,
    key: jnp.ndarray,
    n_sweeps: int,
    step_size: float = 0.5,
    target_accept: float = 0.55,
    burn_in: int = 200,
    thin: int = 1,
):
    """Run Metropolis–Hastings and return samples of shape (n_samples, dim)."""
    state = SamplerState(
        x=x0,
        step_size=jnp.asarray(step_size, dtype=jnp.float64),
        acc_rate=jnp.asarray(0.5, dtype=jnp.float64),
        key=key,
    )

    def body(carry, _):
        return _metropolis_step(logpsi_fn, params, carry, target_accept)

    # burn-in
    state, _ = jax.lax.scan(body, state, None, length=burn_in)

    # production: store every `thin`-th configuration
    def prod_body(carry, _):
        carry, acc = body(carry, None)
        return carry, carry.x

    state, traj = jax.lax.scan(prod_body, state, None, length=n_sweeps)
    samples = traj.reshape(-1, traj.shape[-1])[::thin]
    return samples, state


def integrated_autocorr_time(values: jnp.ndarray, c: float = 5.0) -> jnp.ndarray:
    """Integrated autocorrelation time via Sokal's window.

    ``values`` shape (T,) — a scalar observable.
    """
    x = values - jnp.mean(values)
    var = jnp.mean(x * x)
    var = jnp.where(var < 1e-30, 1e-30, var)
    n = values.shape[0]

    def acf(lag):
        return jnp.sum(x[: n - lag] * x[lag:]) / ((n - lag) * var)

    max_lag = min(n // 2, 2000)
    acfs = jax.vmap(acf)(jnp.arange(max_lag))
    # Sokal window: stop when M >= c * tau
    tau = 1.0 + 2.0 * jnp.cumsum(acfs)
    m = jnp.arange(max_lag) + 1
    idx = jnp.where(m >= c * tau, size=max_lag, fill_value=max_lag).astype(jnp.int32)
    return jnp.where(idx[0] < max_lag, tau[jnp.minimum(idx[0], max_lag - 1)], tau[-1])