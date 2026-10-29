"""Autograd local energy."""

from __future__ import annotations

from collections.abc import Callable

import jax
import jax.numpy as jnp

from . import jax_config  # noqa: F401

_LOG_SQ_CLAMP: float = 60.0


def make_local_energy(
    logpsi_fn: Callable[[dict, jnp.ndarray], jnp.ndarray],
    potential_fn: Callable[[jnp.ndarray], jnp.ndarray],
    g: float = 0.0,
) -> Callable[[dict, jnp.ndarray], jnp.ndarray]:

    def _single(params, xi):
        # xi is shape (dim,). Reshape to flat for JAX derivatives.
        dim = xi.size

        def logpsi_scalar(x_flat):
            return logpsi_fn(params, x_flat.reshape(1, dim))[0]

        grad = jax.grad(logpsi_scalar)(xi)
        hess = jax.hessian(logpsi_scalar)(xi)

        # dim == 1 produces a scalar Hessian. jnp.trace would crash.
        lap = hess if dim == 1 else jnp.trace(hess)

        kinetic = -0.5 * (lap + jnp.sum(grad * grad))
        e = kinetic + jnp.sum(potential_fn(xi.reshape(1, dim)))

        if g != 0.0:
            log_psi2 = 2.0 * logpsi_scalar(xi)
            log_psi2 = jnp.clip(log_psi2, max=_LOG_SQ_CLAMP)
            e = e + g * jnp.exp(log_psi2)
        return e

    # vmap over particles
    return jax.vmap(lambda p, xb: jax.vmap(lambda xi: _single(p, xi))(xb), in_axes=(None, 0))


def energy_and_variance(
    e_loc: jnp.ndarray,
    tau_int: jnp.ndarray | None = None,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    n = e_loc.shape[0]
    mean = jnp.mean(e_loc)
    var = jnp.var(e_loc)
    stderr = jnp.sqrt(var / n)
    if tau_int is not None:
        stderr = stderr * jnp.sqrt(2.0 * tau_int)
    return mean, var, stderr


def overlap_squared(
    logpsi_fn: Callable[[dict, jnp.ndarray], jnp.ndarray],
    params: dict,
    ref_logpsi: Callable[[jnp.ndarray], jnp.ndarray],
    samples: jnp.ndarray,
) -> jnp.ndarray:
    log_psi = logpsi_fn(params, samples)
    log_ref = ref_logpsi(samples)
    ratio = jnp.exp(log_ref - log_psi)
    num = jnp.abs(jnp.mean(ratio)) ** 2
    den = jnp.mean(ratio**2)
    return num / jnp.where(den < 1e-30, 1e-30, den)
