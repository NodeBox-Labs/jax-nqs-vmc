from __future__ import annotations

from typing import Callable, Optional, Tuple

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
        # xi is shape (dim,). We must reshape to (1, dim) for the network.
        dim = xi.shape[0]
        x_2d = xi.reshape(1, dim)
        
        def logpsi_scalar(x_flat):
            # hessian passes a flat array, we reshape back to (1, dim)
            return logpsi_fn(params, x_flat.reshape(1, dim))[0]

        grad = jax.grad(logpsi_scalar)(xi)
        hess = jax.hessian(logpsi_scalar)(xi)
        
        # For dim=1, hessian returns a scalar (0-dim array), so trace fails.
        # We use jnp.trace only if dim > 1, otherwise the Hessian is just the scalar.
        if dim == 1:
            lap = hess
        else:
            lap = jnp.trace(hess)
            
        kinetic = -0.5 * (lap + jnp.sum(grad * grad))
        e = kinetic + potential_fn(x_2d)[0]
        
        if g != 0.0:
            log_psi2 = 2.0 * logpsi_scalar(xi)
            log_psi2 = jnp.clip(log_psi2, max=_LOG_SQ_CLAMP)
            e = e + g * jnp.exp(log_psi2)
        return e

    return jax.vmap(lambda p, xb: jax.vmap(lambda xi: _single(p, xi))(xb), in_axes=(None, 0))


def energy_and_variance(
    e_loc: jnp.ndarray,
    tau_int: Optional[jnp.ndarray] = None,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
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
    den = jnp.mean(ratio ** 2)
    return num / jnp.where(den < 1e-30, 1e-30, den)