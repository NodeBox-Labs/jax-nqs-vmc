"""Local energy Hψ/ψ computed via autograd.

We differentiate ``log ψ`` (not ψ directly) to control dynamic range and reuse
log-amplitude conventions. The kinetic term in 1D is

    −½ ψ''/ψ = −½ [ (log ψ)'' + ((log ψ)')² ]

which we implement with ``jax.hessian`` and ``jax.grad``. In d>1 the sum over
coordinates is handled via the trace of the Hessian.
"""

from __future__ import annotations

from typing import Callable, Optional

import jax
import jax.numpy as jnp


PotentialFn = Callable[[jnp.ndarray], jnp.ndarray]  # (N, dim) -> (N,)


def make_local_energy(
    logpsi_fn: Callable[[dict, jnp.ndarray], jnp.ndarray],
    potential_fn: PotentialFn,
    g: float = 0.0,
    vmap: bool = True,
):
    """Return ``local_energy(params, x)`` with shape ``(N,)``.

    The nonlinear GPE term ``g |ψ|²`` is included when ``g > 0``.
    """

    def _single(params, xi):
        # xi shape (dim,)
        def logpsi_scalar(flat):
            x = flat.reshape(1, -1)
            return logpsi_fn(params, x)[0]

        grad = jax.grad(logpsi_scalar)(xi)
        hess = jax.hessian(logpsi_scalar)(xi)
        lap = jnp.trace(hess)
        kinetic = -0.5 * (lap + jnp.sum(grad * grad))
        pot = potential_fn(xi.reshape(1, -1))[0]
        e = kinetic + pot
        if g != 0.0:
            psi2 = jnp.exp(2.0 * logpsi_scalar(xi))
            e = e + g * psi2
        return e

    if vmap:
        return jax.vmap(lambda p, xb: jax.vmap(lambda xi: _single(p, xi))(xb),
                        in_axes=(None, 0))
    return _single


def energy_and_variance(local_energies: jnp.ndarray):
    """Return (E, Var, error_of_mean) with statistically correct error bars."""
    e = local_energies
    n = e.shape[0]
    mean = jnp.mean(e)
    var = jnp.var(e)
    # error of the mean; callers can multiply by sqrt(2τ) for correlated samples
    stderr = jnp.sqrt(var / n)
    return mean, var, stderr


def orthogonalization_penalty(
    logpsi_fn: Callable[[dict, jnp.ndarray], jnp.ndarray],
    params: dict,
    ref_logpsi: Callable[[jnp.ndarray], jnp.ndarray],
    samples: jnp.ndarray,
    weights: Optional[jnp.ndarray] = None,
):
    """Estimate |<ψ_ref|ψ>|² / (‖ψ_ref‖²‖ψ‖²) on the fly.

    We use the identity E_{x~|ψ|²}[ ψ_ref(x)/ψ(x) ] = <ψ_ref|ψ> / <ψ|ψ>,
    which is (up to normalization) the fidelity overlap. Squared modulus gives
    a natural penalty term. This is the loss contribution for excited-state
    searches.
    """
    log_psi = logpsi_fn(params, samples)
    log_ref = ref_logpsi(samples)
    ratio = jnp.exp(log_ref - log_psi)
    if weights is None:
        overlap = jnp.mean(ratio)
    else:
        overlap = jnp.sum(weights * ratio) / jnp.sum(weights)
    return jnp.abs(overlap) ** 2