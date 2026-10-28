"""Autograd local energy, correlation-aware statistics, and the orthogonality
penalty. All numerical guards from Phase 1 live here."""

from __future__ import annotations

from collections.abc import Callable

import jax
import jax.numpy as jnp

from . import jax_config  # noqa: F401 — enables x64

#: exp(2·logψ) is clamped to exp(60) ≈ 1e26 to stay well inside float64.
_LOG_SQ_CLAMP: float = 60.0


def make_local_energy(
    logpsi_fn: Callable[[dict, jnp.ndarray], jnp.ndarray],
    potential_fn: Callable[[jnp.ndarray], jnp.ndarray],
    g: float = 0.0,
) -> Callable[[dict, jnp.ndarray], jnp.ndarray]:
    """Return ``local_energy(params, x)`` of shape ``(N,)``:

        H ψ / ψ = -½ tr(Hess log ψ) - ½ |∇ log ψ|² + V(x) + g |ψ|²

    The ``g |ψ|²`` term is included only when ``g != 0``. Its argument is
    clamped in the log domain to prevent overflow from early-training blowups.
    """

    def _single(params, xi):
        def logpsi_scalar(flat):
            return logpsi_fn(params, flat.reshape(1, -1))[0]

        grad = jax.grad(logpsi_scalar)(xi)
        hess = jax.hessian(logpsi_scalar)(xi)
        kinetic = -0.5 * (jnp.trace(hess) + jnp.sum(grad * grad))
        e = kinetic + potential_fn(xi.reshape(1, -1))[0]
        if g != 0.0:
            log_psi2 = 2.0 * logpsi_scalar(xi)
            log_psi2 = jnp.clip(log_psi2, max=_LOG_SQ_CLAMP)
            e = e + g * jnp.exp(log_psi2)
        return e

    return jax.vmap(lambda p, xb: jax.vmap(lambda xi: _single(p, xi))(xb),
                    in_axes=(None, 0))


def energy_and_variance(
    e_loc: jnp.ndarray,
    tau_int: jnp.ndarray | None = None,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Return ``(E, Var, stderr)``.

    When ``tau_int`` is provided, the standard error is multiplied by
    ``sqrt(2 * tau_int)`` — the standard correction for a correlated chain
    with estimator ``Var / n``. Without it, the returned ``stderr`` is the
    uncorrelated estimate and *understates* the true error by a factor of
    roughly ``sqrt(2 tau_int)``.
    """
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
    """Self-normalized estimate of ``|<ψ_ref|ψ>|² / (‖ψ_ref‖² ‖ψ‖²)``.

    Uses self-normalized importance sampling:

        r_i = exp(log ψ_ref(x_i) - log ψ(x_i)),  x_i ~ |ψ|²
        F = |E[r]|² / E[r²]

    This is the only estimator that is bounded in ``[0, 1]`` by construction.
    The Phase 1 audit flagged the previous estimator ``|E[r]|²`` as biased.
    """
    log_psi = logpsi_fn(params, samples)
    log_ref = ref_logpsi(samples)
    ratio = jnp.exp(log_ref - log_psi)
    num = jnp.abs(jnp.mean(ratio)) ** 2
    den = jnp.mean(ratio ** 2)
    return num / jnp.where(den < 1e-30, 1e-30, den)