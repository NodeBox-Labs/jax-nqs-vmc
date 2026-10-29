import jax.numpy as jnp

from nqs_vmc.local_energy import overlap_squared


def test_overlap_equals_one_for_identical():
    samples = jnp.linspace(-3, 3, 500)[:, None]

    def logpsi(p, x):
        return -0.5 * jnp.sum(x**2, axis=-1)

    F = overlap_squared(logpsi, {}, lambda x: logpsi({}, x), samples)
    assert 0.99 < float(F) < 1.01


def test_overlap_equals_zero_for_distant():
    samples = jnp.linspace(-3, 3, 500)[:, None]

    def logpsi(p, x):
        return -0.5 * jnp.sum(x**2, axis=-1)

    def ref(x):
        return -0.5 * jnp.sum((x - 100.0) ** 2, axis=-1)

    F = overlap_squared(logpsi, {}, ref, samples)
    assert float(F) < 1e-6
