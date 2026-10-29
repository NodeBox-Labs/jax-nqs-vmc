import jax.numpy as jnp

from nqs_vmc.local_energy import energy_and_variance


def test_tau_int_correction_inflates_error():
    rng = jnp.arange(1000, dtype=jnp.float64) * 0.001
    e = rng + 0.1 * jnp.sin(rng)
    _, _, s0 = energy_and_variance(e)
    _, _, s1 = energy_and_variance(e, tau_int=jnp.asarray(4.0))
    assert s1 > s0
    assert jnp.allclose(s1 / s0, jnp.sqrt(8.0), rtol=1e-6)
