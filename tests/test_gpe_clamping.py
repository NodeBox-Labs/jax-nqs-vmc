import jax.numpy as jnp

from nqs_vmc.local_energy import make_local_energy


def test_gpe_clamps_overflow():
    def logpsi(p, x):
        return jnp.full(x.shape[:-1], 100.0)  # exp(200) overflows float64

    le = make_local_energy(logpsi, lambda x: 0.0, g=1.0)
    e = le({}, jnp.linspace(-1, 1, 5)[:, None])
    assert jnp.isfinite(e).all(), "GPE local energy overflowed"
