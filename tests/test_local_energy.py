import jax.numpy as jnp
import numpy as np

from nqs_vmc.local_energy import make_local_energy


def test_local_energy_gaussian():
    # ψ(x) ∝ exp(-x²/2) is exact ground state of HO with E = 0.5
    def logpsi(params, x):
        return -0.5 * jnp.sum(x**2, axis=-1)

    def V(x):
        return 0.5 * jnp.sum(x**2, axis=-1)

    le = make_local_energy(logpsi, V)
    x = jnp.linspace(-1.5, 1.5, 20)[:, None]
    e = le({}, x)
    np.testing.assert_allclose(e, 0.5, atol=1e-10)