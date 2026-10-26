import jax
import jax.numpy as jnp
import numpy as np

from nqs_vmc.sampler import sample


def test_acceptance_in_band():
    def logpsi(p, x):
        return -0.5 * jnp.sum(x**2, axis=-1)

    key = jax.random.PRNGKey(0)
    x0 = jnp.zeros((32, 1))
    samples, s = sample(logpsi, {}, x0, key, n_sweeps=2000, step_size=0.7, burn_in=200)
    acc = float(s.acc_rate)
    assert 0.3 < acc < 0.8, f"acceptance {acc}"