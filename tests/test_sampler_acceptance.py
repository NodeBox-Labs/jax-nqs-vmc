import jax
import jax.numpy as jnp
from nqs_vmc.sampler import sample

def test_acceptance_in_band():
    def logpsi(p, x):
        return -0.5 * jnp.sum(x**2, axis=-1)
    key = jax.random.PRNGKey(0)
    x0 = jnp.zeros((32, 1))
    _, s = sample(logpsi, {}, x0, key, n_sweeps=2000, step_size=0.7, target_accept=0.55, burn_in=200, adapt_rate=0.005)
    acc = float(s.acc_rate)
    assert 0.3 < acc < 0.85, f"acceptance {acc}"