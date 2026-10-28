import jax
import jax.numpy as jnp
import numpy as np
from scipy.stats import ks_2samp

from nqs_vmc.sampler import sample


def test_sampler_matches_gaussian():
    """Sample from ψ(x) = exp(-x²/2); compare to N(0, 1) via KS."""
    def logpsi(p, x):
        return -0.5 * jnp.sum(x**2, axis=-1)

    key = jax.random.PRNGKey(0)
    samples, _ = sample(logpsi, {}, jnp.zeros((64, 1)), key,
                        n_sweeps=4000, step_size=0.7, target_accept=0.55,
                        burn_in=500, adapt_rate=0.005)
    s = np.asarray(samples).ravel()
    ref = np.random.default_rng(1).standard_normal(s.size)
    _, p = ks_2samp(s, ref)
    assert p > 0.01, f"KS p={p:.3e}; sampler does not match target"