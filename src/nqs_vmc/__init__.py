"""NQS-VMC: neural quantum states with variational Monte Carlo in JAX."""

import jax

# x64 is essential: excited-state gaps and tunneling splittings are tiny.
jax.config.update("jax_enable_x64", True)

__version__ = "0.1.0"