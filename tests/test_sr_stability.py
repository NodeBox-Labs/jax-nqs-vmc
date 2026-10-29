import jax.numpy as jnp
import numpy as np

from nqs_vmc.optimizers import SRUpdater


def test_sr_solvers_are_finite_on_singular_S():
    # Rank-1 S is singular.
    v = jnp.ones((4,), dtype=jnp.float64)
    S = jnp.outer(v, v)
    F = jnp.array([1.0, 0.0, 0.0, 0.0], dtype=jnp.float64)
    for solver in ("solve", "lstsq", "pinv"):
        sr = SRUpdater(lr=1.0, diag_shift=1e-3, solver=solver)
        # Set params to a matching-shape tree
        params = {"w": jnp.zeros((4,), dtype=jnp.float64)}
        new_params, _ = sr.update(params, None, (S, F))
        assert np.isfinite(np.asarray(new_params["w"])).all(), f"{solver} produced NaN"
