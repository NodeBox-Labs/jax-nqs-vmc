import jax.numpy as jnp

from nqs_vmc.checkpoint import load, save


def test_checkpoint_roundtrip(tmp_path):
    params = {"w": jnp.ones((3, 4), dtype=jnp.float64)}
    save(tmp_path / "ckpt", params, step=42)
    loaded = load(tmp_path / "ckpt")
    assert loaded is not None
    params2, step2 = loaded
    assert step2 == 42
    assert jnp.allclose(params2["w"], params["w"])