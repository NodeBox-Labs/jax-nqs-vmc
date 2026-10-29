"""Phase 1 §5: every shipped YAML must still produce the v0.2 numbers."""

from pathlib import Path

import pytest

from nqs_vmc.config import RunConfig

REF_V02 = {
    "ho_ground": 0.5,
    "ho_excited": 1.5,
    "double_well": None,  # ED reference, checked by other tests
    "gross_pitaevskii": None,
    "two_bosons": None,
}


@pytest.mark.slow
@pytest.mark.parametrize("name,target", list(REF_V02.items()))
def test_shipped_config(name: str, target):
    if target is None:
        pytest.skip("energy reference checked by system-specific tests")
    import tempfile

    from scripts.run_vmc import run_seed

    cfg_path = Path("configs") / f"{name}.yaml"
    cfg = RunConfig.from_yaml(cfg_path)
    # reduce cost: 30 iterations, one seed
    cfg.optimizer.n_iter = 30
    cfg.seeds = (0,)
    with tempfile.TemporaryDirectory() as d:
        m = run_seed(cfg, 0, Path(d))
    assert abs(m["final_energy"] - target) < 0.1, (
        f"{name}: energy {m['final_energy']:.4f} drifted from {target}"
    )
