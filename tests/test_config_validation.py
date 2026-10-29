import pytest

from nqs_vmc.config import (
    AnsatzConfig,
    OptimizerConfig,
    ReferenceConfig,
    RunConfig,
    SamplerConfig,
)


def test_samples_divisible_by_chains():
    with pytest.raises(ValueError, match="divisible"):
        SamplerConfig(n_chains=100, n_samples=8192)


def test_cusp_must_match_g():
    with pytest.raises(ValueError, match="cusp"):
        RunConfig(system="two_bosons", g=1.0, ansatz=AnsatzConfig(type="deepsets", cusp=0.5))


def test_empty_seeds_rejected():
    with pytest.raises(ValueError, match="seeds"):
        RunConfig(seeds=())


def test_unknown_optimizer_rejected():
    with pytest.raises(ValueError, match="optimizer.name"):
        OptimizerConfig(name="lbfgs")


def test_sr_solver_rejected():
    with pytest.raises(ValueError, match="sr_solver"):
        OptimizerConfig(sr_solver="cholesky")


def test_odd_grid_rejected():
    with pytest.raises(ValueError, match="n_grid"):
        ReferenceConfig(n_grid=1025)


def test_ortho_requires_penalty():
    from nqs_vmc.config import ExcitedConfig

    with pytest.raises(ValueError, match="penalty"):
        ExcitedConfig(use_ortho=True, penalty=0.0)
