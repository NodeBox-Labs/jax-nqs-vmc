"""Typed configuration with cross-field validation.

Every constraint in this module is either (a) checked in ``__post_init__``, or
(b) checked in :meth:`RunConfig.validate`. There is no constraint enforced
only by documentation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml


def _require(cond: bool, msg: str) -> None:
    if not cond:
        raise ValueError(msg)


def _as_tuple(x, name: str) -> tuple[int, ...]:
    if isinstance(x, int):
        x = (x,)
    _require(len(x) > 0, f"{name} must be non-empty.")
    for v in x:
        _require(isinstance(v, int) and v > 0, f"{name} entries must be positive ints; got {v!r}.")
    return tuple(int(v) for v in x)


@dataclass
class AnsatzConfig:
    type: str = "even"
    hidden: tuple[int, ...] = (64, 64)
    activation: str = "tanh"
    pair_hidden: tuple[int, ...] = (32, 32)
    cusp: float = 0.0

    def __post_init__(self) -> None:
        valid_types = (
            "mlp", "even", "odd", "deepsets", "combo",
            "autoregressive", "transformer",
        )
        _require(
            self.type in valid_types,
            f"ansatz.type must be one of {valid_types}; got {self.type!r}.",
        )
        self.hidden = _as_tuple(self.hidden, "ansatz.hidden")
        self.pair_hidden = _as_tuple(self.pair_hidden, "ansatz.pair_hidden")
        _require(
            self.activation in ("tanh", "gelu", "silu"),
            f"ansatz.activation must be tanh|gelu|silu; got {self.activation!r}.",
        )
        _require(self.cusp >= 0.0, "ansatz.cusp must be >= 0.")


@dataclass
class SamplerConfig:
    n_chains: int = 128
    n_samples: int = 8192
    step_size_init: float = 0.5
    target_accept: float = 0.55
    burn_in: int = 200
    adapt_rate: float = 0.005

    def __post_init__(self) -> None:
        _require(self.n_chains > 0, "sampler.n_chains must be > 0.")
        _require(self.n_samples > 0, "sampler.n_samples must be > 0.")
        _require(
            self.n_samples % self.n_chains == 0,
            f"sampler.n_samples ({self.n_samples}) must be divisible by "
            f"sampler.n_chains ({self.n_chains}).",
        )
        _require(0.0 < self.target_accept < 1.0, "target_accept must be in (0, 1).")
        _require(self.step_size_init > 0.0, "step_size_init must be > 0.")
        _require(self.adapt_rate > 0.0, "adapt_rate must be > 0.")
        _require(self.burn_in >= 0, "burn_in must be >= 0.")


@dataclass
class OptimizerConfig:
    name: str = "adam"
    lr: float = 1e-3
    n_iter: int = 400
    diag_shift: float = 1e-3
    use_diag: bool = False
    sr_solver: str = "solve"  # solve | lstsq | pinv
    clip_norm: float = 0.0  # 0 disables clipping
    warmup_steps: int = 0
    schedule: str = "constant"  # constant | cosine

    def __post_init__(self) -> None:
        _require(
            self.name in ("adam", "sr", "minsr"),
            f"optimizer.name must be adam|sr|minsr; got {self.name!r}.",
        )
        _require(self.lr > 0.0, "optimizer.lr must be > 0.")
        _require(self.n_iter > 0, "optimizer.n_iter must be > 0.")
        _require(self.diag_shift > 0.0, "optimizer.diag_shift must be > 0.")
        _require(
            self.sr_solver in ("solve", "lstsq", "pinv"),
            f"optimizer.sr_solver must be solve|lstsq|pinv; got {self.sr_solver!r}.",
        )
        _require(self.clip_norm >= 0.0, "optimizer.clip_norm must be >= 0.")
        _require(self.warmup_steps >= 0, "optimizer.warmup_steps must be >= 0.")
        _require(
            self.schedule in ("constant", "cosine"),
            f"optimizer.schedule must be constant|cosine; got {self.schedule!r}.",
        )


@dataclass
class ReferenceConfig:
    L: float = 10.0
    n_grid: int = 2048

    def __post_init__(self) -> None:
        _require(self.L > 0.0, "reference.L must be > 0.")
        _require(
            self.n_grid > 0 and self.n_grid % 2 == 0,
            "reference.n_grid must be a positive even integer.",
        )


@dataclass
class ExcitedConfig:
    use_ortho: bool = False
    level: int = 0
    penalty: float = 0.0

    def __post_init__(self) -> None:
        _require(self.level >= 0, "excited.level must be >= 0.")
        _require(self.penalty >= 0.0, "excited.penalty must be >= 0.")
        if self.use_ortho:
            _require(self.penalty > 0.0, "excited.use_ortho=True requires excited.penalty > 0.")


@dataclass
class RunConfig:
    system: str = "harmonic"
    omega: float = 1.0
    g: float = 0.0
    dim: int = 1
    a: float = -0.5
    b: float = 0.25
    ansatz: AnsatzConfig = field(default_factory=AnsatzConfig)
    sampler: SamplerConfig = field(default_factory=SamplerConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    reference: ReferenceConfig = field(default_factory=ReferenceConfig)
    excited: ExcitedConfig = field(default_factory=ExcitedConfig)
    seeds: tuple[int, ...] = (0, 1, 2)

    def __post_init__(self) -> None:
        _require(
            self.system in ("harmonic", "harmonic2d", "double_well", "gpe", "two_bosons", "three_bosons"),
            f"system must be harmonic|harmonic2d|double_well|gpe|two_bosons|three_bosons; got {self.system!r}.",
        )
        _require(self.dim >= 1, "dim must be >= 1.")
        _require(len(self.seeds) > 0, "seeds must be non-empty.")
        for s in self.seeds:
            _require(isinstance(s, int) and s >= 0, f"seeds must be non-negative ints; got {s!r}.")
        self.seeds = tuple(int(s) for s in self.seeds)
        
        # DeepSets cusp must equal g
        if self.ansatz.type == "deepsets":
            _require(
                abs(self.ansatz.cusp - self.g) < 1e-8,
                f"DeepSets requires ansatz.cusp == system.g; "
                f"got cusp={self.ansatz.cusp}, g={self.g}.",
            )

    def validate(self) -> None:
        """Re-run all checks; useful after mutating fields at runtime."""
        self.__post_init__()
        self.ansatz.__post_init__()
        self.sampler.__post_init__()
        self.optimizer.__post_init__()
        self.reference.__post_init__()
        self.excited.__post_init__()

    @classmethod
    def from_yaml(cls, path: Path) -> RunConfig:
        with open(path) as f:
            raw = yaml.safe_load(f) or {}
        raw["ansatz"] = AnsatzConfig(**_norm(raw.get("ansatz", {})))
        raw["sampler"] = SamplerConfig(**raw.get("sampler", {}))
        raw["optimizer"] = OptimizerConfig(**raw.get("optimizer", {}))
        raw["reference"] = ReferenceConfig(**raw.get("reference", {}))
        raw["excited"] = ExcitedConfig(**raw.get("excited", {}))
        raw["seeds"] = tuple(raw.get("seeds", cls.seeds))
        return cls(**raw)


def _norm(d: dict) -> dict:
    out = dict(d)
    for k in ("hidden", "pair_hidden"):
        if k in out and isinstance(out[k], list):
            out[k] = tuple(out[k])
    return out