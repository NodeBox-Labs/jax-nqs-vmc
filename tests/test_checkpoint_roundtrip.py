"""Windows-safe checkpointing using flax.serialization."""

from __future__ import annotations

import pickle
from pathlib import Path


def save(path: Path, params: dict, step: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path.with_suffix(".ckpt"), "wb") as f:
        pickle.dump({"params": params, "step": step}, f)


def load(path: Path) -> tuple[dict, int] | None:
    p = path.with_suffix(".ckpt")
    if not p.exists():
        return None
    with open(p, "rb") as f:
        data = pickle.load(f)
    return data["params"], data["step"]


def exists(path: Path) -> bool:
    return path.with_suffix(".ckpt").exists()
