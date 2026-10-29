"""Orbax-backed checkpointing. Thin wrapper so an Orbax version bump touches
one file."""

from __future__ import annotations

from pathlib import Path

import orbax.checkpoint as ocp


def save(path: Path, params: dict, step: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ckptr = ocp.PyTreeCheckpointer()
    ckptr.save(str(path), {"params": params, "step": step}, force=True)


def load(path: Path) -> tuple[dict, int] | None:
    if not path.exists():
        return None
    ckptr = ocp.PyTreeCheckpointer()
    restored = ckptr.restore(str(path))
    return restored["params"], int(restored["step"])


def exists(path: Path) -> bool:
    return path.exists()
