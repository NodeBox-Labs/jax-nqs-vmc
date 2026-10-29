"""Structured per-iteration logging as JSON Lines."""

from __future__ import annotations

import json
from pathlib import Path


class JsonlLogger:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = open(path, "a", buffering=1)

    def log(self, record: dict[str, float]) -> None:
        self._fh.write(json.dumps(record, separators=(",", ":")) + "\n")

    def close(self) -> None:
        if not self._fh.closed:
            self._fh.close()


def replay(path: Path) -> dict[str, list]:
    """Return a dict-of-columns for plotting."""
    cols: dict[str, list] = {}
    if not path.exists():
        return cols
    with open(path) as f:
        for line in f:
            rec = json.loads(line)
            for k, v in rec.items():
                cols.setdefault(k, []).append(v)
    return cols
