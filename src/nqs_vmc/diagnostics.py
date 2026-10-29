"""Metrics: energy errors, variance, fidelity, autocorrelation, aggregation."""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path

import jax.numpy as jnp
import numpy as np


def fidelity_from_grid(logpsi_fn, params, x_grid: jnp.ndarray, ref_psi_grid: np.ndarray):
    """Overlap between normalized NQS wavefunction and reference on a grid."""
    log_psi = logpsi_fn(params, x_grid[:, None])
    psi = jnp.exp(log_psi - jnp.max(log_psi))
    psi = psi / jnp.sqrt(jnp.trapz(psi**2, x_grid))
    ref = ref_psi_grid / np.sqrt(np.trapz(ref_psi_grid**2, x_grid))
    overlap = jnp.abs(jnp.trapz(psi * ref, x_grid)) ** 2
    return float(overlap)


def aggregate_seeds(results: Iterable[dict]) -> dict:
    results = list(results)

    def _stack(key):
        return np.array([r[key] for r in results])

    keys = [k for k in results[0] if isinstance(results[0][k], (int, float))]
    out = {}
    for k in keys:
        arr = _stack(k)
        out[k] = {"mean": float(arr.mean()), "std": float(arr.std()), "values": arr.tolist()}
    return out


def save_results(outdir: Path, metrics: dict, history: dict):
    outdir.mkdir(parents=True, exist_ok=True)
    with open(outdir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    np.savez(outdir / "history.npz", **{k: np.asarray(v) for k, v in history.items()})
