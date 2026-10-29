"""Matplotlib plotting utilities."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from .metrics import replay


def _integrate_trapezoid(y: Any, x: Any) -> float:
    """Trapezoid integration compatible with numpy < 2.0 and >= 2.0."""
    if hasattr(np, "trapezoid"):
        return float(np.trapezoid(y, x))
    return float(np.trapz(y, x))


def plot_convergence(history_path: Path, out_path: Path) -> None:
    """Plot energy and variance vs iteration."""
    data = replay(history_path)
    if not data:
        return

    fig, ax1 = plt.subplots(figsize=(8, 5))

    # Energy on the left axis
    color = "tab:red"
    ax1.set_xlabel("Iteration")
    ax1.set_ylabel("Energy", color=color)
    ax1.plot(data["iteration"], data["energy"], color=color, label="Energy")
    ax1.tick_params(axis="y", labelcolor=color)
    ax1.axhline(y=data["energy"][-1], color="gray", linestyle="--", alpha=0.5)

    # Variance on the right axis
    ax2 = ax1.twinx()
    color = "tab:blue"
    ax2.set_ylabel("Variance", color=color)
    ax2.plot(data["iteration"], data["variance"], color=color, label="Variance")
    ax2.tick_params(axis="y", labelcolor=color)

    fig.tight_layout()
    plt.savefig(out_path, dpi=140)
    plt.close()


def plot_wavefunction(
    logpsi_fn: Callable[[Any, Any], Any],
    params: Any,
    x_grid: Any,
    ref_psi: Any,
    out_path: Path,
) -> None:
    """Plot |psi(x)|^2 for VMC and reference."""
    import jax.numpy as jnp

    log_psi = logpsi_fn(params, jnp.asarray(x_grid)[:, None])
    psi = jnp.exp(log_psi - jnp.max(log_psi))
    psi_np = np.asarray(psi)
    norm = np.sqrt(_integrate_trapezoid(psi_np**2, x_grid))
    psi_norm = psi_np / norm

    ref_norm = np.sqrt(_integrate_trapezoid(ref_psi**2, x_grid))
    ref_psi_norm = ref_psi / ref_norm

    plt.figure(figsize=(8, 5))
    plt.plot(x_grid, psi_norm**2, label="VMC", linewidth=2)
    plt.plot(x_grid, ref_psi_norm**2, label="Exact", linestyle="--", linewidth=2)
    plt.xlabel("x")
    plt.ylabel(r"$|\psi(x)|^2$")
    plt.legend()
    plt.title("Wavefunction Probability Density")
    plt.savefig(out_path, dpi=140)
    plt.close()
