"""Entry point: run VMC for a given YAML config."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import jax
import jax.numpy as jnp

from nqs_vmc.ansatz import build_ansatz
from nqs_vmc.config import RunConfig
from nqs_vmc.diagnostics import aggregate_seeds, fidelity_from_grid, save_results
from nqs_vmc.reference.exact_diag import double_well_exact, harmonic_exact
from nqs_vmc.systems import build_system
from nqs_vmc.trainer import train


def get_reference(cfg: RunConfig):
    """Return exact-diag reference (vals, x, vecs) for 1-D systems, or None."""
    ref = cfg.reference
    if cfg.system == "harmonic":
        return harmonic_exact(omega=cfg.omega, n_max=6, L=ref.L, n_grid=ref.n_grid)
    if cfg.system == "double_well":
        return double_well_exact(a=cfg.a, b=cfg.b, n_max=6, L=ref.L, n_grid=ref.n_grid)
    return None  # GPE / few-bosons handled separately


def run_seed(cfg: RunConfig, seed: int, outdir: Path) -> dict:
    """Run one VMC seed; returns a metrics dict."""
    system = build_system(cfg)
    ansatz = build_ansatz(cfg.ansatz)

    def logpsi(params, x):
        return ansatz.apply(params, x)

    rng = jax.random.PRNGKey(seed)
    dummy = jnp.zeros((1, system.dim))
    params = ansatz.init(rng, dummy)

    # Build orthogonalisation reference for excited-state runs
    ortho_ref = None
    if cfg.excited.use_ortho:
        ref_result = get_reference(cfg)
        if ref_result is not None:
            _, ref_x, ref_vecs = ref_result
            psi0 = ref_vecs[:, 0]
            grid = jnp.asarray(ref_x)
            psi0_j = jnp.asarray(psi0)

            def ortho_ref(x):  # type: ignore[misc]
                return jnp.interp(x[:, 0], grid, psi0_j)

    outdir.mkdir(parents=True, exist_ok=True)
    state = train(logpsi, params, system, cfg, seed, ortho_ref, outdir=outdir, verbose=True)

    # Build summary metrics dict
    metrics = {
        "seed": seed,
        "final_energy": state.history["energy"][-1] if state.history["energy"] else float("nan"),
        "final_variance": state.history["variance"][-1] if state.history["variance"] else float("nan"),
        "final_acceptance": state.history["acceptance"][-1] if state.history["acceptance"] else 0.0,
        "tau_int": state.history["tau_int"][-1] if state.history["tau_int"] else 0.0,
        "wall_time_s": sum(state.history["wall_time"]),
    }

    ref = get_reference(cfg)
    if ref is not None:
        ref_vals, ref_x, ref_vecs = ref
        idx = cfg.excited.level
        metrics["reference_energy"] = float(ref_vals[idx])
        metrics["energy_error"] = float(state.history["energy"][-1] - ref_vals[idx])
        metrics["fidelity"] = fidelity_from_grid(
            logpsi, state.params, jnp.asarray(ref_x), ref_vecs[:, idx]
        )

    save_results(outdir, metrics, state.history)

    from nqs_vmc.plotting import plot_convergence

    plot_convergence(outdir / "metrics.jsonl", outdir / "convergence.png")
    return metrics


def main():
    ap = argparse.ArgumentParser(description="Run VMC from a YAML config file.")
    ap.add_argument("--config", required=True, type=Path, help="Path to YAML config.")
    args = ap.parse_args()

    cfg = RunConfig.from_yaml(args.config)
    base = Path("results") / args.config.stem

    all_metrics = []
    for seed in cfg.seeds:
        print(f"=== seed {seed} ===")
        m = run_seed(cfg, seed, base / str(seed))
        all_metrics.append(m)
        print(json.dumps(m, indent=2))

    agg = aggregate_seeds(all_metrics)
    with open(base / "aggregate.json", "w") as f:
        json.dump(agg, f, indent=2)
    print("\n=== aggregate ===")
    print(json.dumps(agg, indent=2))


if __name__ == "__main__":
    main()
