"""Entry point: run VMC for a given YAML config."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import yaml

from nqs_vmc.ansatz import build_ansatz
from nqs_vmc.diagnostics import fidelity_from_grid, aggregate_seeds, save_results
from nqs_vmc.reference.exact_diag import double_well_exact, harmonic_exact
from nqs_vmc.systems import build_system
from nqs_vmc.trainer import train


def load_config(path: Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def make_logpsi(ansatz):
    def logpsi(params, x):
        return ansatz.apply(params, x)
    return logpsi


def get_reference(cfg: dict):
    ref = cfg.get("reference", {})
    if cfg["system"] == "harmonic":
        return harmonic_exact(n_max=6, L=ref.get("L", 10.0), n_grid=ref.get("n_grid", 2048))
    if cfg["system"] == "double_well":
        return double_well_exact(
            a=cfg.get("a", -0.5), b=cfg.get("b", 0.25),
            n_max=6, L=ref.get("L", 8.0), n_grid=ref.get("n_grid", 2048),
        )
    return None  # for GPE / few-bosons we handle separately


def run_seed(cfg: dict, seed: int, outdir: Path):
    system = build_system(cfg)
    ansatz = build_ansatz(cfg["ansatz"])
    logpsi = make_logpsi(ansatz)

    rng = jax.random.PRNGKey(seed)
    dummy = jnp.zeros((1, system.dim))
    params = ansatz.init(rng, dummy)

    ortho_ref = None
    if cfg.get("excited", {}).get("use_ortho", False):
        ref_vals, ref_x, ref_vecs = get_reference(cfg) or (None, None, None)
        if ref_vecs is not None:
            psi0 = ref_vecs[:, 0]
            grid = jnp.asarray(ref_x)
            psi0_j = jnp.asarray(psi0)
            def ortho_ref(x):
                # linear interpolation of ground state on the grid
                return jnp.interp(x[:, 0], grid, psi0_j)
            ortho_ref = ortho_ref

    opt_cfg = cfg["optimizer"]
    state = train(
        logpsi, params, system.potential, system.dim,
        g=getattr(system, "g", 0.0),
        n_iter=opt_cfg.get("n_iter", 500),
        n_chains=cfg["sampler"]["n_chains"],
        n_sweeps=cfg["sampler"]["n_samples"] // cfg["sampler"]["n_chains"],
        burn_in=cfg["sampler"].get("burn_in", 100),
        step_size_init=cfg["sampler"]["step_size_init"],
        target_accept=cfg["sampler"]["target_accept"],
        optimizer=opt_cfg["name"],
        lr=opt_cfg["lr"],
        sr_diag_shift=opt_cfg.get("diag_shift", 1e-3),
        sr_use_diag=opt_cfg.get("use_diag", False),
        seed=seed,
        orthogonalize_against=ortho_ref,
        ortho_penalty=cfg.get("excited", {}).get("penalty", 0.0),
        verbose=True,
    )

    # final metrics: energy, variance, fidelity
    metrics = {
        "seed": seed,
        "final_energy": state.history["energy"][-1],
        "final_variance": state.history["variance"][-1],
        "final_acceptance": state.history["acceptance"][-1],
        "tau_int": state.history["tau_int"][-1],
    }

    ref = get_reference(cfg)
    if ref is not None:
        ref_vals, ref_x, ref_vecs = ref
        idx = cfg.get("excited", {}).get("level", 0)
        metrics["reference_energy"] = float(ref_vals[idx])
        metrics["energy_error"] = float(state.history["energy"][-1] - ref_vals[idx])
        metrics["fidelity"] = fidelity_from_grid(
            logpsi, state.params, jnp.asarray(ref_x), ref_vecs[:, idx]
        )

    outdir.mkdir(parents=True, exist_ok=True)
    save_results(outdir, metrics, state.history)
    return metrics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, type=Path)
    args = ap.parse_args()
    cfg = load_config(args.config)

    base = Path("results") / args.config.stem
    all_metrics = []
    for seed in cfg["seeds"]:
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