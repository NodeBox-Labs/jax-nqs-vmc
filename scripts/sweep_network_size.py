"""Sweep hidden layer sizes and plot energy error / variance vs network width."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import yaml

from run_vmc import run_seed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, type=Path)
    ap.add_argument("--widths", nargs="+", type=int, default=[16, 32, 64, 128])
    args = ap.parse_args()

    with open(args.config) as f:
        base_cfg = yaml.safe_load(f)

    curves = {"width": [], "energy_error": [], "variance": [], "fidelity": []}
    for w in args.widths:
        cfg = json.loads(json.dumps(base_cfg))
        cfg["ansatz"]["hidden"] = [w, w]
        outdir = Path("results") / f"sweep_{args.config.stem}" / f"w{w}"
        ms = []
        for seed in cfg["seeds"]:
            m = run_seed(cfg, seed, outdir / str(seed))
            ms.append(m)
        errs = np.array([m.get("energy_error", np.nan) for m in ms])
        vars_ = np.array([m["final_variance"] for m in ms])
        fids = np.array([m.get("fidelity", np.nan) for m in ms])
        curves["width"].append(w)
        curves["energy_error"].append(errs.mean())
        curves["variance"].append(vars_.mean())
        curves["fidelity"].append(fids.mean())

    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    ax[0].plot(curves["width"], np.abs(curves["energy_error"]), "o-")
    ax[0].set_xscale("log"); ax[0].set_yscale("log")
    ax[0].set_xlabel("hidden width"); ax[0].set_ylabel("|ΔE|")
    ax[1].plot(curves["width"], curves["variance"], "o-")
    ax[1].set_xscale("log"); ax[1].set_yscale("log")
    ax[1].set_xlabel("hidden width"); ax[1].set_ylabel("Var(E_L)")
    plt.tight_layout()
    plt.savefig(Path("results") / f"sweep_{args.config.stem}.png", dpi=140)
    print(json.dumps(curves, indent=2))


if __name__ == "__main__":
    main()