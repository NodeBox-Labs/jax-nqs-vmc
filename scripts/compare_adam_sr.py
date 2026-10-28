"""Head-to-head Adam vs SR on the same ansatz & seed set."""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml
from run_vmc import run_seed

from nqs_vmc.diagnostics import aggregate_seeds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, type=Path)
    args = ap.parse_args()
    with open(args.config) as f:
        base_cfg = yaml.safe_load(f)

    results = {"adam": [], "sr": []}
    for opt_name in ("adam", "sr"):
        cfg = dict(base_cfg)
        cfg["optimizer"] = dict(base_cfg["optimizer"])
        cfg["optimizer"]["name"] = opt_name
        outdir = Path("results") / f"compare_{args.config.stem}" / opt_name
        for seed in cfg["seeds"]:
            m = run_seed(cfg, seed, outdir / str(seed))
            results[opt_name].append(m)

    for opt_name, ms in results.items():
        print(f"\n=== {opt_name} ===")
        agg = aggregate_seeds(ms)
        for k, v in agg.items():
            if k in ("final_energy", "final_variance", "energy_error", "fidelity"):
                print(f"  {k}: {v['mean']:.6f} ± {v['std']:.6f}")


if __name__ == "__main__":
    main()