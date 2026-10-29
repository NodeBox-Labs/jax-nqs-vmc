"""Head-to-head Adam vs SR on the same ansatz & seed set."""

from __future__ import annotations

import argparse
import copy
from pathlib import Path

from run_vmc import run_seed

from nqs_vmc.config import RunConfig
from nqs_vmc.diagnostics import aggregate_seeds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, type=Path)
    args = ap.parse_args()

    base_cfg = RunConfig.from_yaml(args.config)

    results = {"adam": [], "sr": []}
    for opt_name in ("adam", "sr"):
        cfg = copy.deepcopy(base_cfg)
        cfg.optimizer.name = opt_name
        outdir = Path("results") / f"compare_{args.config.stem}" / opt_name
        for seed in cfg.seeds:
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
