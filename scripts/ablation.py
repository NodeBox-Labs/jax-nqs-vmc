"""Run a full ablation grid and generate a Markdown table."""

import argparse
from pathlib import Path

import pandas as pd
from nqs_vmc.config import RunConfig
from run_vmc import run_seed


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="configs/ho_ground.yaml")
    args = ap.parse_args()

    ansatze = ["even", "transformer"]
    optimizers = ["adam", "minsr"]
    widths = [32, 64]

    results = []

    for ansatz_type in ansatze:
        for opt in optimizers:
            for w in widths:
                for seed in [0, 1]:
                    cfg = RunConfig(
                        ansatz={"type": ansatz_type, "hidden": [w, w]},
                        optimizer={"name": opt, "lr": 0.005, "n_iter": 100},
                        seeds=(seed,),
                    )
                    outdir = Path(f"results/abl_{ansatz_type}_{opt}_{w}_{seed}")
                    m = run_seed(cfg, seed, outdir)
                    results.append({
                        "ansatz": ansatz_type,
                        "optimizer": opt,
                        "width": w,
                        "seed": seed,
                        "energy": m["final_energy"],
                        "variance": m["final_variance"],
                    })

    df = pd.DataFrame(results)
    summary = (
        df.groupby(["ansatz", "optimizer", "width"])
        .agg(
            mean_energy=("energy", "mean"),
            std_energy=("energy", "std"),
            mean_var=("variance", "mean"),
        )
        .reset_index()
    )

    md = summary.to_markdown(index=False)
    print(md)
    Path("results").mkdir(exist_ok=True)
    with open("results/ablation_table.md", "w") as f:
        f.write(md)


if __name__ == "__main__":
    main()