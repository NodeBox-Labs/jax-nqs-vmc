"""Run a full ablation grid and generate a Markdown table."""

import pandas as pd
from pathlib import Path
from nqs_vmc.config import RunConfig
from run_vmc import run_seed

def main():
    results = []
    ansatze = ["mlp", "even", "transformer"]
    optimizers = ["adam", "sr", "minsr"]
    widths = [32, 64]

    for ansatz in ansatze:
        for opt in optimizers:
            for w in widths:
                for seed in [0, 1, 2]:
                    cfg = RunConfig(
                        system="harmonic",
                        ansatz={"type": ansatz, "hidden": [w, w]},
                        optimizer={"name": opt, "lr": 0.005, "n_iter": 200},
                        seeds=(seed,)
                    )
                    outdir = Path(f"results/abl_{ansatz}_{opt}_{w}_{seed}")
                    m = run_seed(cfg, seed, outdir)
                    results.append({
                        "ansatz": ansatz, "optimizer": opt, "width": w, "seed": seed,
                        "energy": m["final_energy"], "variance": m["final_variance"]
                    })

    df = pd.DataFrame(results)
    summary = df.groupby(["ansatz", "optimizer", "width"]).agg(
        mean_energy=("energy", "mean"),
        std_energy=("energy", "std"),
        mean_var=("variance", "mean")
    ).reset_index()

    md = summary.to_markdown(index=False)
    print(md)
    with open("results/ablation_table.md", "w") as f:
        f.write(md)

if __name__ == "__main__":
    main()