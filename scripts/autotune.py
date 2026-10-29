"""Hyperparameter search with Optuna."""

from pathlib import Path

import optuna
import yaml
from run_vmc import run_seed

from nqs_vmc.config import RunConfig


def objective(trial):
    # Load base config
    with open("configs/ho_ground.yaml") as f:
        cfg_dict = yaml.safe_load(f)

    # Suggest hyperparameters
    cfg_dict["optimizer"]["lr"] = trial.suggest_float("lr", 1e-4, 1e-1, log=True)
    cfg_dict["ansatz"]["hidden"] = [trial.suggest_categorical("width", [16, 32, 64, 128])] * 2
    cfg_dict["optimizer"]["n_iter"] = 200

    # Build and run
    cfg = RunConfig(**cfg_dict)
    outdir = Path(f"results/optuna_trial_{trial.number}")
    metrics = run_seed(cfg, seed=0, outdir=outdir)

    return metrics["final_variance"]


if __name__ == "__main__":
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=20)
    print("Best trial:", study.best_trial.params)
