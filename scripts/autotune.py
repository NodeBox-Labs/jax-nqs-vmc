"""Hyperparameter search with Optuna."""

from pathlib import Path

import optuna
from run_vmc import run_seed

from nqs_vmc.config import AnsatzConfig, OptimizerConfig, RunConfig


def objective(trial):
    # Load base config
    cfg = RunConfig.from_yaml(Path("configs/ho_ground.yaml"))

    # Suggest hyperparameters
    lr = trial.suggest_float("lr", 1e-4, 1e-1, log=True)
    width = trial.suggest_categorical("width", [16, 32, 64, 128])

    cfg.optimizer = OptimizerConfig(
        name=cfg.optimizer.name,
        lr=lr,
        n_iter=200,
        diag_shift=cfg.optimizer.diag_shift,
        use_diag=cfg.optimizer.use_diag,
        sr_solver=cfg.optimizer.sr_solver,
        clip_norm=cfg.optimizer.clip_norm,
        warmup_steps=cfg.optimizer.warmup_steps,
        schedule=cfg.optimizer.schedule,
    )
    cfg.ansatz = AnsatzConfig(
        type=cfg.ansatz.type,
        hidden=(width, width),
        activation=cfg.ansatz.activation,
        pair_hidden=cfg.ansatz.pair_hidden,
        cusp=cfg.ansatz.cusp,
    )

    outdir = Path(f"results/optuna_trial_{trial.number}")
    metrics = run_seed(cfg, seed=0, outdir=outdir)
    return metrics["final_variance"]


if __name__ == "__main__":
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=20)
    print("Best trial:", study.best_trial.params)
