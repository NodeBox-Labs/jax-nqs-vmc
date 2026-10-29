from .optimizers_minsr import MatrixFreeSR

def build_optimizer(cfg) -> Optimizer:
    from .config import OptimizerConfig
    if cfg.name == "adam":
        return AdamOpt(cfg.lr, cfg.clip_norm, cfg.schedule, cfg.warmup_steps, cfg.n_iter)
    elif cfg.name == "sr":
        return SRUpdater(cfg.lr, cfg.diag_shift, cfg.use_diag, cfg.sr_solver, cfg.clip_norm)
    elif cfg.name == "minsr":
        return MatrixFreeSR(cfg.lr, cfg.diag_shift)
    raise ValueError(f"Unknown optimizer {cfg.name}")