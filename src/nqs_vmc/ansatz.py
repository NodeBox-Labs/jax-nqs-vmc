from .ansatz_autoregressive import AutoregressiveMLP
from .ansatz_transformer import TransformerAnsatz

REGISTRY = {
    "mlp": MLP, 
    "even": ParityEven, 
    "odd": ParityOdd, 
    "deepsets": DeepSets,
    "autoregressive": AutoregressiveMLP,
    "transformer": TransformerAnsatz,
}

def build_ansatz(cfg: AnsatzConfig) -> nn.Module:
    cls = REGISTRY[cfg.type]
    kwargs = {"hidden": tuple(cfg.hidden), "activation": cfg.activation}
    if cfg.type == "deepsets":
        kwargs["pair_hidden"] = tuple(cfg.pair_hidden)
        kwargs["cusp"] = cfg.cusp
    elif cfg.type == "transformer":
        kwargs["num_layers"] = 2
    return cls(**kwargs)