from . import anharmonic, few_bosons, gross_pitaevskii, harmonic

SYSTEM_REGISTRY = {
    "harmonic": harmonic.HarmonicOscillator,
    "double_well": anharmonic.DoubleWell,
    "gpe": gross_pitaevskii.GrossPitaevskii,
    "two_bosons": few_bosons.TwoBosons,
}

def build_system(cfg: dict):
    name = cfg["system"]
    return SYSTEM_REGISTRY[name](**{k: v for k, v in cfg.items() if k not in {"system", "ansatz", "sampler", "optimizer", "seeds", "reference", "output"}})