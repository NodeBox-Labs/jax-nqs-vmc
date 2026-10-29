"""Report any missing Phase 2 files."""
import importlib
import sys
from pathlib import Path

EXPECTED_MODULES = [
    "nqs_vmc",
    "nqs_vmc.jax_config",
    "nqs_vmc.config",
    "nqs_vmc.ansatz",
    "nqs_vmc.sampler",
    "nqs_vmc.local_energy",
    "nqs_vmc.optimizers",
    "nqs_vmc.checkpoint",
    "nqs_vmc.metrics",
    "nqs_vmc.trainer",
    "nqs_vmc.cli",
    "nqs_vmc.diagnostics",
    "nqs_vmc.systems",
    "nqs_vmc.reference",
    "nqs_vmc.reference.exact_diag",
    "nqs_vmc.reference.split_step",
    "nqs_vmc.reference.few_body",
]

EXPECTED_SYMBOLS = {
    "nqs_vmc.config": ["RunConfig", "AnsatzConfig", "SamplerConfig",
                       "OptimizerConfig", "ReferenceConfig", "ExcitedConfig"],
    "nqs_vmc.ansatz": ["MLP", "ParityEven", "ParityOdd", "DeepSets", "build_ansatz"],
    "nqs_vmc.sampler": ["SamplerState", "sample", "integrated_autocorr_time"],
    "nqs_vmc.local_energy": ["make_local_energy", "energy_and_variance", "overlap_squared"],
    "nqs_vmc.optimizers": ["AdamOpt", "SRUpdater", "quantum_geometric_tensor", "build_optimizer"],
    "nqs_vmc.checkpoint": ["save", "load", "exists"],
    "nqs_vmc.metrics": ["JsonlLogger", "replay"],
    "nqs_vmc.trainer": ["TrainState", "train"],
    "nqs_vmc.cli": ["main"],
    "nqs_vmc.systems": ["BaseSystem", "Harmonic", "DoubleWell",
                        "GrossPitaevskii", "TwoBosons", "build_system"],
    "nqs_vmc.reference.exact_diag": ["exact_1d", "grid_hamiltonian"],
    "nqs_vmc.reference.split_step": ["gpe_imaginary_time"],
    "nqs_vmc.reference.few_body": ["two_boson_ground_state"],
}

missing = []
for mod_name in EXPECTED_MODULES:
    try:
        mod = importlib.import_module(mod_name)
    except ImportError as e:
        missing.append(f"MODULE {mod_name}: {e}")
        continue
    for sym in EXPECTED_SYMBOLS.get(mod_name, []):
        if not hasattr(mod, sym):
            missing.append(f"SYMBOL {mod_name}.{sym}")

if missing:
    print("MISSING:")
    for m in missing:
        print(f"  - {m}")
    sys.exit(1)
print(f"All {len(EXPECTED_MODULES)} modules and {sum(len(v) for v in EXPECTED_SYMBOLS.values())} symbols present.")