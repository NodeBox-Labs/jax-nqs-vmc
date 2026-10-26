# nqs-vmc — Neural Quantum States with Variational Monte Carlo (JAX)

A reproducible research codebase for representing wavefunctions with neural
networks and minimizing the energy by variational Monte Carlo (VMC). All
problems are 1D (ħ = m = 1) with exact or high-accuracy reference solutions.

## Problems

1. **Harmonic oscillator** — ground state and first excited state (`E_n = n + ½`).
2. **Anharmonic / double-well** — reference by exact diagonalization on a grid
   (`scipy.sparse.linalg.eigsh`).
3. **Gross–Pitaevskii** ground state in a harmonic trap with `g > 0`
   (nonlinear functional `E[ψ] = ∫ ½|ψ'|² + V|ψ|² + (g/2)|ψ|⁴ dx`).
   Reference: imaginary-time split-step solver.
4. **Two interacting bosons** in 1D harmonic trap with contact interaction,
   permutation-symmetric DeepSets ansatz.

## Methods

- **Ansatz**: MLP with `tanh`/GELU on `log|ψ|`; parity-aware variants; DeepSets
  for few-body; Jastrow with exact cusp for contact interactions.
- **Sampling**: Metropolis–Hastings with adaptive step size; reports acceptance
  rate and integrated autocorrelation time.
- **Local energy**: autograd second derivatives (`jax.hessian`), zero-variance
  principle for variance reduction.
- **Optimizers**: Adam (Optax) and Stochastic Reconfiguration (natural gradient)
  with configurable diagonal shift; both tracked in the same run.

## Install

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

JAX will pick CPU by default; install `jax[cuda12]` for GPU.

## Quickstart

```bash
# Harmonic oscillator ground state
python scripts/run_vmc.py --config configs/ho_ground.yaml

# Harmonic oscillator first excited state (orthogonality penalty)
python scripts/run_vmc.py --config configs/ho_excited.yaml

# Double well
python scripts/run_vmc.py --config configs/double_well.yaml

# Gross–Pitaevskii
python scripts/run_vmc.py --config configs/gross_pitaevskii.yaml

# Two bosons with contact interaction
python scripts/run_vmc.py --config configs/two_bosons.yaml

# Adam vs SR head-to-head
python scripts/compare_adam_sr.py --config configs/ho_ground.yaml

# Network-size sweep
python scripts/sweep_network_size.py --config configs/ho_ground.yaml

# Tests
pytest tests/ -v
```

Each run writes `results/<config_name>/<seed>/` containing `metrics.json`
(energies, variances, acceptance, autocorrelation, fidelity), `history.npz`
(per-iteration curves), and `psi_grid.npz` (wavefunction sampled on a grid for
fidelity computation).

## Validation checklist

- `tests/test_harmonic_exact.py` — exact eigenvalues within 1e-6.
- `tests/test_local_energy.py` — analytic `Hψ/ψ` matches autograd for Gaussian.
- `tests/test_sampler_acceptance.py` — acceptance rate sits in target band.

## Known failure modes (documented in the reports)

- **Excited states without orthogonality** → collapse to ground state.
- **Double-well tunneling** → symmetric ansatz misses exponentially small gap;
  need odd-parity or penalty.
- **GPE mode collapse** → too-small SR regularization or too-large learning rate
  leads to `ψ → 0` locally; adaptive step size and normalized local energy help.
- **Contact interaction cusp** → local energy diverges at `x₁ = x₂` unless the
  cusp is built into the Jastrow factor (we do).
