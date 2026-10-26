<div align="center">

**`nqs-vmc` — Neural Quantum States with Variational Monte Carlo in JAX**

_A reproducible reference implementation of neural-network wavefunctions for
interacting quantum systems, validated against exact and mean-field solvers._

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue?style=flat-square)](https://www.python.org/downloads/)
[![JAX](https://img.shields.io/badge/jax-%E2%89%A50.4.26-orange?style=flat-square)](https://github.com/google/jax)
[![Flax](https://img.shields.io/badge/flax-%E2%89%A50.8.4-blueviolet?style=flat-square)](https://github.com/google/flax)
[![Tests](https://img.shields.io/badge/tests-pytest-informational?style=flat-square)](tests/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green?style=flat-square)](#license)

</div>

# `nqs-vmc` — Neural Quantum States with Variational Monte Carlo

**Project status:** active reference implementation. The API is stable; every
number in this README is reproducible from the shipped YAML configs.

## Overview

This repository implements **variational Monte Carlo (VMC) with neural quantum
states (NQS)** from scratch in [JAX](https://github.com/google/jax). It is
written as a **reference codebase** — not a black-box library — for researchers
and students who need to understand, extend, or reproduce NQS calculations in
one spatial dimension with exact reference solutions.

The code is deliberately small, typed, tested, and instrumented. Every
optimization step is logged (energy, variance, acceptance rate, integrated
autocorrelation time, gradient norm), and every run is seeded and reproducible
from a YAML config.

**Scope.** Four problems, in increasing difficulty, covering the main failure
modes of NQS-VMC in one dimension: excited-state collapse, tunneling
splittings, mean-field mode collapse, and the contact-interaction cusp.

---

## Table of contents

- [Problems](#problems)
- [Methods](#methods)
- [Repository layout](#repository-layout)
- [Installation](#installation)
- [Quickstart](#quickstart)
- [Configuration reference](#configuration-reference)
- [Results and artifacts](#results-and-artifacts)
- [Validation strategy](#validation-strategy)
- [Design notes](#design-notes)
- [Known failure modes](#known-failure-modes)
- [Reproducing every figure](#reproducing-every-figure)
- [Extending the code](#extending-the-code)
- [Development](#development)
- [License](#license)
- [Citation](#citation)

---

## Problems

All problems use natural units `ħ = m = 1`. Reference solutions are computed
independently in pure NumPy/SciPy, so they act as a genuine cross-check rather
than a mirror of the VMC code.

| # | Problem | Reference solver | Key challenge |
|---|---------|------------------|---------------|
| 1 | **1D harmonic oscillator** — ground state and first excited state (`E_n = n + ½`) | Analytic; cross-checked by ED | Excited-state collapse without explicit parity or orthogonality |
| 2 | **1D double well** — `V(x) = a x² + b x⁴` with `a < 0`, `b > 0` | `scipy.sparse.linalg.eigsh` on a fine grid | Exponentially small tunneling splitting; symmetric collapse |
| 3 | **1D Gross–Pitaevskii** ground state, `g > 0` — `E[ψ] = ∫ ½|ψ'|² + V|ψ|² + (g/2)|ψ|⁴ dx` | Imaginary-time split-step solver | Nonlinear local energy; mode collapse |
| 4 | **Two interacting bosons** in a 1D harmonic trap with contact interaction | 2D grid ED with smeared delta | Exact cusp `-g/2·|x₁-x₂|` must be in the ansatz |

**Why these four?** Together they cover every standard failure mode of NQS-VMC
in 1D. If the code passes all four, it is a legitimate reference implementation
rather than a demo.

---

## Methods

### Wavefunction ansatz

All ansätze return `log ψ(x)`, following the sign-safe log-amplitude convention:

- **`MLP`** — dense net with `tanh` / GELU / SiLU on `log|ψ(x)|`. Used for the
  double-well ground state.
- **`ParityEven`** — MLP on `|x|`; guarantees `ψ(-x) = ψ(x)` for the HO ground
  state and for the GPE mean-field ground state.
- **`ParityOdd`** — `sign(x)·exp(MLP(|x|))`, i.e. `log|ψ| = MLP(|x|) + log|x|`;
  mathematically orthogonal to the ground state, hence no penalty is needed
  for the HO first excited state.
- **`DeepSets`** — permutation-symmetric for `N` particles:
  `log ψ(x₁…x_N) = ρ(Σᵢ φ(xᵢ), Σᵢ<ⱼ J(|xᵢ-xⱼ|))`.
  The pair function `J` includes the **exact contact cusp** `-g/2·|xᵢ-xⱼ|`, so
  the kinetic local energy is finite at `xᵢ = xⱼ`.
- **`LinearCombination`** — `ψ = ψ_even + e^α·ψ_odd`; useful when an explicit
  orthogonalization penalty is preferred over a symmetry-aware ansatz.

### Sampling

Multi-chain **Metropolis–Hastings** with a Gaussian proposal, jitted end-to-end
via `jax.lax.scan`:

- Chains evolve in lockstep; burn-in and production are separate scans.
- Step size is adapted in the log domain every sweep toward a configurable
  target acceptance rate (default 55%).
- The **integrated autocorrelation time** `τ_int` is estimated with Sokal's
  automatic-windowing estimator, and reported per iteration alongside the
  acceptance rate and step size.

### Local energy

Computed via autograd on `log ψ` (not `ψ`), for numerical stability:

    −½ ψ''/ψ = −½ [ tr(Hess log ψ) + |∇ log ψ|² ]

The nonlinear GPE term `g|ψ|²` is included when `g > 0`, giving the mean-field
local energy whose variance vanishes at the self-consistent ground state (the
**zero-variance principle**). Statistics (`E`, `Var`, `stderr`) and, if
requested, the **orthogonality penalty** `λ|⟨ψ₀|ψ⟩|²` are computed on the same
batch.

### Optimizers

Both optimizers share the same training loop, so comparisons are
apples-to-apples:

- **Adam** via [Optax](https://github.com/google-deepmind/optax) — cheap,
  robust, the default.
- **Stochastic Reconfiguration (SR / natural gradient)** — computes the
  quantum geometric tensor
  `S_ij = ⟨O_i O_j⟩ - ⟨O_i⟩⟨O_j⟩` (with `O_i = ∂ log ψ / ∂θ_i`) and the
  gradient `F_i = ⟨E_L O_i⟩ - ⟨E_L⟩⟨O_i⟩`, then updates
  `δθ = -lr · (S + εI)⁻¹ F`.
  Supports full or diagonal `S`; the diagonal shift `ε` is the only
  regularization knob.

### Diagnostics

Per iteration, the trainer records: energy, variance, standard error,
acceptance rate, adapted step size, `τ_int`, and gradient norm. Post-run:

- **Energy error** vs the exact/reference value, with seed-to-seed error bars.
- **Energy variance** as a convergence diagnostic — must trend to zero for
  eigenstates (zero-variance principle).
- **Fidelity** `|⟨ψ_ref|ψ_NQS⟩|²` computed on the same grid used by the exact
  diagonalization, with no interpolation bias beyond trapezoidal quadrature.

---

## Repository layout

    jax-nqs-vmc/
    ├── README.md
    ├── pyproject.toml
    ├── configs/
    │   ├── ho_ground.yaml
    │   ├── ho_excited.yaml
    │   ├── double_well.yaml
    │   ├── gross_pitaevskii.yaml
    │   └── two_bosons.yaml
    ├── src/nqs_vmc/
    │   ├── __init__.py          # JAX x64 setup, version exports
    │   ├── ansatz.py            # MLP log|ψ|, symmetry-aware variants
    │   ├── sampler.py           # Metropolis–Hastings + adaptive step size
    │   ├── local_energy.py      # autograd second derivatives
    │   ├── optimizers.py        # Adam vs SR (natural gradient)
    │   ├── systems/
    │   │   ├── harmonic.py
    │   │   ├── anharmonic.py
    │   │   ├── gross_pitaevskii.py
    │   │   └── few_bosons.py    # DeepSets / Jastrow + NN
    │   ├── reference/
    │   │   ├── exact_diag.py    # LOBPCG / scipy eigsh
    │   │   └── split_step.py    # imaginary-time GPE solver
    │   └── diagnostics.py       # variance, fidelity, autocorrelation, error bars
    ├── scripts/
    │   ├── run_vmc.py
    │   ├── sweep_network_size.py
    │   └── compare_adam_sr.py
    └── tests/
        ├── test_harmonic_exact.py
        ├── test_local_energy.py
        └── test_sampler_acceptance.py

---

## Installation

**Requirements.** Python ≥ 3.10, `jax ≥ 0.4.26`, `flax ≥ 0.8.4`, `optax ≥ 0.1.9`,
`scipy ≥ 1.11`. CPU-only works out of the box; a CUDA 12 GPU is optional.

    git clone https://github.com/<your-user>/jax-nqs-vmc.git
    cd jax-nqs-vmc
    python -m venv .venv && source .venv/bin/activate
    pip install -e ".[dev]"

For GPU:

    pip install -e ".[dev]" "jax[cuda12]" \
      -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

Verify the install:

    pytest -q

You should see `5 passed` (or more) in a few seconds.

---

## Quickstart

Every problem is run with a single command:

    # 1. HO ground state — should converge to E = 0.5
    python scripts/run_vmc.py --config configs/ho_ground.yaml

    # 2. HO first excited state — should converge to E = 1.5
    python scripts/run_vmc.py --config configs/ho_excited.yaml

    # 3. Double well — reference via scipy eigsh
    python scripts/run_vmc.py --config configs/double_well.yaml

    # 4. Gross–Pitaevskii mean-field ground state
    python scripts/run_vmc.py --config configs/gross_pitaevskii.yaml

    # 5. Two interacting bosons with contact interaction
    python scripts/run_vmc.py --config configs/two_bosons.yaml

Compare Adam and SR on the same ansatz and seeds:

    python scripts/compare_adam_sr.py --config configs/ho_ground.yaml

Sweep hidden-layer width and produce the `|ΔE|` and `Var(E_L)` plots:

    python scripts/sweep_network_size.py --config configs/ho_ground.yaml \
      --widths 16 32 64 128

Each command prints a live progress bar with `E`, `Var(E_L)`, acceptance rate
and `τ_int`, then writes the per-seed and aggregate results under `results/`.

---

## Configuration reference

Every config is a YAML file with the same top-level schema. Example
(`configs/ho_ground.yaml`):

    system: harmonic
    omega: 1.0
    dim: 1

    ansatz:
      type: even
      hidden: [64, 64]
      activation: tanh

    sampler:
      n_chains: 128
      n_samples: 8192
      step_size_init: 0.5
      target_accept: 0.55
      burn_in: 200

    optimizer:
      name: adam          # "adam" or "sr"
      lr: 0.005
      n_iter: 400
      diag_shift: 1.0e-3  # SR only
      use_diag: false     # SR only

    reference:
      L: 10.0
      n_grid: 2048

    excited:
      use_ortho: false
      level: 0
      penalty: 0.0

    seeds: [0, 1, 2, 3, 4]

| Field | Meaning |
|-------|---------|
| `system` | One of `harmonic`, `double_well`, `gpe`, `two_bosons`. |
| `dim` | Spatial dimension per particle (always 1 here). |
| `ansatz.type` | `mlp`, `even`, `odd`, `deepsets`, `combo`. |
| `ansatz.hidden` | Widths of the log-amplitude MLP. |
| `sampler.n_chains` | Number of parallel Metropolis walkers. |
| `sampler.n_samples` | Total samples collected per iteration. |
| `sampler.target_accept` | Step-size adaptation target. |
| `optimizer.name` | `adam` or `sr`. |
| `optimizer.lr` | Learning rate. |
| `optimizer.diag_shift` | SR regularization `ε`. |
| `optimizer.use_diag` | Diagonal approximation for the SR solve. |
| `reference.*` | Grid parameters for the exact solver. |
| `excited.use_ortho` | Enable the orthogonality penalty. |
| `excited.level` | Eigenstate index used for the reference energy. |
| `seeds` | List of PRNG seeds; each is run independently. |

---

## Results and artifacts

Each seed writes to `results/<config_name>/<seed>/`:

- `metrics.json` — final energy, variance, acceptance, `τ_int`, reference
  energy, energy error, fidelity.
- `history.npz` — per-iteration curves: energy, variance, stderr, acceptance,
  step size, `τ_int`, gradient norm.
- `psi_grid.npz` — final wavefunction evaluated on the reference grid
  (used for fidelity and plotting).

`results/<config_name>/aggregate.json` summarizes mean and standard deviation
across seeds for every scalar metric. All artifacts are git-ignored; only the
`results/.gitkeep` placeholder is committed.

### What a successful run looks like

For `ho_ground.yaml`, after ~400 Adam iterations:

| Metric | Expected |
|--------|----------|
| `final_energy` | `0.500 ± 0.005` |
| `final_variance` | `< 1e-3` |
| `final_acceptance` | `0.5–0.6` |
| `fidelity` | `> 0.999` |

For `ho_excited.yaml`: `final_energy ≈ 1.500 ± 0.005`, `fidelity > 0.99`.

For `double_well.yaml`: the reference has two nearly degenerate low-lying
states; the reported `final_energy` matches the **symmetric ground state**, and
`fidelity > 0.95` on a 2048-point grid.

For `gross_pitaevskii.yaml`: `final_energy` matches the split-step chemical
potential within a few `1e-3`, and `final_variance < 5e-3`.

For `two_bosons.yaml`: the contact-interaction energy shifts upward from
`E = 1.0` (noninteracting) with increasing `g`; `final_variance` scales
linearly with `g` because of the cusp — this is expected and documented.

---

## Validation strategy

Three layers of validation, each independent of the others:

1. **Unit tests (`pytest`).**
   - `tests/test_harmonic_exact.py` — the grid ED reproduces `E_n = n + ½` to
     `1e-6`; the double-well ground state is even and the first excited state
     is odd.
   - `tests/test_local_energy.py` — for the analytic Gaussian ground state of
     the HO, the autograd local energy equals `0.5` to `1e-10` at every point.
   - `tests/test_sampler_acceptance.py` — the adaptive MH produces acceptance
     rates in `[0.3, 0.8]` on a Gaussian target.

2. **Independent reference solvers (`src/nqs_vmc/reference/`).**
   - Grid exact diagonalization in NumPy/SciPy — no JAX, no shared code.
   - Split-step imaginary-time GPE solver — pure NumPy.
   - Two-boson 2D grid ED with a smeared delta — pure SciPy.

3. **Statistical comparison across seeds.**
   - Energy error reported as `mean ± std / sqrt(N_seeds)` over 3–5 seeds.
   - Fidelity computed on the same grid the reference solver used, with no
     additional interpolation.

If a claim in the paper, a tutorial, or a review says “NQS-VMC reproduces
`E₀ = 0.5 ± 0.005`”, that claim is directly checkable by running
`run_vmc.py --config configs/ho_ground.yaml` and inspecting
`results/ho_ground/aggregate.json`.

---

## Design notes

A few choices distinguish this codebase from a typical “VMC in 100 lines”
notebook. They are deliberate, and each addresses a specific failure mode of
naive NQS-VMC.

1. **JAX x64 is enabled globally in `__init__.py`.** Excited-state gaps and
   tunneling splittings are `O(10⁻²)` or smaller; `float32` washes them out.

2. **Log-domain everywhere.** Autograd differentiates `log ψ`, not `ψ`. This
   controls the dynamic range, prevents overflow for wide wavefunctions, and
   makes the kinetic term `-½[tr(Hess log ψ) + |∇ log ψ|²]` numerically clean.

3. **Parity-aware ansätze for the HO.** The first excited state uses
   `ParityOdd`, so orthogonality is exact by construction. The penalty path is
   retained and tested separately, but it is not needed for problem 1 — an
   important distinction that many tutorials miss.

4. **Exact cusp in the DeepSets ansatz.** The pair function `J(|xᵢ-xⱼ|)`
   contains `-g/2·|xᵢ-xⱼ|` explicitly. Without it, the kinetic local energy
   diverges at `xᵢ = xⱼ`, variance blows up, and training stalls.

5. **Same training loop for Adam and SR.** No duplicated code, no divergence
   in diagnostics between the two. The optimizer is a strategy object, nothing
   more.

6. **Adaptive step size in log domain.** The Metropolis step size evolves as
   `σ ← σ · exp(η(α - α_target))` with `η` small enough to avoid oscillation.
   Reported in every metrics file so the reader can verify convergence.

7. **Sokal's `τ_int` estimator.** Autocorrelation is not assumed; it is
   measured. Statistical error bars can then be corrected by `sqrt(2 τ_int)`
   if the user wants strictly correlated-aware errors.

**Why not a pip-installable library?** Because the goal is pedagogy, not API
stability. Every file is short enough to read in one sitting, and each design
decision above corresponds to a comment or a test. If you want an industrial
NQS library, use [NetKet](https://github.com/netket/netket); if you want to
understand what NetKet does, read this repository.

---

## Known failure modes

These are **documented failures** — the point is not that they never happen,
but that the code makes them visible.

### Excited states without orthogonality

If `ho_excited.yaml` is run with `ansatz.type: mlp` and `use_ortho: false`,
the optimizer will **collapse to the ground state** (`E → 0.5`, not `1.5`).
This is not a bug — it is the variational principle doing exactly what it
should. The two valid fixes are:

1. Enforce parity in the ansatz (`type: odd`), which is what the shipped
   config does.
2. Add an orthogonalization penalty against the exact ground state
   (`use_ortho: true`, `penalty: 5.0`), which is supported but not needed for
   problem 1.

### Double-well tunneling

For deep double wells (`|a| ≫ b`), the gap `E₁ - E₀` becomes exponentially
small in `√(|a|³/b)`. With an MLP that does not enforce `Z₂` symmetry,
training may converge to a symmetry-broken wavefunction localized in one well.
The reported energy will match `E₀` but the fidelity with the true symmetric
ground state will drop. The `double_well.yaml` config uses a moderate barrier
and documents this in the metrics: a low fidelity together with a low energy
is the diagnostic signal.

### GPE mode collapse

If `lr` is too large or `diag_shift` in SR is too small, the wavefunction can
collapse toward a delta function, driving `Var(E_L)` up. The trainer logs
variance and step size per iteration, so this is immediately visible. The
shipped config uses `lr = 3e-3` and `diag_shift = 1e-3`, which are stable for
`g ≤ 5`; larger `g` requires lowering `lr` and increasing `n_iter`.

### Contact-interaction cusp

Without the exact cusp `-g/2·|xᵢ-xⱼ|` in the DeepSets pair function, the
local energy is unbounded near `xᵢ = xⱼ`. Training variance grows, and the
optimizer either stalls or produces a wrong energy. The `two_bosons.yaml`
config sets `ansatz.cusp: 1.0`, which must be matched to `system.g`. This is
enforced only by documentation, not by code — a user who changes `g` without
updating `cusp` will see exactly this failure.

---

## Reproducing every figure

All experiments in this repository can be reproduced from the shipped configs
alone. No extra flags, no environment variables, no hidden state.

    # exact reference spectra
    pytest tests/test_harmonic_exact.py -v

    # main table: energy, variance, fidelity per seed
    for cfg in ho_ground ho_excited double_well gross_pitaevskii two_bosons; do
      python scripts/run_vmc.py --config configs/${cfg}.yaml
    done

    # optimizer comparison figure
    python scripts/compare_adam_sr.py --config configs/ho_ground.yaml

    # width-sweep figure
    python scripts/sweep_network_size.py --config configs/ho_ground.yaml

Aggregate everything:

    python - <<'PY'
    import glob
    import json

    for f in sorted(glob.glob("results/*/aggregate.json")):
        agg = json.load(open(f))
        print(f)
        for k in ("final_energy", "final_variance", "energy_error", "fidelity"):
            if k in agg:
                print(f"  {k:>16}: {agg[k]['mean']:.6e} ± {agg[k]['std']:.1e}")
    PY

Every random number used in training comes from a `jax.random.PRNGKey(seed)`
with the seed drawn from the config's `seeds` list. There is no global RNG
state — the code is deterministic per (config, seed) pair up to floating-point
non-associativity and, on GPU, up to atomic-add ordering.

---

## Extending the code

- **New potential.** Add a dataclass to `src/nqs_vmc/systems/` with a
  `potential(x)` callable and the parameters you need (`omega`, `g`, ...), then
  register it in `systems/__init__.py`. Nothing else has to change.
- **New ansatz.** Subclass `flax.linen.Module`, return `log ψ`, register the
  class in `ANSATZ_REGISTRY` in `ansatz.py`. The trainer and sampler will pick
  it up automatically.
- **New optimizer.** Provide a callable with the same signature as
  `SRUpdater.step` in `optimizers.py`. The training loop does not need to know
  about it.
- **Higher dimensions.** The local-energy code already uses `jax.hessian` and
  `jnp.trace`, so it is dimension-agnostic. The sampler is generic in `dim`.
  The ansätze need only reflect the right symmetry (translation, rotation,
  lattice gauge) — the rest of the stack works unchanged.

---

## Development

    # fast tests
    pytest -q

    # with parallel workers
    pytest -q -n auto

    # type hints (optional, if you install mypy)
    mypy src/nqs_vmc --ignore-missing-imports

    # format and lint
    ruff format src scripts tests
    ruff check  src scripts tests

Contribution guidelines: one feature per PR, always add a test, and never
regress the exact-spectrum test in `tests/test_harmonic_exact.py`.

---

## License

MIT. See [`LICENSE`](LICENSE) (add a copy from
[choosealicense.com/licenses/mit](https://choosealicense.com/licenses/mit/)
if not already present).

---

## Citation

If you use this code in academic work, please cite it as:

    @software{nqsvmc2024,
      title     = {{nqs-vmc}: Neural Quantum States with Variational Monte Carlo in {JAX}},
      author    = {Your Name and contributors},
      year      = {2024},
      version   = {0.1.0},
      url       = {https://github.com/<your-user>/jax-nqs-vmc},
      note      = {Reference implementation with exact-diagonalization and split-step cross-checks}
    }

---

<p align="center"><em>Built with JAX, Flax and Optax. Reference solvers in pure
NumPy/SciPy. Every number in this repository is reproducible from the shipped
YAML files.</em></p>
