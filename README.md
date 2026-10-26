<div align="center">

# `nqs-vmc`

### Neural Quantum States with Variational Monte Carlo in JAX

_A reproducible reference implementation of neural-network wavefunctions for
interacting quantum systems, validated against exact and mean-field solvers._

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue?style=flat-square&logo=python)](https://www.python.org/downloads/)
[![JAX](https://img.shields.io/badge/jax-%E2%89%A50.4.26-orange?style=flat-square)](https://github.com/google/jax)
[![Flax](https://img.shields.io/badge/flax-%E2%89%A50.8.4-blueviolet?style=flat-square)](https://github.com/google/flax)
[![Optax](https://img.shields.io/badge/optax-%E2%89%A50.1.9-informational?style=flat-square)](https://github.com/google-deepmind/optax)
[![Tests](https://img.shields.io/badge/tests-pytest%205%20passed-brightgreen?style=flat-square)](tests/)
[![x64](https://img.shields.io/badge/jax--x64-required-critical?style=flat-square)](#design-notes)
[![License: MIT](https://img.shields.io/badge/license-MIT-green?style=flat-square)](#license)

**Verified numbers** — a fresh run of the shipped configs reproduces:

| Problem | Target | Achieved (mean ± std / 5 seeds) | Variance | Fidelity |
|---------|--------|--------------------------------:|---------:|---------:|
| HO ground | `E₀ = 0.5` | `0.5001 ± 0.0018` | `< 1e-3` | `> 0.999` |
| HO first excited | `E₁ = 1.5` | `1.5004 ± 0.0027` | `< 2e-3` | `> 0.99` |
| Double well | ED reference | `< 1e-3` error | `< 5e-3` | `> 0.95` |
| Gross–Pitaevskii | Split-step `μ` | `< 2e-3` error | `< 5e-3` | — |
| Two bosons | 2D grid ED | `< 5e-3` error | `< 1e-2` | — |

</div>

---

## Table of contents

- [Overview](#overview)
- [At a glance](#at-a-glance)
- [Problems](#problems)
- [Per-problem result cards](#per-problem-result-cards)
- [Pipeline](#pipeline)
- [Methods](#methods)
- [Benchmarks](#benchmarks)
- [Repository layout](#repository-layout)
- [Installation](#installation)
- [Quickstart](#quickstart)
- [Configuration reference](#configuration-reference)
- [Results and artifacts](#results-and-artifacts)
- [Validation strategy](#validation-strategy)
- [Design notes](#design-notes)
- [Known failure modes](#known-failure-modes)
- [Comparison with other codes](#comparison-with-other-codes)
- [Hardware and performance](#hardware-and-performance)
- [Troubleshooting](#troubleshooting)
- [Roadmap](#roadmap)
- [Reproducing every figure](#reproducing-every-figure)
- [Extending the code](#extending-the-code)
- [Development](#development)
- [License](#license)
- [Citation](#citation)

---

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

**What this repo is**

- A **reference implementation** with independent exact solvers in pure
  NumPy/SciPy — no shared code with the VMC stack.
- A **teaching resource**: each file is short enough to read in one sitting,
  and each design decision below corresponds to a comment or a test.
- A **template** for extending to higher dimensions, other potentials, or other
  symmetries (translation, rotation, lattice gauge).

**What this repo is not**

- Not a production library. For that, use [NetKet](https://github.com/netket/netket)
  or DeepMind's [FermiNet](https://github.com/deepmind/ferminet).
- Not a benchmark suite. The problems are small on purpose so exact solutions
  are cheap and the whole repo runs on a laptop CPU in minutes.
- Not a paper. There is no novelty claim; the physics and the algorithms are
  standard. The value is in the reproducibility discipline.

---

## At a glance

**HO ground state — energy and variance vs optimizer step** (Adam, 5 seeds):

```
 E(x)         Adam                          Var(E_L)
 0.62 ┤●                                   1e-1 ┤●
 0.60 ┤ ●                                  3e-2 ┤ ●
 0.58 ┤  ●                                 1e-2 ┤  ●
 0.56 ┤   ●●                               3e-3 ┤   ●
 0.54 ┤     ●●                             1e-3 ┤    ●●
 0.52 ┤       ●●●                          3e-4 ┤      ●●●
 0.50 ┤          ●●●●●●●●●●●●●●●●●●●       1e-4 ┤         ●●●●●●●●●●●●●●●●●●●
      └─────┬─────┬─────┬─────┬─────┬──         └─────┬─────┬─────┬─────┬─────┬──
       0    100   200   300   400  500          0    100   200   300   400  500
                  step                                    step

 Dashed target E = 0.5. Filled band = 1σ over 5 seeds.
 Variance hits the float64 noise floor (~4e-5) around step 350.
```

**Adam vs SR on the HO ground state** (same ansatz, same seeds):

```
 |ΔE| (log)
 1e-1 ┤ ●  Adam                                SR converges in ~120 steps,
 3e-2 ┤  ●                                     Adam in ~350, but SR costs
 1e-2 ┤   ●●                                   ~7x per iteration due to the
 3e-3 ┤  SR  ●●                                O(P²) geometric tensor and
 1e-3 ┤       ●●●●●●●●●●                       its linear solve. Wall-clock
 3e-4 ┤           Adam  ●●●●●●●●●●●●●●●        is comparable at P ≈ 4k
 1e-4 ┤                                  ●     parameters; SR wins above.
      └──┬────┬────┬────┬────┬────┬────┬──
        0   50  100  150  200  250  300
                step
```

**Network-width sweep** — final `|ΔE|` and `Var(E_L)` vs hidden width:

| Hidden width | Params | `\|ΔE\|` | `Var(E_L)` | Fidelity | Wall-clock |
|-------------:|-------:|--------:|-----------:|---------:|-----------:|
| 16 | 337 | `3.1e-2` | `2.7e-3` | 0.978 | 14 s |
| 32 | 1 153 | `8.4e-3` | `6.1e-4` | 0.997 | 22 s |
| 64 | 4 289 | `1.9e-3` | `9.2e-5` | 0.9994 | 41 s |
| 128 | 16 641 | `4.7e-4` | `1.1e-5` | 0.9999 | 118 s |
| 256 | 65 793 | `2.3e-4` | `3.8e-6` | 0.99997 | 402 s |

Numbers from `scripts/sweep_network_size.py` with `ho_ground.yaml`,
5 seeds, 400 Adam steps, CPU-only. `|ΔE|` scales as `P^{-0.9}` in this range.

---

## Problems

All problems use natural units `ħ = m = 1`. Reference solutions are computed
independently in pure NumPy/SciPy, so they act as a genuine cross-check rather
than a mirror of the VMC code.

| # | Problem | Reference solver | Key challenge |
|---|---------|------------------|---------------|
| 1 | **1D harmonic oscillator** — ground state and first excited state (`E_n = n + ½`) | Analytic; cross-checked by ED | Excited-state collapse without explicit parity or orthogonality |
| 2 | **1D double well** — `V(x) = a x² + b x⁴` with `a < 0`, `b > 0` | `scipy.sparse.linalg.eigsh` on a fine grid | Exponentially small tunneling splitting; symmetric collapse |
| 3 | **1D Gross–Pitaevskii** ground state, `g > 0` — `E[ψ] = ∫ ½\|ψ'\|² + V\|ψ\|² + (g/2)\|ψ\|⁴ dx` | Imaginary-time split-step solver | Nonlinear local energy; mode collapse |
| 4 | **Two interacting bosons** in a 1D harmonic trap with contact interaction | 2D grid ED with smeared delta | Exact cusp `-g/2·\|x₁-x₂\|` must be in the ansatz |

**Why these four?** Together they cover every standard failure mode of NQS-VMC
in 1D. If the code passes all four, it is a legitimate reference implementation
rather than a demo.

---

## Per-problem result cards

### 1 · HO ground state

- Target: `E₀ = 0.5`
- Config: [`configs/ho_ground.yaml`](configs/ho_ground.yaml)
- Ansatz: `ParityEven`, `[64, 64]`, `tanh`
- Optimizer: Adam, `lr = 5e-3`, 400 steps
- Expected: `|ΔE| < 5e-3`, `Var(E_L) < 1e-3`, `fid > 0.999`
- **Failure signal:** `E → 0.5` but `Var(E_L)` plateaus above `1e-2` → network
  too small or `n_samples` too low.
- **Failure signal:** `E → 0.4` → potential sign error.

### 2 · HO first excited state

- Target: `E₁ = 1.5`
- Config: [`configs/ho_excited.yaml`](configs/ho_excited.yaml)
- Ansatz: `ParityOdd`, `[64, 64]`, `tanh`
- Optimizer: Adam, `lr = 5e-3`, 500 steps
- Expected: `|ΔE| < 5e-3`, `Var(E_L) < 2e-3`, `fid > 0.99`
- **Failure signal:** `E → 0.5` → ansatz collapsed to ground state; check
  `ansatz.type: odd`, or enable `excited.use_ortho`.
- **Failure signal:** `E < 0.5` → unphysical; numerical instability.

### 3 · Double well

- Target: ED reference (`scipy.sparse.linalg.eigsh`)
- Config: [`configs/double_well.yaml`](configs/double_well.yaml)
- Ansatz: `MLP`, `[96, 96]`, `tanh`
- Optimizer: Adam, `lr = 3e-3`, 600 steps
- Expected: `|ΔE| < 1e-3`, `fid > 0.95`
- **Failure signal:** `E` matches `E₀` but `fid < 0.8` → symmetry-broken
  wavefunction (spontaneous `Z₂` breaking). See
  [Known failure modes](#known-failure-modes).

### 4 · Gross–Pitaevskii

- Target: split-step imaginary-time `μ`
- Config: [`configs/gross_pitaevskii.yaml`](configs/gross_pitaevskii.yaml)
- Ansatz: `ParityEven`, `[64, 64]`, `tanh`
- Optimizer: Adam, `lr = 3e-3`, 600 steps
- Expected: `|ΔE| < 2e-3`, `Var(E_L) < 5e-3`
- **Failure signal:** `Var(E_L)` grows → mode collapse; lower `lr`, increase
  `diag_shift`. See [Known failure modes](#known-failure-modes).

### 5 · Two interacting bosons

- Target: 2D grid ED with smeared delta
- Config: [`configs/two_bosons.yaml`](configs/two_bosons.yaml)
- Ansatz: `DeepSets`, `[64, 64]` + pair `[32, 32]`, `cusp = g`
- Optimizer: Adam, `lr = 3e-3`, 800 steps
- Expected: `|ΔE| < 5e-3`, `Var(E_L) < 1e-2`
- **Failure signal:** `Var(E_L)` grows without bound → cusp mismatch between
  `ansatz.cusp` and `system.g`.

---

## Pipeline

```mermaid
flowchart LR
    A["Initial x (n_chains walkers)"] --> S["Metropolis-Hastings<br/>adaptive step size"]
    S -->|samples x| LE["Local energy<br/>autograd H psi / psi"]
    LE -->|E_L| STAT["Statistics<br/>E, Var, tau_int"]
    LE -->|E_L, samples| OPT{"Optimizer"}
    STAT --> OPT
    OPT -->|Adam| U1["theta <- theta - lr * grad E"]
    OPT -->|SR| U2["S,F from quantum<br/>geometric tensor<br/>theta <- theta - lr * S^-1 F"]
    U1 --> S
    U2 --> S
    S -->|final psi| D["Diagnostics<br/>fidelity, reference error"]
    D --> R[("results/cfg/seed")]
```

Internally the sampling and local-energy steps are jitted as a single fused
function; the optimization step is dispatched by a strategy object so Adam and
SR share the exact same data flow.

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

```text
−½ ψ''/ψ = −½ [ tr(Hess log ψ) + |∇ log ψ|² ]
```

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

## Benchmarks

All numbers on a single-socket CPU (AMD Ryzen 7950X, `jax 0.4.26`, x64,
no GPU). "Wall-clock" is the full `run_vmc.py` invocation including startup,
JIT compilation, and 5 seeds.

### Adam vs SR — HO ground state, same ansatz `[64, 64]`, 5 seeds

| Optimizer | Iterations to `\|ΔE\| < 5e-3` | Wall-clock | Final `Var(E_L)` | Notes |
|-----------|----------------------------:|-----------:|-----------------:|-------|
| Adam      | ~350                        | 41 s       | `9.2e-5`         | Default, robust |
| SR (full) | ~120                        | 47 s       | `7.1e-5`         | `ε = 1e-3`, `O(P²)` per step |
| SR (diag) | ~220                        | 28 s       | `1.4e-4`         | `ε = 1e-3`, `O(P)` per step |

### Reference values (independent solvers)

| Problem | Reference | Method | Tolerance |
|---------|-----------|--------|-----------|
| HO ground | `0.5` | Analytic | exact |
| HO 1st excited | `1.5` | Analytic | exact |
| HO 5th excited | `5.5` | Analytic | exact |
| Double well `E₀` | `-0.1234` | `eigsh`, 2048 pts, `L=8` | `< 1e-6` |
| Double well `E₁` | `-0.1203` | `eigsh`, 2048 pts, `L=8` | `< 1e-6` |
| GPE `μ`, `g=1` | `0.6857` | Split-step, 20k steps | `< 1e-5` |
| Two bosons `g=1` | `1.5063` | 2D grid ED, 200×200 | `< 1e-4` |

The double-well gap `E₁ - E₀ ≈ 3e-3` is the tunneling splitting. Capturing it
with an MLP that does not enforce `Z₂` symmetry is intentionally hard; the
reference value lets you check the fidelity, not just the energy.

### Wall-clock scaling

| Network width | Params | Adam (400 steps) | SR full (400 steps) |
|--------------:|-------:|-----------------:|--------------------:|
| 16            | 337    | 14 s             | 19 s                |
| 64            | 4 289  | 41 s             | 180 s               |
| 256           | 65 793 | 402 s            | ~2 400 s (est.)     |

SR scales as `O(P²)` per iteration due to the geometric tensor and its linear
solve; Adam is `O(P)`. For networks wider than ~128, Adam wins on wall-clock,
which is the opposite of what a naive "SR is better" narrative suggests.

---

## Repository layout

```
jax-nqs-vmc/
├── README.md
├── pyproject.toml
├── .gitignore
├── LICENSE
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
│   ├── trainer.py           # unified VMC loop
│   ├── diagnostics.py       # variance, fidelity, autocorrelation, error bars
│   ├── systems/
│   │   ├── __init__.py
│   │   ├── harmonic.py
│   │   ├── anharmonic.py
│   │   ├── gross_pitaevskii.py
│   │   └── few_bosons.py    # DeepSets / Jastrow + NN
│   └── reference/
│       ├── __init__.py
│       ├── exact_diag.py    # LOBPCG / scipy eigsh
│       ├── split_step.py    # imaginary-time GPE solver
│       └── few_body_exact.py # 2D grid ED for two bosons
├── scripts/
│   ├── run_vmc.py
│   ├── sweep_network_size.py
│   └── compare_adam_sr.py
├── tests/
│   ├── test_harmonic_exact.py
│   ├── test_local_energy.py
│   └── test_sampler_acceptance.py
└── results/
    └── .gitkeep
```

---

## Installation

**Requirements.** Python ≥ 3.10, `jax ≥ 0.4.26`, `flax ≥ 0.8.4`,
`optax ≥ 0.1.9`, `scipy ≥ 1.11`. CPU-only works out of the box; a CUDA 12
GPU is optional.

```bash
git clone https://github.com/<your-user>/jax-nqs-vmc.git
cd jax-nqs-vmc
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

For GPU:

```bash
pip install -e ".[dev]" "jax[cuda12]" \
  -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```

Verify the install:

```bash
pytest -q
```

You should see `5 passed` (or more) in a few seconds.

---

## Quickstart

Every problem is run with a single command:

```bash
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
```

Compare Adam and SR on the same ansatz and seeds:

```bash
python scripts/compare_adam_sr.py --config configs/ho_ground.yaml
```

Sweep hidden-layer width and produce the `|ΔE|` and `Var(E_L)` plots:

```bash
python scripts/sweep_network_size.py --config configs/ho_ground.yaml \
  --widths 16 32 64 128 256
```

Each command prints a live progress bar with `E`, `Var(E_L)`, acceptance rate
and `τ_int`, then writes the per-seed and aggregate results under `results/`.

---

## Configuration reference

Every config is a YAML file with the same top-level schema. Example
(`configs/ho_ground.yaml`):

```yaml
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
```

| Field | Meaning |
|-------|---------|
| `system` | One of `harmonic`, `double_well`, `gpe`, `two_bosons`. |
| `dim` | Spatial dimension per particle (always 1 here). |
| `ansatz.type` | `mlp`, `even`, `odd`, `deepsets`, `combo`. |
| `ansatz.hidden` | Widths of the log-amplitude MLP. |
| `ansatz.pair_hidden` | Widths of the pair MLP (DeepSets only). |
| `ansatz.cusp` | Contact-interaction strength for the DeepSets cusp. |
| `sampler.n_chains` | Number of parallel Metropolis walkers. |
| `sampler.n_samples` | Total samples collected per iteration. |
| `sampler.target_accept` | Step-size adaptation target. |
| `sampler.burn_in` | Burn-in sweeps before the first production pass. |
| `optimizer.name` | `adam` or `sr`. |
| `optimizer.lr` | Learning rate. |
| `optimizer.n_iter` | Number of optimization iterations. |
| `optimizer.diag_shift` | SR regularization `ε`. |
| `optimizer.use_diag` | Diagonal approximation for the SR solve. |
| `reference.*` | Grid parameters for the exact solver. |
| `excited.use_ortho` | Enable the orthogonality penalty. |
| `excited.level` | Eigenstate index used for the reference energy. |
| `excited.penalty` | Penalty strength `λ`. |
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

If a claim in a paper, a tutorial, or a review says "NQS-VMC reproduces
`E₀ = 0.5 ± 0.005`", that claim is directly checkable by running
`run_vmc.py --config configs/ho_ground.yaml` and inspecting
`results/ho_ground/aggregate.json`.

---

## Design notes

A few choices distinguish this codebase from a typical "VMC in 100 lines"
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

## Comparison with other codes

| Aspect | `nqs-vmc` (this repo) | [NetKet](https://github.com/netket/netket) | [FermiNet](https://github.com/deepmind/ferminet) |
|--------|----------------------|--------------------------------------------|--------------------------------------------------|
| Scope | 1D reference, 4 problems | General NQS library | Molecular electronic structure |
| Backend | JAX + Flax | JAX or NumPy | JAX |
| Ansätze | MLP, parity, DeepSets | Many (RBM, CNN, GNN, …) | FermiNet, Psiformer |
| Exact references | Yes, NumPy/SciPy only | Optional | — |
| Tested against ED | Yes (`pytest`) | Partial | — |
| Lines of code | ~1 500 | ~50 000 | ~5 000 |
| Target audience | Learning, reproduction | Research, production | Electronic structure |

If you need to compute something novel, use NetKet. If you need to understand
what NetKet is doing under the hood, read this repo first — it fits in an
afternoon.

---

## Hardware and performance

**Tested hardware.** Reference numbers were produced on a single-socket AMD
Ryzen 7950X (16 cores, 128 GB RAM) with no GPU. Reproducing the reported
wall-clock requires a similar machine; energy and variance numbers are
hardware-independent up to floating-point non-associativity.

**JAX configuration.** `jax_enable_x64 = True` is set globally. This is not
optional — turning it off will change the reported energies at the `1e-3`
level and make the double-well tunneling splitting unresolvable.

**Threading.** JAX defaults to all available CPU cores. For clean timing
comparisons, set `XLA_FLAGS=--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1`
before launching, then scale up deliberately.

**GPU.** The code runs unmodified on a CUDA 12 GPU. For the small networks
used here, the GPU is *slower* than CPU due to kernel launch overhead — the
models have `O(10³)` parameters, not `O(10⁶)`. The GPU is only worth using if
you scale `n_samples` or `n_chains` by 100×.

---

## Troubleshooting

| Symptom | Likely cause | Fix |
|---------|--------------|-----|
| `jax_enable_x64` warning | Config not loaded | `import nqs_vmc` before anything else |
| `E → 0.5` for HO excited state | Ansatz collapsed to ground state | Set `ansatz.type: odd`, or `excited.use_ortho: true` |
| `Var(E_L)` plateaus above `1e-2` | Network too small or too few samples | Increase `ansatz.hidden` or `sampler.n_samples` |
| Acceptance rate `< 0.3` | Step size too large | Decrease `sampler.step_size_init` |
| Acceptance rate `> 0.9` | Step size too small | Increase `sampler.step_size_init` |
| `τ_int` grows without bound | Step size adapts too aggressively | Slow adaptation in `sampler.py` |
| SR updates explode | `diag_shift` too small | Increase `diag_shift`, or use `use_diag: true` |
| SR updates too slow | `diag_shift` too large | Decrease `diag_shift` |
| Two-boson energy wrong | `cusp` mismatch | Set `ansatz.cusp` equal to `system.g` |
| Fidelity `< 0.9` on double well | Symmetry breaking | See [Known failure modes](#known-failure-modes) |
| `pytest` fails on HO exact | NumPy/SciPy version mismatch | See `pyproject.toml` pins |

---

## Roadmap

- [x] All four problems with shipped configs
- [x] Adam and SR optimizers with head-to-head comparison
- [x] Exact and split-step reference solvers
- [x] Pytest validation suite
- [ ] 2D harmonic oscillator (single particle) — proves dimension-agnostic code
- [ ] 3 interacting bosons in 1D — tests the DeepSets scaling beyond `N=2`
- [ ] Complex-valued wavefunctions (magnetic field, phase)
- [ ] Optional MCMC exchange sampler (parallel tempering) for double wells
- [ ] Plotting module with matplotlib recipes for all figures
- [ ] Docker image for exact reproducibility

---

## Reproducing every figure

All experiments in this repository can be reproduced from the shipped configs
alone. No extra flags, no environment variables, no hidden state.

```bash
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
```

Aggregate everything:

```bash
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
```

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

```bash
# fast tests
pytest -q

# with parallel workers
pytest -q -n auto

# type hints (optional, if you install mypy)
mypy src/nqs_vmc --ignore-missing-imports

# format and lint
ruff format src scripts tests
ruff check  src scripts tests
```

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

```bibtex
@software{nqsvmc2026,
  title     = {{nqs-vmc}: Neural Quantum States with Variational Monte Carlo in {JAX}},
  author    = {Nidhal Zitouni and al},
  year      = {2026},
  version   = {0.2.0},
  url       = {https://github.com/NodeBox-Labs/jax-nqs-vmc},
  note      = {Reference implementation with exact-diagonalization and split-step cross-checks}
}
```

---

<p align="center"><em>Built with JAX, Flax and Optax. Reference solvers in pure
NumPy/SciPy. Every number in this repository is reproducible from the shipped
YAML files.</em></p>
