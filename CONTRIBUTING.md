# Contributing to `nqs-vmc`

Contributions are welcome — whether they are bug fixes, new solver references,
extended ansätze, or improved documentation. This project is designed as a
**reference implementation**, so we prioritize **clarity, correctness, and
reproducibility** over raw performance.

---

## Table of contents

- [Quick start for developers](#quick-start-for-developers)
- [What to contribute](#what-to-contribute)
- [Expectations for a PR](#expectations-for-a-pr)
- [PR checklist](#pr-checklist)
- [Design principles](#design-principles)
- [Reporting bugs](#reporting-bugs)
- [Commit message format](#commit-message-format)
- [Code style](#code-style)
- [Code of conduct](#code-of-conduct)

---

## Quick start for developers

```bash
# 1. Fork and clone
git clone https://github.com/<your-user>/jax-nqs-vmc.git
cd jax-nqs-vmc

# 2. Set up the environment
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

# 3. Verify everything works before you start
pytest -q -m "not slow"
```

You should see all fast tests pass in under 10 seconds. If they do not, stop
and open an issue before writing code — the environment is not in a state
where your changes can be meaningfully tested.

---

## What to contribute

| Area | Examples |
|------|----------|
| **New systems** | A 2D harmonic oscillator, a Morse potential, a tilted double well, a lattice potential. |
| **New ansätze** | A complex-valued wavefunction, a convolutional ansatz for 2D, a transformer on particle coordinates. |
| **New optimizers** | K-FAC, Gauss-Newton, line-search SR, or a variant of the quantum geometric tensor. |
| **New reference solvers** | An analytic solution, a Fourier-space ED, a Bethe-ansatz solver for the 1D Bose gas. |
| **Diagnostics** | Better autocorrelation estimators, batching strategies, R-hat convergence checks. |
| **Docs & tutorials** | A worked example, a notebook, a figure, a better error message. |

If you are unsure whether an idea fits, open an issue with the tag
`discussion` before writing code. It is much cheaper to align on scope than to
reject a large PR.

---

## Expectations for a PR

1. **One feature per PR.** Do not mix a new ansatz with a new solver in the
   same branch.

2. **Add a test.** Every new solver goes into `tests/`; every new ansatz is
   exercised by at least one `run_vmc.py` config. The end-to-end VMC test
   (`pytest -m slow`) must still pass.

3. **Reproduce the numbers.** If you change the physics, update the
   "Verified numbers" table in the README with the new values and the machine
   you ran them on.

4. **Do not break the baseline.** The shipped configs (`configs/*.yaml`) are
   the contract. If `ho_ground.yaml` stops converging to `E₀ = 0.5 ± 0.005`,
   the PR is not ready.

5. **Format.** Run `ruff format .` and `ruff check .` before committing.

6. **Type hints.** Public functions should carry type hints. Optional:
   `mypy src/nqs_vmc --ignore-missing-imports` should pass.

---

## PR checklist

Copy this checklist into your PR description and tick every box:

- [ ] `pytest -q -m "not slow"` passes locally.
- [ ] `pytest -q -m slow` passes (if applicable to your change).
- [ ] A new entry was added to the relevant section of the README
      (methods, results, or roadmap).
- [ ] The roadmap checkbox that corresponds to your PR is ticked.
- [ ] Your commit messages follow the
      [Conventional Commits](https://www.conventionalcommits.org/) format
      (e.g. `feat(ansatz): add convolutional wavefunction for 2D`).
- [ ] The new code has a docstring that explains the *physics* in addition to
      the mechanics (what Hamiltonian, what boundary conditions, what
      reference it validates against).
- [ ] No new dependency was added without a discussion in an issue first.

---

## Design principles

This repository is deliberately opinionated. Please respect the following
when extending it:

- **Reproducibility first.** Every result must be reachable from a YAML config
  and a fixed seed. No magic constants buried in scripts. If a number appears
  in the README, there is a `python scripts/run_vmc.py --config ...` that
  reproduces it.

- **Log-domain everywhere.** `log ψ` is the canonical representation. Do not
  introduce a `ψ`-domain code path without a strong reason; it will overflow
  for wide wavefunctions and it will break the zero-variance diagnostic.

- **Independent references.** Reference solvers must live in
  `src/nqs_vmc/reference/` and must not import anything from `ansatz.py`,
  `sampler.py`, or `trainer.py`. Cross-checks only work if the two sides are
  truly independent. A reference solver that shares code with the VMC stack
  is not a reference solver — it is a mirror.

- **Same training loop.** New optimizers must plug into the existing
  `Optimizer` protocol. Do not fork the training loop. If Adam and SR diverge
  in how they are trained, the head-to-head comparison in the README becomes
  meaningless.

- **Failures are documented.** If a new ansatz or a new potential exhibits a
  known failure mode (excited-state collapse, symmetry breaking, mode
  collapse, cusp blow-up), it belongs in the "Known failure modes" section of
  the README, not hidden in a comment.

- **x64 is not negotiable.** `jax_enable_x64 = True` is set globally. Do not
  propose a float32 default. Excited-state gaps and tunneling splittings are
  `O(10⁻²)` or smaller, and float32 washes them out.

---

## Reporting bugs

Please open an issue with:

- The output of `python -c "import jax, numpy, flax; print(jax.__version__, numpy.__version__, flax.__version__)"`.
- The config file (`configs/*.yaml`) that reproduces the bug.
- The seed.
- The full traceback, or the relevant slice of `metrics.json` / `history.npz`.
- If the bug is a numerical one: the exact command, the number you got, and
  the number you expected.

Physics-related bugs (wrong energy, wrong cusp, wrong reference value) are
especially welcome — those are the hardest to catch and the most valuable to
fix. A reproduction that is 10 lines of Python is worth more than a
paragraph of prose.

---

## Commit message format

We follow [Conventional Commits](https://www.conventionalcommits.org/):

```
<type>(<scope>): <short summary>

<optional body>

<optional footer>
```

**Types:** `feat`, `fix`, `docs`, `test`, `refactor`, `perf`, `chore`, `ci`.

**Scopes:** `ansatz`, `sampler`, `local_energy`, `optimizers`, `trainer`,
`systems`, `reference`, `configs`, `tests`, `scripts`, `docs`, `ci`.

**Examples:**

```
feat(ansatz): add transformer wavefunction for N particles
fix(reference): correct chemical potential sign in split-step GPE
docs(readme): add result card for two-boson config
test(local_energy): check autograd against analytic Gaussian
```

One commit should be one logical change. If you find yourself writing
`and` in the subject line, split the commit.

---

## Code style

- **Formatter and linter:** `ruff format .` and `ruff check .`.
- **Line length:** 100 characters.
- **Type hints:** required on every public function; optional but encouraged on
  internal helpers.
- **Docstrings:** required on every public module and class. The docstring
  should state the *physics* (Hamiltonian, boundary conditions, units) and
  the *contract* (shapes, units, invariants).
- **No bare `except`.** If you catch an exception, catch a specific one.
- **No `print` in library code.** Use the logger, or `tqdm` for progress
  bars. `print` is fine in `scripts/`.
- **Deterministic by default.** Any function that touches randomness must
  take an explicit `key` argument (JAX PRNG), never a global seed.

---

## Code of conduct

Be kind. This is a small project, but it operates on the same norms as the
wider scientific-Python community. Harassment, dismissiveness, or "this is
obviously wrong" without a reference is not acceptable. Constructive
disagreement with a citation is always welcome.

Concretely:

- Disagree with the physics: cite a paper, a textbook, or a calculation.
- Disagree with the code: propose a test that would demonstrate the issue.
- Disagree with a design choice: open an issue, do not open a PR that undoes
  it without discussion.

If someone's contribution does not match the project's standards, say so
plainly and kindly, and point to the specific principle or test that is
violated. If you are on the receiving end, assume good faith and ask for
clarification before responding.

---

<p align="center"><em>This project is a teaching resource first and a research
tool second. Contributions that make it clearer, more correct, or easier to
reproduce are always welcome.</em></p>