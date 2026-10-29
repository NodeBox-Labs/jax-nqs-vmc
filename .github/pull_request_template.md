## What does this PR do?

<!-- One sentence. -->

## Which issue does it close?

<!-- Fixes #123 -->

## Type of change

- [ ] Bug fix
- [ ] New feature
- [ ] Documentation
- [ ] Refactor (no behaviour change)
- [ ] Performance
- [ ] CI / build

## Checklist

- [ ] `pytest -q -m "not slow"` passes locally
- [ ] `pytest -q -m slow` passes locally (if applicable)
- [ ] `mypy src/nqs_vmc` passes
- [ ] `ruff check src scripts tests` passes
- [ ] New public functions have docstrings and type hints
- [ ] A new test covers the change (or explains why not)
- [ ] No `results/`, `__pycache__/`, or `*.egg-info/` in the diff
- [ ] Commit messages follow Conventional Commits

## Backward compatibility

- [ ] No public signature changed, **or** the change is documented in `CHANGELOG.md`
- [ ] All five shipped YAMLs still converge within 2σ of the v0.2 numbers

## Reproducibility

- [ ] The PR runs to completion from a fresh `pip install -e ".[dev]"`
- [ ] Any numbers in the PR description are from a run you actually performed