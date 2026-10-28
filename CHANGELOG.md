# Changelog

## [0.3.0] 

### Added
- CLI (`nqs-vmc run|sweep|compare|plot`) via `pyproject.toml` entry point.
- Orbax checkpointing with resume.
- Structured JSON Lines per-iteration logging (`metrics.jsonl`).
- Gradient clipping, warmup+cosine schedules, three SR solvers
  (`solve`, `lstsq`, `pinv`), configurable via `optimizer.*`.
- Correlation-aware standard errors (multiply by `√(2τ_int)`).
- Self-normalized overlap estimator for the orthogonality penalty.
- 9 new tests covering: sampler correctness, config validation, determinism,
  SR stability, GPE overflow clamping, checkpoint round-trip, `τ_int`
  correction, overlap estimator, CLI smoke.

### Fixed
- x64 now enabled from every `nqs_vmc.*` module, not only the top-level
  package (§3.3).
- `g·exp(2·logψ)` is clamped to `exp(60)` (§3.1B).
- SR no longer silently produces NaN on ill-conditioned `S` (§3.1F).
- Config validation is enforced at construction time, not only by docs
  (§3.3).

### Changed
- `n_samples` must be divisible by `n_chains` (previously silently dropped).
- `ansatz.cusp` must equal `system.g` for `DeepSets` (previously
  documented only).
- `energy_and_variance` takes an optional `tau_int` argument.

### Preserved
- All 5 shipped YAMLs produce the same `final_energy` within 2σ.
- All public signatures are backward-compatible (new args are keyword-only).

## [0.2.0] — earlier
- Initial stable release.