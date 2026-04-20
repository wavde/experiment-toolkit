# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.0] - 2026-04-19

### Added
- `cs_did`: Callaway & Sant'Anna (2021) staggered difference-in-differences
  with never-treated controls and a unit cluster bootstrap for inference.
  Includes `simulate_staggered_panel` for teaching / tests. Robust to
  treatment-effect heterogeneity across cohorts — unlike TWFE.
- `sensitivity`: E-value (VanderWeele & Ding 2017) for risk-ratio / odds-ratio
  / standardised-mean-difference effects and Rosenbaum bounds for matched-pair
  Wilcoxon signed-rank tests (`rosenbaum_wilcoxon_bounds`,
  `rosenbaum_gamma_threshold`).
- `sequential`: `msprt_cuped_pvalue` composes CUPED variance reduction with
  mSPRT to tighten always-valid peeks.
- `sample_size`: `rho` parameter added to `sample_size_for_mde` / `mde_for_n`
  for CUPED-aware power planning. Zero-default keeps v0.1 behaviour.
  New `cuped_sample_size_summary` reports side-by-side naive vs CUPED sample
  sizes and percent savings.

### Notes
- Pure-numpy / scipy; no new runtime dependencies.
- All v0.1.0 APIs remain backwards compatible.

## [0.1.0] - 2026-04-19

### Added
- `sample_size`: two-sample sample size / MDE / power calculations.
- `cuped`: variance reduction via pre-experiment covariate (Deng et al. 2013).
- `ratio`: delta-method variance for ratio metrics (Deng et al. 2018).
- `sequential`: always-valid mSPRT sequential test (Johari et al. 2017).
- `cli`: command-line wrapper exposing `sample-size` and `mde` subcommands.

### Notes
- Tested on Python 3.10, 3.11, 3.12.
- mSPRT formula uses `V = 2σ²/n_per_arm` with shrinkage `τ²/(V+τ²)` — a corrected
  variant of an earlier draft that had an off-by-2 in the prior scaling and
  inflated Type-I error to ~17% at α=0.05. Empirical Type-I error of the
  shipped formula is ≤ α under H0; see `tests/test_toolkit.py::test_msprt_always_valid_type1_error`.
