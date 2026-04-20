# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2026-04-19

### Added
- Initial public release.
- `sample_size`: two-sample sample size / MDE / power calculations.
- `cuped`: variance reduction via pre-experiment covariate (Deng et al. 2013).
- `ratio`: delta-method variance for ratio metrics (Deng et al. 2018).
- `sequential`: always-valid mSPRT sequential test (Johari et al. 2017).
- `cli`: thin command-line wrapper exposing `experiment-toolkit` for quick checks.

### Notes
- Tested on Python 3.10, 3.11, 3.12.
- mSPRT formula uses `V = 2σ²/n_per_arm` with shrinkage `τ²/(V+τ²)` -- a corrected
  variant of an earlier draft that had an off-by-2 in the prior scaling and
  inflated Type-I error to ~17% at α=0.05. Empirical Type-I error of the
  shipped formula is ≤ α under H0; see `tests/test_sequential.py`.
