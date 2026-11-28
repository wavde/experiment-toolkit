# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- `judge_audit`: real-data audit of an LLM judge against expert human
  judgments on MT-Bench (Zheng et al., 2023, CC BY 4.0), downloaded at runtime
  from a pinned revision and checked against SHA-256 hashes. Lines up human
  and GPT-4 verdicts item by item, estimates the judge's win-rate bias with a
  question-level bootstrap, splits agreement by which model humans preferred,
  and runs a known-answer resampling check of judge-only, human-only and
  PPI++ readouts. New `data` extra (`pyarrow`) for reading the parquet files.
- `scripts/judge_audit_mtbench.py`: seeded, single-command audit that writes
  `docs/judge_audit_results.md` and `docs/judge_audit.png`. CI smoke-runs it
  on synthetic data with the real schema.
- `judge_drift`: canary-set drift monitor. `simulate_canary` re-judges a fixed
  set of stored outputs daily under a `SilentDrift`, with a `repeatability`
  setting for how often the judge repeats its verdict. `canary_alarm` is a
  paired, Bonferroni-corrected monitor on those verdicts, and
  `canary_detection_prob` / `canary_items_for_power` give closed-form
  detection probability and canary size.
- `scripts/canary_drift_sim.py`: seeded, single-command simulation that writes
  `docs/canary_drift_results.md` and `docs/canary_drift.png`. CI smoke-runs it.
- `judge_drift`: silent judge drift with no version flag. `SilentDrift` changes
  the judge from an unknown day (step or linear ramp, per arm) or shifts true
  quality with the judge unchanged. `analyze_silent` compares the judge-only
  readout with PPI++ using launch-week or whole-test random labels, and
  `first_alarm` is a windowed, Bonferroni-corrected drift monitor for judge
  scores or for judge-human agreement split by the human label. Closed forms
  are included for the judge-only mean, the large-sample bias of launch-week
  labels, and the chance that one monitor check flags.
- `scripts/silent_drift_sim.py`: seeded, single-command simulation that writes
  `docs/silent_drift_results.md` and `docs/silent_drift.png`. CI smoke-runs it.
- `bandit_eval`: measuring the lift of a creative that a bandit picked from
  many generated variants. Off-policy estimators that use logged propensities
  (`ope_lift` with IPS, SNIPS or doubly robust; `policy_terms`), the naive
  dashboard comparison (`naive_lift`), a selection helper (`pick_winner`), a
  minimal batched Thompson-sampling log generator with click-rate drift and an
  optional evaluation slice whose outcomes the bandit never sees
  (`simulate_bandit_log`), and a closed-form winner's-curse reference for a
  fixed equal split (`expected_winner_bias_fixed`). Evaluation only; still no
  bandit framework.
- `scripts/bandit_eval_sim.py`: seeded, single-command simulation that writes
  `docs/bandit_eval_results.md` and `docs/bandit_eval.png`. CI smoke-runs it.
- `judge_drift`: a judge model upgraded mid-test. Simulates a switch to a new
  judge (with per-arm error rates) partway through an A/B test, and compares
  the pooled judge score, the pre-upgrade judge score, and PPI++ with labels
  spread across the test, labels only before the upgrade, or labels
  stratified by judge version. Closed forms are included for the naive means
  and SEs and for the large-sample bias of front-loaded labels.
- `scripts/judge_drift_sim.py`: seeded, single-command simulation that writes
  `docs/judge_drift_results.md` and `docs/judge_drift.png`. CI smoke-runs it.
- `ppi_covariates`: PPI++ with several control variates, combining the LLM
  judge score with a pre-treatment covariate in the CUPED style. Includes
  closed-form oracle variance `Var(Y)/n · (1 − R² · N/(N+n))`, label budgets
  for a target power (`labels_for_power`), and a simulator. With an
  uninformative covariate it reproduces `llm_judge`'s PPI++ exactly.
- `scripts/ppi_covariates_sim.py`: seeded, single-command simulation that
  writes `docs/ppi_covariates_results.md` and `docs/ppi_covariates.png`.
  CI smoke-runs it.
- `offline_eval`: offline LLM evals vs online A/B tests. Simulates segment
  mix shift between the eval set and live traffic, judge error that differs
  between baseline and candidate outputs, and online-only costs. It compares
  the raw judge readout, a readout post-stratified to the traffic mix, and
  PPI++ with human labels per segment against the online A/B result.
  Closed-form expectations are included for every readout.
- `scripts/offline_online_sim.py`: seeded, single-command simulation that
  writes `docs/offline_online_results.md` and `docs/offline_online.png`.
- `--smoke` mode for both evidence scripts. CI now runs both in smoke mode,
  so a clean-install break in an evidence pipeline fails the build.
- `llm_judge`: A/B tests whose primary metric is scored by an LLM judge.
  Simulates binary true quality and a judge with per-arm sensitivity /
  specificity, which allows differential error between arms. Provides the naive
  judge-only analysis (`naive_judge_diff`) and PPI / PPI++ correction
  (`ppi_mean_diff`; Angelopoulos et al. 2023) that uses a small human-labeled
  sample from each arm. Also adds closed-form helpers for the expected naive
  bias, SEs, rejection rates and sample-size inflation.
- `scripts/llm_judge_sim.py`: seeded, single-command simulation that writes
  `docs/llm_judge_results.md` and `docs/llm_judge.png`, with every simulated
  number shown next to its closed-form value. Known-answer and Type-I tests
  live in `tests/test_llm_judge.py`.

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
