# experiment-toolkit

> Small, well-tested utilities for online controlled experiments.

![CI](https://github.com/wavde/experiment-toolkit/actions/workflows/ci.yml/badge.svg)
[![PyPI](https://img.shields.io/pypi/v/experiment-toolkit.svg)](https://pypi.org/project/experiment-toolkit/)
[![Python](https://img.shields.io/pypi/pyversions/experiment-toolkit.svg)](https://pypi.org/project/experiment-toolkit/)
![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)

![hero](https://raw.githubusercontent.com/wavde/experiment-toolkit/main/docs/hero.png)

A compact Python package covering the calculations that come up in everyday experimentation work: sample sizing, variance reduction, sequential tests, ratio-metric variance, staggered DiD, sensitivity analysis for observational designs, and A/B metrics scored by an LLM judge. The implementations follow standard results from the experimentation and causal-inference literature (referenced in the docstrings and below).

Intended audience: analysts, data scientists, and experimentation platform engineers who want a small, inspectable reference implementation rather than a framework.

## Modules

| Module | Purpose |
|--------|---------|
| `sample_size` | Per-arm sample size and MDE, CUPED-aware (`rho` keyword) |
| `cuped` | Deng et al. (2013) CUPED variance reduction |
| `ratio` | Delta-method variance for ratio metrics (revenue per session, etc.) |
| `sequential` | mSPRT always-valid p-values, with optional CUPED compounding |
| `cs_did` | Callaway & Sant'Anna (2021) staggered DiD, robust to heterogeneous cohort effects |
| `sensitivity` | E-value (VanderWeele & Ding) and Rosenbaum bounds |
| `llm_judge` | LLM-judge-scored A/B metrics: naive bias/power vs. PPI++ correction with a small human-labeled sample |

The public API is typed (the package ships a `py.typed` marker) and unit-tested, and the key methods cite their source in the module docstrings.

## Install

```bash
pip install experiment-toolkit
```

Or from source:

```bash
pip install git+https://github.com/wavde/experiment-toolkit.git
```

## Quick start

```python
from experiment_toolkit import sample_size_for_mde, apply_cuped, msprt_pvalue

# Per-arm sample size to detect a 2% lift at sd = 1.0
n = sample_size_for_mde(mde=0.02, std_dev=1.0, alpha=0.05, power=0.80)
# ~39,000 per arm

# CUPED adjustment with a pre-experiment covariate
y_adj = apply_cuped(y, pre_period_y)

# Always-valid p-value, safe to peek
p = msprt_pvalue(delta_hat=0.015, sigma=1.0, n_per_arm=5000, tau=0.05)
```

## CLI

The CLI exposes `sample-size` and `mde`. The other modules are library-only.

```bash
experiment-toolkit sample-size --mde 0.02 --sd 1.0
# Required per-arm sample size: 39,244

experiment-toolkit mde --n 10000 --sd 1.0
# Detectable effect (MDE): 0.0396
```

## LLM-as-judge simulation

Can you ship on an A/B test whose primary metric is scored by an LLM judge? `llm_judge` simulates binary true quality plus a judge with per-arm error rates, then compares three analyses: the naive judge-only analysis, human labels only, and PPI++ (judge scores corrected with a small human-labeled sample drawn within each arm). There are no model API calls, and the ground truth is known by construction.

```bash
pip install -e ".[dev]"
python scripts/llm_judge_sim.py
```

The script uses a fixed seed and writes [`docs/llm_judge_results.md`](docs/llm_judge_results.md) and [`docs/llm_judge.png`](docs/llm_judge.png). Every simulated number appears next to its closed-form value. When judge noise is the same in both arms, it shrinks the effect by Youden's J and costs power. When the noise differs between arms, the naive test's false-positive rate grows with sample size. Calibration labels have to be drawn within each arm: a single shared or offline calibration cancels out of the difference.

Write-up, with the decision rule: [LLM-as-judge is a noisy metric: what it does to A/B power and bias](https://github.com/wavde/wavde/blob/main/writing/llm-judge-noise-in-ab-tests.md).

```python
from experiment_toolkit import JudgeConfusion, simulate_judged_experiment, analyze_judged_experiment

judge = JudgeConfusion(sensitivity=0.85, specificity=0.85)
lenient = JudgeConfusion(sensitivity=0.85, specificity=0.80)  # judge favours treatment
t, c = simulate_judged_experiment(5000, p_control=0.6, effect=0.0, judge_control=judge,
                                  judge_treatment=lenient, n_labeled_per_arm=500, seed=0)
res = analyze_judged_experiment(t, c)  # {"naive", "human_only", "ppi"}
```

## Out of scope

Not in this package: a bandit framework, a Bayesian posterior sampler, a full experimentation platform, assignment or exposure logging, multiple-comparison correction beyond what the sequential tests provide. Those belong in separate layers.

## Development

```bash
pip install -e ".[dev]"
pytest
ruff check .
```

## References

- Deng, Xu, Kohavi, Walker (2013). CUPED.
- Deng, Knoblich, Lu (2018). Delta method in metric analytics.
- Johari, Pekelis, Walsh (2015). Always-valid inference.
- Callaway & Sant'Anna (2021). Difference-in-differences with multiple time periods.
- VanderWeele & Ding (2017). Sensitivity analysis in observational research.
- Kohavi, Tang, Xu (2020). *Trustworthy Online Controlled Experiments.*
- Angelopoulos, Bates, Fannjiang, Jordan, Zrnic (2023). Prediction-powered inference.
- Angelopoulos, Duchi, Zrnic (2023). PPI++: Efficient prediction-powered inference.
- Chaganty, Mussmann, Liang (2018). The price of debiasing automatic metrics in natural language evaluation.

## License

MIT. See [LICENSE](LICENSE).
