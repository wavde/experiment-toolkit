# experiment-toolkit

> Small, well-tested utilities for online controlled experiments.

![CI](https://github.com/wavde/experiment-toolkit/actions/workflows/ci.yml/badge.svg)
[![PyPI](https://img.shields.io/pypi/v/experiment-toolkit.svg)](https://pypi.org/project/experiment-toolkit/)
[![Python](https://img.shields.io/pypi/pyversions/experiment-toolkit.svg)](https://pypi.org/project/experiment-toolkit/)
![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)

## What's inside

| Module | Purpose |
|--------|---------|
| `sample_size` | Required per-arm sample size for a given MDE, and the inverse |
| `cuped` | Deng et al. (2013) CUPED variance reduction |
| `ratio` | Delta-method variance for ratio metrics (revenue/session, etc.) |
| `sequential` | mSPRT always-valid p-values for peeking-robust experiments |

Every function is tested, typed, and has a reference to the paper it implements.

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

# How many users do I need per arm to detect a 2% lift (sd=1.0)?
n = sample_size_for_mde(mde=0.02, std_dev=1.0, alpha=0.05, power=0.80)
# ~39,000 per arm

# Apply CUPED with a pre-experiment covariate
y_adj = apply_cuped(y, pre_period_y)

# Always-valid p-value — safe to peek
p = msprt_pvalue(delta_hat=0.015, sigma=1.0, n_per_arm=5000, tau=0.05)
```

## CLI

The CLI wraps `sample-size` and `mde`. The other modules (`cuped`, `ratio`, `sequential`) are library-only.

```bash
experiment-toolkit sample-size --mde 0.02 --sd 1.0
# Required per-arm sample size: 39,244

experiment-toolkit mde --n 10000 --sd 1.0
# Detectable effect (MDE): 0.0396
```

## Development

```bash
pip install -e ".[dev]"
pytest
ruff check .
```

## References

- Deng, Xu, Kohavi, Walker (2013) — CUPED
- Deng, Knoblich, Lu (2018) — Delta Method in Metric Analytics
- Johari, Pekelis, Walsh (2015) — Always Valid Inference
- Kohavi, Tang, Xu (2020) — *Trustworthy Online Controlled Experiments*

## License

MIT — see [LICENSE](LICENSE).
