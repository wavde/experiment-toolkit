# Contributing

Thanks for your interest. This package is maintained but small, so scope is narrow.

## Development setup

```bash
git clone https://github.com/wavde/experiment-toolkit.git
cd experiment-toolkit
python -m venv .venv
source .venv/bin/activate        # or .venv\Scripts\activate on Windows
pip install -e ".[dev]"
```

## Before opening a PR

```bash
ruff check .
pytest -q
```

Both must pass. CI runs these on Python 3.10, 3.11, and 3.12.

## What I'll accept

- Bug fixes with a regression test
- Numerically stable reformulations of existing methods
- Narrow feature additions **with a paper reference** (e.g., a new variance-reduction technique from a published paper)
- Documentation improvements
- New CLI subcommands that wrap existing library functions

## What I won't accept (without prior discussion)

- New heavy dependencies (we stay lean on `numpy` + `scipy`)
- Methods that reinvent things `scipy.stats` or `statsmodels` already do well
- Scope expansion into ML / forecasting / dashboards — this package is scoped to **inference for online experiments**

## Style

- Type hints on every public function
- Docstrings state the contract (inputs, outputs, failure modes), not just the function name
- Magic numbers go into named constants
- Each public function gets a known-answer test (not just "it runs")

## Bug reports

Include:
1. A minimal reproducer (fewer than 20 lines)
2. Expected vs observed output
3. Package version (`python -c "import experiment_toolkit; print(experiment_toolkit.__version__)"`)
4. Python version

## License

By contributing you agree your work is released under the MIT license (see `LICENSE`).
