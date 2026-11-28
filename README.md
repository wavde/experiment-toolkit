# experiment-toolkit

> Small, well-tested utilities for online controlled experiments.

![CI](https://github.com/wavde/experiment-toolkit/actions/workflows/ci.yml/badge.svg)
[![PyPI](https://img.shields.io/pypi/v/experiment-toolkit.svg)](https://pypi.org/project/experiment-toolkit/)
[![Python](https://img.shields.io/pypi/pyversions/experiment-toolkit.svg)](https://pypi.org/project/experiment-toolkit/)
![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)

![hero](https://raw.githubusercontent.com/wavde/experiment-toolkit/main/docs/hero.png)

A compact Python package covering the calculations that come up in everyday experimentation work: sample sizing, variance reduction, sequential tests, ratio-metric variance, staggered DiD, sensitivity analysis for observational designs, A/B metrics scored by an LLM judge, and honest lift estimates for creatives picked by a bandit. The implementations follow standard results from the experimentation and causal-inference literature (referenced in the docstrings and below).

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
| `offline_eval` | Offline LLM evals vs online A/B: mix shift, judge bias and online-only costs, with reweighting and PPI++ fixes |
| `ppi_covariates` | PPI++ plus a pre-treatment covariate (CUPED-style): closed-form label budgets and when the covariate stops helping |
| `judge_drift` | A judge that changes mid-test, flagged or silent: which analyses stay valid, why front-loaded human labels don't, and which drift monitors (judge scores, label agreement, a re-judged canary set) can detect it |
| `bandit_eval` | Lift of a creative a bandit picked from many generated variants: winner's curse, time confounding, and off-policy (IPS / SNIPS / DR) estimates from logged propensities |
| `judge_audit` | Real-data audit of an LLM judge against expert humans (MT-Bench): win-rate bias, error that depends on which model humans preferred, and PPI++ checked against a known answer |

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

## Offline evals vs online A/B

When can an offline LLM eval win be trusted to predict the online result? `offline_eval` simulates an eval set whose prompt mix differs from live traffic, a judge whose errors can differ between the baseline and candidate outputs, and online-only costs such as latency. It compares the raw offline judge score, the score reweighted to the traffic mix, judge-plus-human-labels PPI++ reweighted to the mix, and the online A/B result. There are no model API calls.

```bash
python scripts/offline_online_sim.py
```

The script uses a fixed seed and writes [`docs/offline_online_results.md`](docs/offline_online_results.md) and [`docs/offline_online.png`](docs/offline_online.png).

What the results show:

- Reweighting to the traffic mix fixes mix shift.
- Human labels fix judge bias.
- No offline correction can see an online-only cost.
- Each correction costs offline power, so an offline win is best used to choose what to test online, not to skip the online test.

Write-up, with the decision rule: [An offline eval win is evidence about the eval set](https://github.com/wavde/wavde/blob/main/writing/offline-evals-vs-online-ab.md).

## PPI++ with a pre-treatment covariate

How much does a pre-treatment covariate, such as a query difficulty bucket or a user's prior-period outcome, cut the human-label budget of PPI++? `ppi_covariates` generalises PPI++ to several control variates (the judge score and the covariate). It provides the closed-form variance `Var(Y)/n · (1 − R² · N/(N+n))`, label budgets for a target power, and a simulator to check them.

```bash
python scripts/ppi_covariates_sim.py
```

The script uses a fixed seed and writes [`docs/ppi_covariates_results.md`](docs/ppi_covariates_results.md) and [`docs/ppi_covariates.png`](docs/ppi_covariates.png).

The covariate's gain depends on how much it adds beyond the judge. It is largest when the judge is weak, small when the judge is strong, and capped when labels are a large share of each arm.

Write-up, with the decision rule: [A covariate stacks on the judge, but a better judge beats both](https://github.com/wavde/wavde/blob/main/writing/ppi-with-covariates-label-budget.md).

## Judge drift mid-test

What happens to a judge-scored A/B test when the judge model is upgraded partway through? `judge_drift` simulates an upgrade that switches both arms to a new judge at the same time, and compares five analyses: the judge score pooled over the test, the judge score before the upgrade only, and three PPI++ variants that differ in when human labels are drawn.

```bash
python scripts/judge_drift_sim.py
```

The script uses a fixed seed and writes [`docs/judge_drift_results.md`](docs/judge_drift_results.md) and [`docs/judge_drift.png`](docs/judge_drift.png).

- An upgrade that changes both arms' judged rates equally is harmless under the null.
- An upgrade that is more lenient on one arm's outputs biases the pooled judge score.
- It also biases PPI++ if all the human labels were collected before the upgrade.
- PPI++ with labels spread across the test, or stratified by judge version, stays unbiased.

Write-up, with the decision rule: [Upgrading the judge mid-test is safe only if your labels know about it](https://github.com/wavde/wavde/blob/main/writing/judge-drift-mid-test.md).

## Silent judge drift

A hosted judge can change under the same model name, so the analysis has no version flag to stratify on. `SilentDrift` changes the judge from an unknown day, as a step or a gradual ramp, on one or both arms. It can also shift true quality with no judge change, which a drift monitor should not flag. The simulation compares the judge-only readout with PPI++ using launch-week labels or labels drawn at random from the whole test. It also runs three cheap monitors (`first_alarm`): judge pass rate on the control arm, judge pass rate on each arm, and the judge's agreement with human labels split by the human label.

```bash
python scripts/silent_drift_sim.py
```

The script uses a fixed seed and writes [`docs/silent_drift_results.md`](docs/silent_drift_results.md) and [`docs/silent_drift.png`](docs/silent_drift.png).

- PPI++ with labels drawn at random from the whole test stays valid under every drift tested, without knowing when or whether the judge changed. The labels can be drawn at the end from logged outputs.
- Launch-week labels and the judge-only readout are biased when drift hits one arm more than the other.
- A score monitor on the control arm can't see drift that only affects treatment. Score monitors on both arms also fire when true quality changes and the judge doesn't.
- Agreement split by the human label separates judge drift from quality change, but needs thousands of labels per arm to detect a realistic drift.

Write-up, with the decision rule: [When the judge drifts silently, random labels protect the readout and monitors only tell you where to look](https://github.com/wavde/wavde/blob/main/writing/silent-judge-drift.md).

## Canary-set drift monitor

A canary set is a fixed set of stored outputs that the production judge re-scores every day. Their true quality can't change, so a shift in their scores can only come from the judge. `simulate_canary` re-judges stored outputs under a `SilentDrift`, with a `repeatability` setting for how often the judge repeats its verdict on the same output. `canary_alarm` compares each output's recent verdicts with its own early ones. The simulation runs it on the same scenarios as the silent-drift simulation, next to the score and agreement monitors.

```bash
python scripts/canary_drift_sim.py
```

The script uses a fixed seed and writes [`docs/canary_drift_results.md`](docs/canary_drift_results.md) and [`docs/canary_drift.png`](docs/canary_drift.png).

- With 400 stored outputs per arm and no human labels, the canary caught treatment-only drift about as often and as fast as a score monitor on both arms, and caught gradual drift more often.
- Unlike the score monitors, it stayed quiet when true quality fell and the judge didn't change.
- A canary built only from control outputs missed drift that affected only treatment outputs. The canary has to contain outputs from every arm.
- Closed forms give the canary size for a target detection probability (`canary_items_for_power`).

Write-up, with the decision rule: [A canary set catches a drifting judge without blaming real quality changes](https://github.com/wavde/wavde/blob/main/writing/canary-set-judge-drift.md).
## Generated creatives picked by a bandit

Generative tools make it cheap to test 20 or 50 creative variants at once, and a bandit moves traffic toward whichever is ahead. How far is the leader's reported lift from its true lift, and which estimate from the logged decisions can be trusted? `bandit_eval` provides off-policy estimators that use the logged allocation probabilities (IPS, self-normalised IPS, doubly robust), the naive dashboard comparison, and a minimal batched Thompson-sampling log generator with known ground truth. It evaluates bandit logs; it is not a bandit framework.

```bash
python scripts/bandit_eval_sim.py
```

The script uses a fixed seed and writes [`docs/bandit_eval_results.md`](docs/bandit_eval_results.md) and [`docs/bandit_eval.png`](docs/bandit_eval.png). The full run takes about 33 minutes on a laptop CPU; `--smoke` takes seconds.

Two biases are at work, and they need different fixes:

- **Time confounding.** A variant that wins traffic late is measured on late traffic. With click rates drifting ±20% over the test and no variant better than control, the leader's raw lift averages +12.6% and every replication claims a win. IPS, SNIPS and DR with the logged propensities remove this part.
- **Winner's curse.** The leader was chosen because its estimate was high. IPS on the same log still overstates it: by +4.6 pp with 20 null variants (about 22% false wins at a nominal 2.5%), and DR on the same log by +8.0 pp with 50. When variant lifts are spread around zero (SD 4%) and clicks drift, the leader's true lift is +5.7% while the same-log IPS estimate is +7.7% and the raw lift +16.3%.

What the simulation shows about the fixes:

- **Splitting the log after the fact doesn't work.** The bandit allocated traffic using both halves, so the held-out half is biased (here downward) and the variant it picks is worse.
- **Data the bandit never learned from does.** A 20% evaluation slice that the bandit serves but never updates on, and a 50k-per-arm confirmation A/B, are both unbiased with 94–96% coverage. The slice barely hurts the pick, but its SE is 2.3–2.8× the confirmation test's at a similar impression count, because most of its traffic went to variants that lost.

Decision rule: let the bandit choose, but quote the lift only from impressions it didn't learn from. Run a short confirmation A/B when the number feeds a ship or forecast decision. If no extra time is available, reserve an evaluation slice from day one and log propensities with a floor, and treat it as a check on direction and rough size. Never report the leader's raw lift, and don't treat IPS on the same log as a fix for the winner's curse.

```python
from experiment_toolkit.bandit_eval import ope_lift

# One row per impression: arm shown (0 = control), click, logged probability of that arm.
# Rows from the evaluation slice only, for the leader chosen on the other rows.
res = ope_lift(arms[slice_], clicks[slice_], propensity[slice_], target=leader, control=0)
res.estimate, res.ci()
```

Where this breaks: with two to five variants the same-log overstatement is under 2 pp, so the extra machinery buys little. A thin propensity floor (≤1% of variant traffic) drops the slice's coverage to 0.91–0.93. Lifts are held constant over time; if generated creatives have a novelty effect that fades, no estimate from the test window measures the post-launch lift, and only a later confirmation test does.

Write-up, with the decision rule: [Five classic data science case studies, re-read in 2026](https://github.com/wavde/wavde/blob/main/writing/classic-case-studies-reread-2026.md).

## Real-data judge audit (MT-Bench)

Every LLM-judge simulation above assumes a judge whose errors can depend on which arm produced the output. `judge_audit` checks that on real data: the public [MT-Bench human judgments](https://huggingface.co/datasets/lmsys/mt_bench_human_judgments) (Zheng et al., 2023, CC BY 4.0), where expert humans and GPT-4 judged the same pairwise comparisons of six models. Each pair of models is read as a side-by-side test, with the stronger model as treatment.

```bash
pip install -e ".[dev,data]"
python scripts/judge_audit_mtbench.py
```

The data (about 1.4 MB) are downloaded on first run from a pinned revision, checked against SHA-256 hashes and cached in `data/mt_bench/`; they aren't committed. The script uses a fixed seed and writes [`docs/judge_audit_results.md`](docs/judge_audit_results.md) and [`docs/judge_audit.png`](docs/judge_audit.png). CI smoke-runs it on synthetic data with the same schema.

- GPT-4's win rate for the stronger model was above the experts' in all 15 pairs, by up to +0.185, on the same items.
- When experts preferred the stronger model, GPT-4 agreed 82% of the time; when they preferred the weaker one, 22%. The judge's error depends on the arm.
- In a known-answer check that resamples the real verdicts, with 200 human labels per pair, PPI++ kept 93–96% coverage. The judge alone covered the truth in at most 0.5% of replications in 13 of 15 pairs, and at most 44% in the other two.
- The real judge correlates with humans far less than the simulated judges assumed (0.14–0.56 against 0.69), so PPI++ saved only 2–28% of labels.

Write-up, with the decision rule: [A real LLM judge favours the winner: auditing GPT-4 against expert humans](https://github.com/wavde/wavde/blob/main/writing/real-judge-audit-mt-bench.md).

## Out of scope

Not in this package: a bandit framework (`bandit_eval` only evaluates logged bandit traffic; its simulator is a minimal test fixture), a Bayesian posterior sampler, a full experimentation platform, assignment or exposure logging, multiple-comparison correction beyond what the sequential tests provide. Those belong in separate layers.

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
- Dudík, Langford, Li (2011). Doubly robust policy evaluation and learning.
- Swaminathan & Joachims (2015). The self-normalized estimator for counterfactual learning.
- Nie, Tian, Taylor, Zou (2018). Why adaptively collected data have negative bias and how to correct for it.
- Hadad, Hirshberg, Zhan, Wager, Athey (2021). Confidence intervals for policy evaluation in adaptive experiments.
- Andrews, Kitagawa, McCloskey (2024). Inference on winners.

## License

MIT. See [LICENSE](LICENSE).
