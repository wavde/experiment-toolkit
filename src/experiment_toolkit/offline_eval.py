"""
Offline LLM evals vs online A/B tests: when does an offline win predict the
online result?

Before an online test, a team usually runs an offline eval: a fixed prompt
set, outputs from the baseline model ``A`` and the candidate ``B``, and an LLM
judge that scores each output as good or bad. This module simulates both
stages with known ground truth, which isolates three ways the offline answer
can disagree with the online one:

1. **Mix shift.** The eval set's prompt mix (for example, heavy on coding)
   differs from live traffic (mostly chat). If ``B`` helps on one segment and
   hurts on another, the offline average and the online average have
   different signs. Fix: post-stratify the offline estimate to the online mix.
2. **Judge bias.** The judge errs differently on ``B``'s outputs (for
   example, it is more lenient toward longer answers). Reweighting doesn't
   help. Fix: human labels on a random subset of the eval set, combined with
   the judge by PPI++ within each segment (see :mod:`experiment_toolkit.llm_judge`).
3. **Online-only costs.** ``B`` changes something the offline eval can't see,
   such as latency or refusals that make users abandon. No offline correction
   can see it. Only the online test (or a guardrail metric in it) catches it.

Model
-----
Segment ``k`` has offline share ``w_off[k]``, online share ``w_on[k]``, and
true quality probability ``p_a[k]`` for ``A`` and ``p_a[k] + delta[k]`` for
``B``. The judge's confusion matrix may differ for ``A`` and ``B`` outputs.
Online, a user's success probability is
``base + kappa * P(quality) - online_cost * [arm is B]``, so the true online
effect is ``kappa * sum_k w_on[k] * delta[k] - online_cost``.

References: Angelopoulos et al. (2023) for PPI / PPI++; Kohavi, Tang, Xu (2020)
ch. 21 on the gap between offline metrics and online results.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import stats

from experiment_toolkit.llm_judge import JudgeConfusion, ppi_mean_diff

_WEIGHT_TOL = 1e-9


@dataclass(frozen=True)
class OfflineOnlineScenario:
    """Ground truth for one candidate change, segment by segment.

    Attributes
    ----------
    w_off, w_on : segment shares in the offline eval set and in live traffic.
    p_a : true probability that ``A``'s output is good, per segment.
    delta : true quality effect of ``B`` (``p_b - p_a``), per segment.
    judge_a, judge_b : judge confusion matrices for ``A`` and ``B`` outputs.
    kappa : how much one unit of quality moves the online success rate.
    base_online : online success rate at zero quality.
    online_cost : drop in online success caused by ``B`` that offline can't see.
    """

    w_off: np.ndarray
    w_on: np.ndarray
    p_a: np.ndarray
    delta: np.ndarray
    judge_a: JudgeConfusion
    judge_b: JudgeConfusion
    kappa: float = 0.8
    base_online: float = 0.1
    online_cost: float = 0.0
    names: tuple[str, ...] = field(default=())

    def __post_init__(self) -> None:
        arrays = {}
        for name in ("w_off", "w_on", "p_a", "delta"):
            arr = np.asarray(getattr(self, name), dtype=float)
            if arr.ndim != 1:
                raise ValueError(f"{name} must be 1-D")
            arrays[name] = arr
            object.__setattr__(self, name, arr)
        k = len(arrays["w_on"])
        if any(len(a) != k for a in arrays.values()):
            raise ValueError("w_off, w_on, p_a and delta must have the same length")
        for name in ("w_off", "w_on"):
            w = arrays[name]
            if np.any(w < 0) or abs(w.sum() - 1.0) > _WEIGHT_TOL:
                raise ValueError(f"{name} must be non-negative and sum to 1")
        p_b = arrays["p_a"] + arrays["delta"]
        if np.any(arrays["p_a"] < 0) or np.any(arrays["p_a"] > 1):
            raise ValueError("p_a must be in [0, 1]")
        if np.any(p_b < 0) or np.any(p_b > 1):
            raise ValueError("p_a + delta must be in [0, 1]")
        for arm, p in (("A", arrays["p_a"]), ("B", p_b)):
            q = self.base_online + self.kappa * p - (self.online_cost if arm == "B" else 0.0)
            if np.any(q < 0) or np.any(q > 1):
                raise ValueError(f"online success probability for {arm} must be in [0, 1]")
        if self.names and len(self.names) != k:
            raise ValueError("names must match the number of segments")

    @property
    def p_b(self) -> np.ndarray:
        return self.p_a + self.delta


# -- closed-form answers ---------------------------------------------------------


def expected_judged_diff(scenario: OfflineOnlineScenario, weights: np.ndarray) -> float:
    """Expected judged B-minus-A rate under segment ``weights``.

    The per-segment value is ``q_b - q_a``, where ``q = sens*p + (1-spec)*(1-p)``.
    """
    w = np.asarray(weights, dtype=float)
    q_a = np.array([scenario.judge_a.positive_rate(p) for p in scenario.p_a])
    q_b = np.array([scenario.judge_b.positive_rate(p) for p in scenario.p_b])
    return float(w @ (q_b - q_a))


def expected_quality_diff(scenario: OfflineOnlineScenario, weights: np.ndarray) -> float:
    """True B-minus-A quality difference under segment ``weights``."""
    return float(np.asarray(weights, dtype=float) @ scenario.delta)


def expected_online_effect(scenario: OfflineOnlineScenario) -> float:
    """True online effect: ``kappa * sum(w_on * delta) - online_cost``."""
    return scenario.kappa * expected_quality_diff(scenario, scenario.w_on) - scenario.online_cost


# -- simulation ----------------------------------------------------------------


@dataclass(frozen=True)
class OfflineEvalData:
    """Offline eval: one A output and one B output per prompt.

    ``y_*`` are true (human) labels, ``s_*`` are judge scores, and ``labeled``
    marks prompts that received human labels for both outputs.
    """

    segment: np.ndarray
    y_a: np.ndarray
    s_a: np.ndarray
    y_b: np.ndarray
    s_b: np.ndarray
    labeled: np.ndarray


def simulate_offline_eval(
    scenario: OfflineOnlineScenario,
    n_prompts: int,
    n_labeled: int,
    rng: np.random.Generator,
) -> OfflineEvalData:
    """Draw ``n_prompts`` prompts from the offline mix, generate and judge A and B
    outputs, and human-label a simple random sample of ``n_labeled`` prompts."""
    if n_prompts <= 0:
        raise ValueError("n_prompts must be positive")
    if not 0 <= n_labeled <= n_prompts:
        raise ValueError("n_labeled must be in [0, n_prompts]")
    seg = rng.choice(len(scenario.w_off), size=n_prompts, p=scenario.w_off)

    def outputs(p: np.ndarray, judge: JudgeConfusion) -> tuple[np.ndarray, np.ndarray]:
        y = (rng.random(n_prompts) < p[seg]).astype(float)
        p_pos = np.where(y == 1.0, judge.sensitivity, 1.0 - judge.specificity)
        return y, (rng.random(n_prompts) < p_pos).astype(float)

    y_a, s_a = outputs(scenario.p_a, scenario.judge_a)
    y_b, s_b = outputs(scenario.p_b, scenario.judge_b)
    labeled = np.zeros(n_prompts, dtype=bool)
    labeled[rng.choice(n_prompts, size=n_labeled, replace=False)] = True
    return OfflineEvalData(seg, y_a, s_a, y_b, s_b, labeled)


@dataclass(frozen=True)
class Readout:
    """B-minus-A estimate with a Wald 95% CI and a ship / stop / no-decision call."""

    estimate: float
    se: float

    @property
    def ci_low(self) -> float:
        return self.estimate - stats.norm.ppf(0.975) * self.se

    @property
    def ci_high(self) -> float:
        return self.estimate + stats.norm.ppf(0.975) * self.se

    def decision(self) -> str:
        """``"ship"`` if the CI is above 0, ``"stop"`` if below, else ``"none"``."""
        if self.ci_low > 0:
            return "ship"
        if self.ci_high < 0:
            return "stop"
        return "none"


def offline_readouts(
    data: OfflineEvalData, w_on: np.ndarray
) -> dict[str, Readout]:
    """Three offline estimates of B minus A.

    - ``"judge_raw"``: mean judge score difference over the eval set as drawn.
    - ``"judge_reweighted"``: per-segment judge differences, post-stratified to
      the online mix ``w_on``.
    - ``"ppi_reweighted"``: per-segment PPI++ estimates of the true quality
      difference using the human-labeled prompts, post-stratified to ``w_on``.

    Every segment with positive online weight must appear in the eval set
    (and, for PPI++, have at least 2 labeled and 2 unlabeled prompts).
    """
    w_on = np.asarray(w_on, dtype=float)
    n = len(data.segment)
    raw = Readout(
        estimate=float(data.s_b.mean() - data.s_a.mean()),
        se=float(np.sqrt((data.s_b.var(ddof=1) + data.s_a.var(ddof=1)) / n)),
    )
    est_j, var_j, est_p, var_p = 0.0, 0.0, 0.0, 0.0
    for k, w in enumerate(w_on):
        if w == 0:
            continue
        m = data.segment == k
        n_k = int(m.sum())
        if n_k < 2:
            raise ValueError(f"segment {k} has online weight but fewer than 2 eval prompts")
        sb, sa = data.s_b[m], data.s_a[m]
        est_j += w * (sb.mean() - sa.mean())
        var_j += w**2 * (sb.var(ddof=1) + sa.var(ddof=1)) / n_k
        lab, unl = m & data.labeled, m & ~data.labeled
        res = ppi_mean_diff(
            data.y_b[lab], data.s_b[lab], data.s_b[unl],
            data.y_a[lab], data.s_a[lab], data.s_a[unl],
        )
        est_p += w * res.estimate
        var_p += w**2 * res.se**2
    return {
        "judge_raw": raw,
        "judge_reweighted": Readout(float(est_j), float(np.sqrt(var_j))),
        "ppi_reweighted": Readout(float(est_p), float(np.sqrt(var_p))),
    }


def simulate_online_ab(
    scenario: OfflineOnlineScenario, n_per_arm: int, rng: np.random.Generator
) -> Readout:
    """Online A/B on live traffic: B-minus-A difference in user success rate."""
    if n_per_arm <= 1:
        raise ValueError("n_per_arm must be at least 2")
    rates = []
    for p, cost in ((scenario.p_a, 0.0), (scenario.p_b, scenario.online_cost)):
        counts = rng.multinomial(n_per_arm, scenario.w_on)
        prob = scenario.base_online + scenario.kappa * p - cost
        rates.append(rng.binomial(counts, prob).sum() / n_per_arm)
    r_a, r_b = rates
    se = np.sqrt((r_a * (1 - r_a) + r_b * (1 - r_b)) / n_per_arm)
    return Readout(float(r_b - r_a), float(se))
