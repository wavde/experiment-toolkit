"""
What happens to a judge-scored A/B test when the judge model is upgraded mid-test.

Loops 1 and 7 (:mod:`experiment_toolkit.llm_judge`,
:mod:`experiment_toolkit.ppi_covariates`) assume one fixed judge. In practice
the judge model or prompt is often replaced while a test is running. Both arms
switch to the new judge at the same time, so the change looks harmless. This
module checks when it is.

Model
-----
Each arm has ``n`` units in time order. A share ``w_old`` are scored by the old
judge and the rest by the new one. True quality ``Y`` is Bernoulli with a rate
that is constant over time (``p_c`` in control, ``p_c + effect`` in treatment).
A judge is a confusion matrix per arm, so the new judge can err differently on
treatment and control outputs.

Analyses compared
-----------------
- ``naive_pooled``: judge-score difference over all units.
- ``naive_pre``: judge-score difference over the pre-upgrade units only
  (truncate the test at the upgrade).
- ``ppi_spread``: PPI++ with human labels drawn at random across the whole arm.
- ``ppi_front``: PPI++ with all human labels drawn before the upgrade, which is
  common when labeling is front-loaded. The rectifier then compares old-judge
  scores on labeled units with a mix of old- and new-judge scores on unlabeled
  units.
- ``ppi_stratified``: PPI++ run separately before and after the upgrade, with
  labels split in proportion to each period's size. The two estimates are
  combined with the period shares as weights.

``ppi_spread`` and ``ppi_stratified`` are unbiased for the true effect,
whatever the judges do, because labels are a random sample within each
(arm, period) cell that is weighted correctly. ``ppi_front`` is unbiased only
when the upgrade shifts both arms' judged rates by the same amount.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats

from experiment_toolkit.llm_judge import JudgeConfusion, ppi_mean_diff

ANALYSES = ("naive_pooled", "naive_pre", "ppi_spread", "ppi_front", "ppi_stratified")
_Z = float(stats.norm.ppf(0.975))


@dataclass(frozen=True)
class JudgeUpgrade:
    """Old and new judges for each arm, and the pre-upgrade share of units."""

    old_t: JudgeConfusion
    old_c: JudgeConfusion
    new_t: JudgeConfusion
    new_c: JudgeConfusion
    w_old: float = 0.5

    def __post_init__(self) -> None:
        if not 0.0 < self.w_old < 1.0:
            raise ValueError("w_old must be in (0, 1)")


@dataclass(frozen=True)
class DriftArm:
    y: np.ndarray
    s: np.ndarray
    new: np.ndarray  # True for units scored by the new judge


@dataclass(frozen=True)
class Readout:
    estimate: float
    se: float

    def covers(self, truth: float) -> bool:
        return abs(self.estimate - truth) <= _Z * self.se

    def rejects(self) -> bool:
        return abs(self.estimate) > _Z * self.se


def _check_p(p: float) -> None:
    if not 0.0 <= p <= 1.0:
        raise ValueError(f"rate must be in [0, 1], got {p}")


def simulate_drift_arm(
    n: int, p: float, old: JudgeConfusion, new: JudgeConfusion, w_old: float,
    rng: np.random.Generator,
) -> DriftArm:
    """``n`` units; the first ``round(w_old * n)`` are scored by ``old``."""
    _check_p(p)
    if n < 10:
        raise ValueError("n must be at least 10")
    n_old = int(round(w_old * n))
    is_new = np.arange(n) >= n_old
    y = (rng.random(n) < p).astype(float)
    sens = np.where(is_new, new.sensitivity, old.sensitivity)
    fpr = np.where(is_new, 1 - new.specificity, 1 - old.specificity)
    s = (rng.random(n) < np.where(y == 1.0, sens, fpr)).astype(float)
    return DriftArm(y, s, is_new)


def simulate_drift_experiment(
    n_per_arm: int, p_control: float, effect: float, upgrade: JudgeUpgrade,
    rng: np.random.Generator,
) -> tuple[DriftArm, DriftArm]:
    _check_p(p_control + effect)
    t = simulate_drift_arm(n_per_arm, p_control + effect, upgrade.old_t, upgrade.new_t,
                           upgrade.w_old, rng)
    c = simulate_drift_arm(n_per_arm, p_control, upgrade.old_c, upgrade.new_c,
                           upgrade.w_old, rng)
    return t, c


def _naive(st: np.ndarray, sc: np.ndarray) -> Readout:
    return Readout(float(st.mean() - sc.mean()),
                   float(np.sqrt(st.var(ddof=1) / len(st) + sc.var(ddof=1) / len(sc))))


def _labels(n_units: int, k: int, rng: np.random.Generator) -> np.ndarray:
    mask = np.zeros(n_units, dtype=bool)
    mask[rng.choice(n_units, size=k, replace=False)] = True
    return mask


def _ppi(t: DriftArm, c: DriftArm, lab_t: np.ndarray, lab_c: np.ndarray,
         sub_t: np.ndarray | None = None, sub_c: np.ndarray | None = None) -> Readout:
    def parts(arm, lab, sub):
        sub = np.ones(len(arm.y), dtype=bool) if sub is None else sub
        return arm.y[lab & sub], arm.s[lab & sub], arm.s[~lab & sub]

    r = ppi_mean_diff(*parts(t, lab_t, sub_t), *parts(c, lab_c, sub_c))
    return Readout(r.estimate, r.se)


def analyze_drift(
    t: DriftArm, c: DriftArm, n_labeled: int, rng: np.random.Generator
) -> dict[str, Readout]:
    """Run every analysis with the same per-arm human-label budget."""
    out = {
        "naive_pooled": _naive(t.s, c.s),
        "naive_pre": _naive(t.s[~t.new], c.s[~c.new]),
    }
    out["ppi_spread"] = _ppi(t, c, _labels(len(t.y), n_labeled, rng),
                             _labels(len(c.y), n_labeled, rng))

    def front(arm: DriftArm) -> np.ndarray:
        lab = np.zeros(len(arm.y), dtype=bool)
        old_idx = np.flatnonzero(~arm.new)
        lab[rng.choice(old_idx, size=n_labeled, replace=False)] = True
        return lab

    out["ppi_front"] = _ppi(t, c, front(t), front(c))

    w_old = float((~t.new).mean())
    k_old = int(round(n_labeled * w_old))
    est, var = 0.0, 0.0
    for period_new, k, w in ((False, k_old, w_old), (True, n_labeled - k_old, 1 - w_old)):
        def lab(arm: DriftArm, period_new=period_new, k=k) -> np.ndarray:
            mask = np.zeros(len(arm.y), dtype=bool)
            idx = np.flatnonzero(arm.new == period_new)
            mask[rng.choice(idx, size=k, replace=False)] = True
            return mask

        r = _ppi(t, c, lab(t), lab(c), t.new == period_new, c.new == period_new)
        est += w * r.estimate
        var += w**2 * r.se**2
    out["ppi_stratified"] = Readout(est, float(np.sqrt(var)))
    return out


# -- closed-form answers ---------------------------------------------------------


def expected_naive(upgrade: JudgeUpgrade, p_control: float, effect: float,
                   pre_only: bool = False) -> float:
    """Expected judged difference, pooled over both periods or pre-upgrade only."""
    p_t = p_control + effect
    old = upgrade.old_t.positive_rate(p_t) - upgrade.old_c.positive_rate(p_control)
    if pre_only:
        return old
    new = upgrade.new_t.positive_rate(p_t) - upgrade.new_c.positive_rate(p_control)
    return upgrade.w_old * old + (1 - upgrade.w_old) * new


def expected_naive_se(upgrade: JudgeUpgrade, p_control: float, effect: float, n: int,
                      pre_only: bool = False) -> float:
    """SE of the naive difference (period counts are fixed, so the per-arm
    variance is the period-weighted Bernoulli variance)."""
    def arm_var(old: JudgeConfusion, new: JudgeConfusion, p: float) -> float:
        q_o, q_n = old.positive_rate(p), new.positive_rate(p)
        if pre_only:
            return q_o * (1 - q_o) / (upgrade.w_old * n)
        return (upgrade.w_old * q_o * (1 - q_o) + (1 - upgrade.w_old) * q_n * (1 - q_n)) / n

    p_t = p_control + effect
    return float(np.sqrt(arm_var(upgrade.old_t, upgrade.new_t, p_t)
                         + arm_var(upgrade.old_c, upgrade.new_c, p_control)))


def expected_ppi_front(upgrade: JudgeUpgrade, p_control: float, effect: float,
                       n: int, n_labeled: int) -> float:
    """Large-sample mean of ``ppi_front``, using the population version of the
    PPI++ weight that ``ppi_mean_diff`` estimates."""
    n_unlab = n - n_labeled
    n_new = n - int(round(upgrade.w_old * n))

    def arm(old: JudgeConfusion, new: JudgeConfusion, p: float) -> float:
        q_o, q_n = old.positive_rate(p), new.positive_rate(p)
        cov_ys = p * old.sensitivity - p * q_o
        m = upgrade.w_old * q_o + (1 - upgrade.w_old) * q_n
        lam = cov_ys / ((1 + n_labeled / n_unlab) * m * (1 - m))
        unlab_mean = ((n_unlab - n_new) * q_o + n_new * q_n) / n_unlab
        return p - lam * (q_o - unlab_mean)

    return arm(upgrade.old_t, upgrade.new_t, p_control + effect) - arm(
        upgrade.old_c, upgrade.new_c, p_control
    )


def rejection_rate(mean: float, se: float) -> float:
    shift = mean / se
    return float(stats.norm.cdf(shift - _Z) + stats.norm.cdf(-shift - _Z))
