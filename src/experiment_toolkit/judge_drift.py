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

Silent drift (no version boundary)
----------------------------------
A hosted judge can change under the same model name, so the analysis has no
version flag to stratify on. :class:`SilentDrift` moves each arm's judge from a
base confusion matrix to a drifted one, starting on an unknown day and either
as a step or a linear ramp. It can also shift true quality in both arms with
no judge change, which is the case a monitor should *not* call judge drift.
:func:`analyze_silent` compares the judge-only difference with PPI++ using
launch-week labels or labels drawn at random from the whole test.
:func:`first_alarm` is a cheap monitor: each day it compares the latest
``window`` days of a series with a reference period at the start, and the
Bonferroni threshold keeps the chance of any false alarm at or below ``alpha``.

Canary set
----------
A canary set is a fixed set of stored outputs re-judged every day. Because the
outputs never change, a shift in their judge scores can only come from the
judge. :func:`simulate_canary` re-judges ``n_items`` stored outputs of one arm
under a :class:`SilentDrift`. ``repeatability`` sets how often the judge gives
an item the same verdict on different days, from 0 (fresh noise each day) to 1
(deterministic), without changing the judge's error rates.
:func:`canary_alarm` is a paired version of :func:`first_alarm`: it compares
each item's recent verdicts with its own reference-period verdicts.
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


# -- silent drift: no version boundary ---------------------------------------------


def _lerp(a: JudgeConfusion, b: JudgeConfusion, w: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return (a.sensitivity + w * (b.sensitivity - a.sensitivity),
            a.specificity + w * (b.specificity - a.specificity))


@dataclass(frozen=True)
class SilentDrift:
    """Judge drift the analysis is not told about.

    Before ``start_day`` (1-based) both arms are scored by ``base``. From
    ``start_day`` each arm's sensitivity and specificity move linearly towards
    ``after_t`` / ``after_c`` and arrive after ``ramp_days`` days (1 = a step
    change). ``quality_shift`` moves both arms' true rates by the same amount
    on the same schedule, with no change in the judge.
    """

    base: JudgeConfusion
    after_t: JudgeConfusion
    after_c: JudgeConfusion
    start_day: int
    ramp_days: int = 1
    quality_shift: float = 0.0

    def __post_init__(self) -> None:
        if self.start_day < 1 or self.ramp_days < 1:
            raise ValueError("start_day and ramp_days must be at least 1")

    def weight(self, day: np.ndarray) -> np.ndarray:
        """Share of the way from base to drifted on each day (0 before the start)."""
        return np.clip((np.asarray(day) - self.start_day + 1) / self.ramp_days, 0.0, 1.0)

    def path(self, arm: str, p: float, days: np.ndarray
             ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """True rate, sensitivity and specificity on each of ``days`` for one arm."""
        after = {"t": self.after_t, "c": self.after_c}[arm]
        w = self.weight(days)
        sens, spec = _lerp(self.base, after, w)
        return p + self.quality_shift * w, sens, spec


@dataclass(frozen=True)
class DailyArm:
    y: np.ndarray
    s: np.ndarray
    day: np.ndarray  # 1-based day each unit arrived


def simulate_silent_experiment(
    n_days: int, n_per_day: int, p_control: float, effect: float, drift: SilentDrift,
    rng: np.random.Generator,
) -> tuple[DailyArm, DailyArm]:
    """Equal daily traffic in each arm; returns ``(treatment, control)``."""
    if n_days < 2 or n_per_day < 1:
        raise ValueError("need at least 2 days and 1 unit per day")
    days = np.arange(1, n_days + 1)
    for name, p in (("c", p_control), ("t", p_control + effect)):
        rates = drift.path(name, p, days)[0]
        if rates.min() < 0.0 or rates.max() > 1.0:
            raise ValueError("true rate leaves [0, 1]")

    def arm(name: str, p: float) -> DailyArm:
        p_d, sens_d, spec_d = drift.path(name, p, days)
        day = np.repeat(days, n_per_day)
        i = day - 1
        y = (rng.random(len(day)) < p_d[i]).astype(float)
        hit = np.where(y == 1.0, sens_d[i], 1.0 - spec_d[i])
        s = (rng.random(len(day)) < hit).astype(float)
        return DailyArm(y, s, day)

    return arm("t", p_control + effect), arm("c", p_control)


def label_mask(arm: DailyArm, k: int, rng: np.random.Generator,
               last_day: int | None = None) -> np.ndarray:
    """A simple random sample of ``k`` units, from days up to ``last_day`` if given."""
    pool = np.flatnonzero(arm.day <= (last_day if last_day is not None else arm.day.max()))
    if k > len(pool):
        raise ValueError("more labels than units in the labeling window")
    mask = np.zeros(len(arm.y), dtype=bool)
    mask[rng.choice(pool, size=k, replace=False)] = True
    return mask


SILENT_ANALYSES = ("naive_pooled", "ppi_launch", "ppi_spread")


def analyze_silent(
    t: DailyArm, c: DailyArm, spread_t: np.ndarray, spread_c: np.ndarray,
    launch_t: np.ndarray, launch_c: np.ndarray,
) -> dict[str, Readout]:
    """Judge-only difference, and PPI++ with launch-window or spread-out labels."""
    def ppi(lab_t: np.ndarray, lab_c: np.ndarray) -> Readout:
        r = ppi_mean_diff(t.y[lab_t], t.s[lab_t], t.s[~lab_t],
                          c.y[lab_c], c.s[lab_c], c.s[~lab_c])
        return Readout(r.estimate, r.se)

    return {
        "naive_pooled": _naive(t.s, c.s),
        "ppi_launch": ppi(launch_t, launch_c),
        "ppi_spread": ppi(spread_t, spread_c),
    }


def first_alarm(
    series: list[tuple[np.ndarray, np.ndarray]], n_days: int, ref_days: int = 7,
    window: int = 7, alpha: float = 0.05,
) -> int | None:
    """First day any series' recent mean differs from its reference-period mean.

    Each series is ``(values, day)``. On every day ``d`` from
    ``ref_days + window`` to ``n_days``, days ``d - window + 1 .. d`` are
    compared with days ``1 .. ref_days`` by a two-sample z-test with unpooled
    variances. The critical value is Bonferroni-corrected over every check of
    every series, so with no change the chance of any alarm is at most
    ``alpha``. Returns the alarm day, or ``None``.
    """
    checks = np.arange(ref_days + window, n_days + 1)
    if len(checks) == 0:
        raise ValueError("n_days too short for the reference period and window")
    z_crit = float(stats.norm.ppf(1 - alpha / (2 * len(checks) * len(series))))
    first = None
    for values, day in series:
        x = np.asarray(values, dtype=float)
        n_d = np.bincount(day, minlength=n_days + 1)[1:].astype(float)
        s1 = np.bincount(day, weights=x, minlength=n_days + 1)[1:]
        s2 = np.bincount(day, weights=x * x, minlength=n_days + 1)[1:]
        cn, c1, c2 = (np.concatenate([[0.0], np.cumsum(a)]) for a in (n_d, s1, s2))

        def stats_for(lo: np.ndarray, hi: np.ndarray, cn=cn, c1=c1, c2=c2):
            n = cn[hi] - cn[lo]
            with np.errstate(invalid="ignore", divide="ignore"):
                m = (c1[hi] - c1[lo]) / n
                v = ((c2[hi] - c2[lo]) - n * m**2) / (n - 1)
            return n, m, v

        n_r, m_r, v_r = stats_for(np.zeros(1, dtype=int), np.full(1, ref_days))
        n_w, m_w, v_w = stats_for(checks - window, checks)
        with np.errstate(invalid="ignore", divide="ignore"):
            z = (m_w - m_r) / np.sqrt(v_r / n_r + v_w / n_w)
        hits = checks[np.nan_to_num(np.abs(z), nan=0.0, posinf=0.0) > z_crit]
        if len(hits) and (first is None or hits[0] < first):
            first = int(hits[0])
    return first


def score_series(arm: DailyArm) -> tuple[np.ndarray, np.ndarray]:
    """Every judge score in the arm, for a score monitor."""
    return arm.s, arm.day


def agreement_series(arm: DailyArm, labeled: np.ndarray
                     ) -> list[tuple[np.ndarray, np.ndarray]]:
    """Judge pass rate among human-pass and among human-fail labeled units.

    These are the judge's sensitivity and false-positive rate. Unlike the raw
    agreement rate or the judge-minus-human gap, they don't move when true
    quality shifts and the judge stays the same.
    """
    out = []
    for good in (1.0, 0.0):
        keep = labeled & (arm.y == good)
        out.append((arm.s[keep], arm.day[keep]))
    return out


def expected_silent_naive(drift: SilentDrift, p_control: float, effect: float,
                          n_days: int) -> float:
    """Expected judge-only difference with equal traffic every day."""
    days = np.arange(1, n_days + 1)

    def mean_q(arm: str, p: float) -> float:
        p_d, sens, spec = drift.path(arm, p, days)
        return float(np.mean(sens * p_d + (1 - spec) * (1 - p_d)))

    return mean_q("t", p_control + effect) - mean_q("c", p_control)


def expected_ppi_launch(drift: SilentDrift, p_control: float, effect: float, n_days: int,
                        n_per_day: int, n_labeled: int, launch_days: int) -> float:
    """Large-sample mean of PPI++ when every label comes from the first
    ``launch_days`` days (population version of the PPI++ weight)."""
    days = np.arange(1, n_days + 1)
    n = n_days * n_per_day
    n_unlab = n - n_labeled

    def arm(name: str, p: float) -> float:
        p_d, sens, spec = drift.path(name, p, days)
        q_d = sens * p_d + (1 - spec) * (1 - p_d)
        early = days <= launch_days
        p_l, q_l = p_d[early].mean(), q_d[early].mean()
        cov_ys = (p_d * sens)[early].mean() - p_l * q_l
        m = q_d.mean()
        lam = cov_ys / ((1 + n_labeled / n_unlab) * m * (1 - m))
        unlab_mean = (n * m - n_labeled * q_l) / n_unlab
        return float(p_l - lam * (q_l - unlab_mean))

    return arm("t", p_control + effect) - arm("c", p_control)


def window_detection_prob(mean_ref: float, var_ref: float, n_ref: float, mean_win: float,
                          var_win: float, n_win: float, z_crit: float) -> float:
    """Probability one check flags, for a window wholly after the change."""
    shift = (mean_win - mean_ref) / np.sqrt(var_ref / n_ref + var_win / n_win)
    return float(stats.norm.cdf(shift - z_crit) + stats.norm.cdf(-shift - z_crit))


def monitor_z_crit(n_days: int, n_series: int, ref_days: int = 7, window: int = 7,
                   alpha: float = 0.05) -> float:
    """The Bonferroni critical value :func:`first_alarm` uses."""
    n_checks = n_days - ref_days - window + 1
    return float(stats.norm.ppf(1 - alpha / (2 * n_checks * n_series)))


# -- canary set: fixed stored outputs re-judged daily --------------------------------


def simulate_canary(
    drift: SilentDrift, arm: str, n_items: int, p: float, n_days: int,
    repeatability: float, rng: np.random.Generator,
) -> np.ndarray:
    """Judge verdicts, shape ``(n_items, n_days)``, on fixed stored outputs of one arm.

    True quality is drawn once per item at rate ``p`` and never changes, so
    ``drift.quality_shift`` has no effect here. Each verdict is
    ``u < rate`` with ``u = Phi(sqrt(r) z_item + sqrt(1 - r) e_item_day)``, so
    the judge's sensitivity and specificity hold on every day for any ``r``.
    """
    _check_p(p)
    if not 0.0 <= repeatability <= 1.0:
        raise ValueError("repeatability must be in [0, 1]")
    if n_items < 2 or n_days < 2:
        raise ValueError("need at least 2 items and 2 days")
    days = np.arange(1, n_days + 1)
    _, sens, spec = drift.path(arm, p, days)
    y = rng.random(n_items) < p
    rate = np.where(y[:, None], sens[None, :], 1.0 - spec[None, :])
    latent = (np.sqrt(repeatability) * rng.standard_normal((n_items, 1))
              + np.sqrt(1.0 - repeatability) * rng.standard_normal((n_items, n_days)))
    return (stats.norm.cdf(latent) < rate).astype(float)


def canary_z(verdicts: np.ndarray, ref_days: int = 7, window: int = 7
             ) -> tuple[np.ndarray, np.ndarray]:
    """Paired z-statistic on each check day for one canary set.

    For check day ``d``, each item's mean verdict over days
    ``d - window + 1 .. d`` minus its mean over days ``1 .. ref_days``; the
    statistic is the mean of those differences over its standard error across
    items. Returns ``(check_days, z)``; ``z`` is 0 where no item changed.
    """
    s = np.asarray(verdicts, dtype=float)
    n_items, n_days = s.shape
    checks = np.arange(ref_days + window, n_days + 1)
    if len(checks) == 0:
        raise ValueError("n_days too short for the reference period and window")
    ref = s[:, :ref_days].mean(axis=1)
    cs = np.concatenate([np.zeros((n_items, 1)), np.cumsum(s, axis=1)], axis=1)
    d = (cs[:, checks] - cs[:, checks - window]) / window - ref[:, None]
    with np.errstate(invalid="ignore", divide="ignore"):
        z = d.mean(axis=0) / (d.std(axis=0, ddof=1) / np.sqrt(n_items))
    return checks, np.nan_to_num(z, nan=0.0, posinf=0.0, neginf=0.0)


def canary_alarm(sets: list[np.ndarray], ref_days: int = 7, window: int = 7,
                 alpha: float = 0.05) -> int | None:
    """First check day on which any canary set's paired z exceeds the Bonferroni
    threshold over every check and set (the same threshold as
    :func:`first_alarm`). Returns ``None`` if no set ever flags."""
    first = None
    n_days = sets[0].shape[1]
    z_crit = monitor_z_crit(n_days, len(sets), ref_days, window, alpha)
    for s in sets:
        checks, z = canary_z(s, ref_days, window)
        hits = checks[np.abs(z) > z_crit]
        if len(hits) and (first is None or hits[0] < first):
            first = int(hits[0])
    return first


def _canary_item_moments(base: JudgeConfusion, after: JudgeConfusion, p: float,
                         ref_days: int, window: int) -> tuple[float, float]:
    """Mean and variance of one item's paired difference, fully changed window,
    ``repeatability = 0``."""
    out = []
    for good, w in ((True, p), (False, 1.0 - p)):
        r0 = base.sensitivity if good else 1 - base.specificity
        r1 = after.sensitivity if good else 1 - after.specificity
        out.append((w, r1 - r0, r0 * (1 - r0) / ref_days + r1 * (1 - r1) / window))
    mean = sum(w * m for w, m, _ in out)
    var = sum(w * v for w, _, v in out) + sum(w * (m - mean) ** 2 for w, m, _ in out)
    return mean, var


def canary_detection_prob(base: JudgeConfusion, after: JudgeConfusion, p: float,
                          n_items: int, z_crit: float, ref_days: int = 7,
                          window: int = 7) -> float:
    """Chance one check flags a fully changed window (``repeatability = 0``)."""
    mean, var = _canary_item_moments(base, after, p, ref_days, window)
    shift = mean / np.sqrt(var / n_items)
    return float(stats.norm.cdf(shift - z_crit) + stats.norm.cdf(-shift - z_crit))


def canary_items_for_power(base: JudgeConfusion, after: JudgeConfusion, p: float,
                           z_crit: float, power: float = 0.8, ref_days: int = 7,
                           window: int = 7) -> int:
    """Smallest canary set for which one check on a fully changed window flags
    with probability ``power`` (``repeatability = 0``, one-sided approximation)."""
    mean, var = _canary_item_moments(base, after, p, ref_days, window)
    if mean == 0.0:
        raise ValueError("the judge does not change on this arm")
    k = int(np.ceil(var * ((z_crit + stats.norm.ppf(power)) / mean) ** 2))
    while canary_detection_prob(base, after, p, k, z_crit, ref_days, window) < power:
        k += 1
    while k > 2 and canary_detection_prob(base, after, p, k - 1, z_crit, ref_days,
                                          window) >= power:
        k -= 1
    return k
