"""
A/B tests whose primary metric is scored by an LLM judge.

The true per-unit outcome is binary quality ``Y`` (good / not good), which a
human rater would label. The judge produces a binary score ``S`` whose errors
are described by a per-arm confusion matrix:

    P(S = 1 | Y = 1) = sensitivity_a
    P(S = 1 | Y = 0) = 1 - specificity_a

so the judged positive rate in arm ``a`` is

    q_a = sensitivity_a * p_a + (1 - specificity_a) * (1 - p_a).

Consequences for the naive analysis (difference of judged means):

* **Non-differential error** (same confusion matrix in both arms): the
  expected estimate is ``J * (p_t - p_c)`` with Youden's
  ``J = sensitivity + specificity - 1``. The effect is attenuated toward zero,
  but under H0 the test still holds its Type-I rate; the cost is power
  (Bross 1954; Rogan & Gladen 1978).
* **Differential error** (the treatment changes how the judge errs, e.g. the
  judge is more lenient toward the treatment's style): under H0 the naive
  estimate converges to ``q_t - q_c != 0``, so the Type-I error goes to 1 as
  the sample grows.

The correction implemented here is prediction-powered inference with power
tuning (PPI++; Angelopoulos, Bates, Fannjiang, Jordan, Zrnic 2023;
Angelopoulos, Duchi, Zrnic 2023). In each arm, a simple random subsample of
``n`` units also gets a human label and the remaining ``N`` units only get a
judge score. The per-arm estimate is

    theta_hat = mean_lab(Y) - lam * (mean_lab(S) - mean_unlab(S))

with variance ``Var(Y - lam*S)/n + lam^2 * Var(S)/N``. This is a control
variate (the same idea as CUPED, with the judge score as the covariate;
see also Chaganty, Mussmann, Liang 2018). ``theta_hat`` is unbiased for
``p_a`` for any judge, because the labeled subsample is random *within the
arm*. ``lam = 0`` gives the human-labels-only estimate. ``lam = 1`` gives
vanilla PPI. The power-tuned value ``lam* = Cov(Y, S) / ((1 + n/N) Var(S))``
minimises the variance.

A single calibration shared by both arms, such as an offline validation set
or labels drawn from control only, does **not** fix differential error. An
additive correction applied equally to both arms cancels in the difference,
so the result is the naive estimate again. Calibration has to happen within
each arm.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike
from scipy import stats

_MIN_VARIANCE = 1e-12
_MIN_LABELED = 2


@dataclass(frozen=True)
class JudgeConfusion:
    """Binary judge error rates relative to the human (true) label.

    Attributes
    ----------
    sensitivity : P(judge says good | truly good), in [0, 1].
    specificity : P(judge says bad | truly bad), in [0, 1].
    """

    sensitivity: float
    specificity: float

    def __post_init__(self) -> None:
        for name in ("sensitivity", "specificity"):
            value = getattr(self, name)
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1], got {value}")

    @property
    def youden_j(self) -> float:
        """Youden's J = sensitivity + specificity - 1 (attenuation factor)."""
        return self.sensitivity + self.specificity - 1.0

    def positive_rate(self, p: float) -> float:
        """Judged positive rate q given the true positive rate p."""
        _check_prob(p, "p")
        return self.sensitivity * p + (1.0 - self.specificity) * (1.0 - p)


@dataclass(frozen=True)
class JudgedArm:
    """Unit-level data for one arm.

    ``y`` holds the true labels for every unit. In a real experiment only
    ``y[labeled]`` would be observed. ``s`` holds the judge score for every unit.
    """

    y: np.ndarray
    s: np.ndarray
    labeled: np.ndarray


@dataclass(frozen=True)
class JudgeDiffResult:
    """Treatment-minus-control estimate with a Wald z-test and CI.

    ``lambdas`` holds the (treatment, control) PPI weights. It is ``None`` for
    the naive judge-only estimator.
    """

    estimate: float
    se: float
    ci_low: float
    ci_high: float
    p_value: float
    lambdas: tuple[float, float] | None = None


# -- simulation ----------------------------------------------------------------


def simulate_judged_arm(
    n: int,
    p: float,
    judge: JudgeConfusion,
    n_labeled: int = 0,
    seed: int | np.random.Generator | None = None,
) -> JudgedArm:
    """Simulate one arm: true labels, judge scores, and a random labeled subset.

    Parameters
    ----------
    n : number of units in the arm (all are judged).
    p : true probability of good quality.
    judge : confusion matrix of the judge in this arm.
    n_labeled : size of the simple random subsample that also gets a human label.
    seed : seed or ``numpy.random.Generator`` for reproducibility.
    """
    if n <= 0:
        raise ValueError("n must be positive")
    if not 0 <= n_labeled <= n:
        raise ValueError(f"n_labeled must be in [0, {n}], got {n_labeled}")
    _check_prob(p, "p")
    rng = np.random.default_rng(seed)

    y = (rng.random(n) < p).astype(float)
    p_positive = np.where(y == 1.0, judge.sensitivity, 1.0 - judge.specificity)
    s = (rng.random(n) < p_positive).astype(float)
    labeled = np.zeros(n, dtype=bool)
    labeled[rng.choice(n, size=n_labeled, replace=False)] = True
    return JudgedArm(y=y, s=s, labeled=labeled)


def simulate_judged_experiment(
    n_per_arm: int,
    p_control: float,
    effect: float,
    judge_control: JudgeConfusion,
    judge_treatment: JudgeConfusion | None = None,
    n_labeled_per_arm: int = 0,
    seed: int | np.random.Generator | None = None,
) -> tuple[JudgedArm, JudgedArm]:
    """Simulate a two-arm experiment with a judge-scored binary metric.

    The true treatment effect is ``effect = p_treatment - p_control``. When
    ``judge_treatment`` is ``None`` the judge errs the same way in both arms
    (non-differential error).

    Returns
    -------
    (treatment, control) : tuple of :class:`JudgedArm`.
    """
    p_treatment = p_control + effect
    _check_prob(p_treatment, "p_control + effect")
    if judge_treatment is None:
        judge_treatment = judge_control
    rng = np.random.default_rng(seed)
    treatment = simulate_judged_arm(n_per_arm, p_treatment, judge_treatment, n_labeled_per_arm, rng)
    control = simulate_judged_arm(n_per_arm, p_control, judge_control, n_labeled_per_arm, rng)
    return treatment, control


# -- estimators ----------------------------------------------------------------


def naive_judge_diff(s_t: ArrayLike, s_c: ArrayLike, alpha: float = 0.05) -> JudgeDiffResult:
    """Difference in mean judge score, treating the judge as ground truth.

    The estimator is unbiased for ``q_t - q_c``, the difference in *judged*
    rates. It estimates ``p_t - p_c`` only when the judge is perfect, and its
    expectation is ``J * (p_t - p_c)`` when the judge's error is
    non-differential.
    """
    st = _as_1d(s_t, "s_t")
    sc = _as_1d(s_c, "s_c")
    if len(st) < _MIN_LABELED or len(sc) < _MIN_LABELED:
        raise ValueError("each arm needs at least 2 judged units")
    estimate = float(st.mean() - sc.mean())
    se = float(np.sqrt(st.var(ddof=1) / len(st) + sc.var(ddof=1) / len(sc)))
    return _wald(estimate, se, alpha)


def ppi_mean_diff(
    y_lab_t: ArrayLike,
    s_lab_t: ArrayLike,
    s_unlab_t: ArrayLike,
    y_lab_c: ArrayLike,
    s_lab_c: ArrayLike,
    s_unlab_c: ArrayLike,
    lam: float | None = None,
    alpha: float = 0.05,
) -> JudgeDiffResult:
    """Prediction-powered difference in means (PPI / PPI++), arm by arm.

    Parameters
    ----------
    y_lab_t, s_lab_t : human labels and judge scores on the treatment arm's
        labeled subsample (a simple random sample of that arm).
    s_unlab_t : judge scores on the remaining treatment units.
    y_lab_c, s_lab_c, s_unlab_c : the same for control.
    lam : PPI weight. ``None`` (default) estimates the variance-minimising
        PPI++ weight separately in each arm. ``0.0`` gives the
        human-labels-only estimator. ``1.0`` gives vanilla PPI.
    alpha : two-sided significance level for the CI.

    Returns
    -------
    :class:`JudgeDiffResult` with the per-arm weights in ``lambdas``.

    Notes
    -----
    Unbiasedness requires the labeled units to be a random sample *within each
    arm*. Labels from one arm (or an offline set) applied to both arms cannot
    remove differential judge error.
    """
    theta_t, var_t, lam_t = _ppi_arm(y_lab_t, s_lab_t, s_unlab_t, lam, "treatment")
    theta_c, var_c, lam_c = _ppi_arm(y_lab_c, s_lab_c, s_unlab_c, lam, "control")
    result = _wald(theta_t - theta_c, float(np.sqrt(var_t + var_c)), alpha)
    return JudgeDiffResult(
        estimate=result.estimate,
        se=result.se,
        ci_low=result.ci_low,
        ci_high=result.ci_high,
        p_value=result.p_value,
        lambdas=(lam_t, lam_c),
    )


def analyze_judged_experiment(
    treatment: JudgedArm,
    control: JudgedArm,
    alpha: float = 0.05,
) -> dict[str, JudgeDiffResult]:
    """Run the naive, human-only and PPI++ analyses on one simulated experiment.

    Returns a dict with keys ``"naive"`` (judge scores on every unit),
    ``"human_only"`` (human labels on the labeled subsample) and ``"ppi"``
    (PPI++ combining both).
    """
    def split(arm: JudgedArm) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        return arm.y[arm.labeled], arm.s[arm.labeled], arm.s[~arm.labeled]

    yt, st, ut = split(treatment)
    yc, sc, uc = split(control)
    return {
        "naive": naive_judge_diff(treatment.s, control.s, alpha=alpha),
        "human_only": ppi_mean_diff(yt, st, ut, yc, sc, uc, lam=0.0, alpha=alpha),
        "ppi": ppi_mean_diff(yt, st, ut, yc, sc, uc, lam=None, alpha=alpha),
    }


# -- closed-form answers --------------------------------------------------------


def expected_naive_diff(
    p_control: float,
    effect: float,
    judge_control: JudgeConfusion,
    judge_treatment: JudgeConfusion | None = None,
) -> float:
    """Expected value of the naive judge-only estimate, ``q_t - q_c``.

    Equals ``J * effect`` under non-differential error and is non-zero under
    H0 (``effect = 0``) when the judge errs differently by arm.
    """
    judge_treatment = judge_control if judge_treatment is None else judge_treatment
    q_t = judge_treatment.positive_rate(p_control + effect)
    q_c = judge_control.positive_rate(p_control)
    return q_t - q_c


def naive_judge_se(
    n_per_arm: int,
    p_control: float,
    effect: float,
    judge_control: JudgeConfusion,
    judge_treatment: JudgeConfusion | None = None,
) -> float:
    """Standard error of the naive judge-only difference with ``n_per_arm`` units."""
    if n_per_arm <= 0:
        raise ValueError("n_per_arm must be positive")
    judge_treatment = judge_control if judge_treatment is None else judge_treatment
    q_t = judge_treatment.positive_rate(p_control + effect)
    q_c = judge_control.positive_rate(p_control)
    return float(np.sqrt((q_t * (1 - q_t) + q_c * (1 - q_c)) / n_per_arm))


def ppi_expected_se(
    n_per_arm: int,
    n_labeled_per_arm: int,
    p_control: float,
    effect: float,
    judge_control: JudgeConfusion,
    judge_treatment: JudgeConfusion | None = None,
    lam: float | None = None,
) -> float:
    """Standard error of the PPI difference at population moments.

    Each arm contributes
    ``[Var(Y) - 2 lam Cov(Y,S) + lam^2 Var(S)] / n + lam^2 Var(S) / N``, where
    ``n = n_labeled_per_arm`` and ``N = n_per_arm - n``. ``lam=None`` uses
    the oracle PPI++ weight, which gives
    ``Var(Y)/n * (1 - rho^2 * N/(N+n))`` with ``rho = Corr(Y, S)``.
    ``lam=0`` gives the human-labels-only standard error.
    """
    if not 0 < n_labeled_per_arm <= n_per_arm:
        raise ValueError("need 0 < n_labeled_per_arm <= n_per_arm")
    judge_treatment = judge_control if judge_treatment is None else judge_treatment
    p_t = p_control + effect
    _check_prob(p_t, "p_control + effect")
    var = _ppi_arm_variance(n_per_arm, n_labeled_per_arm, p_t, judge_treatment, lam)
    var += _ppi_arm_variance(n_per_arm, n_labeled_per_arm, p_control, judge_control, lam)
    return float(np.sqrt(var))


def two_sided_rejection_rate(mean: float, se: float, alpha: float = 0.05) -> float:
    """P(|Z| > z_{1-alpha/2}) when the estimate ~ N(mean, se^2).

    This is the power when ``mean`` is the true effect. It is the Type-I error
    when ``mean`` is the bias of an estimator under H0.
    """
    _check_alpha(alpha)
    if se <= 0:
        raise ValueError("se must be positive")
    z = stats.norm.ppf(1 - alpha / 2)
    shift = mean / se
    return float(stats.norm.cdf(shift - z) + stats.norm.cdf(-shift - z))


def judge_sample_size_inflation(
    p_control: float,
    effect: float,
    judge: JudgeConfusion,
) -> float:
    """Sample-size multiplier for the naive judge metric vs. the true metric.

    This assumes non-differential error. The ratio of per-arm sample sizes
    needed for equal power is

        [q_t(1-q_t) + q_c(1-q_c)] / (J^2 * [p_t(1-p_t) + p_c(1-p_c)]).

    Under differential error the naive test is biased, so no amount of extra
    sample fixes it and this multiplier does not apply.
    """
    if effect == 0:
        raise ValueError("effect must be non-zero")
    j = judge.youden_j
    if j <= 0:
        raise ValueError("judge must be better than chance (sensitivity + specificity > 1)")
    p_t = p_control + effect
    _check_prob(p_t, "p_control + effect")
    q_t, q_c = judge.positive_rate(p_t), judge.positive_rate(p_control)
    var_true = p_t * (1 - p_t) + p_control * (1 - p_control)
    if var_true < _MIN_VARIANCE:
        raise ValueError("true outcome has zero variance")
    return (q_t * (1 - q_t) + q_c * (1 - q_c)) / (j**2 * var_true)


# -- helpers -------------------------------------------------------------------


def _ppi_arm(
    y_lab: ArrayLike,
    s_lab: ArrayLike,
    s_unlab: ArrayLike,
    lam: float | None,
    name: str,
) -> tuple[float, float, float]:
    y = _as_1d(y_lab, f"y_lab_{name}")
    s = _as_1d(s_lab, f"s_lab_{name}")
    u = _as_1d(s_unlab, f"s_unlab_{name}")
    if y.shape != s.shape:
        raise ValueError(f"{name}: y_lab and s_lab must have the same length")
    n, n_unlab = len(y), len(u)
    if n < _MIN_LABELED:
        raise ValueError(f"{name}: need at least {_MIN_LABELED} labeled units")

    if lam is None:
        var_s = float(np.var(np.concatenate([s, u]), ddof=1))
        if n_unlab < _MIN_LABELED or var_s < _MIN_VARIANCE:
            lam = 0.0
        else:
            cov_ys = float(np.cov(y, s, ddof=1)[0, 1])
            lam = cov_ys / ((1.0 + n / n_unlab) * var_s)
    lam = float(lam)
    if lam != 0.0 and n_unlab < _MIN_LABELED:
        raise ValueError(f"{name}: need at least {_MIN_LABELED} unlabeled units when lam != 0")

    if lam == 0.0:
        return float(y.mean()), float(y.var(ddof=1) / n), 0.0
    theta = float(y.mean() - lam * (s.mean() - u.mean()))
    var = float(np.var(y - lam * s, ddof=1) / n + lam**2 * np.var(u, ddof=1) / n_unlab)
    return theta, var, lam


def _ppi_arm_variance(
    n_total: int,
    n_labeled: int,
    p: float,
    judge: JudgeConfusion,
    lam: float | None,
) -> float:
    n_unlab = n_total - n_labeled
    q = judge.positive_rate(p)
    var_y = p * (1 - p)
    var_s = q * (1 - q)
    cov_ys = p * judge.sensitivity - p * q
    if lam is None:
        if n_unlab == 0 or var_s < _MIN_VARIANCE:
            lam = 0.0
        else:
            lam = cov_ys / ((1.0 + n_labeled / n_unlab) * var_s)
    if lam != 0.0 and n_unlab == 0:
        raise ValueError("lam != 0 requires unlabeled units")
    var = (var_y - 2 * lam * cov_ys + lam**2 * var_s) / n_labeled
    if n_unlab > 0:
        var += lam**2 * var_s / n_unlab
    return var


def _wald(estimate: float, se: float, alpha: float) -> JudgeDiffResult:
    _check_alpha(alpha)
    z = stats.norm.ppf(1 - alpha / 2)
    if se > 0:
        p_value = float(2 * stats.norm.sf(abs(estimate) / se))
    else:
        p_value = 1.0 if estimate == 0 else 0.0
    return JudgeDiffResult(
        estimate=estimate,
        se=se,
        ci_low=estimate - z * se,
        ci_high=estimate + z * se,
        p_value=p_value,
    )


def _as_1d(x: ArrayLike, name: str) -> np.ndarray:
    arr = np.asarray(x, dtype=float)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be a 1-D array")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must be finite (no NaN/inf)")
    return arr


def _check_prob(p: float, name: str) -> None:
    if not 0.0 <= p <= 1.0:
        raise ValueError(f"{name} must be in [0, 1], got {p}")


def _check_alpha(alpha: float) -> None:
    if not 0 < alpha < 1:
        raise ValueError("alpha must be in (0, 1)")
