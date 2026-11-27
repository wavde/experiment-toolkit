"""
How far a pre-treatment covariate cuts the human-label budget of PPI++.

Loop 1 (:mod:`experiment_toolkit.llm_judge`) showed that when an LLM judge
scores the primary metric, PPI++ with human labels drawn within each arm is
unbiased, but the label budget limits power. This module adds a second
control variate: a pre-treatment covariate ``X`` that is known for every unit
(for example the query's difficulty bucket or the user's prior-period
outcome), in the spirit of CUPED (Deng et al. 2013).

Per arm, with ``n`` labeled units, ``N`` unlabeled units and a vector of
control variates ``Z`` (any subset of the judge score ``S`` and ``X``):

    theta_hat = mean_lab(Y) - lam' (mean_lab(Z) - mean_unlab(Z))
    Var       = Var(Y - lam'Z) / n + lam' Cov(Z) lam / N

The variance-minimising weight is ``lam* = Cov(Z)^-1 Cov(Z, Y) / (1 + n/N)``.
At that weight the per-arm variance is

    Var(Y) / n * (1 - R^2 * N / (N + n))

where ``R^2`` is the population R^2 of ``Y`` regressed on ``Z``. Choices of ``Z``:

- ``()``: human labels only.
- ``("x",)``: human labels plus CUPED on the covariate.
- ``("s",)``: PPI++, as in loop 1.
- ``("s", "x")``: PPI++ plus the covariate.

``theta_hat`` stays unbiased for any ``Z`` that is measured the same way on
labeled and unlabeled units, because labels are a random sample within the
arm. Adding ``X`` therefore cannot introduce bias. It can only fail to help.
Two things limit the help: ``R^2`` can only rise to 1, and the factor
``N / (N + n)`` caps the gain when labels are a large share of the arm.

Model
-----
``X`` is binary with ``P(X=1) = pi``. True quality is binary with
``P(Y=1 | X=x) = p[x]``. The judge's score depends on ``Y`` only, through a
confusion matrix, so ``S`` and ``X`` are conditionally independent given
``Y``. All moments of ``(Y, S, X)`` are available in closed form.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike
from scipy import stats

from experiment_toolkit.llm_judge import JudgeConfusion, JudgeDiffResult

CONTROL_VARIATES = ((), ("x",), ("s",), ("s", "x"))
_MIN_UNITS = 3
_RIDGE = 1e-12


@dataclass(frozen=True)
class CovariateArm:
    """Population for one arm.

    Attributes
    ----------
    p_x0, p_x1 : P(Y=1 | X=0) and P(Y=1 | X=1).
    pi : P(X=1).
    judge : the judge's confusion matrix for this arm's outputs.
    """

    p_x0: float
    p_x1: float
    pi: float
    judge: JudgeConfusion

    def __post_init__(self) -> None:
        for name in ("p_x0", "p_x1", "pi"):
            value = getattr(self, name)
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1], got {value}")

    @property
    def p(self) -> float:
        """Marginal P(Y=1)."""
        return self.pi * self.p_x1 + (1 - self.pi) * self.p_x0

    def shifted(self, effect: float, judge: JudgeConfusion | None = None) -> CovariateArm:
        """The same population with ``effect`` added to P(Y=1) at both X levels."""
        return CovariateArm(self.p_x0 + effect, self.p_x1 + effect, self.pi,
                            self.judge if judge is None else judge)

    def moments(self) -> tuple[np.ndarray, np.ndarray]:
        """Mean vector and covariance matrix of ``(Y, S, X)``."""
        sens, fpr = self.judge.sensitivity, 1.0 - self.judge.specificity
        p, pi = self.p, self.pi
        q = sens * p + fpr * (1 - p)
        q_x1 = sens * self.p_x1 + fpr * (1 - self.p_x1)
        cov = np.array(
            [
                [p * (1 - p), sens * p - p * q, pi * self.p_x1 - p * pi],
                [0.0, q * (1 - q), pi * q_x1 - q * pi],
                [0.0, 0.0, pi * (1 - pi)],
            ]
        )
        cov = np.triu(cov) + np.triu(cov, 1).T
        return np.array([p, q, pi]), cov


def covariate_arm(p: float, r2_x: float, judge: JudgeConfusion, pi: float = 0.5) -> CovariateArm:
    """Arm with marginal P(Y=1) = ``p`` whose binary covariate explains ``r2_x`` of Var(Y).

    The two X levels sit symmetrically around ``p``. This gives
    ``R^2(Y; X) = pi (1 - pi) d^2 / (p (1 - p))``, where ``d = p_x1 - p_x0``.
    """
    if not 0.0 <= r2_x < 1.0:
        raise ValueError("r2_x must be in [0, 1)")
    d = np.sqrt(r2_x * p * (1 - p) / (pi * (1 - pi)))
    arm = CovariateArm(p - pi * d, p + (1 - pi) * d, pi, judge)
    return arm


def _index(z: tuple[str, ...]) -> list[int]:
    lookup = {"s": 1, "x": 2}
    try:
        return [lookup[name] for name in z]
    except KeyError as exc:
        raise ValueError(f"control variates must be a subset of ('s', 'x'), got {z}") from exc


def population_r2(arm: CovariateArm, z: tuple[str, ...]) -> float:
    """Population R^2 of Y on the control variates ``z``."""
    if not z:
        return 0.0
    _, cov = arm.moments()
    idx = _index(z)
    c = cov[0, idx]
    return float(c @ np.linalg.solve(cov[np.ix_(idx, idx)], c) / cov[0, 0])


def expected_arm_variance(
    arm: CovariateArm, n_total: int, n_labeled: int, z: tuple[str, ...]
) -> float:
    """Variance of ``theta_hat`` for one arm at the oracle weight ``lam*``."""
    if not 0 < n_labeled <= n_total:
        raise ValueError("need 0 < n_labeled <= n_total")
    _, cov = arm.moments()
    var_y = cov[0, 0]
    n_unlab = n_total - n_labeled
    if not z or n_unlab == 0:
        return var_y / n_labeled
    return var_y / n_labeled * (1 - population_r2(arm, z) * n_unlab / (n_unlab + n_labeled))


def expected_se(
    treatment: CovariateArm,
    control: CovariateArm,
    n_total: int,
    n_labeled: int,
    z: tuple[str, ...],
) -> float:
    """SE of the treatment-minus-control estimate at the oracle weights."""
    return float(
        np.sqrt(
            expected_arm_variance(treatment, n_total, n_labeled, z)
            + expected_arm_variance(control, n_total, n_labeled, z)
        )
    )


def labels_for_power(
    treatment: CovariateArm,
    control: CovariateArm,
    n_total: int,
    z: tuple[str, ...],
    power: float = 0.80,
    alpha: float = 0.05,
) -> int | None:
    """Smallest per-arm label count giving ``power`` for a two-sided test.

    Returns ``None`` when even labeling every unit is not enough.
    """
    effect = treatment.p - control.p
    if effect == 0:
        raise ValueError("treatment and control have the same mean")
    z_a = stats.norm.ppf(1 - alpha / 2)

    def pw(n_lab: int) -> float:
        shift = abs(effect) / expected_se(treatment, control, n_total, n_lab, z)
        return float(stats.norm.cdf(shift - z_a) + stats.norm.cdf(-shift - z_a))

    if pw(n_total) < power:
        return None
    lo, hi = _MIN_UNITS, n_total
    while lo < hi:
        mid = (lo + hi) // 2
        if pw(mid) >= power:
            hi = mid
        else:
            lo = mid + 1
    return lo


# -- simulation and estimation ---------------------------------------------------


@dataclass(frozen=True)
class CovariateArmData:
    """Unit-level data: true labels ``y`` (observed only where ``labeled``),
    judge scores ``s`` and covariate ``x`` for every unit."""

    y: np.ndarray
    s: np.ndarray
    x: np.ndarray
    labeled: np.ndarray

    def controls(self, z: tuple[str, ...]) -> np.ndarray:
        cols = {"s": self.s, "x": self.x}
        _index(z)
        return np.column_stack([cols[name] for name in z]) if z else np.empty((len(self.y), 0))


def simulate_covariate_arm(
    arm: CovariateArm, n: int, n_labeled: int, rng: np.random.Generator
) -> CovariateArmData:
    """Draw ``n`` units and human-label a simple random sample of ``n_labeled``."""
    if n <= 0 or not 0 <= n_labeled <= n:
        raise ValueError("need n > 0 and 0 <= n_labeled <= n")
    x = (rng.random(n) < arm.pi).astype(float)
    y = (rng.random(n) < np.where(x == 1.0, arm.p_x1, arm.p_x0)).astype(float)
    p_pos = np.where(y == 1.0, arm.judge.sensitivity, 1.0 - arm.judge.specificity)
    s = (rng.random(n) < p_pos).astype(float)
    labeled = np.zeros(n, dtype=bool)
    labeled[rng.choice(n, size=n_labeled, replace=False)] = True
    return CovariateArmData(y, s, x, labeled)


def control_variate_arm(
    y_lab: ArrayLike, z_lab: ArrayLike, z_unlab: ArrayLike
) -> tuple[float, float, np.ndarray]:
    """One arm's estimate, its variance and the fitted weights.

    ``z_lab`` / ``z_unlab`` are 2-D arrays (units x control variates) and may
    have zero columns. The weights are the plug-in version of ``lam*``:
    ``Cov(Z)`` comes from all units in the arm and ``Cov(Z, Y)`` from the
    labeled ones.
    """
    y = np.asarray(y_lab, dtype=float)
    zl = np.atleast_2d(np.asarray(z_lab, dtype=float).reshape(len(y), -1))
    zu = np.asarray(z_unlab, dtype=float)
    zu = zu.reshape(len(zu), zl.shape[1])
    n, n_unlab, k = len(y), len(zu), zl.shape[1]
    if n < _MIN_UNITS:
        raise ValueError(f"need at least {_MIN_UNITS} labeled units")
    if k == 0:
        return float(y.mean()), float(y.var(ddof=1) / n), np.zeros(0)
    if n_unlab < _MIN_UNITS:
        raise ValueError(f"need at least {_MIN_UNITS} unlabeled units with control variates")
    cov_zz = np.atleast_2d(np.cov(np.vstack([zl, zu]), rowvar=False))
    c = np.array([np.cov(y, zl[:, j])[0, 1] for j in range(k)])
    lam = np.linalg.solve(cov_zz + _RIDGE * np.eye(k), c) / (1 + n / n_unlab)
    theta = float(y.mean() - lam @ (zl.mean(axis=0) - zu.mean(axis=0)))
    resid = y - zl @ lam
    cov_u = np.atleast_2d(np.cov(zu, rowvar=False))
    var = float(resid.var(ddof=1) / n + lam @ cov_u @ lam / n_unlab)
    return theta, var, lam


def control_variate_diff(
    treatment: CovariateArmData,
    control: CovariateArmData,
    z: tuple[str, ...],
    alpha: float = 0.05,
) -> JudgeDiffResult:
    """Treatment-minus-control estimate with control variates ``z`` in each arm."""
    if not 0 < alpha < 1:
        raise ValueError("alpha must be in (0, 1)")
    parts = []
    for arm in (treatment, control):
        zs = arm.controls(z)
        parts.append(control_variate_arm(arm.y[arm.labeled], zs[arm.labeled], zs[~arm.labeled]))
    est = parts[0][0] - parts[1][0]
    se = float(np.sqrt(parts[0][1] + parts[1][1]))
    z_a = stats.norm.ppf(1 - alpha / 2)
    p_value = float(2 * stats.norm.sf(abs(est) / se)) if se > 0 else 1.0
    return JudgeDiffResult(est, se, est - z_a * se, est + z_a * se, p_value)
