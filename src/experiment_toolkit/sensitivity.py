"""
Sensitivity analysis helpers for causal claims from observational data.

Two independent techniques are provided:

1. **E-value** (VanderWeele & Ding, Annals of Internal Medicine, 2017).
   Converts an observed risk ratio (or a standardised mean difference
   approximated as a risk ratio) into the minimum strength of association
   that an unmeasured confounder would need to have, with *both* the
   treatment and the outcome, to fully explain away the observed effect.
   The E-value has no assumption about the direction or distribution of
   the confounder and is a transparent one-number summary for memos.

2. **Rosenbaum bounds** for matched-pair studies (Rosenbaum, 2002).
   Given matched pairs with outcome differences, compute upper / lower
   bounds on the Wilcoxon signed-rank one-sided p-value under a worst-case
   hidden bias of size ``gamma`` (the odds ratio by which a hidden
   confounder could shift the treatment-assignment probability inside a
   pair). ``rosenbaum_gamma_threshold`` returns the smallest gamma at
   which the upper-bound p-value crosses a significance level — the
   "break point" reported in applied memos.

Both helpers return plain Python / numpy objects; no pandas dependency.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike
from scipy import stats


@dataclass
class EValueResult:
    """Structured result from :func:`e_value`."""

    point_estimate: float
    ci_low: float | None
    ci_high: float | None
    rr_observed: float
    rr_ci: float | None
    e_value_point: float
    e_value_ci: float | None

    def __str__(self) -> str:  # pragma: no cover - convenience
        s = f"E-value (point)={self.e_value_point:.3f}"
        if self.e_value_ci is not None:
            s += f", E-value (CI)={self.e_value_ci:.3f}"
        return s


def e_value(
    point_estimate: float,
    ci_low: float | None = None,
    ci_high: float | None = None,
    kind: str = "rr",
    sd: float | None = None,
    rare: bool = False,
    p0: float | None = None,
    lower_ci: float | None = None,
) -> EValueResult:
    """
    Compute the E-value for an observed effect estimate.

    Parameters
    ----------
    point_estimate : float
        The observed effect. Interpretation depends on ``kind``:

        - ``"rr"``: a risk ratio (must be > 0).
        - ``"or"``: an odds ratio (must be > 0). If ``rare=True`` it is used
          as-is; otherwise it is converted to a risk ratio via
          ``RR = OR / ((1 - p0) + p0 * OR)`` and you **must** supply ``p0``,
          the baseline outcome risk in the unexposed group. There is no
          publication-grade default for ``p0`` — it is outcome-specific.
        - ``"smd"``: a standardised mean difference (Cohen's d), which may be
          negative (a protective effect). Converted to an approximate risk
          ratio via ``RR = exp(0.91 * d)`` (VanderWeele & Ding 2017).

    ci_low, ci_high : optional bounds of the confidence interval on the *same
        scale* as ``point_estimate``. The E-value for the CI is computed from
        the bound closest to the null (the lower bound for a harmful effect,
        ``RR > 1``; the upper bound for a protective effect, ``RR < 1``).
    lower_ci : deprecated alias for ``ci_low``, kept for backwards
        compatibility.
    sd : reserved for ``kind="smd"``; Cohen's d is already standardised so
        callers typically need not set this.
    rare : whether the outcome is rare (only matters for ``kind="or"``).

    Returns
    -------
    EValueResult
    """
    if lower_ci is not None and ci_low is None:
        ci_low = lower_ci
    if kind not in {"rr", "or", "smd"}:
        raise ValueError(f"kind must be one of rr/or/smd, got {kind!r}")
    if kind in {"rr", "or"} and point_estimate <= 0:
        raise ValueError("point_estimate must be positive for kind='rr'/'or'")
    if kind == "or" and not rare:
        if p0 is None:
            raise ValueError(
                "kind='or' with rare=False needs p0 (baseline outcome risk in "
                "the unexposed group) to convert OR to RR; pass p0 or rare=True."
            )
        if not 0 < p0 < 1:
            raise ValueError("p0 must be in (0, 1)")

    def _to_rr(x: float) -> float:
        if kind == "rr":
            return x
        if kind == "or":
            if rare:
                return x
            return x / ((1 - p0) + p0 * x)
        return math.exp(0.91 * x)

    rr = _to_rr(point_estimate)

    def _ev(rr_val: float) -> float:
        rr_val = rr_val if rr_val >= 1 else 1 / rr_val
        return rr_val + math.sqrt(rr_val * (rr_val - 1))

    ev_point = _ev(rr)

    # Pick the CI bound closest to the null (RR = 1) for the direction observed.
    null_side = ci_low if rr >= 1 else ci_high
    rr_ci = _to_rr(null_side) if null_side is not None else None

    ev_ci: float | None
    if rr_ci is None:
        ev_ci = None
    elif (rr > 1 and rr_ci <= 1) or (rr < 1 and rr_ci >= 1):
        # CI crosses the null -> the relevant bound IS the null
        ev_ci = 1.0
    else:
        ev_ci = _ev(rr_ci)

    return EValueResult(
        point_estimate=point_estimate,
        ci_low=ci_low,
        ci_high=ci_high,
        rr_observed=rr,
        rr_ci=rr_ci,
        e_value_point=ev_point,
        e_value_ci=ev_ci,
    )


def _wilcoxon_signed_rank_one_sided_p(differences: np.ndarray, gamma: float) -> float:
    """Upper-bound one-sided p-value under a hidden bias of size ``gamma``.

    Implements the Rosenbaum (2002) bounding argument for the Wilcoxon
    signed-rank statistic, using a normal approximation to the null
    distribution of the signed-rank sum. At ``gamma=1`` this is the
    large-sample normal approximation to the usual one-sided Wilcoxon
    signed-rank p-value (not the exact small-sample value); larger gamma
    widens the null distribution, raising the p-value. Ties are ranked by
    average rank; zero differences are dropped.
    """
    d = differences[differences != 0]
    if len(d) == 0:
        return 1.0
    abs_d = np.abs(d)
    ranks = stats.rankdata(abs_d, method="average")
    s_obs = float(ranks[d > 0].sum())

    # Under Rosenbaum bounds, each positive-sign indicator is Bernoulli with
    # probability p_plus = gamma / (1 + gamma) for the *upper* bound.
    p_plus = gamma / (1.0 + gamma)
    mean = p_plus * ranks.sum()
    var = p_plus * (1.0 - p_plus) * (ranks**2).sum()
    if var <= 0:
        return 1.0
    z = (s_obs - mean) / math.sqrt(var)
    # one-sided upper p-value (testing for positive effect)
    return float(1.0 - stats.norm.cdf(z))


def rosenbaum_wilcoxon_bounds(
    differences: ArrayLike,
    gammas: ArrayLike = (1.0, 1.25, 1.5, 2.0, 3.0),
) -> list[dict[str, float]]:
    """Rosenbaum-bound one-sided Wilcoxon signed-rank p-values.

    Parameters
    ----------
    differences : paired treated-minus-control outcome differences.
    gammas : iterable of hidden-bias magnitudes to evaluate. ``gamma=1``
        corresponds to no hidden bias; larger values represent stronger
        unmeasured confounding.

    Returns
    -------
    list of dicts with keys ``gamma`` and ``p_upper`` (upper-bound on the
    one-sided p-value for a positive effect).
    """
    d = np.asarray(differences, dtype=float)
    out: list[dict[str, float]] = []
    for g in gammas:
        g = float(g)
        if g < 1:
            raise ValueError("gamma must be >= 1")
        out.append({"gamma": g, "p_upper": _wilcoxon_signed_rank_one_sided_p(d, g)})
    return out


def rosenbaum_gamma_threshold(
    differences: ArrayLike,
    alpha: float = 0.05,
    gamma_max: float = 10.0,
    tol: float = 1e-3,
) -> float:
    """Smallest gamma at which the Rosenbaum upper-bound p-value crosses ``alpha``.

    Uses bisection on a monotone-in-gamma bound. Returns ``gamma_max`` if the
    conclusion is robust all the way up to that ceiling, and ``1.0`` if the
    result is already not significant at gamma=1.
    """
    d = np.asarray(differences, dtype=float)
    if _wilcoxon_signed_rank_one_sided_p(d, 1.0) > alpha:
        return 1.0
    if _wilcoxon_signed_rank_one_sided_p(d, gamma_max) <= alpha:
        return gamma_max
    lo, hi = 1.0, gamma_max
    while hi - lo > tol:
        mid = 0.5 * (lo + hi)
        if _wilcoxon_signed_rank_one_sided_p(d, mid) <= alpha:
            lo = mid
        else:
            hi = mid
    return lo
