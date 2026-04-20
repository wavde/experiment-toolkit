"""
Variance of ratio metrics via the delta method.

Real product metrics are often ratios (revenue / session, clicks / impression,
minutes / DAU). A naive two-sample t-test on per-user ratios is usually wrong
because the numerator and denominator are correlated.

Delta-method approximation for R = E[N] / E[D] with per-user (n_i, d_i):

    Var(R) ~= (1/mean(d))^2 * ( Var(N) - 2*R*Cov(N,D) + R^2 * Var(D) )

Reference: Deng, Knoblich, Lu (2018), "Applying the Delta Method in Metric
Analytics: A Practical Guide with Novel Ideas."
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike


def ratio_metric_variance(numerator: ArrayLike, denominator: ArrayLike) -> tuple[float, float]:
    """
    Return (ratio_estimate, standard_error) for the per-unit ratio metric.

    Parameters
    ----------
    numerator : per-unit numerator values (e.g., minutes watched per user)
    denominator : per-unit denominator values (e.g., sessions per user)

    Returns
    -------
    (ratio, standard_error)
    """
    n = np.asarray(numerator, dtype=float)
    d = np.asarray(denominator, dtype=float)
    if n.shape != d.shape:
        raise ValueError("numerator and denominator must have the same shape")
    if len(n) < 2:
        raise ValueError("need at least 2 observations")

    mean_n = n.mean()
    mean_d = d.mean()
    if mean_d == 0:
        raise ValueError("mean of denominator is zero; ratio undefined")

    ratio = mean_n / mean_d

    # Delta-method variance of the sample ratio.
    var_n = np.var(n, ddof=1)
    var_d = np.var(d, ddof=1)
    cov_nd = np.cov(n, d, ddof=1)[0, 1]

    var_ratio = (var_n - 2 * ratio * cov_nd + ratio**2 * var_d) / (len(n) * mean_d**2)
    se = float(np.sqrt(max(var_ratio, 0.0)))

    return float(ratio), se
