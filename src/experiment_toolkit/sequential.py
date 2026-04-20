"""
Always-valid p-values via the mixture Sequential Probability Ratio Test (mSPRT).

Reference: Johari, Pekelis, Walsh (2015), "Always Valid Inference: Bringing
Sequential Analysis to A/B Testing." Netflix and Optimizely both ship variants
of this in production.

The mSPRT mixes SPRTs over a prior on the effect size, yielding a test
statistic whose 1/statistic gives a p-value valid at any stopping time.

For a two-sample z-test with per-arm sample size n, difference of means
delta_hat, pooled variance sigma^2, and a normal mixing prior N(0, tau^2):

    Lambda = sqrt( sigma^2 / (sigma^2 + n*tau^2/2) ) *
             exp( n * delta_hat^2 / (2*sigma^2) *
                  (n*tau^2/2) / (sigma^2 + n*tau^2/2) )

and the always-valid p-value is min(1, 1/Lambda).
"""

from __future__ import annotations

import numpy as np


def msprt_pvalue(
    delta_hat: float,
    sigma: float,
    n_per_arm: int,
    tau: float,
) -> float:
    """
    Compute an always-valid mSPRT p-value for a two-sample mean comparison.

    Parameters
    ----------
    delta_hat : observed difference in means (treatment - control)
    sigma : pooled within-arm standard deviation of the outcome
    n_per_arm : per-arm sample size at the moment of the peek
    tau : standard deviation of the mixing prior on the true effect

    Returns
    -------
    always-valid p-value, in [0, 1].
    """
    if sigma <= 0:
        raise ValueError("sigma must be positive")
    if n_per_arm <= 0:
        raise ValueError("n_per_arm must be positive")
    if tau <= 0:
        raise ValueError("tau must be positive (the prior must be proper)")

    sigma2 = sigma**2
    # Variance of delta_hat under the null: V = 2*sigma^2 / n_per_arm
    v = 2 * sigma2 / n_per_arm
    shrinkage = tau**2 / (v + tau**2)

    log_lambda = (
        0.5 * np.log(v / (v + tau**2))
        + (delta_hat**2 / (2 * v)) * shrinkage
    )
    # p = min(1, 1 / Lambda) => -log(p) = min(0, log(Lambda))
    p = float(min(1.0, np.exp(-log_lambda)))
    return p
