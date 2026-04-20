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

``msprt_cuped_pvalue`` combines mSPRT with CUPED variance reduction
(Deng et al. 2013). CUPED linearly re-expresses the outcome using a
pre-experiment covariate and reduces variance by a factor ``(1 - rho^2)``.
Because CUPED is a deterministic transformation under randomization, running
mSPRT on the CUPED-adjusted outcomes preserves the always-valid Type-I
guarantee while hitting the rejection threshold in roughly ``(1 - rho^2)``
of the samples that plain mSPRT would need.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike

from experiment_toolkit.cuped import compute_theta


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
    delta_hat = float(delta_hat)
    sigma = float(sigma)
    n_per_arm = int(n_per_arm)
    tau = float(tau)

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


def msprt_cuped_pvalue(
    y_t: ArrayLike,
    y_c: ArrayLike,
    x_t: ArrayLike,
    x_c: ArrayLike,
    n_per_arm: int | None = None,
    tau: float | None = None,
    theta: float | None = None,
) -> float:
    """
    Always-valid mSPRT p-value with CUPED variance reduction.

    The CUPED coefficient ``theta`` and pooled mean of ``X`` are either
    supplied by the caller (to preserve the martingale structure when peeking
    repeatedly) or estimated once from the pooled data passed in this call.
    Under a/b randomization, estimating ``theta`` from the full pooled sample
    is unbiased — but re-estimating it at *every* peek from an ever-growing
    stream does violate the always-valid guarantee. Callers doing sequential
    peeking should estimate ``theta`` on a warm-up sample and then pass it in
    (frozen) on every subsequent peek.

    Parameters
    ----------
    y_t, y_c : streamed outcomes for the treatment and control arms.
    x_t, x_c : pre-experiment covariates for the same users (same length).
    n_per_arm : per-arm sample size at the peek time. If ``None``, uses the
        length of ``y_t`` / ``y_c`` (must be equal).
    tau : prior SD for the effect; defaults to SD of the CUPED-adjusted
        outcome on the current peek data.
    theta : optional pre-computed CUPED coefficient. When ``None``, theta
        is estimated from the pooled data in this call.

    Returns
    -------
    always-valid p-value, in [0, 1], computed on the CUPED-adjusted outcome.
    """
    y_t_arr = np.asarray(y_t, dtype=float)
    y_c_arr = np.asarray(y_c, dtype=float)
    x_t_arr = np.asarray(x_t, dtype=float)
    x_c_arr = np.asarray(x_c, dtype=float)
    if y_t_arr.shape != x_t_arr.shape or y_c_arr.shape != x_c_arr.shape:
        raise ValueError("x_* arrays must match y_* shapes")

    if n_per_arm is None:
        if len(y_t_arr) != len(y_c_arr):
            raise ValueError("arms have unequal length; pass n_per_arm explicitly")
        n_per_arm = int(len(y_t_arr))

    yt = y_t_arr[:n_per_arm]
    yc = y_c_arr[:n_per_arm]
    xt = x_t_arr[:n_per_arm]
    xc = x_c_arr[:n_per_arm]

    if theta is None:
        y_pooled = np.concatenate([yt, yc])
        x_pooled = np.concatenate([xt, xc])
        theta = compute_theta(y_pooled, x_pooled)

    x_pooled_mean = float(np.concatenate([xt, xc]).mean())
    yt_adj = yt - theta * (xt - x_pooled_mean)
    yc_adj = yc - theta * (xc - x_pooled_mean)

    delta_hat = float(yt_adj.mean() - yc_adj.mean())
    pooled_sd = float(np.concatenate([yt_adj, yc_adj]).std(ddof=1))
    if pooled_sd <= 0:
        return 1.0
    if tau is None:
        tau = pooled_sd

    return msprt_pvalue(delta_hat=delta_hat, sigma=pooled_sd, n_per_arm=n_per_arm, tau=tau)
