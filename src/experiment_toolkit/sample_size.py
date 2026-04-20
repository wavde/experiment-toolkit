"""
Sample size and MDE calculators for two-sample experiments on continuous
outcomes. Uses the standard normal approximation (valid for n > ~30 per arm).

For a two-sided test at significance alpha and power 1-beta, comparing two
means with equal per-arm sample size n and pooled standard deviation sigma:

    n = 2 * sigma^2 * (z_{1-alpha/2} + z_{1-beta})^2 / mde^2

where mde is the minimum detectable effect in outcome units.
"""

from __future__ import annotations

import math

from scipy import stats


def sample_size_for_mde(
    mde: float,
    std_dev: float,
    alpha: float = 0.05,
    power: float = 0.80,
    two_sided: bool = True,
    rho: float = 0.0,
) -> int:
    """Required per-arm sample size to detect `mde` with the given power.

    When ``rho`` (the correlation between the outcome and a pre-experiment
    covariate) is non-zero, the formula applies the CUPED variance reduction
    factor ``(1 - rho^2)`` — see Deng, Xu, Kohavi, Walker (2013). At ``rho=0``
    the result is the classical two-sample z-formula (backwards compatible).
    """
    if mde <= 0:
        raise ValueError("mde must be positive")
    if std_dev <= 0:
        raise ValueError("std_dev must be positive")
    if not 0 < alpha < 1:
        raise ValueError("alpha must be in (0, 1)")
    if not 0 < power < 1:
        raise ValueError("power must be in (0, 1)")
    if not -1 <= rho <= 1:
        raise ValueError("rho must be in [-1, 1]")

    z_alpha = stats.norm.ppf(1 - alpha / 2) if two_sided else stats.norm.ppf(1 - alpha)
    z_beta = stats.norm.ppf(power)
    variance = std_dev**2 * (1 - rho**2)

    n = 2 * variance * (z_alpha + z_beta) ** 2 / mde**2
    return math.ceil(n)


def mde_for_n(
    n_per_arm: int,
    std_dev: float,
    alpha: float = 0.05,
    power: float = 0.80,
    two_sided: bool = True,
    rho: float = 0.0,
) -> float:
    """Inverse: given a fixed per-arm sample size, return the MDE you can
    detect with the given power. ``rho`` enables CUPED variance reduction."""
    if n_per_arm <= 0:
        raise ValueError("n_per_arm must be positive")
    if std_dev <= 0:
        raise ValueError("std_dev must be positive")
    if not -1 <= rho <= 1:
        raise ValueError("rho must be in [-1, 1]")

    z_alpha = stats.norm.ppf(1 - alpha / 2) if two_sided else stats.norm.ppf(1 - alpha)
    z_beta = stats.norm.ppf(power)
    variance = std_dev**2 * (1 - rho**2)

    return (z_alpha + z_beta) * (2 * variance / n_per_arm) ** 0.5


def cuped_sample_size_summary(
    mde: float,
    std_dev: float,
    rho: float,
    alpha: float = 0.05,
    power: float = 0.80,
) -> dict[str, float | int]:
    """Side-by-side comparison of required sample size with and without CUPED.

    Returns a dict with per-arm sample sizes under the naive two-sample z test
    (``rho=0``) and CUPED at the supplied ``rho``, plus absolute and percent
    sample-size savings and the underlying variance-reduction factor.
    """
    n_naive = sample_size_for_mde(mde, std_dev, alpha=alpha, power=power, rho=0.0)
    n_cuped = sample_size_for_mde(mde, std_dev, alpha=alpha, power=power, rho=rho)
    return {
        "n_naive_per_arm": n_naive,
        "n_cuped_per_arm": n_cuped,
        "absolute_saving": n_naive - n_cuped,
        "percent_reduction": float(1 - n_cuped / n_naive) if n_naive > 0 else 0.0,
        "variance_reduction_factor": 1 - rho**2,
        "rho": rho,
        "mde": mde,
        "std_dev": std_dev,
    }
