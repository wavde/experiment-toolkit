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
) -> int:
    """Required per-arm sample size to detect `mde` with the given power."""
    if mde <= 0:
        raise ValueError("mde must be positive")
    if std_dev <= 0:
        raise ValueError("std_dev must be positive")
    if not 0 < alpha < 1:
        raise ValueError("alpha must be in (0, 1)")
    if not 0 < power < 1:
        raise ValueError("power must be in (0, 1)")

    z_alpha = stats.norm.ppf(1 - alpha / 2) if two_sided else stats.norm.ppf(1 - alpha)
    z_beta = stats.norm.ppf(power)

    n = 2 * std_dev**2 * (z_alpha + z_beta) ** 2 / mde**2
    return math.ceil(n)


def mde_for_n(
    n_per_arm: int,
    std_dev: float,
    alpha: float = 0.05,
    power: float = 0.80,
    two_sided: bool = True,
) -> float:
    """Inverse: given a fixed per-arm sample size, return the MDE you can
    detect with the given power."""
    if n_per_arm <= 0:
        raise ValueError("n_per_arm must be positive")
    if std_dev <= 0:
        raise ValueError("std_dev must be positive")

    z_alpha = stats.norm.ppf(1 - alpha / 2) if two_sided else stats.norm.ppf(1 - alpha)
    z_beta = stats.norm.ppf(power)

    return (z_alpha + z_beta) * std_dev * (2 / n_per_arm) ** 0.5
