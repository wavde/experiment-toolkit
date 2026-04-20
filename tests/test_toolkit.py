from __future__ import annotations

import numpy as np
import pytest

from experiment_toolkit import (
    apply_cuped,
    compute_theta,
    mde_for_n,
    msprt_pvalue,
    ratio_metric_variance,
    sample_size_for_mde,
)


# -- sample size ----------------------------------------------------------

def test_sample_size_roundtrip():
    # n computed from mde should imply that mde back out
    n = sample_size_for_mde(mde=0.1, std_dev=1.0, alpha=0.05, power=0.80)
    mde = mde_for_n(n_per_arm=n, std_dev=1.0, alpha=0.05, power=0.80)
    assert abs(mde - 0.1) < 0.001


def test_sample_size_textbook_value():
    # Canonical rule of thumb: n ~ 16 * sigma^2 / mde^2 for alpha=0.05, power=0.80
    n = sample_size_for_mde(mde=0.1, std_dev=1.0, alpha=0.05, power=0.80)
    assert 1500 < n < 1700  # exact is ~1570


def test_sample_size_rejects_bad_inputs():
    with pytest.raises(ValueError):
        sample_size_for_mde(mde=0, std_dev=1.0)
    with pytest.raises(ValueError):
        sample_size_for_mde(mde=0.1, std_dev=-1.0)


# -- CUPED ----------------------------------------------------------------

def test_theta_zero_when_uncorrelated():
    rng = np.random.default_rng(0)
    y = rng.normal(size=10_000)
    x = rng.normal(size=10_000)
    assert abs(compute_theta(y, x)) < 0.05


def test_cuped_preserves_mean():
    rng = np.random.default_rng(1)
    x = rng.normal(100, 20, size=5_000)
    y = 0.5 * x + rng.normal(0, 10, size=5_000)
    y_adj = apply_cuped(y, x)
    assert abs(y_adj.mean() - y.mean()) < 0.01


def test_cuped_reduces_variance():
    rng = np.random.default_rng(2)
    x = rng.normal(0, 1, size=10_000)
    y = 0.8 * x + rng.normal(0, np.sqrt(1 - 0.8**2), size=10_000)
    # rho = 0.8, expected variance reduction ~ 1 - 0.64 = 0.36
    var_ratio = np.var(apply_cuped(y, x), ddof=1) / np.var(y, ddof=1)
    assert 0.30 < var_ratio < 0.45


# -- ratio metric ----------------------------------------------------------

def test_ratio_metric_variance_basic():
    rng = np.random.default_rng(3)
    # sessions per user ~ Poisson(5), minutes per session ~ Normal(10, 2)
    sessions = rng.poisson(5, size=10_000) + 1
    minutes = sessions * rng.normal(10, 2, size=10_000)
    ratio, se = ratio_metric_variance(minutes, sessions)
    assert abs(ratio - 10) < 0.2
    assert se > 0


def test_ratio_metric_variance_validates():
    with pytest.raises(ValueError):
        ratio_metric_variance([1, 2], [0, 0])
    with pytest.raises(ValueError):
        ratio_metric_variance([1], [1])  # too few observations


# -- sequential / mSPRT ----------------------------------------------------

def test_msprt_pvalue_in_unit_interval():
    p = msprt_pvalue(delta_hat=0.1, sigma=1.0, n_per_arm=1000, tau=0.1)
    assert 0.0 <= p <= 1.0


def test_msprt_pvalue_decreases_with_effect():
    p_small = msprt_pvalue(delta_hat=0.01, sigma=1.0, n_per_arm=1000, tau=0.1)
    p_large = msprt_pvalue(delta_hat=0.20, sigma=1.0, n_per_arm=1000, tau=0.1)
    assert p_large < p_small


def test_msprt_pvalue_capped_at_one():
    # No effect, small n: p should be 1 (can't reject)
    p = msprt_pvalue(delta_hat=0.0, sigma=1.0, n_per_arm=10, tau=0.1)
    assert p == 1.0


def test_msprt_always_valid_type1_error():
    """Under the null and peeking at every step, naive z-test rejects often;
    mSPRT should reject rarely (alpha-controlled at any stopping time)."""
    rng = np.random.default_rng(42)
    n_trials = 200
    rejections = 0
    for _ in range(n_trials):
        # Simulate one null experiment with peeking
        max_n = 2000
        treatment = rng.normal(0, 1, size=max_n)
        control = rng.normal(0, 1, size=max_n)
        rejected = False
        for n in range(50, max_n + 1, 50):
            delta = treatment[:n].mean() - control[:n].mean()
            sigma = np.std(np.concatenate([treatment[:n], control[:n]]), ddof=1)
            p = msprt_pvalue(delta, sigma, n, tau=0.1)
            if p < 0.05:
                rejected = True
                break
        if rejected:
            rejections += 1
    # Always-valid p-values control Type-I error; expect < 10% even with peeking
    assert rejections / n_trials < 0.10
