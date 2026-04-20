import numpy as np
import pytest

from experiment_toolkit import (
    CSResult,
    EValueResult,
    cs_staggered_att,
    cuped_sample_size_summary,
    e_value,
    mde_for_n,
    msprt_cuped_pvalue,
    msprt_pvalue,
    rosenbaum_gamma_threshold,
    rosenbaum_wilcoxon_bounds,
    sample_size_for_mde,
    simulate_staggered_panel,
)


# -- CUPED-aware sample size ----------------------------------------------

def test_sample_size_cuped_halves_with_rho_0_707():
    # rho^2 = 0.5 => variance cut in half => sample size cut in half
    n_naive = sample_size_for_mde(mde=0.1, std_dev=1.0, rho=0.0)
    n_cuped = sample_size_for_mde(mde=0.1, std_dev=1.0, rho=float(np.sqrt(0.5)))
    assert abs(n_cuped / n_naive - 0.5) < 0.005


def test_mde_for_n_honours_rho():
    mde_naive = mde_for_n(n_per_arm=1000, std_dev=1.0, rho=0.0)
    mde_cuped = mde_for_n(n_per_arm=1000, std_dev=1.0, rho=0.5)
    assert mde_cuped < mde_naive
    # variance factor is (1 - 0.25) = 0.75, so MDE scales by sqrt(0.75)
    assert abs(mde_cuped / mde_naive - np.sqrt(0.75)) < 1e-6


def test_sample_size_rejects_bad_rho():
    with pytest.raises(ValueError):
        sample_size_for_mde(mde=0.1, std_dev=1.0, rho=1.5)


def test_cuped_sample_size_summary_shape():
    s = cuped_sample_size_summary(mde=0.1, std_dev=1.0, rho=0.5)
    assert s["n_cuped_per_arm"] < s["n_naive_per_arm"]
    assert 0 < s["percent_reduction"] < 1
    assert s["variance_reduction_factor"] == pytest.approx(0.75)


# -- mSPRT + CUPED --------------------------------------------------------

def test_msprt_cuped_matches_plain_msprt_when_x_is_independent():
    rng = np.random.default_rng(0)
    n = 2_000
    x_t = rng.normal(size=n)
    x_c = rng.normal(size=n)
    y_t = rng.normal(size=n) + 0.05  # small real effect
    y_c = rng.normal(size=n)
    p_cuped = msprt_cuped_pvalue(y_t, y_c, x_t, x_c, tau=1.0)
    p_plain = msprt_pvalue(
        delta_hat=float(y_t.mean() - y_c.mean()),
        sigma=float(np.concatenate([y_t, y_c]).std(ddof=1)),
        n_per_arm=n,
        tau=1.0,
    )
    # When X ⟂ Y, CUPED ~= plain (difference should be small)
    assert abs(p_cuped - p_plain) < 0.05


def test_msprt_cuped_tighter_when_x_correlates_with_y():
    rng = np.random.default_rng(1)
    n = 2_000
    x_t = rng.normal(size=n)
    x_c = rng.normal(size=n)
    # Strong correlation ~0.8 between X and Y => CUPED should reduce the p-value
    y_t = 0.8 * x_t + rng.normal(size=n) * 0.6 + 0.05
    y_c = 0.8 * x_c + rng.normal(size=n) * 0.6
    p_cuped = msprt_cuped_pvalue(y_t, y_c, x_t, x_c, tau=1.0)
    p_plain = msprt_pvalue(
        delta_hat=float(y_t.mean() - y_c.mean()),
        sigma=float(np.concatenate([y_t, y_c]).std(ddof=1)),
        n_per_arm=n,
        tau=1.0,
    )
    assert p_cuped < p_plain


# -- E-value --------------------------------------------------------------

def test_e_value_vanderweele_ding_example():
    # From the paper: RR=3.9 -> E-value ~= 7.26
    res = e_value(point_estimate=3.9, lower_ci=1.8)
    assert isinstance(res, EValueResult)
    assert res.e_value_point == pytest.approx(7.26, rel=0.01)
    # lower CI of 1.8 -> E-value for CI ~= 3.0
    assert res.e_value_ci == pytest.approx(3.0, rel=0.02)


def test_e_value_protective_effect_is_symmetric():
    # RR < 1 should be treated as 1/RR for the E-value
    res_harm = e_value(point_estimate=2.0)
    res_prot = e_value(point_estimate=0.5)
    assert res_harm.e_value_point == pytest.approx(res_prot.e_value_point, rel=1e-9)


def test_e_value_ci_crossing_null_returns_one():
    res = e_value(point_estimate=1.5, lower_ci=0.9)
    assert res.e_value_ci == 1.0


def test_e_value_smd_conversion():
    res = e_value(point_estimate=0.5, kind="smd")
    # exp(0.91 * 0.5) ~= 1.577
    assert res.rr_observed == pytest.approx(np.exp(0.91 * 0.5), rel=1e-6)


# -- Rosenbaum bounds -----------------------------------------------------

def test_rosenbaum_bounds_gamma_1_matches_classical():
    rng = np.random.default_rng(0)
    d = rng.normal(0.3, 1.0, size=50)
    rows = rosenbaum_wilcoxon_bounds(d, gammas=[1.0, 2.0])
    assert rows[0]["gamma"] == 1.0
    # monotone: gamma=2 worse than gamma=1
    assert rows[1]["p_upper"] > rows[0]["p_upper"]


def test_rosenbaum_gamma_threshold_bisection():
    rng = np.random.default_rng(0)
    d = rng.normal(0.5, 1.0, size=100)  # strong signal
    gamma_star = rosenbaum_gamma_threshold(d, alpha=0.05)
    assert gamma_star > 1.0


def test_rosenbaum_gamma_threshold_returns_1_if_not_sig():
    rng = np.random.default_rng(0)
    d = rng.normal(0.0, 1.0, size=20)  # no effect
    gamma_star = rosenbaum_gamma_threshold(d, alpha=0.05)
    assert gamma_star == 1.0


# -- Callaway-Sant'Anna ---------------------------------------------------

def test_cs_recovers_cohort_effects():
    panel = simulate_staggered_panel(
        cohort_sizes={6: 20, 10: 20, 14: 20},
        n_never_treated=30,
        n_periods=20,
        cohort_effects={6: 1.0, 10: 3.0, 14: 5.0},
        seed=42,
    )
    res = cs_staggered_att(
        unit=panel["unit"],
        period=panel["period"],
        y=panel["y"],
        cohort=panel["cohort"],
        n_bootstrap=100,
        seed=0,
    )
    assert isinstance(res, CSResult)
    # True overall ATT is roughly mean of per-cohort average post-treatment
    # effects. With 20 periods and static effects 1/3/5, equal weighting over
    # (g, t) cells gives an overall ~ 3.
    assert abs(res.overall_att - 3.0) < 1.0
    # We should have an entry for each of the three cohorts
    cohorts_seen = {r["cohort"] for r in res.group_time_att}
    assert cohorts_seen == {6, 10, 14}


def test_cs_raises_when_no_never_treated():
    # cohort=0 required for never-treated controls in this implementation
    panel = simulate_staggered_panel(
        cohort_sizes={6: 10}, n_never_treated=0, n_periods=10, seed=0
    )
    with pytest.raises(ValueError):
        cs_staggered_att(
            unit=panel["unit"],
            period=panel["period"],
            y=panel["y"],
            cohort=panel["cohort"],
            n_bootstrap=10,
        )


def test_cs_rejects_unbalanced_panel():
    panel = simulate_staggered_panel(
        cohort_sizes={6: 5}, n_never_treated=5, n_periods=10, seed=0
    )
    # drop one row to unbalance
    mask = np.ones(len(panel["y"]), dtype=bool)
    mask[0] = False
    with pytest.raises(ValueError):
        cs_staggered_att(
            unit=panel["unit"][mask],
            period=panel["period"][mask],
            y=panel["y"][mask],
            cohort=panel["cohort"][mask],
            n_bootstrap=10,
        )
