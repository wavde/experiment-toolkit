import numpy as np
import pytest

from experiment_toolkit import JudgeConfusion
from experiment_toolkit.llm_judge import ppi_expected_se
from experiment_toolkit.ppi_covariates import (
    CONTROL_VARIATES,
    CovariateArm,
    control_variate_arm,
    control_variate_diff,
    covariate_arm,
    expected_arm_variance,
    expected_se,
    labels_for_power,
    population_r2,
    simulate_covariate_arm,
)

JUDGE = JudgeConfusion(0.85, 0.85)


def _arms(r2, effect=0.03, judge=JUDGE):
    c = covariate_arm(0.6, r2, judge)
    return c.shifted(effect), c


# -- closed form -----------------------------------------------------------------

def test_covariate_arm_hits_target_r2_and_mean():
    arm = covariate_arm(0.6, 0.25, JUDGE)
    assert arm.p == pytest.approx(0.6)
    assert population_r2(arm, ("x",)) == pytest.approx(0.25)


def test_judge_only_matches_loop1_closed_form():
    # With an uninformative covariate, PPI++ here must equal llm_judge's PPI++.
    for acc in (0.95, 0.85, 0.70):
        j = JudgeConfusion(acc, acc)
        t, c = _arms(0.0, judge=j)
        ours = expected_se(t, c, 20_000, 500, ("s",))
        assert ours == pytest.approx(ppi_expected_se(20_000, 500, 0.6, 0.03, j))


def test_loop1_label_budgets_reproduced():
    expected = {0.95: 982, 0.85: 2_397, 0.70: 3_610}
    for acc, n in expected.items():
        j = JudgeConfusion(acc, acc)
        assert labels_for_power(*_arms(0.0, judge=j), 20_000, ("s",)) == n
    assert labels_for_power(*_arms(0.0), 20_000, ()) == 4_126


def test_r2_of_independent_signals_is_subadditive():
    arm = covariate_arm(0.6, 0.25, JUDGE)
    r_s, r_x, r_sx = (population_r2(arm, z) for z in (("s",), ("x",), ("s", "x")))
    assert max(r_s, r_x) < r_sx < r_s + r_x


def test_adding_a_control_variate_never_increases_oracle_variance():
    arm = covariate_arm(0.6, 0.1, JUDGE)
    v = {z: expected_arm_variance(arm, 20_000, 500, z) for z in CONTROL_VARIATES}
    assert v[("s", "x")] <= v[("s",)] <= v[()]
    assert v[("s", "x")] <= v[("x",)] <= v[()]


def test_variance_formula_is_r2_times_unlabeled_share():
    arm = covariate_arm(0.6, 0.5, JUDGE)
    n, n_lab = 5_000, 2_000
    r2 = population_r2(arm, ("x",))
    expected = arm.p * (1 - arm.p) / n_lab * (1 - r2 * (n - n_lab) / n)
    assert expected_arm_variance(arm, n, n_lab, ("x",)) == pytest.approx(expected)


# -- simulation vs closed form ---------------------------------------------------

def test_estimators_unbiased_with_closed_form_sd():
    t, c = _arms(0.25)
    rng = np.random.default_rng(0)
    est = {z: [] for z in CONTROL_VARIATES}
    for _ in range(300):
        a = simulate_covariate_arm(t, 20_000, 500, rng)
        b = simulate_covariate_arm(c, 20_000, 500, rng)
        for z in CONTROL_VARIATES:
            est[z].append(control_variate_diff(a, b, z).estimate)
    for z, e in est.items():
        se = expected_se(t, c, 20_000, 500, z)
        assert np.mean(e) == pytest.approx(0.03, abs=4 * se / np.sqrt(300)), z
        assert np.std(e, ddof=1) == pytest.approx(se, rel=0.12), z


def test_type1_controlled_with_lenient_judge():
    c = covariate_arm(0.6, 0.25, JUDGE)
    t = c.shifted(0.0, JudgeConfusion(0.85, 0.80))
    rng = np.random.default_rng(1)
    rej = []
    for _ in range(400):
        a = simulate_covariate_arm(t, 10_000, 400, rng)
        b = simulate_covariate_arm(c, 10_000, 400, rng)
        rej.append(control_variate_diff(a, b, ("s", "x")).p_value < 0.05)
    assert np.mean(rej) < 0.075


# -- mechanics and validation ----------------------------------------------------

def test_no_control_variates_is_labeled_mean():
    y = np.array([1.0, 0.0, 1.0, 1.0])
    theta, var, lam = control_variate_arm(y, np.empty((4, 0)), np.empty((10, 0)))
    assert theta == pytest.approx(0.75)
    assert var == pytest.approx(y.var(ddof=1) / 4)
    assert lam.size == 0


def test_validation():
    with pytest.raises(ValueError):
        CovariateArm(0.5, 1.2, 0.5, JUDGE)
    with pytest.raises(ValueError):
        covariate_arm(0.6, 1.0, JUDGE)
    with pytest.raises(ValueError, match="subset"):
        population_r2(covariate_arm(0.6, 0.1, JUDGE), ("q",))
    with pytest.raises(ValueError):
        labels_for_power(*_arms(0.1, effect=0.0), 20_000, ("s",))
