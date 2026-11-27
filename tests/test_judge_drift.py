import numpy as np
import pytest

from experiment_toolkit.judge_drift import (
    ANALYSES,
    JudgeUpgrade,
    analyze_drift,
    expected_naive,
    expected_naive_se,
    expected_ppi_front,
    simulate_drift_arm,
    simulate_drift_experiment,
)
from experiment_toolkit.llm_judge import JudgeConfusion

OLD = JudgeConfusion(0.85, 0.85)
LENIENT = JudgeConfusion(0.85, 0.75)
SAME = JudgeUpgrade(OLD, OLD, OLD, OLD)
BOTH = JudgeUpgrade(OLD, OLD, LENIENT, LENIENT)
T_ONLY = JudgeUpgrade(OLD, OLD, LENIENT, OLD)


def _means(upgrade, effect, reps, seed, n=8_000, n_lab=400):
    rng = np.random.default_rng(seed)
    out = {a: [] for a in ANALYSES}
    for _ in range(reps):
        t, c = simulate_drift_experiment(n, 0.6, effect, upgrade, rng)
        for a, r in analyze_drift(t, c, n_lab, rng).items():
            out[a].append((r.estimate, r.rejects()))
    return {a: np.array(v) for a, v in out.items()}


# -- closed form -----------------------------------------------------------------

def test_naive_bias_is_half_the_new_judge_gap():
    # Under H0 only the new judge's extra false positives on treatment matter:
    # 0.10 * (1 - 0.6), weighted by the 50% post-upgrade share.
    assert expected_naive(T_ONLY, 0.6, 0.0) == pytest.approx(0.5 * 0.10 * 0.4)
    assert expected_naive(T_ONLY, 0.6, 0.0, pre_only=True) == 0.0
    assert expected_naive(BOTH, 0.6, 0.0) == 0.0


def test_front_loaded_ppi_is_unbiased_without_upgrade():
    assert expected_ppi_front(SAME, 0.6, 0.03, 10_000, 500) == pytest.approx(0.03)


def test_front_loaded_ppi_bias_under_differential_upgrade():
    bias = expected_ppi_front(T_ONLY, 0.6, 0.0, 10_000, 500)
    assert 0.0 < bias < expected_naive(T_ONLY, 0.6, 0.0)


def test_naive_se_pre_only_is_larger():
    full = expected_naive_se(SAME, 0.6, 0.0, 10_000)
    pre = expected_naive_se(SAME, 0.6, 0.0, 10_000, pre_only=True)
    assert pre == pytest.approx(full * np.sqrt(2))


# -- simulation vs closed form ---------------------------------------------------

def test_simulation_matches_closed_form_under_differential_upgrade():
    m = _means(T_ONLY, 0.0, reps=200, seed=1)
    assert m["naive_pooled"][:, 0].mean() == pytest.approx(0.02, abs=0.002)
    assert m["naive_pooled"][:, 1].mean() > 0.6
    assert m["ppi_front"][:, 0].mean() == pytest.approx(
        expected_ppi_front(T_ONLY, 0.6, 0.0, 8_000, 400), abs=0.005
    )
    for a in ("naive_pre", "ppi_spread", "ppi_stratified"):
        assert abs(m[a][:, 0].mean()) < 0.005, a
        assert m[a][:, 1].mean() < 0.09, a


def test_uniform_upgrade_is_harmless_under_null():
    m = _means(BOTH, 0.0, reps=200, seed=2)
    for a in ANALYSES:
        assert m[a][:, 1].mean() < 0.09, a


def test_unbiased_ppi_recovers_effect():
    m = _means(T_ONLY, 0.03, reps=200, seed=3)
    for a in ("ppi_spread", "ppi_stratified"):
        assert m[a][:, 0].mean() == pytest.approx(0.03, abs=0.006), a


# -- mechanics and validation ----------------------------------------------------

def test_periods_and_judges():
    arm = simulate_drift_arm(200_000, 0.6, OLD, LENIENT, 0.25, np.random.default_rng(4))
    assert arm.new.mean() == pytest.approx(0.75)
    assert not arm.new[:1000].any()
    fpr_new = arm.s[arm.new & (arm.y == 0)].mean()
    fpr_old = arm.s[~arm.new & (arm.y == 0)].mean()
    assert fpr_new == pytest.approx(0.25, abs=0.01)
    assert fpr_old == pytest.approx(0.15, abs=0.01)


def test_validation():
    with pytest.raises(ValueError):
        JudgeUpgrade(OLD, OLD, OLD, OLD, w_old=1.0)
    with pytest.raises(ValueError):
        simulate_drift_experiment(1_000, 0.99, 0.05, SAME, np.random.default_rng(0))
