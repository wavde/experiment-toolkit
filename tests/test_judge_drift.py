import numpy as np
import pytest

from scipy import stats

from experiment_toolkit.judge_drift import (
    ANALYSES,
    JudgeUpgrade,
    SilentDrift,
    agreement_series,
    analyze_drift,
    analyze_silent,
    canary_alarm,
    canary_detection_prob,
    canary_items_for_power,
    canary_z,
    expected_naive,
    expected_naive_se,
    expected_ppi_front,
    expected_ppi_launch,
    expected_silent_naive,
    first_alarm,
    label_mask,
    monitor_z_crit,
    rejection_rate,
    score_series,
    simulate_drift_arm,
    simulate_drift_experiment,
    simulate_canary,
    simulate_silent_experiment,
    window_detection_prob,
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


# -- silent drift (no version boundary) --------------------------------------------

S_NONE = SilentDrift(OLD, OLD, OLD, 15)
S_T_STEP = SilentDrift(OLD, LENIENT, OLD, 15)
S_T_RAMP = SilentDrift(OLD, LENIENT, OLD, 15, ramp_days=14)
S_QUALITY = SilentDrift(OLD, OLD, OLD, 15, quality_shift=-0.08)


def test_silent_naive_closed_form():
    # Step on day 15 of 28: half the days carry the 0.10 * 0.4 treatment gap.
    assert expected_silent_naive(S_T_STEP, 0.6, 0.0, 28) == pytest.approx(0.02)
    # Linear ramp over days 15-28: mean weight (1 + ... + 14) / 14 / 28 = 7.5 / 28.
    assert expected_silent_naive(S_T_RAMP, 0.6, 0.0, 28) == pytest.approx(0.04 * 7.5 / 28)
    # A quality shift common to both arms with an unchanged judge adds no gap.
    assert expected_silent_naive(S_QUALITY, 0.6, 0.0, 28) == pytest.approx(0.0)


def test_launch_week_ppi_closed_form():
    assert expected_ppi_launch(S_NONE, 0.6, 0.03, 28, 500, 500, 7) == pytest.approx(0.03)
    bias = expected_ppi_launch(S_T_STEP, 0.6, 0.0, 28, 500, 500, 7)
    assert 0.0 < bias < expected_silent_naive(S_T_STEP, 0.6, 0.0, 28)


def _silent(drift, reps, seed, n_per_day=300, n_lab=400):
    rng = np.random.default_rng(seed)
    out = {a: [] for a in ("naive_pooled", "ppi_launch", "ppi_spread")}
    for _ in range(reps):
        t, c = simulate_silent_experiment(28, n_per_day, 0.6, 0.0, drift, rng)
        res = analyze_silent(t, c, label_mask(t, n_lab, rng), label_mask(c, n_lab, rng),
                             label_mask(t, n_lab, rng, 7), label_mask(c, n_lab, rng, 7))
        for a, r in res.items():
            out[a].append((r.estimate, r.rejects()))
    return {a: np.array(v) for a, v in out.items()}, n_per_day, n_lab


def test_silent_simulation_matches_closed_form():
    m, n_per_day, n_lab = _silent(S_T_STEP, reps=200, seed=11)
    assert m["naive_pooled"][:, 0].mean() == pytest.approx(0.02, abs=0.002)
    assert m["ppi_launch"][:, 0].mean() == pytest.approx(
        expected_ppi_launch(S_T_STEP, 0.6, 0.0, 28, n_per_day, n_lab, 7), abs=0.005
    )
    assert abs(m["ppi_spread"][:, 0].mean()) < 0.005
    assert m["ppi_spread"][:, 1].mean() < 0.09


def test_silent_drift_schedule():
    t, c = simulate_silent_experiment(28, 80_000, 0.6, 0.0, S_T_RAMP, np.random.default_rng(5))
    for day, fpr in ((14, 0.15), (21, 0.15 + 0.10 * 7 / 14), (28, 0.25)):
        on = (t.day == day) & (t.y == 0)
        assert t.s[on].mean() == pytest.approx(fpr, abs=0.01), day
    assert c.s[(c.day == 28) & (c.y == 0)].mean() == pytest.approx(0.15, abs=0.01)


def test_monitor_threshold_and_closed_form():
    assert monitor_z_crit(28, 2) == pytest.approx(stats.norm.ppf(1 - 0.05 / (2 * 15 * 2)))
    assert window_detection_prob(0.5, 0.25, 1_000, 0.55, 0.2475, 1_000, 1.959964) == (
        pytest.approx(rejection_rate(0.05, np.sqrt(0.25 / 1_000 + 0.2475 / 1_000)), abs=1e-6))


def test_first_alarm_finds_a_clear_step():
    day = np.repeat(np.arange(1, 29), 50)
    x = np.where(day >= 20, 1.0, 0.0) + np.tile(np.array([0.0, 0.1]), len(day) // 2)
    assert first_alarm([(x, day)], 28) == 20
    assert first_alarm([(np.tile(np.array([0.0, 1.0]), len(day) // 2), day)], 28) is None


def test_monitors_separate_judge_drift_from_quality_shift():
    rng = np.random.default_rng(12)
    t, c = simulate_silent_experiment(28, 3_000, 0.6, 0.0, S_QUALITY, rng)
    labs = (label_mask(t, 20_000, rng), label_mask(c, 20_000, rng))
    assert first_alarm([score_series(t), score_series(c)], 28) is not None
    assert first_alarm(agreement_series(t, labs[0]) + agreement_series(c, labs[1]), 28) is None

    t, c = simulate_silent_experiment(28, 3_000, 0.6, 0.0, S_T_STEP, rng)
    labs = (label_mask(t, 20_000, rng), label_mask(c, 20_000, rng))
    assert first_alarm([score_series(c)], 28) is None
    assert first_alarm(agreement_series(t, labs[0]) + agreement_series(c, labs[1]), 28) is not None


def test_score_monitor_false_alarm_rate():
    rng = np.random.default_rng(13)
    alarms = []
    for _ in range(200):
        t, c = simulate_silent_experiment(28, 200, 0.6, 0.0, S_NONE, rng)
        alarms.append(first_alarm([score_series(t), score_series(c)], 28) is not None)
    assert np.mean(alarms) < 0.08


def test_silent_validation():
    with pytest.raises(ValueError):
        SilentDrift(OLD, OLD, OLD, 0)
    with pytest.raises(ValueError):
        simulate_silent_experiment(28, 10, 0.05, 0.0, S_QUALITY, np.random.default_rng(0))
    t, _ = simulate_silent_experiment(28, 10, 0.6, 0.0, S_NONE, np.random.default_rng(0))
    with pytest.raises(ValueError):
        label_mask(t, 71, np.random.default_rng(0), last_day=7)
    with pytest.raises(ValueError):
        first_alarm([score_series(t)], 13)


# -- canary set ----------------------------------------------------------------------


def test_canary_ignores_quality_shift_and_keeps_error_rates():
    a = simulate_canary(S_NONE, "t", 300, 0.6, 28, 0.5, np.random.default_rng(21))
    b = simulate_canary(S_QUALITY, "t", 300, 0.6, 28, 0.5, np.random.default_rng(21))
    assert np.array_equal(a, b)
    rng = np.random.default_rng(22)
    s = np.vstack([simulate_canary(S_T_STEP, "t", 2_000, 0.0, 28, 0.9, rng) for _ in range(5)])
    assert s[:, :14].mean() == pytest.approx(0.15, abs=0.01)
    assert s[:, 14:].mean() == pytest.approx(0.25, abs=0.01)
    c = simulate_canary(S_T_STEP, "c", 10_000, 0.0, 28, 0.9, rng)
    assert c[:, 14:].mean() == pytest.approx(0.15, abs=0.01)


def test_canary_repeatability():
    rng = np.random.default_rng(23)
    det = simulate_canary(S_NONE, "t", 500, 0.6, 28, 1.0, rng)
    assert (det == det[:, :1]).all()
    assert canary_alarm([det]) is None

    def day_corr(r):
        s = simulate_canary(S_NONE, "t", 20_000, 0.0, 2, r, rng)
        return np.corrcoef(s[:, 0], s[:, 1])[0, 1]

    assert abs(day_corr(0.0)) < 0.03
    assert day_corr(0.9) > day_corr(0.5) > 0.2


def test_canary_closed_form_matches_one_check():
    rng = np.random.default_rng(24)
    z_crit = monitor_z_crit(28, 2)
    hits = [abs(canary_z(simulate_canary(S_T_STEP, "t", 300, 0.6, 28, 0.0, rng))[1][-1])
            > z_crit for _ in range(600)]
    assert np.mean(hits) == pytest.approx(
        canary_detection_prob(OLD, LENIENT, 0.6, 300, z_crit), abs=0.05)


def test_canary_items_for_power_is_minimal():
    z_crit = monitor_z_crit(28, 2)
    k = canary_items_for_power(OLD, LENIENT, 0.6, z_crit)
    assert canary_detection_prob(OLD, LENIENT, 0.6, k, z_crit) >= 0.8
    assert canary_detection_prob(OLD, LENIENT, 0.6, k - 1, z_crit) < 0.8
    with pytest.raises(ValueError):
        canary_items_for_power(OLD, OLD, 0.6, z_crit)


def test_canary_alarm_rates():
    rng = np.random.default_rng(25)

    def rate(drift, arms, n=400):
        return np.mean([canary_alarm([simulate_canary(drift, a, n, 0.6, 28, 0.0, rng)
                                      for a in arms]) is not None for _ in range(150)])

    assert rate(S_NONE, "tc") < 0.08
    assert rate(S_QUALITY, "tc") < 0.08
    assert rate(S_T_STEP, "tc") > 0.8
    # A canary built only from control outputs can't see treatment-only drift.
    assert rate(S_T_STEP, "c", 800) < 0.08


def test_canary_validation():
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError):
        simulate_canary(S_NONE, "t", 10, 0.6, 28, 1.5, rng)
    with pytest.raises(ValueError):
        canary_z(np.zeros((10, 13)))
