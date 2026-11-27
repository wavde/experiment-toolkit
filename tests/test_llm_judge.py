import numpy as np
import pytest

from experiment_toolkit import (
    JudgeConfusion,
    analyze_judged_experiment,
    expected_naive_diff,
    judge_sample_size_inflation,
    naive_judge_diff,
    naive_judge_se,
    ppi_expected_se,
    ppi_mean_diff,
    simulate_judged_arm,
    simulate_judged_experiment,
    two_sided_rejection_rate,
)

P_CONTROL = 0.6
NOISY = JudgeConfusion(sensitivity=0.85, specificity=0.85)
# Judge that is more lenient toward treatment: more false positives in that arm.
LENIENT = JudgeConfusion(sensitivity=0.85, specificity=0.80)


def _run(n_reps, n_per_arm, effect, judge_t, n_labeled, seed):
    rng = np.random.default_rng(seed)
    out = {"naive": [], "human_only": [], "ppi": []}
    for _ in range(n_reps):
        t, c = simulate_judged_experiment(
            n_per_arm, P_CONTROL, effect, NOISY, judge_t, n_labeled, rng
        )
        for name, res in analyze_judged_experiment(t, c).items():
            covered = res.ci_low <= effect <= res.ci_high
            out[name].append((res.estimate, res.p_value < 0.05, covered))
    return {k: np.array(v) for k, v in out.items()}


# -- closed-form known answers -------------------------------------------------

def test_naive_expectation_is_youden_attenuation_when_nondifferential():
    effect = 0.05
    expected = expected_naive_diff(P_CONTROL, effect, NOISY)
    assert expected == pytest.approx(NOISY.youden_j * effect)


def test_naive_expectation_nonzero_under_h0_with_differential_error():
    # q_t - q_c = (1 - spec_t)(1 - p) - (1 - spec_c)(1 - p) = 0.05 * 0.4
    assert expected_naive_diff(P_CONTROL, 0.0, NOISY, LENIENT) == pytest.approx(0.02)


def test_sample_size_inflation_is_one_for_perfect_judge():
    perfect = JudgeConfusion(1.0, 1.0)
    assert judge_sample_size_inflation(P_CONTROL, 0.03, perfect) == pytest.approx(1.0)


def test_sample_size_inflation_approx_inverse_j_squared():
    # At p=0.5, var(S) = var(Y) = 0.25 so the ratio is exactly 1/J^2 as effect -> 0
    infl = judge_sample_size_inflation(0.5, 1e-6, NOISY)
    assert infl == pytest.approx(1 / NOISY.youden_j**2, rel=1e-4)


def test_ppi_oracle_se_matches_rho_formula():
    n, n_lab = 5000, 500
    q = NOISY.positive_rate(P_CONTROL)
    cov = P_CONTROL * NOISY.sensitivity - P_CONTROL * q
    rho2 = cov**2 / (P_CONTROL * (1 - P_CONTROL) * q * (1 - q))
    n_unlab = n - n_lab
    per_arm = P_CONTROL * (1 - P_CONTROL) / n_lab * (1 - rho2 * n_unlab / (n_unlab + n_lab))
    expected = np.sqrt(2 * per_arm)
    assert ppi_expected_se(n, n_lab, P_CONTROL, 0.0, NOISY) == pytest.approx(expected)


def test_ppi_se_with_lam_zero_is_human_only():
    se = ppi_expected_se(5000, 500, P_CONTROL, 0.0, NOISY, lam=0.0)
    assert se == pytest.approx(np.sqrt(2 * P_CONTROL * (1 - P_CONTROL) / 500))


def test_rejection_rate_at_zero_mean_is_alpha():
    assert two_sided_rejection_rate(0.0, 1.0, alpha=0.05) == pytest.approx(0.05)


# -- simulation vs. known answers ----------------------------------------------

def test_simulated_judge_rates_match_confusion_matrix():
    arm = simulate_judged_arm(200_000, P_CONTROL, NOISY, seed=0)
    assert arm.y.mean() == pytest.approx(P_CONTROL, abs=0.005)
    assert arm.s[arm.y == 1].mean() == pytest.approx(NOISY.sensitivity, abs=0.005)
    assert 1 - arm.s[arm.y == 0].mean() == pytest.approx(NOISY.specificity, abs=0.005)


def test_nondifferential_naive_keeps_type1_but_attenuates():
    res = _run(n_reps=400, n_per_arm=2000, effect=0.0, judge_t=None, n_labeled=200, seed=1)
    assert res["naive"][:, 1].mean() < 0.075

    res = _run(n_reps=400, n_per_arm=2000, effect=0.05, judge_t=None, n_labeled=200, seed=2)
    mc_se = naive_judge_se(2000, P_CONTROL, 0.05, NOISY) / np.sqrt(400)
    assert res["naive"][:, 0].mean() == pytest.approx(NOISY.youden_j * 0.05, abs=4 * mc_se)


def test_naive_power_matches_analytic():
    n, effect = 2000, 0.05
    res = _run(n_reps=400, n_per_arm=n, effect=effect, judge_t=None, n_labeled=200, seed=3)
    analytic = two_sided_rejection_rate(
        expected_naive_diff(P_CONTROL, effect, NOISY), naive_judge_se(n, P_CONTROL, effect, NOISY)
    )
    mc_se = np.sqrt(analytic * (1 - analytic) / 400)
    assert res["naive"][:, 1].mean() == pytest.approx(analytic, abs=4 * mc_se)


def test_differential_error_breaks_naive_but_not_ppi():
    """Under H0 with a judge that is more lenient toward treatment, the naive
    test rejects far more often than alpha. PPI++ with per-arm human labels
    keeps Type-I at alpha and its CI covers the true (zero) effect."""
    res = _run(n_reps=400, n_per_arm=5000, effect=0.0, judge_t=LENIENT, n_labeled=300, seed=4)
    assert res["naive"][:, 1].mean() > 0.4
    # Same Monte-Carlo slack as the mSPRT Type-I test
    assert res["ppi"][:, 1].mean() < 0.075
    assert res["human_only"][:, 1].mean() < 0.075
    assert res["ppi"][:, 2].mean() > 0.92


def test_ppi_unbiased_under_differential_error_with_effect():
    effect = 0.04
    res = _run(n_reps=400, n_per_arm=3000, effect=effect, judge_t=LENIENT, n_labeled=300, seed=5)
    mc_se = ppi_expected_se(3000, 300, P_CONTROL, effect, NOISY, LENIENT) / np.sqrt(400)
    assert res["ppi"][:, 0].mean() == pytest.approx(effect, abs=4 * mc_se)
    # Naive converges to the closed-form q_t - q_c, not to the true effect
    naive_expected = expected_naive_diff(P_CONTROL, effect, NOISY, LENIENT)
    naive_mc_se = naive_judge_se(3000, P_CONTROL, effect, NOISY, LENIENT) / np.sqrt(400)
    assert res["naive"][:, 0].mean() == pytest.approx(naive_expected, abs=4 * naive_mc_se)


def test_ppi_tighter_than_human_only_when_judge_informative():
    res = _run(n_reps=400, n_per_arm=5000, effect=0.0, judge_t=None, n_labeled=500, seed=6)
    sd_ppi = res["ppi"][:, 0].std()
    sd_human = res["human_only"][:, 0].std()
    assert sd_ppi < 0.85 * sd_human
    analytic = ppi_expected_se(5000, 500, P_CONTROL, 0.0, NOISY)
    assert sd_ppi == pytest.approx(analytic, rel=0.15)


def test_shared_calibration_cancels_and_stays_biased():
    """Using one set of labels (here: control's) to correct both arms adds the
    same rectifier to each arm, so the difference is just the naive estimate
    on the unlabeled units, and differential bias remains."""
    t, c = simulate_judged_experiment(
        5000, P_CONTROL, 0.0, NOISY, LENIENT, n_labeled_per_arm=500, seed=7
    )
    y_lab_c, s_lab_c = c.y[c.labeled], c.s[c.labeled]
    shared = ppi_mean_diff(
        y_lab_c, s_lab_c, t.s[~t.labeled], y_lab_c, s_lab_c, c.s[~c.labeled], lam=1.0
    )
    naive_unlab = t.s[~t.labeled].mean() - c.s[~c.labeled].mean()
    assert shared.estimate == pytest.approx(naive_unlab)


# -- estimator mechanics -------------------------------------------------------

def test_ppi_lam_zero_equals_human_only_difference():
    t, c = simulate_judged_experiment(1000, P_CONTROL, 0.05, NOISY, n_labeled_per_arm=200, seed=8)
    res = ppi_mean_diff(
        t.y[t.labeled], t.s[t.labeled], t.s[~t.labeled],
        c.y[c.labeled], c.s[c.labeled], c.s[~c.labeled],
        lam=0.0,
    )
    assert res.estimate == pytest.approx(t.y[t.labeled].mean() - c.y[c.labeled].mean())
    assert res.lambdas == (0.0, 0.0)


def test_ppi_with_perfect_judge_recovers_full_sample_precision():
    """With S == Y, the oracle PPI++ SE equals the SE of labeling every unit,
    and the estimate sits close to the full-sample truth (the gap is only the
    noise in the estimated lambda)."""
    perfect = JudgeConfusion(1.0, 1.0)
    n, n_lab, effect = 2000, 200, 0.05
    full_se = np.sqrt(
        (0.65 * 0.35 + P_CONTROL * (1 - P_CONTROL)) / n
    )
    assert ppi_expected_se(n, n_lab, P_CONTROL, effect, perfect) == pytest.approx(full_se)

    t, c = simulate_judged_experiment(
        n, P_CONTROL, effect, perfect, n_labeled_per_arm=n_lab, seed=9
    )
    res = analyze_judged_experiment(t, c)
    assert res["naive"].estimate == pytest.approx(t.y.mean() - c.y.mean())
    assert res["ppi"].se == pytest.approx(res["naive"].se, rel=0.1)
    assert abs(res["ppi"].estimate - (t.y.mean() - c.y.mean())) < 0.5 * res["naive"].se


def test_naive_judge_diff_values():
    res = naive_judge_diff([1, 1, 0, 0], [1, 0, 0, 0])
    assert res.estimate == pytest.approx(0.25)
    assert res.ci_low < res.estimate < res.ci_high
    assert res.lambdas is None


# -- validation ----------------------------------------------------------------

def test_judge_confusion_validates():
    with pytest.raises(ValueError):
        JudgeConfusion(1.2, 0.9)
    with pytest.raises(ValueError):
        JudgeConfusion(0.9, -0.1)


def test_simulation_validates():
    with pytest.raises(ValueError):
        simulate_judged_arm(100, 0.5, NOISY, n_labeled=101)
    with pytest.raises(ValueError):
        simulate_judged_experiment(100, 0.9, 0.2, NOISY)


def test_ppi_validates():
    with pytest.raises(ValueError, match="labeled"):
        ppi_mean_diff([1], [1], [0, 1], [1, 0], [1, 0], [0, 1])
    with pytest.raises(ValueError, match="same length"):
        ppi_mean_diff([1, 0], [1], [0, 1], [1, 0], [1, 0], [0, 1])
    with pytest.raises(ValueError, match="finite"):
        naive_judge_diff([1, np.nan], [0, 1])


def test_inflation_validates():
    with pytest.raises(ValueError):
        judge_sample_size_inflation(P_CONTROL, 0.0, NOISY)
    with pytest.raises(ValueError, match="better than chance"):
        judge_sample_size_inflation(P_CONTROL, 0.05, JudgeConfusion(0.5, 0.5))
