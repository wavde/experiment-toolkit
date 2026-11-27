import numpy as np
import pytest

from experiment_toolkit import (
    JudgeConfusion,
    OfflineOnlineScenario,
    Readout,
    expected_judged_diff,
    expected_online_effect,
    expected_quality_diff,
    offline_readouts,
    simulate_offline_eval,
    simulate_online_ab,
)

JUDGE = JudgeConfusion(0.85, 0.85)
LENIENT = JudgeConfusion(0.85, 0.80)
W_OFF = np.array([0.2, 0.6, 0.2])
W_ON = np.array([0.7, 0.2, 0.1])
P_A = np.array([0.6, 0.5, 0.4])


def _sc(delta, judge_b=JUDGE, online_cost=0.0):
    return OfflineOnlineScenario(W_OFF, W_ON, P_A, np.array(delta, float), JUDGE, judge_b,
                                 online_cost=online_cost)


MIX_SHIFT = _sc([-0.04, 0.10, 0.0])


def _mean_readouts(sc, reps, seed, n_prompts=10_000, n_labeled=1_000):
    rng = np.random.default_rng(seed)
    est = {k: [] for k in ("judge_raw", "judge_reweighted", "ppi_reweighted", "online")}
    dec = {k: [] for k in est}
    for _ in range(reps):
        reads = offline_readouts(simulate_offline_eval(sc, n_prompts, n_labeled, rng), sc.w_on)
        reads["online"] = simulate_online_ab(sc, 200_000, rng)
        for k, r in reads.items():
            est[k].append(r.estimate)
            dec[k].append(r.decision())
    return ({k: float(np.mean(v)) for k, v in est.items()},
            {k: np.array(v) for k, v in dec.items()})


# -- closed form -----------------------------------------------------------------

def test_expected_values_mix_shift():
    # sum(w_on * delta) = 0.7*-0.04 + 0.2*0.10 = -0.008
    assert expected_quality_diff(MIX_SHIFT, W_ON) == pytest.approx(-0.008)
    assert expected_quality_diff(MIX_SHIFT, W_OFF) == pytest.approx(0.052)
    # non-differential judge attenuates by Youden's J = 0.7
    assert expected_judged_diff(MIX_SHIFT, W_ON) == pytest.approx(0.7 * -0.008)
    assert expected_online_effect(MIX_SHIFT) == pytest.approx(0.8 * -0.008)


def test_online_cost_enters_online_effect_only():
    sc = _sc([0.03] * 3, online_cost=0.03)
    assert expected_online_effect(sc) == pytest.approx(0.8 * 0.03 - 0.03)
    assert expected_quality_diff(sc, W_ON) == pytest.approx(0.03)


def test_lenient_judge_shifts_judged_diff_not_quality():
    sc = _sc([0.0] * 3, judge_b=LENIENT)
    # extra false positives on B: 0.05 * (1 - p_a) per segment
    assert expected_judged_diff(sc, W_OFF) == pytest.approx(0.05 * (W_OFF @ (1 - P_A)))
    assert expected_quality_diff(sc, W_ON) == 0.0


# -- simulation vs closed form ---------------------------------------------------

def test_estimators_match_closed_form_under_mix_shift():
    est, dec = _mean_readouts(MIX_SHIFT, reps=60, seed=1)
    assert est["judge_raw"] == pytest.approx(expected_judged_diff(MIX_SHIFT, W_OFF), abs=0.002)
    assert est["judge_reweighted"] == pytest.approx(
        expected_judged_diff(MIX_SHIFT, W_ON), abs=0.002
    )
    assert est["ppi_reweighted"] == pytest.approx(
        expected_quality_diff(MIX_SHIFT, W_ON), abs=0.003
    )
    assert est["online"] == pytest.approx(expected_online_effect(MIX_SHIFT), abs=0.001)
    # Raw offline eval ships a change that hurts live traffic; reweighting doesn't.
    assert np.mean(dec["judge_raw"] == "ship") > 0.9
    assert np.mean(dec["judge_reweighted"] == "ship") < 0.1
    assert np.mean(dec["online"] == "stop") > 0.9


def test_ppi_removes_lenient_judge_bias():
    sc = _sc([0.0] * 3, judge_b=LENIENT)
    est, dec = _mean_readouts(sc, reps=60, seed=2)
    assert np.mean(dec["judge_reweighted"] == "ship") > 0.3
    assert np.mean(dec["ppi_reweighted"] == "ship") < 0.12
    assert abs(est["ppi_reweighted"]) < 0.005


def test_only_online_catches_online_cost():
    sc = _sc([0.03] * 3, online_cost=0.03)
    _, dec = _mean_readouts(sc, reps=40, seed=3)
    for k in ("judge_raw", "judge_reweighted", "ppi_reweighted"):
        assert np.mean(dec[k] == "stop") == 0.0
    assert np.mean(dec["online"] == "stop") > 0.85


def test_null_type1_is_controlled():
    _, dec = _mean_readouts(_sc([0.0] * 3), reps=100, seed=4, n_prompts=4_000, n_labeled=600)
    for k, d in dec.items():
        assert np.mean(d != "none") < 0.12, k


# -- mechanics and validation ----------------------------------------------------

def test_readout_decision():
    assert Readout(1.0, 0.1).decision() == "ship"
    assert Readout(-1.0, 0.1).decision() == "stop"
    assert Readout(0.1, 0.1).decision() == "none"


def test_offline_eval_draws_offline_mix():
    data = simulate_offline_eval(MIX_SHIFT, 100_000, 100, np.random.default_rng(5))
    shares = np.bincount(data.segment, minlength=3) / 100_000
    assert shares == pytest.approx(W_OFF, abs=0.01)
    assert data.labeled.sum() == 100


def test_scenario_validation():
    with pytest.raises(ValueError, match="sum to 1"):
        OfflineOnlineScenario([0.5, 0.6], [0.5, 0.5], [0.5, 0.5], [0, 0], JUDGE, JUDGE)
    with pytest.raises(ValueError, match="same length"):
        OfflineOnlineScenario([1.0], [0.5, 0.5], [0.5, 0.5], [0, 0], JUDGE, JUDGE)
    with pytest.raises(ValueError, match="p_a \\+ delta"):
        OfflineOnlineScenario([1.0], [1.0], [0.95], [0.1], JUDGE, JUDGE)


def test_missing_segment_raises():
    sc = OfflineOnlineScenario([1.0, 0.0], [0.5, 0.5], [0.5, 0.5], [0.0, 0.0], JUDGE, JUDGE)
    data = simulate_offline_eval(sc, 500, 100, np.random.default_rng(6))
    with pytest.raises(ValueError, match="segment 1"):
        offline_readouts(data, sc.w_on)
