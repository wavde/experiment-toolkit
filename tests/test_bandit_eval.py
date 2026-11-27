import numpy as np
import pytest

from experiment_toolkit.bandit_eval import (
    expected_max_std_normal,
    expected_winner_bias_fixed,
    naive_lift,
    ope_lift,
    pick_winner,
    policy_terms,
    simulate_bandit_log,
)

P0 = 0.1
LIFTS = np.array([0.0, 0.2, -0.1, 0.05])  # arm 2 (lift +20%) is the best variant


def _log(rng, lifts=LIFTS, **kw):
    kw = {"control_share": 0.2, "floor": 0.2, "drift": 0.0, **kw}
    return simulate_bandit_log(lifts, P0, 2_000, 8, rng, **kw)


def _dr(log, rows, arm):
    return ope_lift(log.arms[rows], log.rewards[rows], log.propensities[rows], arm,
                    method="dr", baseline_target=log.baseline(arm)[rows],
                    baseline_control=log.baseline(0)[rows])


# -- estimators on small hand-checked logs ---------------------------------------

def test_ips_and_snips_by_hand():
    arms = np.array([1, 0, 1, 0])
    y = np.array([1.0, 1.0, 0.0, 0.0])
    p = np.array([0.5, 0.5, 0.25, 0.25])
    assert policy_terms(arms, y, p, 1).tolist() == [2.0, 0.0, 0.0, 0.0]
    snips = policy_terms(arms, y, p, 1, method="snips")
    assert snips.mean() == pytest.approx(2.0 / (2.0 + 4.0))
    lift = ope_lift(arms, y, p, target=1, control=0)
    assert lift.estimate == pytest.approx(0.5 - 0.5)


def test_dr_with_good_baseline_beats_ips():
    rng = np.random.default_rng(0)
    n = 40_000
    arms = rng.integers(0, 4, n)
    y = (rng.random(n) < 0.5).astype(float)
    p = np.full(n, 0.25)
    ips = policy_terms(arms, y, p, 1)
    dr = policy_terms(arms, y, p, 1, method="dr", baseline=np.full(n, 0.5))
    assert dr.mean() == pytest.approx(0.5, abs=0.02)
    # Theory: IPS p/pi - p^2 = 1.75; DR p(1-p)/pi = 1.0.
    assert dr.var() == pytest.approx(1.0, rel=0.05)
    assert ips.var() == pytest.approx(1.75, rel=0.05)


def test_naive_lift_welch():
    arms = np.array([0] * 4 + [1] * 4)
    y = np.array([0, 0, 1, 1, 1, 1, 1, 0], dtype=float)
    r = naive_lift(arms, y, target=1)
    assert r.estimate == pytest.approx(0.25)
    assert r.se == pytest.approx(np.sqrt(1 / 3 / 4 + 0.25 / 4))


# -- the two biases ----------------------------------------------------------------

def test_ips_fixes_time_confounding_that_naive_misses():
    """Fixed target chosen in advance; click rate rises over the test."""
    rng = np.random.default_rng(1)
    naive, ips, snips, dr = [], [], [], []
    every = np.ones(16_000, dtype=bool)
    for _ in range(150):
        log = _log(rng, drift=0.3)
        a, y, p = log.arms, log.rewards, log.propensities
        naive.append(naive_lift(a, y, 2).estimate)
        ips.append(ope_lift(a, y, p, 2).estimate)
        snips.append(ope_lift(a, y, p, 2, method="snips").estimate)
        dr.append(_dr(log, every, 2).estimate)
    truth = P0 * LIFTS[1]
    for est in (ips, snips, dr):
        mc_se = np.std(est, ddof=1) / np.sqrt(len(est))
        assert abs(np.mean(est) - truth) < 3.5 * mc_se
    # The best arm wins traffic late, when clicks are more likely.
    assert np.mean(naive) - truth > 0.004


def test_winners_curse_survives_ips_but_not_an_evaluation_slice():
    rng = np.random.default_rng(2)
    same, held = [], []
    lifts = np.zeros(10)
    for _ in range(150):
        log = _log(rng, lifts=lifts, eval_share=0.3)
        lead = log.leader()
        learn, ev = ~log.eval_slice, log.eval_slice
        same.append(ope_lift(log.arms[learn], log.rewards[learn], log.propensities[learn],
                             lead).estimate)
        held.append(_dr(log, ev, lead).estimate)
    assert np.mean(same) > 4 * np.std(same, ddof=1) / np.sqrt(len(same))
    assert abs(np.mean(held)) < 3.5 * np.std(held, ddof=1) / np.sqrt(len(held))


def test_pick_winner_uses_weighted_means():
    arms = np.array([1, 1, 2, 2])
    y = np.array([1.0, 0.0, 1.0, 1.0])
    p = np.array([0.5, 0.1, 0.5, 0.5])
    assert pick_winner(arms, y, p, [1, 2]) == 2
    with pytest.raises(ValueError):
        pick_winner(arms, y, p, [3])


# -- simulator -------------------------------------------------------------------

def test_simulator_logs_exact_propensities_and_hides_the_slice():
    rng = np.random.default_rng(3)
    log = _log(rng, eval_share=0.25, prior=(2.0, 18.0))
    assert np.allclose(log.allocation.sum(axis=1), 1.0)
    assert np.all(log.allocation[:, 0] == 0.2)
    assert log.allocation[:, 1:].min() >= 0.8 * 0.2 / 4 - 1e-12
    assert np.array_equal(log.propensities, log.allocation[log.batch, log.arms])
    assert log.eval_slice.mean() == pytest.approx(0.25, abs=0.02)
    learn = ~log.eval_slice
    clicks = np.bincount(log.arms[learn], weights=log.rewards[learn], minlength=5)
    shows = np.bincount(log.arms[learn], minlength=5)
    assert np.allclose(log.final_posterior_mean, (2 + clicks) / (20 + shows))
    assert np.allclose(log.multiplier, 1.0)


def test_drift_multiplier_averages_one():
    log = _log(np.random.default_rng(4), drift=0.2)
    assert log.multiplier[0] == pytest.approx(0.8)
    assert log.multiplier[-1] == pytest.approx(1.2)
    assert log.multiplier.mean() == pytest.approx(1.0)


# -- closed form -----------------------------------------------------------------

def test_expected_max_std_normal():
    assert expected_max_std_normal(1) == 0.0
    assert expected_max_std_normal(2) == pytest.approx(1 / np.sqrt(np.pi))
    assert expected_max_std_normal(3) == pytest.approx(1.5 / np.sqrt(np.pi))
    assert expected_winner_bias_fixed(2, 10_000, 0.05) == pytest.approx(
        np.sqrt(0.05 * 0.95 / 10_000) / np.sqrt(np.pi))


def test_validation():
    a, y = np.array([0, 1, 1]), np.array([1.0, 0.0, 1.0])
    with pytest.raises(ValueError):
        policy_terms(a, y, np.array([0.5, 0.0, 0.5]), 1)
    with pytest.raises(ValueError):
        policy_terms(a, y, np.array([0.5, 0.5, 0.5]), 1, method="dr")
    with pytest.raises(ValueError):
        policy_terms(a, y, np.array([0.5, 0.5, 0.5]), 1, method="nope")
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError):
        simulate_bandit_log([0.0], 0.9, 100, 4, rng, drift=0.5)
    with pytest.raises(ValueError):
        simulate_bandit_log([0.0], 0.1, 100, 4, rng, eval_share=1.0)
    with pytest.raises(ValueError):
        expected_max_std_normal(0)
