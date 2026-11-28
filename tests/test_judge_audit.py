import numpy as np
import pytest

from experiment_toolkit import judge_audit
from experiment_toolkit.judge_audit import (
    PairedItems,
    bias_with_ci,
    directional_agreement,
    judge_human_correlation,
    pair_items,
    ppi_variance_ratio,
    resample_check,
    verdict,
)
from experiment_toolkit.llm_judge import JudgeConfusion


def _row(q, a, b, winner, judge="author_1", turn=1):
    return {"question_id": q, "model_a": a, "model_b": b, "winner": winner, "judge": judge,
            "turn": turn}


def test_verdict_orientation_and_ties():
    r = _row(81, "alpaca-13b", "gpt-4", "model_b")
    assert verdict(r, "gpt-4") == 1.0
    assert verdict(r, "alpaca-13b") == 0.0
    assert verdict(_row(81, "a", "b", "tie (inconsistent)"), "a") == 0.5
    with pytest.raises(ValueError):
        verdict(_row(81, "a", "b", "both"), "a")


def test_pair_items_orients_by_strength_and_averages_raters():
    human = [
        _row(81, "alpaca-13b", "gpt-4", "model_b", "author_1"),
        _row(81, "gpt-4", "alpaca-13b", "tie", "author_2"),
        _row(82, "gpt-4", "alpaca-13b", "model_b"),
        _row(83, "gpt-4", "alpaca-13b", "model_a"),  # no judge verdict: dropped
    ]
    judge = [
        _row(81, "alpaca-13b", "gpt-4", "model_b", "gpt4_pair"),
        _row(82, "alpaca-13b", "gpt-4", "tie", "gpt4_pair"),
    ]
    pairs = pair_items(human, judge)
    it = pairs[("gpt-4", "alpaca-13b")]
    assert (it.first, it.second) == ("gpt-4", "alpaca-13b")
    np.testing.assert_array_equal(it.question, [81, 82])
    np.testing.assert_allclose(it.y, [0.75, 0.0])
    np.testing.assert_allclose(it.s, [1.0, 0.5])


def _items(y, s, q=None):
    y, s = np.asarray(y, float), np.asarray(s, float)
    return PairedItems("a", "b", np.arange(len(y)) if q is None else np.asarray(q), y, s)


def test_bias_with_ci_constant_gap():
    it = _items([0.0, 0.5, 1.0, 0.0], [0.5, 1.0, 1.0, 0.5], q=[1, 1, 2, 3])
    bias, lo, hi = bias_with_ci(it, 200, np.random.default_rng(0))
    assert bias == pytest.approx(0.375)
    assert lo <= bias <= hi
    flat = _items([0.0, 0.5, 0.5], [0.2, 0.7, 0.7])
    assert bias_with_ci(flat, 100, np.random.default_rng(0)) == pytest.approx((0.2, 0.2, 0.2))


def test_directional_agreement():
    a = _items([1, 1, 1, 0, 0, 0.5], [1, 0.5, 1, 1, 0, 0])
    b = _items([0.75], [1])
    out = directional_agreement([a, b])
    assert out == {"agree_first": 0.75, "agree_second": 0.5, "n_first": 4, "n_second": 2}


def test_judge_human_correlation_known_values():
    assert judge_human_correlation(JudgeConfusion(1.0, 1.0), 0.6) == pytest.approx(1.0)
    assert judge_human_correlation(JudgeConfusion(0.7, 0.3), 0.6) == pytest.approx(0.0)
    rng = np.random.default_rng(1)
    y = rng.random(400_000) < 0.6
    s = np.where(y, rng.random(len(y)) < 0.85, rng.random(len(y)) < 0.15)
    assert judge_human_correlation(JudgeConfusion(0.85, 0.85), 0.6) == pytest.approx(
        np.corrcoef(y, s)[0, 1], abs=0.005)


def test_resample_check_known_answer_and_variance_ratio():
    rng = np.random.default_rng(2)
    y = (rng.random(300) < 0.6).astype(float)
    s = np.where(rng.random(300) < 0.6, y, 1.0)  # judge leans to "first wins"
    it = _items(y, s)
    rho = np.corrcoef(y, s)[0, 1]
    r = resample_check(it, 2_000, 200, 400, rng)
    truth = y.mean()
    assert np.mean([x.covers(truth) for x in r["judge_only"]]) < 0.05
    for a in ("human_only", "ppi"):
        assert np.mean([x.covers(truth) for x in r[a]]) > 0.9, a
        assert np.mean([x.estimate for x in r[a]]) == pytest.approx(truth, abs=0.01), a
    ratio = np.mean([x.se for x in r["ppi"]]) / np.mean([x.se for x in r["human_only"]])
    assert ratio == pytest.approx(np.sqrt(ppi_variance_ratio(rho, 200, 2_000)), abs=0.02)
    assert ppi_variance_ratio(0.0, 200, 2_000) == 1.0
    with pytest.raises(ValueError):
        resample_check(it, 100, 100, 1, rng)


def test_load_rejects_bad_checksum(tmp_path, monkeypatch):
    monkeypatch.setattr(judge_audit.urllib.request, "urlretrieve",
                        lambda url, path: path.write_bytes(b"not the dataset"))
    with pytest.raises(ValueError, match="checksum"):
        judge_audit.load_mt_bench(tmp_path)
