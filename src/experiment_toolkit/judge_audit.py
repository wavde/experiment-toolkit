"""
Audit an LLM judge against human judgments on real data.

Loops 1, 5, 7, 9, 11 and 12 simulate the judge. This module checks the
assumption they share, that a judge's errors can depend on which arm produced
the output, on the public MT-Bench human judgments (Zheng et al., 2023,
CC BY 4.0). Expert humans and GPT-4 judged the same pairwise comparisons:
the same question, conversation turn and pair of models.

Each pair of models is treated as a side-by-side test, with the stronger model
as "treatment". Per item, ``y`` is the human verdict for the first model
(1 = it wins, 0.5 = tie, 0 = it loses; averaged when several experts judged
the item) and ``s`` is the judge's verdict on the same scale. The target is
the human win rate, the mean of ``y``.

Analyses
--------
- :func:`pair_items` lines up human and judge verdicts item by item.
- :func:`bias_with_ci`: judge win rate minus human win rate on the same items,
  with a bootstrap CI that resamples questions.
- :func:`directional_agreement`: how often the judge agrees with a strict
  human preference, separately when humans prefer each model.
- :func:`resample_check`: a semi-synthetic check with a known answer. Items
  are drawn with replacement from the real ``(y, s)`` pairs, so the true win
  rate is the mean of ``y`` exactly, and the judge-only estimate, human labels
  only and PPI++ are compared against it.
"""

from __future__ import annotations

import hashlib
import urllib.request
from collections import defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy import stats

from experiment_toolkit.llm_judge import JudgeConfusion, _ppi_arm

MT_BENCH_REVISION = "f7d2896d2cc5d80f8b55c2bbc722613555233c25"
MT_BENCH_URL = (f"https://huggingface.co/datasets/lmsys/mt_bench_human_judgments/resolve/"
                f"{MT_BENCH_REVISION}/")
MT_BENCH_FILES = {
    "human": ("data/human-00000-of-00001-25f4910818759289.parquet",
              "4877bc46a40929f4082c3c79593700fb897b1d6c7f4c473032694a01322f5769"),
    "gpt4_pair": ("data/gpt4_pair-00000-of-00001-c0b431264a82ddc0.parquet",
                  "57068327ffb5f8fddb15fedbbfe40b26fe862da2326331e5cf64a42320a00e71"),
}
MT_BENCH_COLUMNS = ("question_id", "model_a", "model_b", "winner", "judge", "turn")
# Ordered by the human win rates in Zheng et al. (2023); the earlier model in a
# pair is "first" (treatment).
MT_BENCH_MODELS = ("gpt-4", "claude-v1", "gpt-3.5-turbo", "vicuna-13b-v1.2", "alpaca-13b",
                   "llama-13b")
_Z = float(stats.norm.ppf(0.975))


def load_mt_bench(data_dir: str | Path) -> dict[str, list[dict]]:
    """Download (once) and read the pinned MT-Bench human and GPT-4 judgments.

    Files are fetched from the Hugging Face Hub at a fixed revision and checked
    against their SHA-256. Needs ``pyarrow`` (``pip install -e ".[data]"``).
    Returns ``{"human": rows, "gpt4_pair": rows}`` with the columns in
    :data:`MT_BENCH_COLUMNS`.
    """
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    paths = {}
    for split, (rel, sha) in MT_BENCH_FILES.items():
        path = data_dir / Path(rel).name
        if not path.exists() or _sha256(path) != sha:
            urllib.request.urlretrieve(MT_BENCH_URL + rel, path)
        if _sha256(path) != sha:
            raise ValueError(f"{path.name}: checksum mismatch")
        paths[split] = path

    import pyarrow.parquet as pq

    return {split: pq.read_table(path, columns=list(MT_BENCH_COLUMNS)).to_pylist()
            for split, path in paths.items()}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verdict(row: dict, first: str) -> float:
    """1 if ``first`` won the comparison, 0 if it lost, 0.5 for any tie."""
    winner = row["winner"]
    if winner.startswith("tie"):
        return 0.5
    if winner not in ("model_a", "model_b"):
        raise ValueError(f"unknown winner {winner!r}")
    return 1.0 if row[winner] == first else 0.0


@dataclass(frozen=True)
class PairedItems:
    """Items judged by both humans and the judge for one pair of models."""

    first: str
    second: str
    question: np.ndarray
    y: np.ndarray  # human verdict for ``first``, averaged over raters
    s: np.ndarray  # judge verdict for ``first``


def pair_items(human_rows: Iterable[dict], judge_rows: Iterable[dict],
               order: Sequence[str] = MT_BENCH_MODELS) -> dict[tuple[str, str], PairedItems]:
    """Line up human and judge verdicts by (pair, question, turn).

    Only items with at least one human verdict and exactly one judge verdict
    are kept. ``order`` decides which model in a pair is ``first``.
    """
    rank = {m: i for i, m in enumerate(order)}

    def key(row: dict) -> tuple:
        a, b = sorted((row["model_a"], row["model_b"]), key=rank.__getitem__)
        return a, b, int(row["question_id"]), int(row["turn"])

    judge: dict[tuple, list[float]] = defaultdict(list)
    for row in judge_rows:
        k = key(row)
        judge[k].append(verdict(row, k[0]))
    human: dict[tuple, list[float]] = defaultdict(list)
    for row in human_rows:
        k = key(row)
        human[k].append(verdict(row, k[0]))

    rows: dict[tuple[str, str], list[tuple[int, float, float]]] = defaultdict(list)
    for k in sorted(human):
        if len(judge.get(k, ())) == 1:
            rows[k[:2]].append((k[2], float(np.mean(human[k])), judge[k][0]))
    return {
        pair: PairedItems(pair[0], pair[1], np.array([r[0] for r in v]),
                          np.array([r[1] for r in v]), np.array([r[2] for r in v]))
        for pair, v in sorted(rows.items(), key=lambda kv: (rank[kv[0][0]], rank[kv[0][1]]))
    }


def bias_with_ci(items: PairedItems, n_boot: int, rng: np.random.Generator,
                 level: float = 0.95) -> tuple[float, float, float]:
    """Judge minus human win rate, with a percentile CI from resampling questions."""
    d = items.s - items.y
    qs, inv = np.unique(items.question, return_inverse=True)
    sums = np.bincount(inv, weights=d)
    counts = np.bincount(inv).astype(float)
    draws = rng.integers(len(qs), size=(n_boot, len(qs)))
    boot = sums[draws].sum(axis=1) / counts[draws].sum(axis=1)
    tail = 100 * (1 - level) / 2
    lo, hi = np.percentile(boot, [tail, 100 - tail])
    return float(d.mean()), float(lo), float(hi)


def directional_agreement(items: Iterable[PairedItems]) -> dict[str, float]:
    """Judge agreement with strict human preferences, split by which model humans preferred.

    Pools the given pairs. Returns the share of items where humans prefer the
    first model (``y > 0.5``) that the judge also gives to the first model, the
    same for the second model (``y < 0.5``), and the counts.
    """
    y = np.concatenate([i.y for i in items])
    s = np.concatenate([i.s for i in items]) if len(y) else np.array([])
    first, second = y > 0.5, y < 0.5
    return {
        "agree_first": float((s[first] == 1.0).mean()),
        "agree_second": float((s[second] == 0.0).mean()),
        "n_first": int(first.sum()),
        "n_second": int(second.sum()),
    }


@dataclass(frozen=True)
class Readout:
    estimate: float
    se: float

    def covers(self, truth: float) -> bool:
        return abs(self.estimate - truth) <= _Z * self.se

    def ships_first(self) -> bool:
        """The CI for the first model's win rate lies above 0.5."""
        return self.estimate - _Z * self.se > 0.5


RESAMPLE_ANALYSES = ("judge_only", "human_only", "ppi")


def resample_check(items: PairedItems, n_units: int, n_labeled: int, reps: int,
                   rng: np.random.Generator) -> dict[str, list[Readout]]:
    """Semi-synthetic check with a known answer.

    Each replication draws ``n_units`` items with replacement from the real
    ``(y, s)`` pairs (so the true win rate is ``items.y.mean()``), labels a
    random ``n_labeled`` of them with their human verdict, and reads out the
    judge-only win rate, the human labels alone, and PPI++.
    """
    if not 2 <= n_labeled < n_units:
        raise ValueError("need 2 <= n_labeled < n_units")
    out: dict[str, list[Readout]] = {a: [] for a in RESAMPLE_ANALYSES}
    for _ in range(reps):
        idx = rng.integers(len(items.y), size=n_units)
        y, s = items.y[idx], items.s[idx]
        lab = np.zeros(n_units, dtype=bool)
        lab[rng.choice(n_units, size=n_labeled, replace=False)] = True
        out["judge_only"].append(Readout(float(s.mean()), float(s.std(ddof=1) / np.sqrt(n_units))))
        out["human_only"].append(Readout(float(y[lab].mean()),
                                         float(y[lab].std(ddof=1) / np.sqrt(n_labeled))))
        est, var, _ = _ppi_arm(y[lab], s[lab], s[~lab], None, "audit")
        out["ppi"].append(Readout(est, float(np.sqrt(var))))
    return out


# -- closed forms ----------------------------------------------------------------------


def judge_human_correlation(judge: JudgeConfusion, p: float) -> float:
    """Correlation between a binary judge verdict and the true label in loop 1's model."""
    q = judge.positive_rate(p)
    cov = p * judge.sensitivity - p * q
    return float(cov / np.sqrt(p * (1 - p) * q * (1 - q)))


def ppi_variance_ratio(rho: float, n_labeled: int, n_units: int) -> float:
    """Large-sample variance of PPI++ relative to human labels alone, with one
    judge score correlated ``rho`` with the human verdict: ``1 - rho^2 N/(N+n)``
    for ``N = n_units - n_labeled`` unlabeled and ``n = n_labeled`` labeled units."""
    n_unlab = n_units - n_labeled
    return float(1 - rho**2 * n_unlab / (n_unlab + n_labeled))
