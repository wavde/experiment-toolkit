"""
Callaway & Sant'Anna (2021) staggered difference-in-differences.

Robust to treatment-effect heterogeneity across adoption cohorts — unlike
two-way fixed-effects DiD, which is biased when effects differ across
cohorts or over time (Goodman-Bacon 2021; de Chaisemartin & D'Haultfoeuille
2020). Implementation uses a never-treated control group and the
outcome-regression (unconditional) identifier:

    ATT(g, t) = mean[Y_t - Y_{g-1} | cohort = g]
              - mean[Y_t - Y_{g-1} | never treated]

for each treated cohort ``g`` and post-treatment period ``t >= g``.

Standard errors come from a unit cluster bootstrap, so the implementation
is pure numpy (no statsmodels / pandas dependency for the package).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import ArrayLike
from scipy import stats


@dataclass
class CSResult:
    """Structured Callaway-Sant'Anna estimation result.

    Attributes
    ----------
    group_time_att : list of dicts (one per (g, t) cell) with keys
        ``cohort``, ``period``, ``relative_time``, ``att``, ``se``,
        ``ci_low``, ``ci_high``, ``n_treated``, ``n_control``.
    event_study : list of dicts (one per relative time ``e``) with keys
        ``relative_time``, ``att``, ``se``, ``ci_low``, ``ci_high``.
    overall_att, overall_se, overall_ci_low, overall_ci_high : simple
        equal-weighted aggregation across all (g, t) with ``t >= g``.
    """

    group_time_att: list[dict[str, Any]] = field(default_factory=list)
    event_study: list[dict[str, Any]] = field(default_factory=list)
    overall_att: float = float("nan")
    overall_se: float = float("nan")
    overall_ci_low: float = float("nan")
    overall_ci_high: float = float("nan")
    n_units: int = 0
    n_bootstrap: int = 0

    def __repr__(self) -> str:  # pragma: no cover - convenience
        return (
            f"CS(overall ATT={self.overall_att:+.3f}, "
            f"95% CI [{self.overall_ci_low:+.3f}, {self.overall_ci_high:+.3f}], "
            f"n_units={self.n_units}, boot={self.n_bootstrap})"
        )


def simulate_staggered_panel(
    cohort_sizes: dict[int, int] | None = None,
    n_never_treated: int = 20,
    n_periods: int = 20,
    cohort_effects: dict[int, float] | None = None,
    dynamic_slope: float = 0.0,
    seed: int = 0,
) -> dict[str, np.ndarray]:
    """Simulate a staggered-adoption panel.

    Returns a dict of numpy arrays (all length ``n_units * n_periods``):
    ``unit`` (int), ``period`` (int), ``y`` (float), ``cohort`` (int — the
    treatment period, ``0`` for never-treated).

    Default DGP has three cohorts treated at periods 6/10/14 with static
    effects 1/3/5 — a textbook setting where TWFE is biased but CS is not.
    """
    if cohort_sizes is None:
        cohort_sizes = {6: 15, 10: 15, 14: 15}
    if cohort_effects is None:
        cohort_effects = {6: 1.0, 10: 3.0, 14: 5.0}

    rng = np.random.default_rng(seed)
    common_shocks = rng.normal(0, 1.0, size=n_periods).cumsum() * 0.3
    common_trend = 0.4

    units: list[int] = []
    periods: list[int] = []
    ys: list[float] = []
    cohorts: list[int] = []

    unit_id = 0

    def add_unit(cohort: int, effect: float) -> None:
        nonlocal unit_id
        intercept = rng.normal(100, 8)
        noise = rng.normal(0, 1.0, size=n_periods)
        for p in range(n_periods):
            rel_time = p - cohort if cohort > 0 else -9999
            is_post = cohort > 0 and p >= cohort
            te = effect + dynamic_slope * rel_time if is_post else 0.0
            y = intercept + common_trend * p + common_shocks[p] + noise[p] + te
            units.append(unit_id)
            periods.append(p)
            ys.append(float(y))
            cohorts.append(cohort)
        unit_id += 1

    for g, n in cohort_sizes.items():
        for _ in range(n):
            add_unit(g, cohort_effects[g])
    for _ in range(n_never_treated):
        add_unit(0, 0.0)

    return {
        "unit": np.asarray(units, dtype=int),
        "period": np.asarray(periods, dtype=int),
        "y": np.asarray(ys, dtype=float),
        "cohort": np.asarray(cohorts, dtype=int),
    }


def _to_wide(
    unit: np.ndarray, period: np.ndarray, y: np.ndarray, cohort: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Reshape a balanced long panel into wide form. Returns (unit_ids,
    period_grid, y_matrix [n_units x n_periods], cohort_per_unit)."""
    unit_ids, unit_idx = np.unique(unit, return_inverse=True)
    period_grid, period_idx = np.unique(period, return_inverse=True)
    n_u, n_p = len(unit_ids), len(period_grid)
    y_mat = np.full((n_u, n_p), np.nan, dtype=float)
    y_mat[unit_idx, period_idx] = y
    if np.isnan(y_mat).any():
        raise ValueError("panel is unbalanced; every (unit, period) must be present")
    cohort_per_unit = np.zeros(n_u, dtype=int)
    # cohort is constant within unit — take the first occurrence
    seen = np.zeros(n_u, dtype=bool)
    for u_idx, c in zip(unit_idx, cohort, strict=False):
        if not seen[u_idx]:
            cohort_per_unit[u_idx] = c
            seen[u_idx] = True
    return unit_ids, period_grid, y_mat, cohort_per_unit


def _cs_point_estimates(
    y_mat: np.ndarray,
    cohort_per_unit: np.ndarray,
    period_grid: np.ndarray,
    cohorts: list[int],
) -> dict[tuple[int, int], float]:
    """ATT(g, t) point estimates via the 2x2 CS identifier."""
    period_to_col = {int(p): i for i, p in enumerate(period_grid.tolist())}
    never_mask = cohort_per_unit == 0
    out: dict[tuple[int, int], float] = {}
    for g in cohorts:
        baseline = g - 1
        if baseline not in period_to_col:
            continue
        treated_mask = cohort_per_unit == g
        if not treated_mask.any() or not never_mask.any():
            continue
        b_col = period_to_col[baseline]
        for p in period_grid:
            p_int = int(p)
            if p_int < g or p_int not in period_to_col:
                continue
            t_col = period_to_col[p_int]
            diff_treated = float(
                (y_mat[treated_mask, t_col] - y_mat[treated_mask, b_col]).mean()
            )
            diff_control = float(
                (y_mat[never_mask, t_col] - y_mat[never_mask, b_col]).mean()
            )
            out[(g, p_int)] = diff_treated - diff_control
    return out


def cs_staggered_att(
    unit: ArrayLike,
    period: ArrayLike,
    y: ArrayLike,
    cohort: ArrayLike,
    n_bootstrap: int = 500,
    alpha: float = 0.05,
    seed: int = 0,
) -> CSResult:
    """Callaway-Sant'Anna staggered DiD with never-treated controls.

    Parameters
    ----------
    unit, period, y, cohort : equal-length arrays forming a long panel.
        ``cohort`` must be the treatment period for treated units and
        ``0`` for never-treated units. Panel must be balanced.
    n_bootstrap : number of unit-cluster bootstrap replicates.
    alpha : two-sided confidence level (0.05 = 95% CI).
    seed : RNG seed for the bootstrap.
    """
    unit_arr = np.asarray(unit)
    period_arr = np.asarray(period, dtype=int)
    y_arr = np.asarray(y, dtype=float)
    cohort_arr = np.asarray(cohort, dtype=int)

    unit_ids, period_grid, y_mat, cohort_per_unit = _to_wide(
        unit_arr, period_arr, y_arr, cohort_arr
    )
    n_units = len(unit_ids)
    cohorts = sorted({int(c) for c in cohort_per_unit.tolist() if c > 0})

    point_gt = _cs_point_estimates(y_mat, cohort_per_unit, period_grid, cohorts)
    if not point_gt:
        raise ValueError("No identifiable (g, t) cells — check cohort/period coding.")

    rng = np.random.default_rng(seed)
    boot_gt: dict[tuple[int, int], list[float]] = {k: [] for k in point_gt}
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n_units, size=n_units)
        y_boot = y_mat[idx]
        c_boot = cohort_per_unit[idx]
        b = _cs_point_estimates(y_boot, c_boot, period_grid, cohorts)
        for k in boot_gt:
            if k in b:
                boot_gt[k].append(b[k])

    z = float(stats.norm.ppf(1 - alpha / 2))

    gt_rows: list[dict[str, Any]] = []
    for (g, t), point in point_gt.items():
        boots = np.asarray(boot_gt[(g, t)])
        se = float(boots.std(ddof=1)) if len(boots) > 1 else float("nan")
        gt_rows.append(
            {
                "cohort": g,
                "period": t,
                "relative_time": t - g,
                "att": point,
                "se": se,
                "ci_low": point - z * se,
                "ci_high": point + z * se,
                "n_treated": int((cohort_per_unit == g).sum()),
                "n_control": int((cohort_per_unit == 0).sum()),
            }
        )
    gt_rows.sort(key=lambda r: (r["cohort"], r["period"]))

    # Event-study aggregation
    es_points: dict[int, list[float]] = {}
    es_boots: dict[int, list[list[float]]] = {}
    for (g, t), point in point_gt.items():
        es_points.setdefault(t - g, []).append(point)
        es_boots.setdefault(t - g, []).append(boot_gt[(g, t)])

    es_rows: list[dict[str, Any]] = []
    for e in sorted(es_points):
        att_e = float(np.mean(es_points[e]))
        valid = [np.asarray(b) for b in es_boots[e] if len(b) > 1]
        if valid:
            min_len = min(len(v) for v in valid)
            stacked = np.vstack([v[:min_len] for v in valid])
            replicate_means = stacked.mean(axis=0)
            se_e = float(replicate_means.std(ddof=1))
        else:
            se_e = float("nan")
        es_rows.append(
            {
                "relative_time": e,
                "att": att_e,
                "se": se_e,
                "ci_low": att_e - z * se_e,
                "ci_high": att_e + z * se_e,
            }
        )

    overall_point = float(np.mean(list(point_gt.values())))
    valid_keys = [k for k in point_gt if len(boot_gt[k]) > 1]
    if valid_keys:
        min_len = min(len(boot_gt[k]) for k in valid_keys)
        boot_mat = np.vstack([np.asarray(boot_gt[k][:min_len]) for k in valid_keys])
        overall_replicates = boot_mat.mean(axis=0)
        overall_se = float(overall_replicates.std(ddof=1))
    else:
        overall_se = float("nan")

    return CSResult(
        group_time_att=gt_rows,
        event_study=es_rows,
        overall_att=overall_point,
        overall_se=overall_se,
        overall_ci_low=overall_point - z * overall_se,
        overall_ci_high=overall_point + z * overall_se,
        n_units=n_units,
        n_bootstrap=n_bootstrap,
    )
