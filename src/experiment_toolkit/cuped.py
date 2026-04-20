"""CUPED variance reduction — see Deng et al. (2013)."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike


def compute_theta(y: ArrayLike, x: ArrayLike) -> float:
    """Optimal CUPED coefficient theta = Cov(Y, X) / Var(X)."""
    y_arr = np.asarray(y, dtype=float)
    x_arr = np.asarray(x, dtype=float)
    var_x = np.var(x_arr, ddof=1)
    if var_x == 0:
        return 0.0
    return float(np.cov(y_arr, x_arr, ddof=1)[0, 1] / var_x)


def apply_cuped(
    y: ArrayLike,
    x: ArrayLike,
    theta: float | None = None,
) -> np.ndarray:
    """Return CUPED-adjusted outcome: Y - theta * (X - mean(X))."""
    y_arr = np.asarray(y, dtype=float)
    x_arr = np.asarray(x, dtype=float)
    if theta is None:
        theta = compute_theta(y_arr, x_arr)
    return y_arr - theta * (x_arr - x_arr.mean())
