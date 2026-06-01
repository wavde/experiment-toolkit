"""CUPED variance reduction — see Deng et al. (2013)."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike


def _validate_pair(y_arr: np.ndarray, x_arr: np.ndarray) -> None:
    if y_arr.shape != x_arr.shape:
        raise ValueError("y and x must have the same shape")
    if y_arr.ndim != 1:
        raise ValueError("y and x must be 1-D arrays")
    if len(y_arr) < 2:
        raise ValueError("need at least 2 observations")
    if not (np.all(np.isfinite(y_arr)) and np.all(np.isfinite(x_arr))):
        raise ValueError("y and x must be finite (no NaN/inf)")


def compute_theta(y: ArrayLike, x: ArrayLike) -> float:
    """Optimal CUPED coefficient theta = Cov(Y, X) / Var(X)."""
    y_arr = np.asarray(y, dtype=float)
    x_arr = np.asarray(x, dtype=float)
    _validate_pair(y_arr, x_arr)
    var_x = np.var(x_arr, ddof=1)
    if var_x < 1e-12:
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
    _validate_pair(y_arr, x_arr)
    if theta is None:
        theta = compute_theta(y_arr, x_arr)
    return y_arr - theta * (x_arr - x_arr.mean())
