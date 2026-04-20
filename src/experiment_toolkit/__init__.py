"""experiment-toolkit: small utilities for online controlled experiments."""

from experiment_toolkit.cuped import apply_cuped, compute_theta
from experiment_toolkit.ratio import ratio_metric_variance
from experiment_toolkit.sample_size import mde_for_n, sample_size_for_mde
from experiment_toolkit.sequential import msprt_pvalue

__version__ = "0.1.0"

__all__ = [
    "apply_cuped",
    "compute_theta",
    "mde_for_n",
    "msprt_pvalue",
    "ratio_metric_variance",
    "sample_size_for_mde",
]
