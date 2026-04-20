"""experiment-toolkit: small utilities for online controlled experiments."""

from experiment_toolkit.cs_did import (
    CSResult,
    cs_staggered_att,
    simulate_staggered_panel,
)
from experiment_toolkit.cuped import apply_cuped, compute_theta
from experiment_toolkit.ratio import ratio_metric_variance
from experiment_toolkit.sample_size import (
    cuped_sample_size_summary,
    mde_for_n,
    sample_size_for_mde,
)
from experiment_toolkit.sensitivity import (
    EValueResult,
    e_value,
    rosenbaum_gamma_threshold,
    rosenbaum_wilcoxon_bounds,
)
from experiment_toolkit.sequential import msprt_cuped_pvalue, msprt_pvalue

__version__ = "0.2.0"

__all__ = [
    "CSResult",
    "EValueResult",
    "apply_cuped",
    "compute_theta",
    "cs_staggered_att",
    "cuped_sample_size_summary",
    "e_value",
    "mde_for_n",
    "msprt_cuped_pvalue",
    "msprt_pvalue",
    "ratio_metric_variance",
    "rosenbaum_gamma_threshold",
    "rosenbaum_wilcoxon_bounds",
    "sample_size_for_mde",
    "simulate_staggered_panel",
]
