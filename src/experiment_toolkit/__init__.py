"""experiment-toolkit: small utilities for online controlled experiments."""

from experiment_toolkit.cs_did import (
    CSResult,
    cs_staggered_att,
    simulate_staggered_panel,
)
from experiment_toolkit.cuped import apply_cuped, compute_theta
from experiment_toolkit.llm_judge import (
    JudgeConfusion,
    JudgeDiffResult,
    JudgedArm,
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
from experiment_toolkit.offline_eval import (
    OfflineEvalData,
    OfflineOnlineScenario,
    Readout,
    expected_judged_diff,
    expected_online_effect,
    expected_quality_diff,
    offline_readouts,
    simulate_offline_eval,
    simulate_online_ab,
)
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

__version__ = "0.3.0"

__all__ = [
    "CSResult",
    "EValueResult",
    "JudgeConfusion",
    "JudgeDiffResult",
    "JudgedArm",
    "OfflineEvalData",
    "OfflineOnlineScenario",
    "Readout",
    "analyze_judged_experiment",
    "apply_cuped",
    "compute_theta",
    "cs_staggered_att",
    "cuped_sample_size_summary",
    "e_value",
    "expected_judged_diff",
    "expected_naive_diff",
    "expected_online_effect",
    "expected_quality_diff",
    "judge_sample_size_inflation",
    "mde_for_n",
    "msprt_cuped_pvalue",
    "msprt_pvalue",
    "naive_judge_diff",
    "naive_judge_se",
    "offline_readouts",
    "ppi_expected_se",
    "ppi_mean_diff",
    "ratio_metric_variance",
    "rosenbaum_gamma_threshold",
    "rosenbaum_wilcoxon_bounds",
    "sample_size_for_mde",
    "simulate_judged_arm",
    "simulate_judged_experiment",
    "simulate_offline_eval",
    "simulate_online_ab",
    "simulate_staggered_panel",
    "two_sided_rejection_rate",
]
