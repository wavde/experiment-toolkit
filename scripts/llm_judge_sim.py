"""LLM-as-judge noise in A/B tests: what it does to bias, Type-I error and power.

Writes docs/llm_judge_results.md and docs/llm_judge.png. There are no model
API calls. True quality is binary and simulated, and the "LLM judge" is a
per-arm confusion matrix, so the ground truth is known by construction. Each
simulated number is printed next to its closed-form value from
experiment_toolkit.llm_judge.

Run:       python scripts/llm_judge_sim.py
CI smoke:  python scripts/llm_judge_sim.py --smoke   (few reps, writes to a temp dir)
"""

from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy import stats  # noqa: E402

from experiment_toolkit import (  # noqa: E402
    JudgeConfusion,
    analyze_judged_experiment,
    expected_naive_diff,
    judge_sample_size_inflation,
    naive_judge_se,
    ppi_expected_se,
    simulate_judged_experiment,
    two_sided_rejection_rate,
)

SEED = 20261126
ROOT = Path(__file__).resolve().parents[1]
OUT_MD = ROOT / "docs" / "llm_judge_results.md"
OUT_PNG = ROOT / "docs" / "llm_judge.png"

ALPHA = 0.05
TARGET_POWER = 0.80
P_CONTROL = 0.60
EFFECT = 0.03  # true lift in the share of good responses (60% -> 63%)
N_PER_ARM = 5_000
N_LABELED = 500
N_REPS = 2_000

NOISY = JudgeConfusion(sensitivity=0.85, specificity=0.85)
# Judge that is more lenient toward treatment (e.g. it favours the new
# model's longer answers): 5 points more false positives in that arm.
LENIENT = JudgeConfusion(sensitivity=0.85, specificity=0.80)
PERFECT = JudgeConfusion(sensitivity=1.0, specificity=1.0)

ESTIMATORS = ("naive", "human_only", "ppi")
LABELS = {"naive": "Naive judge", "human_only": "Human labels only", "ppi": "PPI++"}

# Figure / sweep settings
SWEEP_N = (1_000, 2_000, 5_000, 10_000, 20_000, 50_000)
SWEEP_REPS = 500
LABEL_SWEEP_N_PER_ARM = 20_000
LABEL_SWEEP = (250, 500, 1_000, 2_000, 4_000)
ACCURACIES = (1.0, 0.95, 0.90, 0.85, 0.80, 0.70)


def simulate(n_reps, n_per_arm, n_labeled, effect, judge_c, judge_t, rng):
    est = {k: np.empty(n_reps) for k in ESTIMATORS}
    rej = {k: np.empty(n_reps, dtype=bool) for k in ESTIMATORS}
    cov = {k: np.empty(n_reps, dtype=bool) for k in ESTIMATORS}
    for r in range(n_reps):
        t, c = simulate_judged_experiment(
            n_per_arm, P_CONTROL, effect, judge_c, judge_t, n_labeled, rng
        )
        for name, res in analyze_judged_experiment(t, c, alpha=ALPHA).items():
            est[name][r] = res.estimate
            rej[name][r] = res.p_value < ALPHA
            cov[name][r] = res.ci_low <= effect <= res.ci_high
    return est, rej, cov


def expected(name, n_per_arm, n_labeled, effect, judge_c, judge_t):
    if name == "naive":
        mean = expected_naive_diff(P_CONTROL, effect, judge_c, judge_t)
        se = naive_judge_se(n_per_arm, P_CONTROL, effect, judge_c, judge_t)
    else:
        lam = 0.0 if name == "human_only" else None
        mean = effect
        se = ppi_expected_se(n_per_arm, n_labeled, P_CONTROL, effect, judge_c, judge_t, lam=lam)
    return mean, se, two_sided_rejection_rate(mean, se, ALPHA)


def labels_for_power(n_per_arm, effect, judge, lam):
    """Smallest per-arm label count reaching TARGET_POWER, or None if > n_per_arm."""
    def power(n_lab):
        se = ppi_expected_se(n_per_arm, n_lab, P_CONTROL, effect, judge, lam=lam)
        return two_sided_rejection_rate(effect, se, ALPHA)

    if power(n_per_arm) < TARGET_POWER:
        return None
    lo, hi = 2, n_per_arm
    while lo < hi:
        mid = (lo + hi) // 2
        if power(mid) >= TARGET_POWER:
            hi = mid
        else:
            lo = mid + 1
    return lo


def naive_n_for_power(effect, judge):
    q_t, q_c = judge.positive_rate(P_CONTROL + effect), judge.positive_rate(P_CONTROL)
    z = stats.norm.ppf(1 - ALPHA / 2) + stats.norm.ppf(TARGET_POWER)
    return int(np.ceil(z**2 * (q_t * (1 - q_t) + q_c * (1 - q_c)) / (q_t - q_c) ** 2))


def table_scenarios(rng):
    scenarios = [
        ("Perfect judge", PERFECT, PERFECT),
        ("Noisy judge, same in both arms", NOISY, NOISY),
        ("Judge 5 pts more lenient on treatment", NOISY, LENIENT),
    ]
    lines = [
        "| Scenario | True effect | Estimator | Mean estimate (expected) | SD (expected SE) "
        "| Reject rate (expected) | 95% CI coverage |",
        "|---|---:|---|---:|---:|---:|---:|",
    ]
    for label, judge_c, judge_t in scenarios:
        for effect in (0.0, EFFECT):
            est, rej, cov = simulate(N_REPS, N_PER_ARM, N_LABELED, effect, judge_c, judge_t, rng)
            for name in ESTIMATORS:
                mean, se, rate = expected(name, N_PER_ARM, N_LABELED, effect, judge_c, judge_t)
                lines.append(
                    f"| {label} | {effect:.2f} | {LABELS[name]} "
                    f"| {est[name].mean():+.4f} ({mean:+.4f}) "
                    f"| {est[name].std(ddof=1):.4f} ({se:.4f}) "
                    f"| {rej[name].mean():.3f} ({rate:.3f}) "
                    f"| {cov[name].mean():.3f} |"
                )
    return lines


def table_inflation():
    n_perfect = naive_n_for_power(EFFECT, PERFECT)
    lines = [
        "| Judge sensitivity = specificity | Youden J | Sample-size multiplier "
        "| Per-arm n for 80% power |",
        "|---:|---:|---:|---:|",
    ]
    for acc in ACCURACIES:
        judge = JudgeConfusion(acc, acc)
        infl = judge_sample_size_inflation(P_CONTROL, EFFECT, judge)
        n_needed = naive_n_for_power(EFFECT, judge)
        lines.append(
            f"| {acc:.2f} | {judge.youden_j:.2f} | {infl:.2f}x | {n_needed:,} |"
        )
    return lines, n_perfect


def table_labels():
    lines = [
        "| Judge sensitivity = specificity | Corr(judge, human) | Human labels only "
        "| PPI++ labels | Label saving |",
        "|---:|---:|---:|---:|---:|",
    ]
    for acc in ACCURACIES[1:]:
        judge = JudgeConfusion(acc, acc)
        q = judge.positive_rate(P_CONTROL)
        cov = P_CONTROL * (judge.sensitivity - q)
        rho = cov / np.sqrt(P_CONTROL * (1 - P_CONTROL) * q * (1 - q))
        human = labels_for_power(LABEL_SWEEP_N_PER_ARM, EFFECT, judge, lam=0.0)
        ppi = labels_for_power(LABEL_SWEEP_N_PER_ARM, EFFECT, judge, lam=None)
        lines.append(
            f"| {acc:.2f} | {rho:.2f} | {human:,} | {ppi:,} | {1 - ppi / human:.0%} |"
        )
    return lines


def type1_sweep(rng):
    """Simulated naive and PPI++ Type-I error vs N under the lenient judge."""
    naive, ppi = [], []
    for n in SWEEP_N:
        _, rej, _ = simulate(SWEEP_REPS, n, N_LABELED, 0.0, NOISY, LENIENT, rng)
        naive.append(rej["naive"].mean())
        ppi.append(rej["ppi"].mean())
    return np.array(naive), np.array(ppi)


def label_sweep(rng):
    """Simulated power vs labels per arm at LABEL_SWEEP_N_PER_ARM, noisy judge."""
    human, ppi = [], []
    for n_lab in LABEL_SWEEP:
        _, rej, _ = simulate(
            SWEEP_REPS, LABEL_SWEEP_N_PER_ARM, n_lab, EFFECT, NOISY, NOISY, rng
        )
        human.append(rej["human_only"].mean())
        ppi.append(rej["ppi"].mean())
    return np.array(human), np.array(ppi)


def make_figure(sim_type1, sim_power):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))

    ax = axes[0]
    n_grid = np.logspace(np.log10(SWEEP_N[0]), np.log10(SWEEP_N[-1]), 100)
    for drop, color in ((0.01, "#fdae6b"), (0.02, "#f16913"), (0.05, "#a63603")):
        lenient = JudgeConfusion(NOISY.sensitivity, NOISY.specificity - drop)
        bias = expected_naive_diff(P_CONTROL, 0.0, NOISY, lenient)
        rate = [
            two_sided_rejection_rate(bias, naive_judge_se(int(n), P_CONTROL, 0.0, NOISY, lenient))
            for n in n_grid
        ]
        ax.plot(n_grid, np.array(rate) * 100, color=color, lw=2,
                label=f"Naive, treatment specificity {drop * 100:.0f} pt lower")
    naive_sim, ppi_sim = sim_type1
    ax.plot(SWEEP_N, naive_sim * 100, "o", color="#a63603", ms=5, label="Naive, simulated (5 pts)")
    ax.plot(SWEEP_N, ppi_sim * 100, "s", color="#2ca02c", ms=5,
            label=f"PPI++, {N_LABELED} labels/arm, simulated")
    ax.axhline(ALPHA * 100, color="black", ls="--", lw=1, alpha=0.6)
    ax.set_xscale("log")
    ax.set_ylim(0, 100)
    ax.set_xlabel("Units per arm (all judged)")
    ax.set_ylabel("Type-I error (%)")
    ax.set_title("No true effect: differential judge error\nfalse positives grow with N")
    ax.legend(fontsize=7.5, loc="upper left")
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    lab_grid = np.linspace(LABEL_SWEEP[0], LABEL_SWEEP[-1], 100).astype(int)
    for lam, name, color in ((0.0, "Human labels only", "#1f77b4"), (None, "PPI++", "#2ca02c")):
        power = [
            two_sided_rejection_rate(
                EFFECT, ppi_expected_se(LABEL_SWEEP_N_PER_ARM, n, P_CONTROL, EFFECT, NOISY, lam=lam)
            )
            for n in lab_grid
        ]
        ax.plot(lab_grid, np.array(power) * 100, color=color, lw=2, label=f"{name} (analytic)")
    human_sim, ppi_sim = sim_power
    ax.plot(LABEL_SWEEP, human_sim * 100, "o", color="#1f77b4", ms=5, label="Simulated")
    ax.plot(LABEL_SWEEP, ppi_sim * 100, "s", color="#2ca02c", ms=5)
    ax.axhline(TARGET_POWER * 100, color="black", ls="--", lw=1, alpha=0.6)
    ax.set_xlabel("Human-labeled units per arm")
    ax.set_ylabel("Power (%)")
    ax.set_title(
        f"True +{EFFECT * 100:.0f} pt effect, {LABEL_SWEEP_N_PER_ARM:,} judged units/arm\n"
        f"judge sens = spec = {NOISY.sensitivity:.2f}"
    )
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 100)

    fig.suptitle("LLM-as-judge as an A/B metric: bias you can't out-sample, and labels that fix it",
                 fontsize=12, y=1.02)
    fig.tight_layout()
    OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PNG, dpi=140, bbox_inches="tight", metadata={"Software": None})
    plt.close(fig)


def main() -> None:
    global N_REPS, SWEEP_REPS, OUT_MD, OUT_PNG
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--smoke", action="store_true",
                        help="few replications, outputs to a temp dir (for CI)")
    if parser.parse_args().smoke:
        N_REPS, SWEEP_REPS = 20, 10
        tmp = Path(tempfile.mkdtemp())
        OUT_MD, OUT_PNG = tmp / OUT_MD.name, tmp / OUT_PNG.name

    seeds = np.random.SeedSequence(SEED).spawn(3)
    scenario_lines = table_scenarios(np.random.default_rng(seeds[0]))
    inflation_lines, n_human_full = table_inflation()
    label_lines = table_labels()
    sim_type1 = type1_sweep(np.random.default_rng(seeds[1]))
    sim_power = label_sweep(np.random.default_rng(seeds[2]))
    make_figure(sim_type1, sim_power)

    type1_rows = ["| Units per arm | Naive (expected) | PPI++ |", "|---:|---:|---:|"]
    bias = expected_naive_diff(P_CONTROL, 0.0, NOISY, LENIENT)
    for n, naive, ppi in zip(SWEEP_N, *sim_type1, strict=True):
        rate = two_sided_rejection_rate(bias, naive_judge_se(n, P_CONTROL, 0.0, NOISY, LENIENT))
        type1_rows.append(f"| {n:,} | {naive:.3f} ({rate:.3f}) | {ppi:.3f} |")

    power_rows = ["| Labels per arm | Human only | PPI++ |", "|---:|---:|---:|"]
    for n_lab, human, ppi in zip(LABEL_SWEEP, *sim_power, strict=True):
        power_rows.append(f"| {n_lab:,} | {human:.3f} | {ppi:.3f} |")

    md = [
        "# LLM-as-judge noise in A/B tests: simulation results",
        "",
        "<!-- Generated by scripts/llm_judge_sim.py. Do not edit by hand. -->",
        "",
        "Reproduce from a clean checkout:",
        "",
        "```bash",
        'pip install -e ".[dev]"',
        "python scripts/llm_judge_sim.py",
        "```",
        "",
        f"Seed `{SEED}`. No model API calls. True quality is Bernoulli "
        f"(control rate {P_CONTROL:.2f}). The judge is a per-arm confusion matrix, so the "
        "true effect is known by construction. Values in parentheses are the closed-form "
        "answers from `experiment_toolkit.llm_judge`. Estimators:",
        "",
        "- **Naive judge**: difference in mean judge score over all units.",
        "- **Human labels only**: difference in mean human label over the labeled subsample.",
        "- **PPI++**: prediction-powered inference with a power-tuned weight, using the human "
        "labels in each arm to correct the judge (Angelopoulos et al. 2023).",
        "",
        "![LLM-as-judge simulation](llm_judge.png)",
        "",
        f"## 1. Bias, Type-I error, power and coverage ({N_PER_ARM:,} judged units and "
        f"{N_LABELED} human labels per arm, {N_REPS:,} replications)",
        "",
        f"Noisy judge: sensitivity = specificity = {NOISY.sensitivity:.2f}. Lenient judge on "
        f"treatment: specificity {LENIENT.specificity:.2f}. Monte Carlo SE of a rejection rate "
        f"near 0.05 is about {np.sqrt(0.05 * 0.95 / N_REPS):.3f}.",
        "",
        *scenario_lines,
        "",
        "## 2. Differential error: Type-I error grows with sample size",
        "",
        f"No true effect. The judge is {NOISY.specificity - LENIENT.specificity:.2f} less specific "
        f"on treatment. PPI++ uses {N_LABELED} human labels per arm. {SWEEP_REPS} replications "
        "per row.",
        "",
        *type1_rows,
        "",
        "## 3. Non-differential noise: extra sample the naive metric needs",
        "",
        f"Analytic. True effect +{EFFECT:.2f} on a {P_CONTROL:.2f} base rate, two-sided "
        f"alpha = {ALPHA}, power = {TARGET_POWER}. With a perfect metric the requirement is "
        f"{n_human_full:,} per arm. These multipliers apply only when the judge errs the same "
        "way in both arms. Under differential error, no sample size fixes the bias (section 2).",
        "",
        *inflation_lines,
        "",
        f"## 4. Human labels needed for 80% power ({LABEL_SWEEP_N_PER_ARM:,} judged units per arm)",
        "",
        f"Analytic. True effect +{EFFECT:.2f}. PPI++ remains unbiased under differential "
        "error, as long as the labels are a random sample within each arm.",
        "",
        *label_lines,
        "",
        f"Simulated power at sensitivity = specificity = {NOISY.sensitivity:.2f} "
        f"({SWEEP_REPS} replications per row):",
        "",
        *power_rows,
        "",
    ]
    OUT_MD.write_text("\n".join(md), encoding="utf-8", newline="\n")
    print(f"wrote {OUT_MD}")
    print(f"wrote {OUT_PNG}")


if __name__ == "__main__":
    main()
