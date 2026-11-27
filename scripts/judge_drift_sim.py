"""Judge drift mid-test: what an LLM-judge upgrade does to an A/B test.

Writes docs/judge_drift_results.md and docs/judge_drift.png. There are no model
API calls. Quality, judge scores and human labels are simulated, so every true
effect is known. Simulated means appear next to their closed-form values from
experiment_toolkit.judge_drift.

Run:       python scripts/judge_drift_sim.py
CI smoke:  python scripts/judge_drift_sim.py --smoke   (few reps, writes to a temp dir)
"""

from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from experiment_toolkit.judge_drift import (  # noqa: E402
    ANALYSES,
    JudgeUpgrade,
    analyze_drift,
    expected_naive,
    expected_naive_se,
    expected_ppi_front,
    rejection_rate,
    simulate_drift_experiment,
)
from experiment_toolkit.llm_judge import JudgeConfusion  # noqa: E402

SEED = 20261127
ROOT = Path(__file__).resolve().parents[1]
P_CONTROL = 0.60
EFFECT = 0.03
N_PER_ARM = 10_000
N_LABELED = 500
OLD = JudgeConfusion(0.85, 0.85)
UPGRADES = {
    "No upgrade": JudgeUpgrade(OLD, OLD, OLD, OLD),
    "New judge more lenient on both arms": JudgeUpgrade(
        OLD, OLD, JudgeConfusion(0.85, 0.75), JudgeConfusion(0.85, 0.75)
    ),
    "New judge more accurate on both arms": JudgeUpgrade(
        OLD, OLD, JudgeConfusion(0.92, 0.92), JudgeConfusion(0.92, 0.92)
    ),
    "New judge more lenient on treatment only": JudgeUpgrade(
        OLD, OLD, JudgeConfusion(0.85, 0.75), OLD
    ),
}
LABELS = {
    "naive_pooled": "Judge only, pooled",
    "naive_pre": "Judge only, pre-upgrade data",
    "ppi_spread": "PPI++, labels across whole test",
    "ppi_front": "PPI++, labels before upgrade only",
    "ppi_stratified": "PPI++, stratified by judge version",
}
SWEEP_N = 20_000
SWEEP_LABELS = (250, 500, 1_000, 2_000, 4_000)
SWEEP_ANALYSES = ("ppi_front", "ppi_spread", "ppi_stratified")


def expected(a, u, effect, n, n_lab):
    if a == "naive_pooled":
        return expected_naive(u, P_CONTROL, effect)
    if a == "naive_pre":
        return expected_naive(u, P_CONTROL, effect, pre_only=True)
    if a == "ppi_front":
        return expected_ppi_front(u, P_CONTROL, effect, n, n_lab)
    return effect


def simulate(u, effect, n, n_lab, reps, rng, analyses=ANALYSES):
    out = {a: [] for a in analyses}
    for _ in range(reps):
        t, c = simulate_drift_experiment(n, P_CONTROL, effect, u, rng)
        res = analyze_drift(t, c, n_lab, rng)
        for a in analyses:
            r = res[a]
            out[a].append((r.estimate, r.se, r.covers(effect), r.rejects()))
    return {a: np.array(v) for a, v in out.items()}


def main_table(rng, reps):
    lines = [
        "| Judge upgrade | True effect | Analysis | Mean estimate (expected) | Emp. SD "
        "| Mean SE | Coverage | Rejects 0 (expected) |",
        "|---|---:|---|---:|---:|---:|---:|---:|",
    ]
    type1 = {}
    for name, u in UPGRADES.items():
        for effect in (0.0, EFFECT):
            d = simulate(u, effect, N_PER_ARM, N_LABELED, reps, rng)
            for a in ANALYSES:
                x = d[a]
                exp = expected(a, u, effect, N_PER_ARM, N_LABELED)
                rej = f"{x[:, 3].mean():.3f}"
                if a.startswith("naive"):
                    se = expected_naive_se(u, P_CONTROL, effect, N_PER_ARM,
                                           pre_only=a == "naive_pre")
                    rej += f" ({rejection_rate(exp, se):.3f})"
                lines.append(
                    f"| {name} | {effect:+.2f} | {LABELS[a]} | {x[:, 0].mean():+.4f} "
                    f"({exp:+.4f}) | {x[:, 0].std(ddof=1):.4f} | {x[:, 1].mean():.4f} "
                    f"| {x[:, 2].mean():.3f} | {rej} |"
                )
                if effect == 0.0:
                    type1[(name, a)] = x[:, 3].mean()
    return lines, type1


def sweep_table(rng, reps):
    u = UPGRADES["New judge more lenient on treatment only"]
    lines = [
        "| Labels per arm | "
        + " | ".join(f"{LABELS[a]}: Type-I" for a in SWEEP_ANALYSES)
        + " | Labels-before-upgrade bias (expected) |",
        "|---:|" + "---:|" * (len(SWEEP_ANALYSES) + 1),
    ]
    rates = {a: [] for a in SWEEP_ANALYSES}
    for n_lab in SWEEP_LABELS:
        d = simulate(u, 0.0, SWEEP_N, n_lab, reps, rng, SWEEP_ANALYSES)
        for a in SWEEP_ANALYSES:
            rates[a].append(d[a][:, 3].mean())
        bias = d["ppi_front"][:, 0].mean()
        exp = expected_ppi_front(u, P_CONTROL, 0.0, SWEEP_N, n_lab)
        lines.append(
            f"| {n_lab:,} | " + " | ".join(f"{rates[a][-1]:.3f}" for a in SWEEP_ANALYSES)
            + f" | {bias:+.4f} ({exp:+.4f}) |"
        )
    return lines, rates


def make_figure(type1, rates, out_png):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4))
    ax = axes[0]
    names = list(UPGRADES)
    x = np.arange(len(names))
    width = 0.16
    colors = ("#d62728", "#ff7f0e", "#1f77b4", "#9467bd", "#2ca02c")
    for i, (a, color) in enumerate(zip(ANALYSES, colors, strict=True)):
        ax.bar(x + (i - 2) * width, [100 * type1[(n, a)] for n in names], width,
               color=color, label=LABELS[a])
    ax.axhline(5, color="black", ls="--", lw=1)
    ax.set_xticks(x, ["No\nupgrade", "Lenient\nboth arms", "More accurate\nboth arms",
                      "Lenient on\ntreatment"], fontsize=9)
    ax.set_ylabel("Type-I error (%)")
    ax.set_ylim(0, 100)
    ax.set_title(f"No true effect, judge upgraded halfway\n{N_PER_ARM:,} units, "
                 f"{N_LABELED} labels per arm")
    ax.legend(fontsize=7.5, loc="upper left")
    ax.grid(True, axis="y", alpha=0.3)

    ax = axes[1]
    for a, color in zip(SWEEP_ANALYSES, ("#9467bd", "#1f77b4", "#2ca02c"), strict=True):
        ax.plot(SWEEP_LABELS, 100 * np.array(rates[a]), "-o", color=color, label=LABELS[a])
    ax.axhline(5, color="black", ls="--", lw=1)
    ax.set_xscale("log")
    ax.set_xticks(SWEEP_LABELS, [f"{n:,}" for n in SWEEP_LABELS])
    ax.minorticks_off()
    ax.set_xlabel("Human labels per arm")
    ax.set_ylabel("Type-I error (%)")
    ax.set_ylim(0, 100)
    ax.set_title(f"New judge lenient on treatment, no true effect\n{SWEEP_N:,} units per arm")
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(True, alpha=0.3)
    fig.suptitle("A mid-test judge upgrade is safe only if the analysis knows about it",
                 fontsize=12, y=1.02)
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=140, bbox_inches="tight", metadata={"Software": None})
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--smoke", action="store_true",
                        help="few replications, outputs to a temp dir (for CI)")
    args = parser.parse_args()
    reps, sweep_reps = (2_000, 500) if not args.smoke else (4, 3)
    out_dir = Path(tempfile.mkdtemp()) if args.smoke else ROOT / "docs"
    out_md, out_png = out_dir / "judge_drift_results.md", out_dir / "judge_drift.png"

    seeds = np.random.SeedSequence(SEED).spawn(2)
    main_lines, type1 = main_table(np.random.default_rng(seeds[0]), reps)
    sweep_lines, rates = sweep_table(np.random.default_rng(seeds[1]), sweep_reps)
    make_figure(type1, rates, out_png)

    md = [
        "# Judge drift mid-test: simulation results",
        "",
        "<!-- Generated by scripts/judge_drift_sim.py. Do not edit by hand. -->",
        "",
        "Reproduce from a clean checkout:",
        "",
        "```bash",
        'pip install -e ".[dev]"',
        "python scripts/judge_drift_sim.py",
        "```",
        "",
        f"Seed `{SEED}`. No model API calls. True quality is binary with control rate "
        f"{P_CONTROL:.2f}; the true effect is +{EFFECT:.2f} or 0 and doesn't change over "
        "time. The old judge has sensitivity = specificity = 0.85 on both arms. Halfway "
        "through the test, both arms switch to a new judge (sensitivity/specificity): "
        + "; ".join(
            f"*{name}*: treatment {u.new_t.sensitivity:.2f}/{u.new_t.specificity:.2f}, "
            f"control {u.new_c.sensitivity:.2f}/{u.new_c.specificity:.2f}"
            for name, u in UPGRADES.items()
        )
        + ". Every unit is judged; human "
        "labels are the same budget for every PPI++ variant. Values in parentheses are "
        "closed-form answers from `experiment_toolkit.judge_drift` (large-sample for "
        "\"labels before upgrade only\"). PPI++ variants with labels spread across the test "
        "or stratified by judge version are unbiased by construction, so their expected "
        f"value is the true effect. Monte Carlo SE of a rate near 0.05 is about "
        f"{np.sqrt(0.05 * 0.95 / reps):.3f}.",
        "",
        "![Judge drift simulation](judge_drift.png)",
        "",
        f"## 1. Every analysis against the truth ({N_PER_ARM:,} units and {N_LABELED} human "
        f"labels per arm, {reps:,} replications)",
        "",
        *main_lines,
        "",
        f"## 2. More labels make front-loaded labeling worse ({SWEEP_N:,} units per arm, new "
        f"judge lenient on treatment only, no true effect, {sweep_reps} replications per row)",
        "",
        "The bias of labels-before-upgrade-only doesn't shrink with more labels, but its "
        "standard error does.",
        "",
        *sweep_lines,
        "",
    ]
    out_md.write_text("\n".join(md), encoding="utf-8", newline="\n")
    print(f"wrote {out_md}")
    print(f"wrote {out_png}")


if __name__ == "__main__":
    main()
