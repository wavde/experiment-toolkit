"""How far a pre-treatment covariate cuts the human-label budget of PPI++.

Writes docs/ppi_covariates_results.md and docs/ppi_covariates.png. There are no
model API calls. Quality, judge scores and the covariate are simulated, so
every answer is known in closed form. Each simulated number appears next to its
closed-form value from experiment_toolkit.ppi_covariates.

Run:       python scripts/ppi_covariates_sim.py
CI smoke:  python scripts/ppi_covariates_sim.py --smoke   (few reps, writes to a temp dir)
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

from experiment_toolkit import JudgeConfusion  # noqa: E402
from experiment_toolkit.ppi_covariates import (  # noqa: E402
    CONTROL_VARIATES,
    control_variate_diff,
    covariate_arm,
    expected_se,
    labels_for_power,
    population_r2,
    simulate_covariate_arm,
)

SEED = 20261127
ROOT = Path(__file__).resolve().parents[1]

ALPHA = 0.05
P_CONTROL = 0.60
EFFECT = 0.03
N_PER_ARM = 20_000
N_LABELED = 500
JUDGE = JudgeConfusion(0.85, 0.85)
LENIENT = JudgeConfusion(0.85, 0.80)
R2_MAIN = 0.25
ACCURACIES = (0.95, 0.90, 0.85, 0.80, 0.70)
R2_GRID = (0.0, 0.10, 0.25, 0.50)
N_GRID = (5_000, 20_000, 100_000)
SMALL_LABELS = (30, 100, 300)
LABEL_SWEEP = (250, 500, 1_000, 2_000, 3_000)

LABELS = {
    (): "Human labels only",
    ("x",): "Human labels + covariate",
    ("s",): "PPI++ (judge)",
    ("s", "x"): "PPI++ + covariate",
}


def arms(effect, r2=R2_MAIN, judge_t=JUDGE, judge_c=JUDGE):
    control = covariate_arm(P_CONTROL, r2, judge_c)
    return control.shifted(effect, judge_t), control


def simulate(reps, treatment, control, n_total, n_lab, rng, effect):
    out = {z: {"est": [], "rej": [], "cov": []} for z in CONTROL_VARIATES}
    for _ in range(reps):
        t = simulate_covariate_arm(treatment, n_total, n_lab, rng)
        c = simulate_covariate_arm(control, n_total, n_lab, rng)
        for z in CONTROL_VARIATES:
            r = control_variate_diff(t, c, z, alpha=ALPHA)
            out[z]["est"].append(r.estimate)
            out[z]["rej"].append(r.p_value < ALPHA)
            out[z]["cov"].append(r.ci_low <= effect <= r.ci_high)
    return {z: {k: np.asarray(v, dtype=float) for k, v in d.items()} for z, d in out.items()}


def rejection(effect, se):
    za = stats.norm.ppf(1 - ALPHA / 2)
    return float(stats.norm.cdf(effect / se - za) + stats.norm.cdf(-effect / se - za))


def validation_table(rng, reps):
    scenarios = [
        ("No effect", 0.0, JUDGE),
        ("True effect +0.03", EFFECT, JUDGE),
        ("No effect, judge 5 pts more lenient on treatment", 0.0, LENIENT),
    ]
    lines = [
        "| Scenario | Estimator | Mean estimate (truth) | SD (expected SE) "
        "| Reject rate (expected) | 95% CI coverage |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for name, effect, judge_t in scenarios:
        t, c = arms(effect, judge_t=judge_t)
        res = simulate(reps, t, c, N_PER_ARM, N_LABELED, rng, effect)
        for z in CONTROL_VARIATES:
            se = expected_se(t, c, N_PER_ARM, N_LABELED, z)
            r = res[z]
            lines.append(
                f"| {name} | {LABELS[z]} | {r['est'].mean():+.4f} ({effect:+.4f}) "
                f"| {r['est'].std(ddof=1):.4f} ({se:.4f}) "
                f"| {r['rej'].mean():.3f} ({rejection(effect, se):.3f}) | {r['cov'].mean():.3f} |"
            )
    return lines


def small_label_table(rng, reps):
    lines = [
        "| Labels per arm | PPI++ coverage | PPI++ + covariate coverage | PPI++ Type-I "
        "| PPI++ + covariate Type-I |",
        "|---:|---:|---:|---:|---:|",
    ]
    t, c = arms(0.0)
    for n_lab in SMALL_LABELS:
        res = simulate(reps, t, c, N_PER_ARM, n_lab, rng, 0.0)
        a, b = res[("s",)], res[("s", "x")]
        lines.append(
            f"| {n_lab} | {a['cov'].mean():.3f} | {b['cov'].mean():.3f} "
            f"| {a['rej'].mean():.3f} | {b['rej'].mean():.3f} |"
        )
    return lines


def budget_tables():
    r2_head = " | ".join(f"R²(Y; X) = {r:.2f}" for r in R2_GRID)
    lines = [
        f"| Judge accuracy (sens = spec) | R²(Y; judge) | {r2_head} |",
        "|---:|---:|" + "---:|" * len(R2_GRID),
    ]
    saving = [
        "| Judge accuracy (sens = spec) | "
        + " | ".join(f"R²(Y; X) = {r:.2f}" for r in R2_GRID[1:])
        + " |",
        "|---:|" + "---:|" * (len(R2_GRID) - 1),
    ]
    grid = {}
    no_judge = []
    for r2 in R2_GRID:
        t, c = arms(EFFECT, r2)
        no_judge.append(labels_for_power(t, c, N_PER_ARM, ("x",) if r2 > 0 else ()))
    lines.append("| no judge | — | " + " | ".join(f"{n:,}" for n in no_judge) + " |")
    for acc in ACCURACIES:
        j = JudgeConfusion(acc, acc)
        row = []
        for r2 in R2_GRID:
            t, c = arms(EFFECT, r2, j, j)
            row.append(labels_for_power(t, c, N_PER_ARM, ("s", "x") if r2 > 0 else ("s",)))
        grid[acc] = row
        r2_s = population_r2(covariate_arm(P_CONTROL, 0.0, j), ("s",))
        lines.append(f"| {acc:.2f} | {r2_s:.2f} | " + " | ".join(f"{n:,}" for n in row) + " |")
        saving.append(
            f"| {acc:.2f} | " + " | ".join(f"{1 - n / row[0]:.0%}" for n in row[1:]) + " |"
        )
    saving.insert(
        2,
        "| no judge | " + " | ".join(f"{1 - n / no_judge[0]:.0%}" for n in no_judge[1:]) + " |",
    )
    return lines, saving, grid, no_judge


def _fmt(v: int | None) -> str:
    return f"{v:,}" if v is not None else "not reachable"


def n_table():
    lines = [
        "| Judged units per arm | PPI++ labels | PPI++ + covariate labels | Saving "
        "| Labels as share of arm (PPI++) |",
        "|---:|---:|---:|---:|---:|",
    ]
    t, c = arms(EFFECT)
    for n in N_GRID:
        a = labels_for_power(t, c, n, ("s",))
        b = labels_for_power(t, c, n, ("s", "x"))
        fmt = _fmt
        save = f"{1 - b / a:.0%}" if a and b else "—"
        share = f"{a / n:.0%}" if a else "—"
        lines.append(f"| {n:,} | {fmt(a)} | {fmt(b)} | {save} | {share} |")
    return lines


def power_sweep(rng, reps):
    t, c = arms(EFFECT)
    sim = {("s",): [], ("s", "x"): []}
    for n_lab in LABEL_SWEEP:
        res = simulate(reps, t, c, N_PER_ARM, n_lab, rng, EFFECT)
        for z in sim:
            sim[z].append(res[z]["rej"].mean())
    lines = ["| Labels per arm | PPI++ (expected) | PPI++ + covariate (expected) |",
             "|---:|---:|---:|"]
    for i, n_lab in enumerate(LABEL_SWEEP):
        cells = []
        for z in sim:
            exp = rejection(EFFECT, expected_se(t, c, N_PER_ARM, n_lab, z))
            cells.append(f"{sim[z][i]:.3f} ({exp:.3f})")
        lines.append(f"| {n_lab:,} | " + " | ".join(cells) + " |")
    return lines, sim


def make_figure(grid, no_judge, sim, out_png):
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.3))
    ax = axes[0]
    colors = ("#9ecae1", "#6baed6", "#3182bd", "#08519c")
    for i, (r2, color) in enumerate(zip(R2_GRID, colors, strict=True)):
        ys = [grid[a][i] for a in ACCURACIES]
        ax.plot(ACCURACIES, ys, "-o", color=color, label=f"PPI++, covariate R² = {r2:.2f}")
        ax.axhline(no_judge[i], color=color, ls=":", lw=1)
    ax.invert_xaxis()
    ax.set_xlabel("Judge accuracy (sensitivity = specificity)")
    ax.set_ylabel("Human labels per arm for 80% power")
    ax.set_title(f"Label budget, +{EFFECT:.2f} effect, {N_PER_ARM:,} judged units/arm\n"
                 "dotted: no judge, covariate only")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, None)

    ax = axes[1]
    t, c = arms(EFFECT)
    grid_n = np.linspace(LABEL_SWEEP[0], LABEL_SWEEP[-1], 100).astype(int)
    for z, color in ((("s",), "#2ca02c"), (("s", "x"), "#08519c")):
        ax.plot(grid_n, [100 * rejection(EFFECT, expected_se(t, c, N_PER_ARM, n, z))
                         for n in grid_n], color=color, label=f"{LABELS[z]} (closed form)")
        ax.plot(LABEL_SWEEP, np.array(sim[z]) * 100, "o", color=color, ms=5)
    ax.axhline(80, color="black", ls="--", lw=1, alpha=0.6)
    ax.set_xlabel("Human-labeled units per arm")
    ax.set_ylabel("Power (%)")
    ax.set_ylim(0, 100)
    ax.set_title(f"Judge accuracy {JUDGE.sensitivity:.2f}, covariate R² = {R2_MAIN:.2f}\n"
                 "dots: simulated")
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(True, alpha=0.3)
    fig.suptitle("A pre-treatment covariate stacks on PPI++, with diminishing returns",
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
    reps, small_reps, sweep_reps = (2_000, 2_000, 500) if not args.smoke else (4, 4, 3)
    out_dir = Path(tempfile.mkdtemp()) if args.smoke else ROOT / "docs"
    out_md, out_png = out_dir / "ppi_covariates_results.md", out_dir / "ppi_covariates.png"

    seeds = np.random.SeedSequence(SEED).spawn(3)
    val = validation_table(np.random.default_rng(seeds[0]), reps)
    small = small_label_table(np.random.default_rng(seeds[1]), small_reps)
    budget, saving, grid, no_judge = budget_tables()
    ns = n_table()
    sweep, sim = power_sweep(np.random.default_rng(seeds[2]), sweep_reps)
    make_figure(grid, no_judge, sim, out_png)

    t, c = arms(EFFECT)
    r2_s, r2_sx = population_r2(c, ("s",)), population_r2(c, ("s", "x"))
    md = [
        "# PPI++ with a pre-treatment covariate: simulation results",
        "",
        "<!-- Generated by scripts/ppi_covariates_sim.py. Do not edit by hand. -->",
        "",
        "Reproduce from a clean checkout:",
        "",
        "```bash",
        'pip install -e ".[dev]"',
        "python scripts/ppi_covariates_sim.py",
        "```",
        "",
        f"Seed `{SEED}`. No model API calls. True quality is binary with control rate "
        f"{P_CONTROL:.2f}; the true effect is +{EFFECT:.2f} or 0. The covariate X is a binary "
        "pre-treatment attribute known for every unit (for example a difficulty bucket), "
        "set so that it explains a chosen share R²(Y; X) of the variance of true quality. "
        "The judge errs on Y only, so X and the judge score are independent given Y. Every "
        "unit is judged; human labels go to a random sample within each arm. Values in "
        "parentheses are closed-form answers from `experiment_toolkit.ppi_covariates`.",
        "",
        "The closed form for PPI++ alone reproduces loop 1's label budgets exactly "
        "(see `tests/test_ppi_covariates.py`).",
        "",
        "![PPI++ with a covariate](ppi_covariates.png)",
        "",
        f"## 1. Known-answer check ({N_PER_ARM:,} judged units, {N_LABELED} labels per arm, "
        f"judge accuracy {JUDGE.sensitivity:.2f}, R²(Y; X) = {R2_MAIN:.2f}, {reps:,} "
        "replications)",
        "",
        f"R²(Y; judge) = {r2_s:.3f}; R²(Y; judge, X) = {r2_sx:.3f}. Monte Carlo SE of a "
        f"rejection rate near 0.05 is about {np.sqrt(0.05 * 0.95 / reps):.3f}.",
        "",
        *val,
        "",
        "## 2. Small label budgets: does estimating an extra weight cost coverage?",
        "",
        f"No effect; {N_PER_ARM:,} judged units per arm; {small_reps:,} replications per row.",
        "",
        *small,
        "",
        f"## 3. Human labels per arm for 80% power ({N_PER_ARM:,} judged units per arm, "
        f"effect +{EFFECT:.2f})",
        "",
        "Closed form. R²(Y; X) = 0.00 is PPI++ alone (loop 1). Other columns add the covariate.",
        "",
        *budget,
        "",
        "Label saving from adding the covariate, relative to the same row at R²(Y; X) = 0.00:",
        "",
        *saving,
        "",
        f"## 4. Small arms cap the gain (judge accuracy {JUDGE.sensitivity:.2f}, "
        f"R²(Y; X) = {R2_MAIN:.2f})",
        "",
        "Closed form. The variance reduction is R² × N / (N + n), so the gain shrinks when "
        "labels are a large share of the arm.",
        "",
        *ns,
        "",
        f"## 5. Simulated power vs labels (judge accuracy {JUDGE.sensitivity:.2f}, "
        f"R²(Y; X) = {R2_MAIN:.2f}, {sweep_reps} replications per row)",
        "",
        *sweep,
        "",
    ]
    out_md.write_text("\n".join(md), encoding="utf-8", newline="\n")
    print(f"wrote {out_md}")
    print(f"wrote {out_png}")


if __name__ == "__main__":
    main()
