"""Silent judge drift: a judge that changes mid-test with no version flag.

Writes docs/silent_drift_results.md and docs/silent_drift.png. There are no
model API calls. Quality, judge scores and human labels are simulated, so the
true effect and the drift day are known; the analyses and monitors are not told
either. Simulated means appear next to closed-form values from
experiment_toolkit.judge_drift.

Run:       python scripts/silent_drift_sim.py
CI smoke:  python scripts/silent_drift_sim.py --smoke   (few reps, writes to a temp dir)
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

from experiment_toolkit.judge_drift import (  # noqa: E402
    SILENT_ANALYSES,
    SilentDrift,
    agreement_series,
    analyze_silent,
    expected_ppi_launch,
    expected_silent_naive,
    first_alarm,
    label_mask,
    monitor_z_crit,
    score_series,
    simulate_silent_experiment,
    window_detection_prob,
)
from experiment_toolkit.llm_judge import JudgeConfusion  # noqa: E402

SEED = 20261127
ROOT = Path(__file__).resolve().parents[1]
P_CONTROL = 0.60
EFFECT = 0.03
N_DAYS = 28
N_PER_DAY = 500
N_LABELED = 500
LAUNCH_DAYS = 7
REF_DAYS = WINDOW = 7
START = 15
BASE = JudgeConfusion(0.85, 0.85)
LENIENT = JudgeConfusion(0.85, 0.75)
QUALITY_SHIFT = -0.08
SCENARIOS = {
    "No change": SilentDrift(BASE, BASE, BASE, START),
    "Judge drifts on both arms (step)": SilentDrift(BASE, LENIENT, LENIENT, START),
    "Judge drifts on treatment only (step)": SilentDrift(BASE, LENIENT, BASE, START),
    "Judge drifts on treatment only (gradual)": SilentDrift(
        BASE, LENIENT, BASE, START, ramp_days=N_DAYS - START + 1
    ),
    "True quality falls, judge unchanged": SilentDrift(
        BASE, BASE, BASE, START, quality_shift=QUALITY_SHIFT
    ),
}
SHORT = ("No\nchange", "Judge drift\nboth arms", "Judge drift\ntreatment,\nstep",
         "Judge drift\ntreatment,\ngradual", "Quality falls,\njudge\nunchanged")
LABELS = {
    "naive_pooled": "Judge only",
    "ppi_launch": "PPI++, labels from launch week",
    "ppi_spread": "PPI++, labels across whole test",
}
MONITORS = {
    "score_c": "Score monitor, control arm",
    "score_tc": "Score monitor, both arms",
    "agree": "Agreement monitor, both arms",
}
SWEEP_LABELS = (500, 1_000, 2_000, 4_000, 8_000)


def run_once(drift, effect, n_lab, rng, monitors_only=False):
    t, c = simulate_silent_experiment(N_DAYS, N_PER_DAY, P_CONTROL, effect, drift, rng)
    spread_t, spread_c = label_mask(t, n_lab, rng), label_mask(c, n_lab, rng)
    alarms = {
        "score_c": first_alarm([score_series(c)], N_DAYS, REF_DAYS, WINDOW),
        "score_tc": first_alarm([score_series(t), score_series(c)], N_DAYS, REF_DAYS, WINDOW),
        "agree": first_alarm(agreement_series(t, spread_t) + agreement_series(c, spread_c),
                             N_DAYS, REF_DAYS, WINDOW),
    }
    if monitors_only:
        return None, alarms
    launch_t = label_mask(t, n_lab, rng, LAUNCH_DAYS)
    launch_c = label_mask(c, n_lab, rng, LAUNCH_DAYS)
    return analyze_silent(t, c, spread_t, spread_c, launch_t, launch_c), alarms


def alarm_summary(days, has_change):
    fired = [d for d in days if d is not None]
    rate = len(fired) / len(days)
    if not has_change or not fired:
        return rate, "—"
    return rate, f"{np.median(np.array(fired) - START + 1):.0f}"


def main_tables(rng, reps):
    read = [
        "| Scenario | True effect | Analysis | Mean estimate (expected) | Emp. SD | Mean SE "
        "| Coverage | Rejects 0 |",
        "|---|---:|---|---:|---:|---:|---:|---:|",
    ]
    mon = [
        "| Scenario | True effect | "
        + " | ".join(f"{MONITORS[m]}: alarm rate | {MONITORS[m]}: median days after "
                     "change" for m in MONITORS) + " |",
        "|---|---:|" + "---:|" * (2 * len(MONITORS)),
    ]
    type1, alarm0 = {}, {}
    for name, drift in SCENARIOS.items():
        has_change = name != "No change"
        for effect in (0.0, EFFECT):
            est = {a: [] for a in SILENT_ANALYSES}
            alarms = {m: [] for m in MONITORS}
            for _ in range(reps):
                res, al = run_once(drift, effect, N_LABELED, rng)
                for a in SILENT_ANALYSES:
                    r = res[a]
                    est[a].append((r.estimate, r.se, r.covers(effect), r.rejects()))
                for m in MONITORS:
                    alarms[m].append(al[m])
            for a in SILENT_ANALYSES:
                x = np.array(est[a])
                if a == "naive_pooled":
                    exp = expected_silent_naive(drift, P_CONTROL, effect, N_DAYS)
                elif a == "ppi_launch":
                    exp = expected_ppi_launch(drift, P_CONTROL, effect, N_DAYS, N_PER_DAY,
                                              N_LABELED, LAUNCH_DAYS)
                else:
                    exp = effect
                read.append(
                    f"| {name} | {effect:+.2f} | {LABELS[a]} | {x[:, 0].mean():+.4f} "
                    f"({exp:+.4f}) | {x[:, 0].std(ddof=1):.4f} | {x[:, 1].mean():.4f} "
                    f"| {x[:, 2].mean():.3f} | {x[:, 3].mean():.3f} |"
                )
                if effect == 0.0:
                    type1[(name, a)] = x[:, 3].mean()
            cells = []
            for m in MONITORS:
                rate, delay = alarm_summary(alarms[m], has_change)
                cells += [f"{rate:.3f}", delay]
                if effect == 0.0:
                    alarm0[(name, m)] = rate
            mon.append(f"| {name} | {effect:+.2f} | " + " | ".join(cells) + " |")
    return read, mon, type1, alarm0


def sweep_table(rng, reps):
    drift = SCENARIOS["Judge drifts on treatment only (step)"]
    quality = SCENARIOS["True quality falls, judge unchanged"]
    z4 = monitor_z_crit(N_DAYS, 4, REF_DAYS, WINDOW)
    fpr0, fpr1 = 1 - BASE.specificity, 1 - LENIENT.specificity
    lines = [
        "| Labels per arm | Alarm rate, judge drifts on treatment only | Median days after "
        "change | One fully drifted week flags (closed form) | Alarm rate, quality falls with "
        "judge unchanged |",
        "|---:|---:|---:|---:|---:|",
    ]
    for n_lab in SWEEP_LABELS:
        hit = [run_once(drift, 0.0, n_lab, rng, True)[1]["agree"] for _ in range(reps)]
        false = [run_once(quality, 0.0, n_lab, rng, True)[1]["agree"] for _ in range(reps)]
        rate, delay = alarm_summary(hit, True)
        n_fail_week = n_lab * WINDOW / N_DAYS * (1 - P_CONTROL)
        exp = window_detection_prob(fpr0, fpr0 * (1 - fpr0), n_fail_week, fpr1,
                                    fpr1 * (1 - fpr1), n_fail_week, z4)
        f_rate = alarm_summary(false, True)[0]
        lines.append(f"| {n_lab:,} | {rate:.3f} | {delay} | {exp:.3f} | {f_rate:.3f} |")
    return lines


def score_closed_forms():
    """One-check detection chance for a fully drifted week, score monitors."""
    n_week = N_PER_DAY * WINDOW
    q0, q1 = BASE.positive_rate(P_CONTROL), LENIENT.positive_rate(P_CONTROL)
    out = {}
    for m, n_series in (("score_c", 1), ("score_tc", 2)):
        z = monitor_z_crit(N_DAYS, n_series, REF_DAYS, WINDOW)
        out[m] = window_detection_prob(q0, q0 * (1 - q0), n_week, q1, q1 * (1 - q1), n_week, z)
    return q0, q1, out


def burst_labels(power=0.8, alpha=0.05):
    """Labels per arm in each of two weeks for a one-off two-proportion test of
    the judge's false-positive rate (reference week vs flagged week)."""
    fpr0, fpr1 = 1 - BASE.specificity, 1 - LENIENT.specificity
    z = stats.norm.ppf(1 - alpha / 2) + stats.norm.ppf(power)
    n_fail = z**2 * (fpr0 * (1 - fpr0) + fpr1 * (1 - fpr1)) / (fpr1 - fpr0) ** 2
    return int(np.ceil(n_fail / (1 - P_CONTROL)))


def make_figure(type1, alarm0, out_png):
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))
    names = list(SCENARIOS)
    x = np.arange(len(names))
    width = 0.26
    ax = axes[0]
    for i, (a, color) in enumerate(zip(SILENT_ANALYSES, ("#d62728", "#9467bd", "#2ca02c"),
                                       strict=True)):
        ax.bar(x + (i - 1) * width, [100 * type1[(n, a)] for n in names], width,
               color=color, label=LABELS[a])
    ax.axhline(5, color="black", ls="--", lw=1)
    ax.set_xticks(x, SHORT, fontsize=8.5)
    ax.set_ylabel("Type-I error (%)")
    ax.set_ylim(0, 100)
    ax.set_title(f"Readout: no true effect, change on day {START} of {N_DAYS}\n"
                 f"{N_DAYS * N_PER_DAY:,} units and {N_LABELED} labels per arm")
    ax.legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.3), ncol=2)
    ax.grid(True, axis="y", alpha=0.3)

    ax = axes[1]
    for i, (m, color) in enumerate(zip(MONITORS, ("#ff7f0e", "#1f77b4", "#8c564b"),
                                       strict=True)):
        ax.bar(x + (i - 1) * width, [100 * alarm0[(n, m)] for n in names], width,
               color=color, label=MONITORS[m])
    ax.axhline(5, color="black", ls="--", lw=1)
    ax.set_xticks(x, SHORT, fontsize=8.5)
    ax.set_ylabel("Alarm by end of test (%)")
    ax.set_ylim(0, 100)
    ax.set_title(f"Monitors: weekly window vs launch week, Bonferroni 5%\n"
                 f"agreement monitor uses the same {N_LABELED} labels per arm")
    ax.legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.3), ncol=2)
    ax.grid(True, axis="y", alpha=0.3)
    fig.suptitle("Silent judge drift: random labels keep the readout valid; "
                 "score monitors can't tell drift from real change", fontsize=12, y=1.02)
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
    out_md, out_png = out_dir / "silent_drift_results.md", out_dir / "silent_drift.png"

    seeds = np.random.SeedSequence(SEED).spawn(2)
    read, mon, type1, alarm0 = main_tables(np.random.default_rng(seeds[0]), reps)
    sweep = sweep_table(np.random.default_rng(seeds[1]), sweep_reps)
    make_figure(type1, alarm0, out_png)
    q0, q1, score_cf = score_closed_forms()
    n_checks = N_DAYS - REF_DAYS - WINDOW + 1

    md = [
        "# Silent judge drift: simulation results",
        "",
        "<!-- Generated by scripts/silent_drift_sim.py. Do not edit by hand. -->",
        "",
        "Reproduce from a clean checkout:",
        "",
        "```bash",
        'pip install -e ".[dev]"',
        "python scripts/silent_drift_sim.py",
        "```",
        "",
        f"Seed `{SEED}`. No model API calls. A {N_DAYS}-day test with {N_PER_DAY} units per "
        f"arm per day ({N_DAYS * N_PER_DAY:,} per arm). True quality is binary with control "
        f"rate {P_CONTROL:.2f}; the true effect is +{EFFECT:.2f} or 0. The judge starts at "
        f"sensitivity = specificity = {BASE.sensitivity:.2f} on both arms. From day {START}, "
        "with no version flag, the judge's specificity drops to "
        f"{LENIENT.specificity:.2f} (more lenient) on the arms named in the scenario, either "
        f"at once (step) or linearly until day {N_DAYS} (gradual). In the last scenario the "
        f"judge never changes but true quality falls by {-QUALITY_SHIFT:.2f} on both arms from "
        f"day {START}, so the true effect is unchanged. Human labels: {N_LABELED} per arm, "
        f"drawn either from the first {LAUNCH_DAYS} days (launch week) or at random from the "
        "whole test, which can be done at the end from logged outputs. Values in parentheses "
        "are closed-form answers from `experiment_toolkit.judge_drift` (large-sample for "
        "launch-week labels); PPI++ with labels across the whole test is unbiased by "
        f"construction. Monte Carlo SE of a rate near 0.05 is about "
        f"{np.sqrt(0.05 * 0.95 / reps):.3f}.",
        "",
        "Monitors compare each day's latest "
        f"{WINDOW}-day window with the first {REF_DAYS} days, from day {REF_DAYS + WINDOW} to "
        f"day {N_DAYS} ({n_checks} checks), with a two-sample z-test whose threshold is "
        "Bonferroni-corrected over every check and series, so the chance of any false alarm "
        "is at most 0.05 when nothing changes. The *score monitor* tracks the judge's pass "
        "rate on all units (control arm only, or each arm). The *agreement monitor* tracks, "
        "in each arm, the judge's pass rate among labeled units humans passed and among "
        "those humans failed (sensitivity and false-positive rate), using the labels spread "
        "across the test. \"Median days after change\" counts the changed days seen when "
        f"the alarm fires (day of alarm − {START - 1}).",
        "",
        "![Silent drift simulation](silent_drift.png)",
        "",
        f"## 1. Readouts against the truth ({reps:,} replications)",
        "",
        *read,
        "",
        f"## 2. Monitors ({reps:,} replications, same runs as section 1)",
        "",
        "Closed form for one check on a fully changed week: the judge's pass rate on control "
        f"moves from {q0:.3f} to {q1:.3f} when it turns lenient, so a single check flags with "
        f"probability {score_cf['score_c']:.3f} (control-arm monitor) or "
        f"{score_cf['score_tc']:.3f} (both-arm monitor, stricter threshold).",
        "",
        *mon,
        "",
        f"## 3. Labels the agreement monitor needs ({sweep_reps} replications per row, no "
        "true effect)",
        "",
        "Labels are drawn at random across the whole test, as in section 1. To confirm a "
        "score alarm instead with a one-off label burst (launch week vs flagged week, "
        "two-sided 5% test of the judge's false-positive rate, 80% power) needs about "
        f"{burst_labels():,} labels per arm in each of the two weeks (closed form).",
        "",
        *sweep,
        "",
    ]
    out_md.write_text("\n".join(md), encoding="utf-8", newline="\n")
    print(f"wrote {out_md}")
    print(f"wrote {out_png}")


if __name__ == "__main__":
    main()
