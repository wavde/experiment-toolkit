"""Canary set: does re-judging fixed stored outputs catch silent judge drift?

Writes docs/canary_drift_results.md and docs/canary_drift.png. There are no
model API calls. Quality, judge verdicts and human labels are simulated, so the
drift day is known; the monitors are not told it. The canary monitor is
compared with loop 11's score and agreement monitors on the same simulated
tests, and closed-form detection probabilities from
experiment_toolkit.judge_drift appear next to the simulated rates.

Run:       python scripts/canary_drift_sim.py
CI smoke:  python scripts/canary_drift_sim.py --smoke   (few reps, writes to a temp dir)
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
    SilentDrift,
    agreement_series,
    canary_alarm,
    canary_detection_prob,
    canary_items_for_power,
    first_alarm,
    label_mask,
    monitor_z_crit,
    score_series,
    simulate_canary,
    simulate_silent_experiment,
)
from experiment_toolkit.llm_judge import JudgeConfusion  # noqa: E402

SEED = 20261127
ROOT = Path(__file__).resolve().parents[1]
P_CONTROL = 0.60
N_DAYS = 28
N_PER_DAY = 500
N_LABELED = 500
REF_DAYS = WINDOW = 7
START = 15
CANARY = 400
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
JUDGE_DRIFT = {"Judge drifts on both arms (step)", "Judge drifts on treatment only (step)",
               "Judge drifts on treatment only (gradual)"}
SHORT = ("No\nchange", "Judge drift\nboth arms", "Judge drift\ntreatment,\nstep",
         "Judge drift\ntreatment,\ngradual", "Quality falls,\njudge\nunchanged")
MONITORS = {
    "score": "Score monitor, both arms",
    "agree": f"Agreement monitor, {N_LABELED} labels per arm",
    "canary_c": f"Canary, {2 * CANARY} control outputs",
    "canary_tc": f"Canary, {CANARY} outputs per arm",
}
SWEEP_ITEMS = (50, 100, 200, 400, 800)
SWEEP_R = (0.0, 0.5, 0.9)


def canary_sets(drift, n_items, r, rng, control_only=False):
    if control_only:
        return [simulate_canary(drift, "c", n_items, P_CONTROL, N_DAYS, r, rng)]
    return [simulate_canary(drift, arm, n_items, P_CONTROL, N_DAYS, r, rng)
            for arm in ("t", "c")]


def run_once(drift, rng):
    t, c = simulate_silent_experiment(N_DAYS, N_PER_DAY, P_CONTROL, 0.0, drift, rng)
    lab_t, lab_c = label_mask(t, N_LABELED, rng), label_mask(c, N_LABELED, rng)
    return {
        "score": first_alarm([score_series(t), score_series(c)], N_DAYS, REF_DAYS, WINDOW),
        "agree": first_alarm(agreement_series(t, lab_t) + agreement_series(c, lab_c),
                             N_DAYS, REF_DAYS, WINDOW),
        "canary_c": canary_alarm(canary_sets(drift, 2 * CANARY, 0.0, rng, True),
                                 REF_DAYS, WINDOW),
        "canary_tc": canary_alarm(canary_sets(drift, CANARY, 0.0, rng), REF_DAYS, WINDOW),
    }


def summary(days, show_delay):
    fired = [d for d in days if d is not None]
    rate = len(fired) / len(days)
    if not show_delay or not fired:
        return rate, "—"
    return rate, f"{np.median(np.array(fired) - START + 1):.0f}"


def main_table(rng, reps):
    lines = [
        "| Scenario | " + " | ".join(f"{MONITORS[m]}: alarm rate | {MONITORS[m]}: median "
                                     "days after change" for m in MONITORS) + " |",
        "|---|" + "---:|" * (2 * len(MONITORS)),
    ]
    rates = {}
    for name, drift in SCENARIOS.items():
        alarms = {m: [] for m in MONITORS}
        for _ in range(reps):
            for m, day in run_once(drift, rng).items():
                alarms[m].append(day)
        cells = []
        for m in MONITORS:
            rate, delay = summary(alarms[m], name in JUDGE_DRIFT)
            rates[(name, m)] = rate
            cells += [f"{rate:.3f}", delay]
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
    return lines, rates


def sweep_table(rng, reps):
    drift, none = SCENARIOS["Judge drifts on treatment only (step)"], SCENARIOS["No change"]
    z2 = monitor_z_crit(N_DAYS, 2, REF_DAYS, WINDOW)
    lines = [
        "| Canary outputs per arm | Judge calls per day | "
        + " | ".join(f"Treatment-only drift, repeatability {r:.1f}: alarm rate (median days)"
                     for r in SWEEP_R)
        + " | No change, repeatability 0.0: alarm rate | No change, repeatability 0.9: "
        "alarm rate | One fully drifted week flags, repeatability 0.0 (closed form) |",
        "|---:|---:|" + "---:|" * (len(SWEEP_R) + 3),
    ]
    detect = {r: [] for r in SWEEP_R}
    for k in SWEEP_ITEMS:
        cells = []
        for r in SWEEP_R:
            days = [canary_alarm(canary_sets(drift, k, r, rng), REF_DAYS, WINDOW)
                    for _ in range(reps)]
            rate, delay = summary(days, True)
            detect[r].append(rate)
            cells.append(f"{rate:.3f} ({delay})")
        for r in (0.0, 0.9):
            days = [canary_alarm(canary_sets(none, k, r, rng), REF_DAYS, WINDOW)
                    for _ in range(reps)]
            cells.append(f"{summary(days, False)[0]:.3f}")
        cf = canary_detection_prob(BASE, LENIENT, P_CONTROL, k, z2, REF_DAYS, WINDOW)
        lines.append(f"| {k:,} | {2 * k:,} | " + " | ".join(cells) + f" | {cf:.3f} |")
    return lines, detect


def make_figure(rates, detect, out_png):
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    names = list(SCENARIOS)
    x = np.arange(len(names))
    width = 0.2
    ax = axes[0]
    colors = ("#1f77b4", "#8c564b", "#bcbd22", "#2ca02c")
    for i, (m, color) in enumerate(zip(MONITORS, colors, strict=True)):
        ax.bar(x + (i - 1.5) * width, [100 * rates[(n, m)] for n in names], width,
               color=color, label=MONITORS[m])
    ax.axhline(5, color="black", ls="--", lw=1)
    ax.set_xticks(x, SHORT, fontsize=8.5)
    ax.set_ylabel("Alarm by end of test (%)")
    ax.set_ylim(0, 105)
    ax.set_title(f"Monitors, change on day {START} of {N_DAYS}, Bonferroni 5%\n"
                 "canary judge re-runs fully independent (repeatability 0)")
    ax.legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.3), ncol=2)
    ax.grid(True, axis="y", alpha=0.3)

    ax = axes[1]
    for r, color in zip(SWEEP_R, ("#9467bd", "#17becf", "#2ca02c"), strict=True):
        ax.plot(SWEEP_ITEMS, 100 * np.array(detect[r]), "-o", color=color,
                label=f"repeatability {r:.1f}")
    ax.axhline(80, color="grey", ls=":", lw=1)
    ax.set_xscale("log")
    ax.set_xticks(SWEEP_ITEMS, [f"{k:,}" for k in SWEEP_ITEMS])
    ax.minorticks_off()
    ax.set_xlabel("Canary outputs per arm (re-judged daily)")
    ax.set_ylabel("Alarm by end of test (%)")
    ax.set_ylim(0, 105)
    ax.set_title("Canary size: judge drifts on treatment only (step)\n"
                 "repeatability = how often the judge repeats its verdict")
    ax.legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=3)
    ax.grid(True, alpha=0.3)
    fig.suptitle("A canary set of stored outputs flags judge drift without "
                 "flagging real quality change", fontsize=12, y=1.02)
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=140, bbox_inches="tight", metadata={"Software": None})
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--smoke", action="store_true",
                        help="few replications, outputs to a temp dir (for CI)")
    args = parser.parse_args()
    reps, sweep_reps = (2_000, 500) if not args.smoke else (3, 3)
    out_dir = Path(tempfile.mkdtemp()) if args.smoke else ROOT / "docs"
    out_md, out_png = out_dir / "canary_drift_results.md", out_dir / "canary_drift.png"

    seeds = np.random.SeedSequence(SEED).spawn(2)
    main_lines, rates = main_table(np.random.default_rng(seeds[0]), reps)
    sweep_lines, detect = sweep_table(np.random.default_rng(seeds[1]), sweep_reps)
    make_figure(rates, detect, out_png)
    z2 = monitor_z_crit(N_DAYS, 2, REF_DAYS, WINDOW)
    k80 = canary_items_for_power(BASE, LENIENT, P_CONTROL, z2, 0.8, REF_DAYS, WINDOW)
    cf_main = canary_detection_prob(BASE, LENIENT, P_CONTROL, CANARY, z2, REF_DAYS, WINDOW)
    n_checks = N_DAYS - REF_DAYS - WINDOW + 1

    md = [
        "# Canary-set drift monitor: simulation results",
        "",
        "<!-- Generated by scripts/canary_drift_sim.py. Do not edit by hand. -->",
        "",
        "Reproduce from a clean checkout:",
        "",
        "```bash",
        'pip install -e ".[dev]"',
        "python scripts/canary_drift_sim.py",
        "```",
        "",
        f"Seed `{SEED}`. No model API calls. The test and scenarios are the same as in "
        f"[`silent_drift_results.md`](silent_drift_results.md): {N_DAYS} days, {N_PER_DAY} "
        f"units per arm per day, control quality rate {P_CONTROL:.2f}, no true effect. The "
        f"judge starts at sensitivity = specificity = {BASE.sensitivity:.2f}; from day "
        f"{START}, with no version flag, its specificity drops to {LENIENT.specificity:.2f} "
        f"on the arms named, at once (step) or linearly until day {N_DAYS} (gradual). In the "
        f"last scenario the judge never changes but true quality falls by "
        f"{-QUALITY_SHIFT:.2f} on both arms from day {START}.",
        "",
        "A *canary set* is a fixed set of stored outputs, saved before the change, that the "
        "production judge re-scores every day. Their true quality never changes, so only a "
        "judge change can move their scores. No human labels are used. The canary monitor "
        "compares each output's mean verdict over the latest "
        f"{WINDOW} days with its own mean over the first {REF_DAYS} days, and tests the mean "
        "of those paired differences across outputs. Every monitor checks daily from day "
        f"{REF_DAYS + WINDOW} to day {N_DAYS} ({n_checks} checks) with a Bonferroni "
        "threshold over every check and series, so the chance of any false alarm is at most "
        "0.05 when the judge doesn't change. The score and agreement monitors are loop 11's: "
        "the judge pass rate on all units of each arm, and in each arm the judge pass rate "
        f"among the {N_LABELED} randomly labeled units humans passed and among those humans "
        "failed. *Repeatability* is how often the judge gives a stored output the same "
        "verdict on different days (0 = a fresh, independent verdict each day; 1 = fully "
        "deterministic). It doesn't change the judge's error rates. Section 1 uses 0, the "
        "worst case for the canary. \"Median days after change\" counts the changed days "
        f"seen when the alarm fires (day of alarm − {START - 1}). Monte Carlo SE of a rate "
        f"near 0.05 is about {np.sqrt(0.05 * 0.95 / reps):.3f}.",
        "",
        "![Canary drift simulation](canary_drift.png)",
        "",
        f"## 1. Monitors against known drift ({reps:,} replications, repeatability 0)",
        "",
        "The canary with control outputs only re-judges the same number of outputs per day "
        f"({2 * CANARY}) as the canary with {CANARY} outputs per arm. Closed form for one "
        f"check on a fully drifted week, {CANARY} treatment outputs: the canary flags with "
        f"probability {cf_main:.3f}.",
        "",
        *main_lines,
        "",
        f"## 2. Canary size ({sweep_reps} replications per cell)",
        "",
        "Judge calls per day count both arms. With fully independent re-judging, one check "
        f"on a fully drifted week flags with 80% probability at {k80:,} canary outputs per "
        "arm (closed form).",
        "",
        *sweep_lines,
        "",
    ]
    out_md.write_text("\n".join(md), encoding="utf-8", newline="\n")
    print(f"wrote {out_md}")
    print(f"wrote {out_png}")


if __name__ == "__main__":
    main()
