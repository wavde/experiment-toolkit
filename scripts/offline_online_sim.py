"""Offline LLM evals vs online A/B: when does an offline win predict the online result?

Writes docs/offline_online_results.md and docs/offline_online.png. There are no
model API calls. Prompts, outputs, judge scores, human labels and online users
are all simulated, so the true online effect is known by construction. Each
simulated number appears next to its closed-form value from
experiment_toolkit.offline_eval.

Run:        python scripts/offline_online_sim.py
CI smoke:   python scripts/offline_online_sim.py --smoke   (few reps, writes to a temp dir)
"""

from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from experiment_toolkit import (  # noqa: E402
    JudgeConfusion,
    OfflineOnlineScenario,
    expected_judged_diff,
    expected_online_effect,
    expected_quality_diff,
    offline_readouts,
    simulate_offline_eval,
    simulate_online_ab,
)

SEED = 20261127
ROOT = Path(__file__).resolve().parents[1]

SEGMENTS = ("chat", "coding", "reasoning")
W_OFF = (0.2, 0.6, 0.2)  # eval set over-represents coding
W_ON = (0.7, 0.2, 0.1)  # live traffic is mostly chat
P_A = (0.6, 0.5, 0.4)
JUDGE = JudgeConfusion(sensitivity=0.85, specificity=0.85)
LENIENT_ON_B = JudgeConfusion(sensitivity=0.85, specificity=0.80)

N_PROMPTS = 10_000
N_LABELED = 1_000
N_ONLINE = 200_000
REPS = 500
SWEEP_PROMPTS = (2_000, 5_000, 10_000, 20_000)
SWEEP_REPS = 300

METHODS = ("judge_raw", "judge_reweighted", "ppi_reweighted", "online")
LABELS = {
    "judge_raw": "Offline judge, eval mix",
    "judge_reweighted": "Offline judge, reweighted to traffic",
    "ppi_reweighted": "Offline judge + human labels (PPI++), reweighted",
    "online": "Online A/B",
}


def scenario(delta, judge_b=JUDGE, online_cost=0.0) -> OfflineOnlineScenario:
    return OfflineOnlineScenario(
        w_off=np.array(W_OFF), w_on=np.array(W_ON), p_a=np.array(P_A),
        delta=np.array(delta, dtype=float), judge_a=JUDGE, judge_b=judge_b,
        online_cost=online_cost, names=SEGMENTS,
    )


SCENARIOS = {
    "Aligned win": scenario([0.03, 0.03, 0.03]),
    "No effect": scenario([0.0, 0.0, 0.0]),
    "Mix shift (B helps coding, hurts chat)": scenario([-0.04, 0.10, 0.0]),
    "Judge lenient on B, no true effect": scenario([0.0, 0.0, 0.0], judge_b=LENIENT_ON_B),
    "Online-only cost (B adds latency)": scenario([0.03, 0.03, 0.03], online_cost=0.03),
}


def expected(method: str, sc: OfflineOnlineScenario) -> float:
    if method == "judge_raw":
        return expected_judged_diff(sc, sc.w_off)
    if method == "judge_reweighted":
        return expected_judged_diff(sc, sc.w_on)
    if method == "ppi_reweighted":
        return expected_quality_diff(sc, sc.w_on)
    return expected_online_effect(sc)


def run_scenario(sc, n_prompts, reps, rng, online=True):
    est = {m: np.empty(reps) for m in METHODS}
    dec = {m: np.empty(reps, dtype=object) for m in METHODS}
    for r in range(reps):
        data = simulate_offline_eval(sc, n_prompts, N_LABELED, rng)
        reads = offline_readouts(data, sc.w_on)
        if online:
            reads["online"] = simulate_online_ab(sc, N_ONLINE, rng)
        for m, read in reads.items():
            est[m][r] = read.estimate
            dec[m][r] = read.decision()
    return est, dec


def truth_call(effect: float) -> str:
    if effect > 0:
        return "ship"
    return "stop" if effect < 0 else "don't ship"


def signed(x: float) -> str:
    """'+.4f' without a stray '-0.0000'."""
    text = f"{x:+.4f}"
    return "+0.0000" if text == "-0.0000" else text


def main_table(rng, reps):
    lines = [
        "| Scenario | True online effect (call) | Method | Mean estimate (expected) "
        "| Ships | Stops | No decision |",
        "|---|---:|---|---:|---:|---:|---:|",
    ]
    ship_rates = {}
    for name, sc in SCENARIOS.items():
        est, dec = run_scenario(sc, N_PROMPTS, reps, rng)
        effect = expected_online_effect(sc)
        ship_rates[name] = {m: float(np.mean(dec[m] == "ship")) for m in METHODS}
        for m in METHODS:
            d = dec[m]
            lines.append(
                f"| {name} | {signed(effect)} ({truth_call(effect)}) | {LABELS[m]} "
                f"| {signed(est[m].mean())} ({signed(expected(m, sc))}) "
                f"| {np.mean(d == 'ship'):.0%} | {np.mean(d == 'stop'):.0%} "
                f"| {np.mean(d == 'none'):.0%} |"
            )
    return lines, ship_rates


def sweep_table(rng, reps):
    win, shift = SCENARIOS["Aligned win"], SCENARIOS["Mix shift (B helps coding, hurts chat)"]
    rows = {}
    for n in SWEEP_PROMPTS:
        _, d_win = run_scenario(win, n, reps, rng, online=False)
        _, d_shift = run_scenario(shift, n, reps, rng, online=False)
        rows[n] = {
            m: (float(np.mean(d_win[m] == "ship")), float(np.mean(d_shift[m] == "ship")))
            for m in METHODS[:3]
        }
    lines = [
        "| Eval prompts | Method | Ships a real win (aligned) | Ships a change that hurts "
        "traffic (mix shift) |",
        "|---:|---|---:|---:|",
    ]
    for n, by_method in rows.items():
        for m, (w, s) in by_method.items():
            lines.append(f"| {n:,} | {LABELS[m]} | {w:.0%} | {s:.0%} |")
    return lines, rows


def make_figure(ship_rates, sweep, out_png):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4))
    colors = {"judge_raw": "#d62728", "judge_reweighted": "#ff7f0e",
              "ppi_reweighted": "#2ca02c", "online": "#1f77b4"}

    ax = axes[0]
    names = list(ship_rates)
    x = np.arange(len(names))
    width = 0.2
    for i, m in enumerate(METHODS):
        ax.bar(x + (i - 1.5) * width, [ship_rates[n][m] * 100 for n in names], width,
               color=colors[m], label=LABELS[m])
    short = ["Aligned\nwin", "No\neffect", "Mix\nshift", "Judge\nlenient on B", "Online-only\ncost"]
    ax.set_xticks(x, short, fontsize=8)
    ax.set_ylabel("Ships B (%)")
    ax.set_ylim(0, 135)
    ax.set_yticks(range(0, 101, 20))
    ax.set_title("How often each readout ships B\n(only 'Aligned win' should ship)")
    ax.legend(fontsize=7, loc="upper center", ncol=2)
    ax.grid(True, axis="y", alpha=0.3)

    ax = axes[1]
    ns = list(sweep)
    for m in METHODS[:3]:
        ax.plot(ns, [sweep[n][m][0] * 100 for n in ns], "-o", color=colors[m], label=LABELS[m])
        ax.plot(ns, [sweep[n][m][1] * 100 for n in ns], "--s", color=colors[m], alpha=0.8)
    ax.set_xscale("log")
    ax.set_xticks(ns, [f"{n:,}" for n in ns])
    ax.minorticks_off()
    ax.set_xlabel("Offline eval prompts")
    ax.set_ylabel("Ships B (%)")
    ax.set_ylim(0, 105)
    ax.set_title(
        "Solid: ships a real win.  Dashed: ships a change\nthat hurts live traffic (mix shift)"
    )
    ax.legend(fontsize=7, loc="center left")
    ax.grid(True, alpha=0.3)

    fig.suptitle("Offline LLM evals vs online A/B: an offline win is evidence about the eval set",
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
    reps, sweep_reps = (REPS, SWEEP_REPS) if not args.smoke else (5, 3)
    out_dir = Path(tempfile.mkdtemp()) if args.smoke else ROOT / "docs"
    out_md, out_png = out_dir / "offline_online_results.md", out_dir / "offline_online.png"

    seeds = np.random.SeedSequence(SEED).spawn(2)
    main_lines, ship_rates = main_table(np.random.default_rng(seeds[0]), reps)
    sweep_lines, sweep = sweep_table(np.random.default_rng(seeds[1]), sweep_reps)
    make_figure(ship_rates, sweep, out_png)

    mix = " / ".join(f"{s} {a:.0%} vs {b:.0%}" for s, a, b in zip(SEGMENTS, W_OFF, W_ON,
                                                                   strict=True))
    md = [
        "# Offline LLM evals vs online A/B: simulation results",
        "",
        "<!-- Generated by scripts/offline_online_sim.py. Do not edit by hand. -->",
        "",
        "Reproduce from a clean checkout:",
        "",
        "```bash",
        'pip install -e ".[dev]"',
        "python scripts/offline_online_sim.py",
        "```",
        "",
        f"Seed `{SEED}`. No model API calls. Segment shares, eval set vs live traffic: {mix}. "
        f"Baseline quality by segment: {', '.join(f'{p:.2f}' for p in P_A)}. Judge sensitivity "
        f"= specificity = {JUDGE.sensitivity:.2f} for A's outputs; in the lenient scenario its "
        f"specificity on B's outputs is {LENIENT_ON_B.specificity:.2f}. Online success = "
        "0.1 + 0.8 x quality - online cost. A readout **ships** when its 95% CI is above 0 and "
        "**stops** when it is below 0. Values in parentheses are closed-form expectations from "
        "`experiment_toolkit.offline_eval`. The three offline estimates are in different units "
        "(judged rate, judged rate, true quality), so compare their signs and decisions with the "
        "online effect, not their sizes.",
        "",
        "![Offline vs online simulation](offline_online.png)",
        "",
        f"## 1. Which readouts ship B ({N_PROMPTS:,} eval prompts, {N_LABELED:,} human-labeled, "
        f"{N_ONLINE:,} online users per arm, {reps:,} replications)",
        "",
        *main_lines,
        "",
        "## 2. Offline eval size: power vs protection",
        "",
        f"{N_LABELED:,} human-labeled prompts at every size. {sweep_reps} replications per row. "
        "Reweighting to the traffic mix leans on the chat segment, which is only "
        f"{W_OFF[0]:.0%} of the eval set, so it needs more prompts for the same power.",
        "",
        *sweep_lines,
        "",
    ]
    out_md.write_text("\n".join(md), encoding="utf-8", newline="\n")
    print(f"wrote {out_md}")
    print(f"wrote {out_png}")


if __name__ == "__main__":
    main()
