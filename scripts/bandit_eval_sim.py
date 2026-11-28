"""Generated creatives + a bandit: how much does the leader's lift overstate the truth?

Writes docs/bandit_eval_results.md and docs/bandit_eval.png. There are no model
API calls. Creative variants, a batched Thompson-sampling allocator and clicks
are simulated, so every true lift is known. Uses experiment_toolkit.bandit_eval.

Run:       python scripts/bandit_eval_sim.py   (about 33 minutes on a laptop CPU)
CI smoke:  python scripts/bandit_eval_sim.py --smoke   (few reps, writes to a temp dir)
"""

from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from experiment_toolkit.bandit_eval import (  # noqa: E402
    Readout,
    expected_winner_bias_fixed,
    naive_lift,
    ope_lift,
    pick_winner,
    simulate_bandit_log,
)

SEED = 20261127
ROOT = Path(__file__).resolve().parents[1]
P0 = 0.05
N_BATCHES = 12
N_PER_BATCH = 40_000
CONTROL_SHARE = 0.2
FLOOR = 0.2
DRIFT = 0.2
K = 20
TAU = 0.04
N_CONFIRM = 50_000
EVAL_SHARE = 0.2
PRIOR = (10.0, 190.0)  # Beta prior centred on the incumbent's click rate
SWEEP_K = (2, 5, 10, 20, 50)
SWEEP_FLOOR = (0.2, 0.05, 0.01, 0.0)
ESTIMATORS = ("naive", "ips_same", "snips_same", "dr_same", "dr_posthoc", "ips_slice",
              "dr_slice", "confirm")
LABELS = {
    "naive": "Leader's raw mean minus control's",
    "ips_same": "IPS, same log",
    "snips_same": "SNIPS, same log",
    "dr_same": "DR, same log",
    "dr_posthoc": "DR, random half of log split after the fact",
    "ips_slice": f"IPS, {EVAL_SHARE:.0%} evaluation slice",
    "dr_slice": f"DR, {EVAL_SHARE:.0%} evaluation slice",
    "confirm": f"Confirmation A/B, {N_CONFIRM // 1000}k per arm",
}
SHORT = {
    "naive": "Naive", "ips_same": "IPS\nsame log", "snips_same": "SNIPS\nsame log",
    "dr_same": "DR\nsame log", "dr_posthoc": "DR\npost-hoc\nsplit",
    "ips_slice": "IPS\neval\nslice", "dr_slice": "DR\neval\nslice", "confirm": "Confirm\nA/B",
}
PRIORS = {
    "null": "Every variant equals the control",
    "spread": f"Variant lifts ~ Normal(0, {TAU:.0%})",
}


def draw_lifts(prior, k, rng):
    return np.zeros(k) if prior == "null" else rng.normal(0.0, TAU, k)


def dr(log, rows, arm):
    a, y, p = log.arms[rows], log.rewards[rows], log.propensities[rows]
    return ope_lift(a, y, p, arm, method="dr", baseline_target=log.baseline(arm)[rows],
                    baseline_control=log.baseline(0)[rows])


def one_rep(lifts, rng, drift=DRIFT, floor=FLOOR, standard=True):
    """All estimators on one simulated test. Returns {estimator: (est, se, truth)}
    in relative-lift units (absolute lift / control click rate).

    ``standard``: also simulate a bandit that learns from every impression
    (needed for every estimator except the evaluation-slice ones)."""
    k = len(lifts)
    out: dict[str, tuple[Readout, int]] = {}
    kw = dict(control_share=CONTROL_SHARE, floor=floor, drift=drift, prior=PRIOR)
    if standard:
        log = simulate_bandit_log(lifts, P0, N_PER_BATCH, N_BATCHES, rng, **kw)
        a, y, p = log.arms, log.rewards, log.propensities
        lead = log.leader()
        every = np.ones(len(a), dtype=bool)
        out["naive"] = (naive_lift(a, y, lead), lead)
        for m in ("ips", "snips"):
            out[f"{m}_same"] = (ope_lift(a, y, p, lead, method=m), lead)
        out["dr_same"] = (dr(log, every, lead), lead)

        half = rng.random(len(a)) < 0.5
        pick = pick_winner(a[~half], y[~half], p[~half], range(1, k + 1))
        out["dr_posthoc"] = (dr(log, half, pick), pick)

        rt, rc = P0 * (1 + lifts[lead - 1]), P0
        ct = rng.binomial(N_CONFIRM, rt) / N_CONFIRM
        cc = rng.binomial(N_CONFIRM, rc) / N_CONFIRM
        se = np.sqrt(ct * (1 - ct) / N_CONFIRM + cc * (1 - cc) / N_CONFIRM)
        out["confirm"] = (Readout(ct - cc, float(se)), lead)

    log = simulate_bandit_log(lifts, P0, N_PER_BATCH, N_BATCHES, rng,
                              eval_share=EVAL_SHARE, **kw)
    lead = log.leader()
    ev = log.eval_slice
    out["ips_slice"] = (ope_lift(log.arms[ev], log.rewards[ev], log.propensities[ev], lead),
                        lead)
    out["dr_slice"] = (dr(log, ev, lead), lead)
    return {e: (r.estimate / P0, r.se / P0, float(lifts[arm - 1])) for e, (r, arm) in out.items()}


def simulate(prior, k, reps, rng, drift=DRIFT, floor=FLOOR, standard=True):
    rows: dict[str, list] = {}
    best = []
    for _ in range(reps):
        lifts = draw_lifts(prior, k, rng)
        best.append(lifts.max())
        for e, v in one_rep(lifts, rng, drift, floor, standard).items():
            rows.setdefault(e, []).append(v)
    return {e: np.array(v) for e, v in rows.items()}, float(np.mean(best))


def summarise(x):
    est, se, truth = x[:, 0], x[:, 1], x[:, 2]
    err = est - truth
    win = est - 1.96 * se > 0
    return {
        "est": est.mean(), "truth": truth.mean(), "bias": err.mean(),
        "rmse": np.sqrt((err**2).mean()), "sd": err.std(ddof=1), "se": se.mean(),
        "cover": (np.abs(err) <= 1.96 * se).mean(), "win": win.mean(),
        "false_win": (win & (truth <= 0)).mean(),
    }


def pct(v, signed=True):
    return f"{100 * v:+.1f}%" if signed else f"{100 * v:.1f}%"


def main_table(rng, reps):
    lines = [
        "| Variants' true lifts | Click-rate drift | Estimator | Mean estimate | Mean true lift "
        "of evaluated arm | Bias | RMSE | Mean SE | 95% CI coverage | Claims a win "
        "| False win |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    stats_out, bests = {}, {}
    for prior in PRIORS:
        for drift in (0.0, DRIFT):
            d, best = simulate(prior, K, reps, rng, drift=drift)
            bests[(prior, drift)] = best
            drift_lbl = "none" if drift == 0 else f"±{drift:.0%}"
            for e in ESTIMATORS:
                s = summarise(d[e])
                stats_out[(prior, drift, e)] = s
                lines.append(
                    f"| {PRIORS[prior]} | {drift_lbl} | {LABELS[e]} | {pct(s['est'])} "
                    f"| {pct(s['truth'])} | {pct(s['bias'])} | {pct(s['rmse'], False)} "
                    f"| {pct(s['se'], False)} | {s['cover']:.3f} | {s['win']:.3f} "
                    f"| {s['false_win']:.3f} |"
                )
    return lines, stats_out, bests


CURVE = ("naive", "dr_same", "dr_posthoc", "dr_slice", "confirm")


def k_sweep(rng, reps):
    fw = ("dr_same", "dr_posthoc", "dr_slice", "confirm")
    null_lines = [
        "| Variants | Fixed equal split, naive (closed form) | "
        + " | ".join(f"{LABELS[e]}: bias" for e in CURVE) + " | "
        + " | ".join(f"{LABELS[e]}: false win" for e in fw) + " |",
        "|---:|" + "---:|" * (len(CURVE) + len(fw) + 1),
    ]
    spread_lines = [
        "| Variants | Best variant's true lift | True lift of leader, bandit learns from "
        "everything | True lift of leader, bandit with evaluation slice "
        f"| {LABELS['dr_same']}: bias | {LABELS['dr_slice']}: bias "
        f"| {LABELS['dr_slice']}: coverage | {LABELS['dr_slice']}: mean SE "
        f"| {LABELS['confirm']}: mean SE |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    curve = {e: [] for e in CURVE}
    closed = []
    n_total = N_BATCHES * N_PER_BATCH
    for k in SWEEP_K:
        d, _ = simulate("null", k, reps, rng, drift=0.0)
        s = {e: summarise(d[e]) for e in ESTIMATORS}
        n_arm = int((1 - CONTROL_SHARE) * n_total / k)
        cf = expected_winner_bias_fixed(k, n_arm, P0) / P0
        closed.append(cf)
        for e in CURVE:
            curve[e].append(s[e]["bias"])
        null_lines.append(
            f"| {k} | {pct(cf)} | " + " | ".join(pct(s[e]["bias"]) for e in CURVE) + " | "
            + " | ".join(f"{s[e]['false_win']:.3f}" for e in fw) + " |"
        )
        d, best = simulate("spread", k, reps, rng, drift=0.0)
        s = {e: summarise(d[e]) for e in ESTIMATORS}
        spread_lines.append(
            f"| {k} | {pct(best)} | {pct(s['naive']['truth'])} | {pct(s['dr_slice']['truth'])} "
            f"| {pct(s['dr_same']['bias'])} | {pct(s['dr_slice']['bias'])} "
            f"| {s['dr_slice']['cover']:.3f} | {pct(s['dr_slice']['se'], False)} "
            f"| {pct(s['confirm']['se'], False)} |"
        )
    return null_lines, spread_lines, curve, closed


def floor_sweep(rng, reps):
    lines = [
        "| Uniform floor (share of variant traffic) | Smallest propensity | Estimator | Bias "
        "| Emp. SD | Mean SE | 95% CI coverage |",
        "|---:|---:|---|---:|---:|---:|---:|",
    ]
    for floor in SWEEP_FLOOR:
        d, _ = simulate("spread", K, reps, rng, drift=DRIFT, floor=floor, standard=False)
        min_p = (1 - CONTROL_SHARE) * floor / K
        min_lbl = f"{min_p:.4f}" if floor > 0 else "0 (no floor)"
        for e in ("ips_slice", "dr_slice"):
            s = summarise(d[e])
            lines.append(
                f"| {floor:.0%} | {min_lbl} | {LABELS[e]} | {pct(s['bias'])} "
                f"| {pct(s['sd'], False)} | {pct(s['se'], False)} | {s['cover']:.3f} |"
            )
    return lines


def make_figure(stats_out, curve, closed, out_png):
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.6))
    ax = axes[0]
    colors = {"naive": "#d62728", "dr_same": "#ff7f0e", "dr_posthoc": "#9467bd",
              "dr_slice": "#1f77b4", "confirm": "#2ca02c"}
    ax.plot(SWEEP_K, 100 * np.array(closed), ":", color="grey",
            label="Fixed equal split, naive (closed form)")
    for e, c in colors.items():
        ax.plot(SWEEP_K, 100 * np.array(curve[e]), "-o", color=c, label=LABELS[e])
    ax.axhline(0, color="black", lw=1)
    ax.set_xscale("log")
    ax.set_xticks(SWEEP_K, [str(k) for k in SWEEP_K])
    ax.minorticks_off()
    ax.set_xlabel("Generated variants in the test")
    ax.set_ylabel("Reported lift minus true lift (pp of relative lift)")
    ax.set_title("No variant is better than control, no drift\n"
                 "Overstatement comes from picking the leader")
    ax.legend(fontsize=7.5, loc="upper left")
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    x = np.arange(len(ESTIMATORS))
    est = [100 * stats_out[("spread", DRIFT, e)]["est"] for e in ESTIMATORS]
    tru = [100 * stats_out[("spread", DRIFT, e)]["truth"] for e in ESTIMATORS]
    ax.bar(x - 0.2, est, 0.4, color="#1f77b4", label="Mean reported lift")
    ax.bar(x + 0.2, tru, 0.4, color="#7f7f7f", label="Mean true lift of that arm")
    ax.set_xticks(x, [SHORT[e] for e in ESTIMATORS], fontsize=8)
    ax.set_ylabel("Relative lift vs control (%)")
    ax.set_title(f"{K} variants, lifts ~ Normal(0, {TAU:.0%}), click rate drifts "
                 f"±{DRIFT:.0%}\nOnly data the bandit never learned from reports the lift honestly")
    ax.legend(fontsize=8)
    ax.grid(True, axis="y", alpha=0.3)
    fig.suptitle("The bandit's leader is a good pick and a bad measurement", fontsize=12, y=1.02)
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=140, bbox_inches="tight", metadata={"Software": None})
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--smoke", action="store_true",
                        help="few replications, outputs to a temp dir (for CI)")
    args = parser.parse_args()
    reps, sweep_reps = (1_000, 400) if not args.smoke else (3, 2)
    out_dir = Path(tempfile.mkdtemp()) if args.smoke else ROOT / "docs"
    out_md, out_png = out_dir / "bandit_eval_results.md", out_dir / "bandit_eval.png"

    seeds = np.random.SeedSequence(SEED).spawn(3)
    main_lines, stats_out, bests = main_table(np.random.default_rng(seeds[0]), reps)
    null_lines, spread_lines, curve, closed = k_sweep(np.random.default_rng(seeds[1]),
                                                      sweep_reps)
    floor_lines = floor_sweep(np.random.default_rng(seeds[2]), sweep_reps)
    make_figure(stats_out, curve, closed, out_png)

    n_total = N_BATCHES * N_PER_BATCH
    md = [
        "# Generated creatives and a bandit: simulation results",
        "",
        "<!-- Generated by scripts/bandit_eval_sim.py. Do not edit by hand. -->",
        "",
        "Reproduce from a clean checkout:",
        "",
        "```bash",
        'pip install -e ".[dev]"',
        "python scripts/bandit_eval_sim.py",
        "```",
        "",
        "The full run takes about 33 minutes on a laptop CPU (`--smoke` takes seconds).",
        "",
        "## Setup",
        "",
        f"Seed `{SEED}`. No model API calls. The incumbent creative (control) has click "
        f"rate {P0:.0%}. A test runs {n_total:,} impressions in {N_BATCHES} batches of "
        f"{N_PER_BATCH:,}. The control keeps a fixed {CONTROL_SHARE:.0%} of traffic. "
        f"Batched Thompson sampling with a Beta({PRIOR[0]:.0f}, {PRIOR[1]:.0f}) prior (centred "
        "on the incumbent's rate) allocates the rest across the "
        f"generated variants, with {FLOOR:.0%} of variant traffic spread uniformly as a "
        "floor. The allocation probabilities are logged and are the exact assignment "
        "probabilities. With drift, every arm's click rate is multiplied by a factor that "
        f"rises linearly from {1 - DRIFT:.1f} to {1 + DRIFT:.1f} over the test (it averages "
        "1); the bandit assumes rates are constant. Variant lifts are relative and constant "
        "over time.",
        "",
        "The bandit's pick (the \"leader\") is the variant with the highest final posterior "
        "mean. Each replication simulates two bandits on the same variants:",
        "",
        "- **A standard bandit** that learns from every impression. Same-log estimators "
        "evaluate its leader on the log that chose it. The post-hoc split divides that log "
        "into random halves, picks the variant with the highest SNIPS value on one half, "
        "and evaluates it on the other. The confirmation A/B is a fresh 50/50 test of its "
        f"leader against control with {N_CONFIRM:,} impressions per arm, run after the "
        "bandit at the average click rate.",
        f"- **A bandit with an evaluation slice**: a random {EVAL_SHARE:.0%} of every batch "
        "is served with the same probabilities, but those clicks are never fed back to the "
        "bandit. Its leader is evaluated on the slice only. The slice uses "
        f"{EVAL_SHARE * n_total:,.0f} impressions, about the size of the confirmation A/B, "
        "but needs no extra traffic or time.",
        "",
        "DR uses the bandit's posterior mean at decision time as its outcome model.",
        "",
        "All lifts are relative to the control's true click rate. Bias is the mean of "
        "(estimate − true lift of the arm being evaluated). \"Claims a win\" means the 95% "
        "CI is entirely above zero; \"false win\" means it claims a win and the arm's true "
        f"lift is ≤ 0. Monte Carlo SE of a rate near 0.05 is about "
        f"{np.sqrt(0.05 * 0.95 / reps):.3f} in section 1 and "
        f"{np.sqrt(0.05 * 0.95 / sweep_reps):.3f} in sections 2 and 3.",
        "",
        "![Bandit evaluation simulation](bandit_eval.png)",
        "",
        f"## 1. Every estimator against the truth ({K} variants, {reps:,} replications)",
        "",
        "Mean true lift of the best variant: "
        + "; ".join(f"{PRIORS[pr]}, drift {'none' if dr == 0 else f'±{dr:.0%}'}: "
                    f"{pct(v)}" for (pr, dr), v in bests.items())
        + ".",
        "",
        *main_lines,
        "",
        f"## 2. More variants, bigger winner's curse (no drift, {sweep_reps} replications "
        "per row)",
        "",
        "### 2a. Every variant equals the control",
        "",
        "The closed-form column is the winner's-curse bias if the variant traffic were split "
        "equally and never reallocated (`expected_winner_bias_fixed`), for reference.",
        "",
        *null_lines,
        "",
        f"### 2b. Variant lifts ~ Normal(0, {TAU:.0%}): what the evaluation slice costs",
        "",
        *spread_lines,
        "",
        f"## 3. A thin propensity floor costs the evaluation slice its coverage ({K} variants, "
        f"lifts "
        f"~ Normal(0, {TAU:.0%}), drift ±{DRIFT:.0%}, {sweep_reps} replications per row)",
        "",
        *floor_lines,
        "",
    ]
    out_md.write_text("\n".join(md), encoding="utf-8", newline="\n")
    print(f"wrote {out_md}")
    print(f"wrote {out_png}")


if __name__ == "__main__":
    main()
