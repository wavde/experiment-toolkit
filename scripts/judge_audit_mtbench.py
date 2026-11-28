"""Real-data audit of an LLM judge: MT-Bench human vs GPT-4 judgments.

Writes docs/judge_audit_results.md and docs/judge_audit.png. There are no model
API calls. The data are the public MT-Bench human judgments (Zheng et al.,
2023, CC BY 4.0), downloaded at runtime from a pinned Hugging Face revision and
checked against their SHA-256. Expert humans and GPT-4 judged the same pairwise
comparisons, so the judge's error is measured against humans item by item.

Run:       pip install -e ".[dev,data]" && python scripts/judge_audit_mtbench.py
CI smoke:  python scripts/judge_audit_mtbench.py --smoke   (synthetic data with the real
           schema, no download, writes to a temp dir)
"""

from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from experiment_toolkit.judge_audit import (  # noqa: E402
    MT_BENCH_MODELS,
    MT_BENCH_REVISION,
    RESAMPLE_ANALYSES,
    bias_with_ci,
    directional_agreement,
    judge_human_correlation,
    load_mt_bench,
    pair_items,
    ppi_variance_ratio,
    resample_check,
)
from experiment_toolkit.llm_judge import JudgeConfusion  # noqa: E402

SEED = 20261127
ROOT = Path(__file__).resolve().parents[1]
N_UNITS = 2_000
LABEL_BUDGETS = (100, 200, 400)
MAIN_LABELS = 200
LOOP1_JUDGE = JudgeConfusion(0.85, 0.85)
LOOP1_P = 0.60
LABELS = {
    "judge_only": "GPT-4 judge only",
    "human_only": "Human labels only",
    "ppi": "PPI++ (judge + human labels)",
}
SHORT = {"gpt-4": "GPT-4", "claude-v1": "Claude-v1", "gpt-3.5-turbo": "GPT-3.5",
         "vicuna-13b-v1.2": "Vicuna-13B", "alpaca-13b": "Alpaca-13B", "llama-13b": "LLaMA-13B"}


def synthetic_mt_bench(rng: np.random.Generator) -> dict[str, list[dict]]:
    """Small fake data with the real schema, for the CI smoke run."""
    human, judge = [], []
    strength = {m: 1.0 - 0.15 * i for i, m in enumerate(MT_BENCH_MODELS)}
    for i, a in enumerate(MT_BENCH_MODELS):
        for b in MT_BENCH_MODELS[i + 1:]:
            for q in range(81, 111):
                for turn in (1, 2):
                    p_a = 0.5 + 0.3 * (strength[a] - strength[b])
                    row = {"question_id": q, "model_a": b, "model_b": a, "turn": turn}
                    for r in range(int(rng.integers(1, 3))):
                        u = rng.random()
                        w = "model_b" if u < p_a - 0.1 else "tie" if u < p_a + 0.1 else "model_a"
                        human.append({**row, "winner": w, "judge": f"author_{r}"})
                    u = rng.random()
                    w = "model_b" if u < p_a + 0.05 else "tie (inconsistent)" if u < p_a + 0.15 \
                        else "model_a"
                    judge.append({**row, "winner": w, "judge": "gpt4_pair"})
    return {"human": human, "gpt4_pair": judge}


def name(pair):
    return f"{SHORT[pair[0]]} vs {SHORT[pair[1]]}"


def bias_table(pairs, rng, n_boot):
    lines = [
        "| Pair (first = treatment) | Items | Human win rate | GPT-4 win rate | Bias (95% CI) "
        "| Judge–human correlation |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    out = {}
    for pair, it in pairs.items():
        bias, lo, hi = bias_with_ci(it, n_boot, rng)
        rho = float(np.corrcoef(it.y, it.s)[0, 1])
        out[pair] = (bias, lo, hi, rho)
        lines.append(f"| {name(pair)} | {len(it.y)} | {it.y.mean():.3f} | {it.s.mean():.3f} "
                     f"| {bias:+.3f} ({lo:+.3f}, {hi:+.3f}) | {rho:.2f} |")
    return lines, out


def agreement_table(pairs):
    groups = {
        "All pairs": list(pairs.values()),
        "Pairs with GPT-4 as a contestant": [v for k, v in pairs.items() if "gpt-4" in k],
        "Pairs without GPT-4": [v for k, v in pairs.items() if "gpt-4" not in k],
    }
    lines = [
        "| Pairs | Humans prefer the stronger model: items | GPT-4 agrees | Humans prefer "
        "the weaker model: items | GPT-4 agrees |",
        "|---|---:|---:|---:|---:|",
    ]
    out = {}
    for label, items in groups.items():
        a = directional_agreement(items)
        out[label] = a
        lines.append(f"| {label} | {a['n_first']:,} | {a['agree_first']:.3f} | "
                     f"{a['n_second']:,} | {a['agree_second']:.3f} |")
    return lines, out


def resample_tables(pairs, rng, reps):
    detail = [
        "| Pair | True win rate | "
        + " | ".join(f"{LABELS[a]}: coverage" for a in RESAMPLE_ANALYSES)
        + " | " + " | ".join(f"{LABELS[a]}: CI above 0.5" for a in RESAMPLE_ANALYSES) + " |",
        "|---|---:|" + "---:|" * (2 * len(RESAMPLE_ANALYSES)),
    ]
    sweep = [
        "| Human labels per pair | " + " | ".join(f"{LABELS[a]}: coverage, range over pairs"
                                                for a in RESAMPLE_ANALYSES)
        + " | PPI++ SE / human-only SE, range (closed form) | Labels PPI++ saves, range |",
        "|---:|" + "---:|" * (len(RESAMPLE_ANALYSES) + 2),
    ]
    saved_main = {}
    for k in LABEL_BUDGETS:
        cov = {a: [] for a in RESAMPLE_ANALYSES}
        ratio, cf, saved = [], [], []
        for pair, it in pairs.items():
            r = resample_check(it, N_UNITS, k, reps, rng)
            truth = float(it.y.mean())
            for a in RESAMPLE_ANALYSES:
                cov[a].append(np.mean([x.covers(truth) for x in r[a]]))
            rho = float(np.corrcoef(it.y, it.s)[0, 1])
            se_ppi = np.mean([x.se for x in r["ppi"]])
            ratio.append(se_ppi / np.mean([x.se for x in r["human_only"]]))
            cf.append(np.sqrt(ppi_variance_ratio(rho, k, N_UNITS)))
            saved.append(1 - ratio[-1] ** 2)
            if k == MAIN_LABELS:
                saved_main[pair] = saved[-1]
                detail.append(
                    f"| {name(pair)} | {truth:.3f} | "
                    + " | ".join(f"{cov[a][-1]:.3f}" for a in RESAMPLE_ANALYSES) + " | "
                    + " | ".join(f"{np.mean([x.ships_first() for x in r[a]]):.3f}"
                                 for a in RESAMPLE_ANALYSES) + " |")
        sweep.append(
            f"| {k} | " + " | ".join(f"{min(cov[a]):.3f}–{max(cov[a]):.3f}"
                                     for a in RESAMPLE_ANALYSES)
            + f" | {min(ratio):.3f}–{max(ratio):.3f} ({min(cf):.3f}–{max(cf):.3f}) "
            f"| {100 * min(saved):.0f}%–{100 * max(saved):.0f}% |")
    return detail, sweep, saved_main


def make_figure(pairs, bias, out_png):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))
    ax = axes[0]
    names = [name(p) for p in pairs]
    yy = np.arange(len(names))[::-1]
    h = [it.y.mean() for it in pairs.values()]
    j = [it.s.mean() for it in pairs.values()]
    for y0, a, b in zip(yy, h, j, strict=True):
        ax.plot([a, b], [y0, y0], color="grey", lw=1.5, zorder=1)
    ax.scatter(h, yy, color="#1f77b4", label="Expert humans", zorder=2)
    ax.scatter(j, yy, color="#d62728", label="GPT-4 judge", zorder=2)
    ax.axvline(0.5, color="black", ls="--", lw=1)
    ax.set_yticks(yy, names, fontsize=8)
    ax.set_xlabel("Win rate of the stronger model (ties count half)")
    ax.set_xlim(0.4, 1.0)
    ax.set_title("Same items, same pairs: GPT-4 inflates the stronger model's win rate")
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(True, axis="x", alpha=0.3)

    ax = axes[1]
    rho_grid = np.linspace(0, 0.8, 81)
    n_lab = MAIN_LABELS
    ax.plot(rho_grid, [100 * (1 - ppi_variance_ratio(r, n_lab, N_UNITS)) for r in rho_grid],
            color="black", lw=1, label="Closed form, 1 − ρ²·N/(N+n)")
    rhos = [bias[p][3] for p in pairs]
    ax.scatter(rhos, [100 * (1 - ppi_variance_ratio(r, n_lab, N_UNITS)) for r in rhos],
               color="#2ca02c", zorder=3, label="MT-Bench pairs (GPT-4 judge)")
    r1 = judge_human_correlation(LOOP1_JUDGE, LOOP1_P)
    ax.scatter([r1], [100 * (1 - ppi_variance_ratio(r1, n_lab, N_UNITS))], marker="*", s=180,
               color="#9467bd", zorder=3, label="Loop 1's simulated judge")
    ax.set_xlabel("Correlation between judge and human verdicts")
    ax.set_ylabel("Labels PPI++ saves vs human labels alone (%)")
    ax.set_xlim(0, 0.8)
    ax.set_ylim(0, 60)
    ax.set_title(f"The real judge saves few labels\n{n_lab} labels out of {N_UNITS:,} prompts")
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=140, bbox_inches="tight", metadata={"Software": None})
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--smoke", action="store_true",
                        help="synthetic data with the real schema, few reps, temp dir (for CI)")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data" / "mt_bench",
                        help="where the MT-Bench parquet files are cached")
    args = parser.parse_args()
    seeds = np.random.SeedSequence(SEED).spawn(3)
    if args.smoke:
        data = synthetic_mt_bench(np.random.default_rng(seeds[2]))
        reps, n_boot, out_dir = 3, 50, Path(tempfile.mkdtemp())
    else:
        data = load_mt_bench(args.data_dir)
        reps, n_boot, out_dir = 1_000, 2_000, ROOT / "docs"
    out_md, out_png = out_dir / "judge_audit_results.md", out_dir / "judge_audit.png"

    pairs = pair_items(data["human"], data["gpt4_pair"])
    bias_lines, bias = bias_table(pairs, np.random.default_rng(seeds[0]), n_boot)
    agree_lines, _ = agreement_table(pairs)
    detail, sweep, _ = resample_tables(pairs, np.random.default_rng(seeds[1]), reps)
    make_figure(pairs, bias, out_png)

    n_items = sum(len(it.y) for it in pairs.values())
    positive = sum(b[0] > 0 for b in bias.values())
    excl = sum(b[1] > 0 for b in bias.values())
    gpt4 = [b[0] for p, b in bias.items() if "gpt-4" in p]
    other = [b[0] for p, b in bias.items() if "gpt-4" not in p]
    rhos = [b[3] for b in bias.values()]
    r1 = judge_human_correlation(LOOP1_JUDGE, LOOP1_P)
    save1 = 1 - ppi_variance_ratio(r1, MAIN_LABELS, N_UNITS)
    save_med = 1 - ppi_variance_ratio(float(np.median(rhos)), MAIN_LABELS, N_UNITS)

    md = [
        "# Real-data judge audit: MT-Bench human vs GPT-4 judgments",
        "",
        "<!-- Generated by scripts/judge_audit_mtbench.py. Do not edit by hand. -->",
        "",
        "Reproduce from a clean checkout (downloads about 1.4 MB on first run):",
        "",
        "```bash",
        'pip install -e ".[dev,data]"',
        "python scripts/judge_audit_mtbench.py",
        "```",
        "",
        "**Data.** MT-Bench human judgments, Zheng et al. (2023), *Judging LLM-as-a-judge "
        "with MT-Bench and Chatbot Arena*, released under CC BY 4.0 at "
        "[lmsys/mt_bench_human_judgments](https://huggingface.co/datasets/lmsys/"
        f"mt_bench_human_judgments) (revision `{MT_BENCH_REVISION[:7]}`). It is downloaded at "
        "runtime and checked against pinned SHA-256 hashes, not committed. Expert annotators "
        "(mostly graduate students in each question's field) and GPT-4 (pairwise prompt, "
        "run with both answer orders; an inconsistent pair counts as a tie) compared answers "
        "from six 2023 models to 80 two-turn MT-Bench questions. No model API calls are made "
        "here.",
        "",
        f"**Setup.** Seed `{SEED}`. Each pair of models is treated as a side-by-side test "
        "with the stronger model (by the paper's human ranking) as treatment. An item is one "
        "question and turn for one pair. Per item, the human verdict is 1 if the stronger "
        "model wins, 0.5 for a tie and 0 if it loses, averaged when several experts judged "
        "it; the GPT-4 verdict is on the same scale. Only items both judged are used: "
        f"{n_items:,} items over {len(pairs)} pairs. The target is the human win rate.",
        "",
        "![Judge audit](judge_audit.png)",
        "",
        f"## 1. GPT-4 versus experts on the same items ({n_boot:,} bootstrap draws of "
        "questions per pair)",
        "",
        f"GPT-4's win rate for the stronger model is above the experts' in {positive} of "
        f"{len(pairs)} pairs, and the 95% CI excludes zero in {excl}. Mean bias is "
        f"{np.mean(gpt4):+.3f} in the {len(gpt4)} pairs where GPT-4 is a contestant and "
        f"{np.mean(other):+.3f} in the other {len(other)}.",
        "",
        *bias_lines,
        "",
        "## 2. The judge's error depends on which model the humans preferred",
        "",
        "Items with a strict human preference (average verdict above or below 0.5). "
        "\"GPT-4 agrees\" is the share where GPT-4 strictly prefers the same model.",
        "",
        *agree_lines,
        "",
        f"## 3. Known-answer check on the real verdicts ({reps:,} replications per cell)",
        "",
        f"Each replication draws {N_UNITS:,} items with replacement from one pair's real "
        "(human, GPT-4) verdicts, so the true win rate is the mean human verdict exactly. "
        "Every item has a GPT-4 verdict; a random subset gets its human verdict as a label. "
        "Coverage is the share of 95% CIs containing the truth. \"CI above 0.5\" is the "
        "share that would ship the stronger model.",
        "",
        f"### {MAIN_LABELS} human labels per pair",
        "",
        *detail,
        "",
        "### By label budget",
        "",
        *sweep,
        "",
        "## 4. The real judge is less informative than loop 1's simulated judge",
        "",
        f"Judge–human correlation across pairs: {min(rhos):.2f} to {max(rhos):.2f} (median "
        f"{np.median(rhos):.2f}). Loop 1's simulated judge (sensitivity = specificity = "
        f"{LOOP1_JUDGE.sensitivity:.2f}, quality rate {LOOP1_P:.2f}) has correlation "
        f"{r1:.2f}. With {MAIN_LABELS} labels out of {N_UNITS:,} prompts, PPI++ saves "
        f"{100 * save1:.0f}% of the labels at loop 1's correlation and {100 * save_med:.0f}% "
        "at the median MT-Bench correlation (closed form).",
        "",
    ]
    out_md.write_text("\n".join(md), encoding="utf-8", newline="\n")
    print(f"wrote {out_md}")
    print(f"wrote {out_png}")


if __name__ == "__main__":
    main()
