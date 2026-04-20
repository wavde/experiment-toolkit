"""Generate the hero chart shown in README.md.

Two panels that tell the package's story:
  (1) Sample size vs MDE — the classic power-calc curve every DS needs.
  (2) mSPRT Type-I error under H0 over peeks — the always-valid story,
      with the buggy variant overlaid to show why the fix mattered.

Run:  python scripts/make_hero.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

RNG = np.random.default_rng(7)
OUT = Path(__file__).resolve().parents[1] / "docs" / "hero.png"


def sample_size_curve(ax: plt.Axes) -> None:
    baseline = 0.10
    mdes_rel = np.linspace(0.01, 0.10, 50)
    sigma2 = baseline * (1 - baseline)
    z_alpha = stats.norm.ppf(1 - 0.025)
    z_beta = stats.norm.ppf(0.80)
    n_per_arm = 2 * sigma2 * (z_alpha + z_beta) ** 2 / (baseline * mdes_rel) ** 2

    ax.plot(mdes_rel * 100, n_per_arm / 1000, lw=2.2, color="#1f77b4")
    ax.set_xlabel("Minimum detectable effect (% relative)")
    ax.set_ylabel("Sample size per arm (thousands)")
    ax.set_title("Power calculation\nα=0.05, power=0.80, p=10%")
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3, which="both")


def msprt_type1(ax: plt.Axes) -> None:
    n_max = 2000
    tau2 = 1.0
    alpha = 0.05
    ns = np.arange(1, n_max + 1)
    trials = 500

    def simulate(use_bug: bool) -> np.ndarray:
        cum_reject = np.zeros(n_max)
        for _ in range(trials):
            x = RNG.normal(size=n_max)
            y = RNG.normal(size=n_max)
            diff = np.cumsum(x) / ns - np.cumsum(y) / ns
            V = (1.0 if use_bug else 2.0) / ns
            logbf = 0.5 * np.log(V / (V + tau2)) + 0.5 * (diff**2) * (tau2 / (V + tau2)) / V
            pval = np.minimum(1.0, 1.0 / np.exp(logbf))
            running_min = np.minimum.accumulate(pval)
            cum_reject += (running_min < alpha).astype(float)
        return cum_reject / trials

    buggy = simulate(True)
    fixed = simulate(False)

    ax.plot(ns, buggy * 100, lw=2, color="#d62728", label="Buggy variant (pre-fix)")
    ax.plot(ns, fixed * 100, lw=2, color="#2ca02c", label="Shipped formula")
    ax.axhline(5, color="black", ls="--", lw=1, alpha=0.6, label="α = 5%")
    ax.set_xlabel("Sample size per arm (peeks)")
    ax.set_ylabel("Cumulative Type-I error (%)")
    ax.set_title("mSPRT always-valid control\nType-I under H0 across all peeks")
    ax.legend(loc="center right", fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 25)


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    sample_size_curve(axes[0])
    msprt_type1(axes[1])
    fig.suptitle("experiment-toolkit — power calcs and always-valid sequential tests",
                 fontsize=12, y=1.02)
    fig.tight_layout()
    fig.savefig(OUT, dpi=140, bbox_inches="tight")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
