"""Minimal CLI for experiment-toolkit.

Example usage:
    experiment-toolkit sample-size --mde 0.02 --sd 1.0
    experiment-toolkit mde --n 10000 --sd 1.0
"""

from __future__ import annotations

import argparse
import sys

from experiment_toolkit.sample_size import mde_for_n, sample_size_for_mde


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="experiment-toolkit")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_ss = sub.add_parser("sample-size", help="compute required per-arm sample size")
    p_ss.add_argument("--mde", type=float, required=True, help="minimum detectable effect")
    p_ss.add_argument("--sd", type=float, required=True, help="outcome standard deviation")
    p_ss.add_argument("--alpha", type=float, default=0.05)
    p_ss.add_argument("--power", type=float, default=0.80)

    p_mde = sub.add_parser("mde", help="compute detectable effect given sample size")
    p_mde.add_argument("--n", type=int, required=True, help="per-arm sample size")
    p_mde.add_argument("--sd", type=float, required=True)
    p_mde.add_argument("--alpha", type=float, default=0.05)
    p_mde.add_argument("--power", type=float, default=0.80)

    args = parser.parse_args(argv)

    if args.cmd == "sample-size":
        n = sample_size_for_mde(args.mde, args.sd, args.alpha, args.power)
        print(f"Required per-arm sample size: {n:,}")
    elif args.cmd == "mde":
        mde = mde_for_n(args.n, args.sd, args.alpha, args.power)
        print(f"Detectable effect (MDE): {mde:.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
