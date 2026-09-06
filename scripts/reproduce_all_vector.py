#!/usr/bin/env python3
"""Run the public vector-domain reproduction pipeline end to end."""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def run(cmd):
    print("\n" + "=" * 88)
    print("RUN:", " ".join(cmd))
    print("=" * 88)
    subprocess.run(cmd, cwd=ROOT, check=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--data-root",
        default="/data/dataset/AgenticTraffic",
        help="Prepared dataset root.",
    )
    ap.add_argument("--budget", type=int, default=8)
    ap.add_argument("--device", default=None)
    ap.add_argument(
        "--require-paper-aggregate",
        action="store_true",
        help="Require end-to-end and exhaustive live aggregates to match paper.",
    )
    args = ap.parse_args()

    py = sys.executable
    common = ["--data-root", args.data_root, "--budget", str(args.budget)]
    if args.device:
        common += ["--device", args.device]

    run([py, "scripts/verify_public_original40.py", *common])

    cmd = [py, "scripts/run_evaluation.py", *common]
    if args.require_paper_aggregate:
        cmd.append("--require-paper-aggregate")
    run(cmd)

    cmd = [py, "scripts/reproduce_exhaustive_unsupported.py", *common]
    if args.require_paper_aggregate:
        cmd.append("--require-paper-aggregate")
    run(cmd)

    run([py, "scripts/reproduce_tables.py"])

    print("\nPublic vector-domain reproduction pipeline completed.")
    print("Outputs: results/reproduction/")


if __name__ == "__main__":
    main()
