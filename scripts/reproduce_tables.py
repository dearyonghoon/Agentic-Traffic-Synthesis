#!/usr/bin/env python3
"""Build compact public reproduction tables from generated CSV outputs."""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def pct(n, d):
    return 100.0 * float(n) / float(d) if d else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--agent-results",
        default=str(
            ROOT / "results" / "reproduction" / "original40_agent_results.csv"
        ),
    )
    ap.add_argument(
        "--exhaustive-summary",
        default=str(
            ROOT / "results" / "reproduction" / "exhaustive697_summary.csv"
        ),
    )
    ap.add_argument(
        "--output-dir",
        default=str(ROOT / "results" / "reproduction" / "tables"),
    )
    args = ap.parse_args()

    agent_path = Path(args.agent_results)
    exhaustive_path = Path(args.exhaustive_summary)

    if not agent_path.exists():
        raise FileNotFoundError(
            f"{agent_path} not found. Run scripts/run_evaluation.py first."
        )
    if not exhaustive_path.exists():
        raise FileNotFoundError(
            f"{exhaustive_path} not found. "
            "Run scripts/reproduce_exhaustive_unsupported.py first."
        )

    agent = pd.read_csv(agent_path)
    exhaustive = pd.read_csv(exhaustive_path)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    agent_rows = []
    for ds, g in agent.groupby("dataset", sort=False):
        n = len(g)
        single = int(g["paper_single_shot_success"].astype(bool).sum())
        paper = int(g["paper_union_success"].astype(bool).sum())
        live = int(g["live_agent_success"].astype(bool).sum())
        preserved = int(g["intent_preserved"].astype(bool).sum())
        agent_rows.append(
            {
                "dataset": ds,
                "n": n,
                "single_shot_reference_n": single,
                "single_shot_reference_percent": pct(single, n),
                "paper_agent_n": paper,
                "paper_agent_percent": pct(paper, n),
                "live_agent_n": live,
                "live_agent_percent": pct(live, n),
                "intent_preserved_n": preserved,
                "intent_preserved_percent": pct(preserved, n),
            }
        )

    n = len(agent)
    single = int(agent["paper_single_shot_success"].astype(bool).sum())
    paper = int(agent["paper_union_success"].astype(bool).sum())
    live = int(agent["live_agent_success"].astype(bool).sum())
    preserved = int(agent["intent_preserved"].astype(bool).sum())
    agent_rows.append(
        {
            "dataset": "overall",
            "n": n,
            "single_shot_reference_n": single,
            "single_shot_reference_percent": pct(single, n),
            "paper_agent_n": paper,
            "paper_agent_percent": pct(paper, n),
            "live_agent_n": live,
            "live_agent_percent": pct(live, n),
            "intent_preserved_n": preserved,
            "intent_preserved_percent": pct(preserved, n),
        }
    )

    agent_summary = pd.DataFrame(agent_rows)

    exhaustive_cols = [
        "dataset",
        "n_unique",
        "profile_first",
        "copula_success_n",
        "cfm_success_n",
        "both_n",
        "copula_only_n",
        "cfm_only_n",
        "neither_n",
        "union_n",
        "copula_success",
        "cfm_success",
        "union_success",
        "second_tool_gain",
        "mean_profile_calls",
        "mean_reverse_calls",
        "relative_call_reduction",
    ]
    missing = [c for c in exhaustive_cols if c not in exhaustive.columns]
    if missing:
        raise KeyError(
            f"{exhaustive_path} missing expected columns: {missing}"
        )
    exhaustive_table = exhaustive[exhaustive_cols].copy()

    agent_csv = out_dir / "table_original40_agent_summary.csv"
    exhaustive_csv = out_dir / "table_exhaustive697_summary.csv"
    markdown_path = out_dir / "PUBLIC_REPRODUCTION_SUMMARY.md"

    agent_summary.to_csv(agent_csv, index=False)
    exhaustive_table.to_csv(exhaustive_csv, index=False)

    lines = [
        "# Public Reproduction Summary",
        "",
        "## Frozen original-40 end-to-end audit",
        "",
        "| Dataset | N | Single-shot reference | Paper agent | Live agent | Intent preserved |",
        "|---|---:|---:|---:|---:|---:|",
    ]

    for r in agent_summary.itertuples(index=False):
        lines.append(
            f"| {r.dataset} | {r.n} | "
            f"{r.single_shot_reference_n}/{r.n} "
            f"({r.single_shot_reference_percent:.2f}%) | "
            f"{r.paper_agent_n}/{r.n} ({r.paper_agent_percent:.2f}%) | "
            f"{r.live_agent_n}/{r.n} ({r.live_agent_percent:.2f}%) | "
            f"{r.intent_preserved_n}/{r.n} "
            f"({r.intent_preserved_percent:.1f}%) |"
        )

    lines.extend(
        [
            "",
            "## Complete unsupported semantic space",
            "",
            "| Dataset | N | Copula | CFM | Union | Fallback-only gain |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )

    for r in exhaustive_table.itertuples(index=False):
        lines.append(
            f"| {r.dataset} | {r.n_unique} | "
            f"{r.copula_success_n}/{r.n_unique} "
            f"({100*r.copula_success:.2f}%) | "
            f"{r.cfm_success_n}/{r.n_unique} "
            f"({100*r.cfm_success:.2f}%) | "
            f"{r.union_n}/{r.n_unique} ({100*r.union_success:.2f}%) | "
            f"{100*r.second_tool_gain:.2f} pp |"
        )

    lines.extend(
        [
            "",
            "Generated from the public reproduction scripts. "
            "Numerical-library constraints are documented in the repository README.",
            "",
        ]
    )

    markdown_path.write_text("\n".join(lines), encoding="utf-8")

    print("saved:", agent_csv)
    print("saved:", exhaustive_csv)
    print("saved:", markdown_path)


if __name__ == "__main__":
    main()
