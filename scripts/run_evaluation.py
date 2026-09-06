#!/usr/bin/env python3
"""Reproduce the frozen original-40 end-to-end agent audit.

This script evaluates the public intent-preserving agent on the 40 frozen
unsupported intents from each vector domain (80 total). Requests are supplied
as structured canonical intents so that this audit isolates execution,
verification, routing, fallback, and safe failure rather than LLM compilation.

The frozen paper reference is derived from original40_reference.csv:
  - single-shot registry: 17/80 = 21.25%
  - intent-preserving two-tool agent: 21/80 = 26.25%
  - CESNET union: 14/40 = 35.0%
  - GAViST5G union: 7/40 = 17.5%

Use the reproduction constraints documented in README.md for the closest
numerical match to the frozen experiments.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent.agent import AgenticTrafficSynthesis
from agent.schema import canonical_intent_copy, intents_exact
from agent.tool_registry import DEFAULT_TOOL_ORDER
from scripts.verify_public_original40 import (
    DATASETS,
    QUERY_PATH,
    REFERENCE_PATH,
    build_public_pipeline,
    load_jsonl,
    as_bool,
)

DEFAULT_BUDGET = 8


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--data-root",
        default="/data/dataset/AgenticTraffic",
        help="Prepared dataset root.",
    )
    ap.add_argument("--budget", type=int, default=DEFAULT_BUDGET)
    ap.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    ap.add_argument(
        "--output",
        default=str(
            ROOT / "results" / "reproduction" / "original40_agent_results.csv"
        ),
    )
    ap.add_argument(
        "--require-paper-aggregate",
        action="store_true",
        help=(
            "Fail unless per-dataset live agent success counts equal the frozen "
            "paper union counts."
        ),
    )
    args = ap.parse_args()

    query_rows = load_jsonl(QUERY_PATH)
    query_by_id = {
        str(r["query_id"]): canonical_intent_copy(r["intent"])
        for r in query_rows
    }

    ref = pd.read_csv(REFERENCE_PATH, low_memory=False)
    required = {"dataset", "base_query_id", "recorded_copula", "recorded_cfm"}
    missing = required - set(ref.columns)
    if missing:
        raise KeyError(f"{REFERENCE_PATH} missing columns: {sorted(missing)}")

    tools_by_dataset = {}
    verifiers = {}

    print("=" * 88)
    print("PUBLIC ORIGINAL-40 END-TO-END AGENT REPRODUCTION")
    print("=" * 88)
    print("data root:", args.data_root)
    print("budget:", args.budget)
    print("device:", args.device)

    for ds in DATASETS:
        verifier, copula, cfm = build_public_pipeline(
            dataset=ds,
            data_root=args.data_root,
            device=args.device,
        )
        verifiers[ds] = verifier
        tools_by_dataset[ds] = {"copula": copula, "cfm": cfm}

    agent = AgenticTrafficSynthesis(
        tools_by_dataset=tools_by_dataset,
        verifiers=verifiers,
        attempt_budget=args.budget,
    )

    rows = []
    for ds in DATASETS:
        ds_ref = (
            ref[ref["dataset"].astype(str) == ds]
            .drop_duplicates("base_query_id")
            .copy()
        )
        ds_ref["base_query_id"] = ds_ref["base_query_id"].astype(str)
        ds_ref = ds_ref.sort_values("base_query_id")

        if len(ds_ref) != 40:
            raise RuntimeError(f"{ds}: expected 40 frozen intents, got {len(ds_ref)}")

        primary = DEFAULT_TOOL_ORDER[ds][0]
        print(f"\n[{ds}] primary={primary}; N={len(ds_ref)}")

        for i, row in enumerate(ds_ref.itertuples(index=False), start=1):
            qid = str(row.base_query_id)
            if qid not in query_by_id:
                raise KeyError(f"Query ID not found in query bank: {qid}")

            intent = canonical_intent_copy(query_by_id[qid])
            recorded = {
                "copula": as_bool(row.recorded_copula),
                "cfm": as_bool(row.recorded_cfm),
            }
            paper_single = bool(recorded[primary])
            paper_union = bool(recorded["copula"] or recorded["cfm"])

            result = agent.run(intent)
            live_verified = result.get("status") == "verified"
            final_intent = result.get("final_intent", result.get("intent", intent))
            preserved = intents_exact(intent, final_intent)

            output = result.get("output") or {}
            verification = output.get("verification") or {}

            rows.append(
                {
                    "dataset": ds,
                    "base_query_id": qid,
                    "intent_json": json.dumps(
                        intent, sort_keys=True, ensure_ascii=False
                    ),
                    "primary_tool": primary,
                    "paper_single_shot_success": paper_single,
                    "paper_union_success": paper_union,
                    "live_agent_success": bool(live_verified),
                    "paper_union_match": bool(live_verified) == paper_union,
                    "intent_preserved": bool(preserved),
                    "status": result.get("status"),
                    "action": result.get("action"),
                    "final_tool": result.get("final_tool"),
                    "tool_calls": int(result.get("tool_calls", 0) or 0),
                    "failed_tool_calls": int(
                        result.get("failed_tool_calls", 0) or 0
                    ),
                    "route_trace": json.dumps(
                        result.get("route_trace", []), ensure_ascii=False
                    ),
                    "accepted_attempt": output.get("attempt"),
                    "semantic_ok": verification.get("semantic_ok"),
                    "plausibility_ok": verification.get("plausibility_ok"),
                    "plausibility_distance": verification.get("distance"),
                    "plausibility_threshold": verification.get("threshold"),
                }
            )

            if i % 10 == 0 or i == len(ds_ref):
                print(f"  {i:2d}/{len(ds_ref)}")

    out = pd.DataFrame(rows)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output, index=False)

    print("\n" + "=" * 88)
    print("SUMMARY")
    print("=" * 88)

    paper_aggregate_ok = True
    for ds, g in out.groupby("dataset", sort=False):
        n = len(g)
        single = int(g["paper_single_shot_success"].sum())
        paper_union = int(g["paper_union_success"].sum())
        live = int(g["live_agent_success"].sum())
        preserved = int(g["intent_preserved"].sum())
        matches = int(g["paper_union_match"].sum())
        paper_aggregate_ok = paper_aggregate_ok and (live == paper_union)

        print(
            f"{ds}: N={n} | "
            f"single-shot reference={single}/{n} ({100*single/n:.1f}%) | "
            f"paper union={paper_union}/{n} ({100*paper_union/n:.1f}%) | "
            f"live agent={live}/{n} ({100*live/n:.1f}%) | "
            f"intent-preserved={preserved}/{n} | "
            f"case-match={matches}/{n}"
        )

    n = len(out)
    single = int(out["paper_single_shot_success"].sum())
    paper_union = int(out["paper_union_success"].sum())
    live = int(out["live_agent_success"].sum())
    preserved = int(out["intent_preserved"].sum())

    print(
        f"OVERALL: single-shot reference={single}/{n} ({100*single/n:.2f}%) | "
        f"paper agent={paper_union}/{n} ({100*paper_union/n:.2f}%) | "
        f"live agent={live}/{n} ({100*live/n:.2f}%) | "
        f"intent-preserved={preserved}/{n} ({100*preserved/n:.1f}%) | "
        f"paper aggregate={'PASS' if paper_aggregate_ok else 'DIFF'}"
    )
    print("saved:", output)

    if preserved != n:
        raise SystemExit("FAIL: intent-preservation invariant was violated.")

    if args.require_paper_aggregate and not paper_aggregate_ok:
        raise SystemExit(
            "FAIL: live agent aggregate differs from the frozen paper reference."
        )


if __name__ == "__main__":
    main()
