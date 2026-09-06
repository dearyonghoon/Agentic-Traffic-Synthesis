#!/usr/bin/env python3
"""Reproduce Experiment 18: complete unsupported semantic space (697 intents).

For each vector dataset, this script reconstructs the exact unsupported semantic
space as the complement of semantic combinations observed in the frozen train
split:

  - CESNET-TimeSeries24: 3^6 total combinations - 448 observed = 281 unsupported
  - GAViST5G: 7 applications * 3^4 total combinations - 151 observed
              = 416 unsupported

Each unsupported intent is evaluated with the frozen Copula and Residual
FiLM-CFM generators using the same verifier and attempt budget. The script
writes case-level results and the aggregate routing summary used in the paper.

Use the reproduction constraints documented in README.md for the closest
numerical match to the frozen experiments.
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent.schema import (
    CESNET_KEYS,
    GAVIST_APPS,
    GAVIST_KEYS,
    LEVELS,
    canonical_intent_copy,
)
from agent.verifier import field_map
from scripts.verify_public_original40 import (
    PROCESSED_BASE,
    build_public_pipeline,
    first_verified,
    load_table,
)

DATASETS = ("cesnet_ts24", "gavist5g")
PROFILE_FIRST = {"cesnet_ts24": "cfm", "gavist5g": "copula"}
REFERENCE_PATH = (
    ROOT
    / "benchmarks"
    / "unsupported_intents"
    / "exhaustive697_reference.csv"
)


def _supported_set(train_df: pd.DataFrame, dataset: str):
    fmap = field_map(dataset)

    if dataset == "cesnet_ts24":
        cols = [fmap[k]["label_col"] for k in CESNET_KEYS]
    elif dataset == "gavist5g":
        cols = ["application_canonical"] + [
            fmap[k]["label_col"] for k in GAVIST_KEYS
        ]
    else:
        raise ValueError(dataset)

    return {
        tuple(str(row[c]) for c in cols)
        for _, row in train_df[cols].drop_duplicates().iterrows()
    }


def build_unsupported_intents(train_df: pd.DataFrame, dataset: str):
    supported = _supported_set(train_df, dataset)
    intents = []

    if dataset == "cesnet_ts24":
        for levels in itertools.product(LEVELS, repeat=len(CESNET_KEYS)):
            if tuple(levels) in supported:
                continue
            intent = {"dataset": dataset}
            intent.update(dict(zip(CESNET_KEYS, levels)))
            intents.append(canonical_intent_copy(intent))

    elif dataset == "gavist5g":
        for app in GAVIST_APPS:
            for levels in itertools.product(LEVELS, repeat=len(GAVIST_KEYS)):
                if (app,) + tuple(levels) in supported:
                    continue
                intent = {"dataset": dataset, "application": app}
                intent.update(dict(zip(GAVIST_KEYS, levels)))
                intents.append(canonical_intent_copy(intent))
    else:
        raise ValueError(dataset)

    return intents


def summarize(dataset: str, case_df: pd.DataFrame):
    n = len(case_df)
    cop = case_df["copula_success"].astype(bool)
    cfm = case_df["cfm_success"].astype(bool)

    both = cop & cfm
    cop_only = cop & ~cfm
    cfm_only = ~cop & cfm
    neither = ~cop & ~cfm
    union = cop | cfm

    first = PROFILE_FIRST[dataset]
    if first == "cfm":
        first_success = cfm
        second_gain = cop_only
        reverse_first = cop
    else:
        first_success = cop
        second_gain = cfm_only
        reverse_first = cfm

    mean_profile_calls = float((1 + (~first_success).astype(int)).mean())
    mean_reverse_calls = float((1 + (~reverse_first).astype(int)).mean())
    relative_call_reduction = (
        (mean_reverse_calls - mean_profile_calls) / mean_reverse_calls
    )

    return {
        "dataset": dataset,
        "n_unique": n,
        "profile_first": first.upper(),
        "copula_success_n": int(cop.sum()),
        "cfm_success_n": int(cfm.sum()),
        "both_n": int(both.sum()),
        "copula_only_n": int(cop_only.sum()),
        "cfm_only_n": int(cfm_only.sum()),
        "neither_n": int(neither.sum()),
        "union_n": int(union.sum()),
        "copula_success": float(cop.mean()),
        "cfm_success": float(cfm.mean()),
        "both_rate": float(both.mean()),
        "copula_only_rate": float(cop_only.mean()),
        "cfm_only_rate": float(cfm_only.mean()),
        "neither_rate": float(neither.mean()),
        "union_success": float(union.mean()),
        "profile_first_success": float(first_success.mean()),
        "second_tool_gain": float(second_gain.mean()),
        "mean_profile_calls": mean_profile_calls,
        "mean_reverse_calls": mean_reverse_calls,
        "relative_call_reduction": float(relative_call_reduction),
        "orchestration_regret_vs_union": 0.0,
    }


def compare_reference(summary_df: pd.DataFrame, reference_df: pd.DataFrame):
    count_cols = [
        "n_unique",
        "copula_success_n",
        "cfm_success_n",
        "both_n",
        "copula_only_n",
        "cfm_only_n",
        "neither_n",
        "union_n",
    ]
    refs = reference_df.set_index("dataset")
    checks = {}

    for row in summary_df.itertuples(index=False):
        if row.dataset not in refs.index:
            checks[row.dataset] = False
            continue
        ref = refs.loc[row.dataset]
        ok = str(row.profile_first).lower() == str(ref["profile_first"]).lower()
        for col in count_cols:
            ok = ok and int(getattr(row, col)) == int(ref[col])
        checks[row.dataset] = bool(ok)

    return checks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--data-root",
        default="/data/dataset/AgenticTraffic",
        help="Prepared dataset root.",
    )
    ap.add_argument("--budget", type=int, default=8)
    ap.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    ap.add_argument(
        "--output-dir",
        default=str(ROOT / "results" / "reproduction"),
    )
    ap.add_argument(
        "--require-paper-aggregate",
        action="store_true",
        help="Fail unless frozen Experiment-18 aggregate counts match exactly.",
    )
    args = ap.parse_args()

    if not REFERENCE_PATH.exists():
        raise FileNotFoundError(REFERENCE_PATH)
    reference_df = pd.read_csv(REFERENCE_PATH)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_case_rows = []
    all_intent_rows = []
    summaries = []

    print("=" * 88)
    print("PUBLIC EXPERIMENT-18 EXHAUSTIVE UNSUPPORTED-SPACE REPRODUCTION")
    print("=" * 88)
    print("reference:", REFERENCE_PATH)
    print("data root:", args.data_root)
    print("budget:", args.budget)
    print("device:", args.device)

    for ds in DATASETS:
        ds_dir, filename = PROCESSED_BASE[ds]
        base = Path(args.data_root) / ds_dir / "processed" / filename
        df, loaded_path = load_table(base)
        if "split_refined" not in df.columns:
            raise KeyError(f"{loaded_path}: missing split_refined")
        train_df = df[df["split_refined"].astype(str) == "train"].copy()
        if train_df.empty:
            raise RuntimeError(f"{ds}: no train rows")

        intents = build_unsupported_intents(train_df, ds)
        expected_n = int(
            reference_df.loc[
                reference_df["dataset"].astype(str) == ds, "n_unique"
            ].iloc[0]
        )
        if len(intents) != expected_n:
            raise RuntimeError(
                f"{ds}: reconstructed {len(intents)} unsupported intents; "
                f"expected {expected_n}. Check dataset preparation/split."
            )

        print(
            f"\n[{ds}] table={loaded_path} train={len(train_df):,} "
            f"unsupported={len(intents)}"
        )

        verifier, copula, cfm = build_public_pipeline(
            dataset=ds,
            data_root=args.data_root,
            device=args.device,
        )

        ds_case_rows = []
        for i, intent in enumerate(intents):
            uid = f"{ds}_unsupported_{i:03d}"

            cop = first_verified(
                copula.generate_candidates(intent, budget=args.budget),
                verifier,
                intent,
            )
            cfm_result = first_verified(
                cfm.generate_candidates(intent, budget=args.budget),
                verifier,
                intent,
            )

            intent_row = {
                "dataset": ds,
                "unsupported_id": uid,
                "intent_json": json.dumps(
                    intent, sort_keys=True, ensure_ascii=False
                ),
            }
            all_intent_rows.append(intent_row)

            case = {
                **intent_row,
                "copula_success": bool(cop["success"]),
                "copula_attempt": int(cop["attempt"]),
                "copula_semantic_ok": bool(cop["semantic_ok"]),
                "copula_plausibility_ok": bool(cop["plausibility_ok"]),
                "copula_distance": cop["distance"],
                "copula_threshold": cop["threshold"],
                "cfm_success": bool(cfm_result["success"]),
                "cfm_attempt": int(cfm_result["attempt"]),
                "cfm_semantic_ok": bool(cfm_result["semantic_ok"]),
                "cfm_plausibility_ok": bool(cfm_result["plausibility_ok"]),
                "cfm_distance": cfm_result["distance"],
                "cfm_threshold": cfm_result["threshold"],
            }
            ds_case_rows.append(case)
            all_case_rows.append(case)

            done = i + 1
            if done % 25 == 0 or done == len(intents):
                print(f"  {done:3d}/{len(intents)}")

        summaries.append(summarize(ds, pd.DataFrame(ds_case_rows)))

        del cfm
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    case_df = pd.DataFrame(all_case_rows)
    intents_df = pd.DataFrame(all_intent_rows)
    summary_df = pd.DataFrame(summaries)

    intents_path = output_dir / "exhaustive697_intents.csv"
    case_path = output_dir / "exhaustive697_case_results.csv"
    summary_path = output_dir / "exhaustive697_summary.csv"
    intents_df.to_csv(intents_path, index=False)
    case_df.to_csv(case_path, index=False)
    summary_df.to_csv(summary_path, index=False)

    checks = compare_reference(summary_df, reference_df)
    aggregate_ok = all(checks.values())

    print("\n" + "=" * 88)
    print("SUMMARY")
    print("=" * 88)
    for row in summary_df.itertuples(index=False):
        match = checks[row.dataset]
        print(
            f"{row.dataset}: N={row.n_unique} | "
            f"Copula={row.copula_success_n}/{row.n_unique} "
            f"({100*row.copula_success:.2f}%) | "
            f"CFM={row.cfm_success_n}/{row.n_unique} "
            f"({100*row.cfm_success:.2f}%) | "
            f"union={row.union_n}/{row.n_unique} "
            f"({100*row.union_success:.2f}%) | "
            f"fallback-only gain={100*row.second_tool_gain:.2f} pp | "
            f"paper aggregate={'PASS' if match else 'DIFF'}"
        )

    print(
        f"OVERALL: N={len(case_df)} | "
        f"paper aggregate={'PASS' if aggregate_ok else 'DIFF'}"
    )
    print("saved:", intents_path)
    print("saved:", case_path)
    print("saved:", summary_path)

    if len(case_df) != int(reference_df["n_unique"].sum()):
        raise SystemExit(
            f"FAIL: expected {int(reference_df['n_unique'].sum())} intents, "
            f"got {len(case_df)}"
        )

    if args.require_paper_aggregate and not aggregate_ok:
        raise SystemExit(
            "FAIL: live exhaustive aggregate differs from the frozen paper reference."
        )


if __name__ == "__main__":
    main()
