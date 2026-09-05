#!/usr/bin/env python3
"""Public paper-reproduction audit for frozen original-40 intents.

This script uses only public-repository assets plus the prepared traffic datasets.
It reproduces the aggregate Copula/CFM success counts reported for the frozen
unsupported-intent audit. By default, aggregate equality is the pass criterion.
Use --require-case-exact to additionally require every per-intent tool outcome
to match the frozen reference.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent.schema import canonical_intent_copy
from agent.verifier import VectorVerifier
from generators.copula import SemanticCopulaSynthesizer
from generators.residual_film_cfm import ResidualFiLMCFMSynthesizer

SEED = 2026
DEFAULT_BUDGET = 8
DATASETS = ("cesnet_ts24", "gavist5g")

QUERY_PATH = ROOT / "benchmarks" / "unsupported_intents" / "hard_intent_query_bank_v1.jsonl"
REFERENCE_PATH = ROOT / "benchmarks" / "unsupported_intents" / "original40_reference.csv"
PRETRAINED_DIR = ROOT / "pretrained"

PROCESSED_BASE = {
    "cesnet_ts24": ("cesnet_ts24", "cesnet_windows_12rows_contiguous_v2"),
    "gavist5g": ("gavist5g", "gavist_60s_windows_v2"),
}
CHECKPOINT_NAMES = {
    "cesnet_ts24": "cesnet_ts24_residual_film_cfm_best.pt",
    "gavist5g": "gavist5g_residual_film_cfm_best.pt",
}


def load_jsonl(path: Path):
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def as_bool(x):
    if isinstance(x, (bool, np.bool_)):
        return bool(x)
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return False
    if isinstance(x, (int, np.integer, float, np.floating)):
        return float(x) != 0.0
    return str(x).strip().lower() in {"1", "true", "yes", "y"}


def load_table(base: Path):
    parquet_path = Path(str(base) + ".parquet")
    csv_path = Path(str(base) + ".csv")
    if parquet_path.exists():
        try:
            return pd.read_parquet(parquet_path), parquet_path
        except Exception as e:
            print(
                f"[WARN] Could not read {parquet_path.name}: "
                f"{type(e).__name__}. Trying CSV fallback."
            )
    if csv_path.exists():
        return pd.read_csv(csv_path, low_memory=False), csv_path
    raise FileNotFoundError(
        f"Neither {parquet_path} nor {csv_path} is readable."
    )


def first_verified(candidates, verifier, intent):
    last_v = None
    last_attempt = 0
    for cand in candidates:
        last_attempt = int(cand.get("attempt", last_attempt + 1))
        v = verifier.verify(cand, intent)
        last_v = v
        if bool(v.get("verified", False)):
            return {
                "success": True,
                "attempt": last_attempt,
                "semantic_ok": bool(v.get("semantic_ok", False)),
                "plausibility_ok": bool(v.get("plausibility_ok", False)),
                "distance": float(v.get("distance", np.nan)),
                "threshold": float(v.get("threshold", np.nan)),
            }

    last_v = last_v or {}
    return {
        "success": False,
        "attempt": int(last_attempt),
        "semantic_ok": bool(last_v.get("semantic_ok", False)),
        "plausibility_ok": bool(last_v.get("plausibility_ok", False)),
        "distance": float(last_v.get("distance", np.nan)),
        "threshold": float(last_v.get("threshold", np.nan)),
    }


def build_public_pipeline(dataset, data_root, device):
    ds_dir, filename = PROCESSED_BASE[dataset]
    table_base = Path(data_root) / ds_dir / "processed" / filename
    df, loaded_path = load_table(table_base)

    if "split_refined" not in df.columns:
        raise KeyError(f"{loaded_path}: missing split_refined")

    train_df = df[df["split_refined"].astype(str) == "train"].copy()
    if train_df.empty:
        raise RuntimeError(f"{dataset}: no training rows")

    print(
        f"[{dataset}] table={loaded_path} "
        f"rows={len(df):,} train={len(train_df):,}"
    )

    verifier = VectorVerifier(dataset=dataset, seed=SEED).fit(train_df)
    copula = SemanticCopulaSynthesizer(
        dataset=dataset,
        seed=SEED,
    ).fit(train_df)

    ckpt = PRETRAINED_DIR / CHECKPOINT_NAMES[dataset]
    if not ckpt.exists():
        raise FileNotFoundError(
            f"Missing public checkpoint: {ckpt}\n"
            "Check that pretrained/ was cloned completely."
        )

    cfm = ResidualFiLMCFMSynthesizer(
        dataset=dataset,
        checkpoint_path=str(ckpt),
        device=device,
        seed=SEED,
    )
    cfm.fit_semantic_stats(train_df)
    cfm.load()
    print(f"[{dataset}] CFM checkpoint={ckpt}")
    return verifier, copula, cfm


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
            ROOT
            / "results"
            / "reproduction"
            / "original40_public_reproduction.csv"
        ),
    )
    ap.add_argument(
        "--require-case-exact",
        action="store_true",
        help="Fail unless every individual tool outcome matches the frozen reference.",
    )
    args = ap.parse_args()

    if not QUERY_PATH.exists():
        raise FileNotFoundError(QUERY_PATH)
    if not REFERENCE_PATH.exists():
        raise FileNotFoundError(REFERENCE_PATH)

    query_rows = load_jsonl(QUERY_PATH)
    query_by_id = {
        str(r["query_id"]): r["intent"]
        for r in query_rows
    }

    ref = pd.read_csv(REFERENCE_PATH, low_memory=False)
    required_ref_cols = {
        "dataset",
        "base_query_id",
        "recorded_copula",
        "recorded_cfm",
    }
    missing = required_ref_cols - set(ref.columns)
    if missing:
        raise KeyError(
            f"{REFERENCE_PATH} is missing columns: {sorted(missing)}"
        )

    print("=" * 88)
    print("PUBLIC ORIGINAL-40 PAPER REPRODUCTION")
    print("=" * 88)
    print("reference:", REFERENCE_PATH)
    print("query bank:", QUERY_PATH)
    print("pretrained:", PRETRAINED_DIR)
    print("data root:", args.data_root)
    print("budget:", args.budget)
    print("device:", args.device)
    print()

    counts = (
        ref.groupby("dataset")["base_query_id"]
        .nunique()
        .to_dict()
    )
    for ds in DATASETS:
        if int(counts.get(ds, 0)) != 40:
            raise RuntimeError(
                f"{ds}: expected 40 frozen intents, found {counts.get(ds, 0)}"
            )

    result_rows = []

    for ds in DATASETS:
        verifier, copula, cfm = build_public_pipeline(
            dataset=ds,
            data_root=args.data_root,
            device=args.device,
        )

        ds_ref = (
            ref[ref["dataset"].astype(str) == ds]
            .drop_duplicates("base_query_id")
            .copy()
        )
        ds_ref["base_query_id"] = ds_ref["base_query_id"].astype(str)
        ds_ref = ds_ref.sort_values("base_query_id")

        print(f"\n[{ds}] replaying {len(ds_ref)} frozen intents")

        for i, row in enumerate(ds_ref.itertuples(index=False), start=1):
            qid = str(row.base_query_id)
            if qid not in query_by_id:
                raise KeyError(f"Query ID not found: {qid}")

            intent = canonical_intent_copy(query_by_id[qid])

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

            recorded_cop = as_bool(row.recorded_copula)
            recorded_cfm = as_bool(row.recorded_cfm)

            result_rows.append({
                "dataset": ds,
                "base_query_id": qid,
                "intent_json": json.dumps(
                    intent, sort_keys=True, ensure_ascii=False
                ),
                "recorded_copula": recorded_cop,
                "public_copula": bool(cop["success"]),
                "copula_match": recorded_cop == bool(cop["success"]),
                "public_copula_attempt": int(cop["attempt"]),
                "public_copula_semantic_ok": bool(cop["semantic_ok"]),
                "public_copula_plausibility_ok": bool(cop["plausibility_ok"]),
                "public_copula_distance": cop["distance"],
                "public_copula_threshold": cop["threshold"],
                "recorded_cfm": recorded_cfm,
                "public_cfm": bool(cfm_result["success"]),
                "cfm_match": recorded_cfm == bool(cfm_result["success"]),
                "public_cfm_attempt": int(cfm_result["attempt"]),
                "public_cfm_semantic_ok": bool(cfm_result["semantic_ok"]),
                "public_cfm_plausibility_ok": bool(
                    cfm_result["plausibility_ok"]
                ),
                "public_cfm_distance": cfm_result["distance"],
                "public_cfm_threshold": cfm_result["threshold"],
            })

            if i % 10 == 0 or i == len(ds_ref):
                print(f"  {i:2d}/{len(ds_ref)}")

        del cfm
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    result = pd.DataFrame(result_rows)
    result["case_exact"] = result["copula_match"] & result["cfm_match"]

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output, index=False)

    print("\n" + "=" * 88)
    print("RESULT")
    print("=" * 88)

    aggregate_ok = True
    for ds, g in result.groupby("dataset"):
        rec_cop = int(g["recorded_copula"].sum())
        pub_cop = int(g["public_copula"].sum())
        rec_cfm = int(g["recorded_cfm"].sum())
        pub_cfm = int(g["public_cfm"].sum())
        ds_aggregate_ok = (rec_cop == pub_cop) and (rec_cfm == pub_cfm)
        aggregate_ok = aggregate_ok and ds_aggregate_ok

        print(
            f"{ds}: N={len(g)} | "
            f"Copula recorded/public={rec_cop}/{pub_cop} "
            f"({100*pub_cop/len(g):.1f}%) | "
            f"CFM recorded/public={rec_cfm}/{pub_cfm} "
            f"({100*pub_cfm/len(g):.1f}%) | "
            f"case-exact={int(g['case_exact'].sum())}/{len(g)} "
            f"({100*g['case_exact'].mean():.1f}%)"
        )

    exact_n = int(result["case_exact"].sum())
    print(
        f"OVERALL: paper-level aggregate reproduction="
        f"{'PASS' if aggregate_ok else 'FAIL'}; "
        f"case-exact={exact_n}/{len(result)} "
        f"({100*exact_n/len(result):.1f}%)"
    )
    print("saved:", output)

    mismatches = result[~result["case_exact"]]
    if len(mismatches):
        print("\nCASE-LEVEL MISMATCHES:")
        cols = [
            "dataset",
            "base_query_id",
            "recorded_copula",
            "public_copula",
            "recorded_cfm",
            "public_cfm",
        ]
        print(mismatches[cols].to_string(index=False))

    if not aggregate_ok:
        raise SystemExit(
            "FAIL: public implementation does not reproduce the frozen aggregate counts."
        )

    if args.require_case_exact and len(mismatches):
        raise SystemExit(
            "FAIL: aggregate counts match, but --require-case-exact was requested."
        )

    print("\nPASS: frozen paper-level aggregate outcomes are reproduced.")
    if len(mismatches):
        print(
            "NOTE: aggregate counts match exactly, while some boundary cases "
            "differ at the per-intent level; see the mismatch table above."
        )


if __name__ == "__main__":
    main()
