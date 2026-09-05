#!/usr/bin/env python3
"""
Local fidelity audit for the public Agentic-Traffic-Synthesis implementation.

This script replays the frozen Experiment-18 "original 40" unsupported intents
for each vector domain (40 CESNET + 40 GAViST = 80 intents total) using the
PUBLIC generator/verifier classes and compares tool success/failure outcomes
against the frozen reference CSV produced by the research environment.

This is a local release-validation script. It intentionally reads the original
research results/checkpoints from /data/code/ttt and should not be committed
until the referenced benchmark/checkpoint assets are packaged for public use.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from agent.schema import canonical_intent_copy
from agent.verifier import VectorVerifier
from generators.copula import SemanticCopulaSynthesizer
from generators.residual_film_cfm import ResidualFiLMCFMSynthesizer


SEED = 2026
DEFAULT_BUDGET = 8

DATASETS = ("cesnet_ts24", "gavist5g")

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


def build_public_pipeline(dataset, data_root, research_root, device):
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

    ckpt = (
        Path(research_root)
        / "results"
        / "residual_film_cfm_08b"
        / "checkpoints"
        / CHECKPOINT_NAMES[dataset]
    )
    if not ckpt.exists():
        raise FileNotFoundError(ckpt)

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
        "--research-root",
        default="/data/code/ttt",
        help="Original frozen research environment.",
    )
    ap.add_argument(
        "--data-root",
        default="/data/dataset/AgenticTraffic",
        help="Prepared public dataset root.",
    )
    ap.add_argument(
        "--budget",
        type=int,
        default=DEFAULT_BUDGET,
    )
    ap.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    ap.add_argument(
        "--output",
        default="results/local_validation/public_original40_replication.csv",
    )
    args = ap.parse_args()

    research_root = Path(args.research_root)

    query_path = (
        research_root
        / "results"
        / "hard_intent_benchmark"
        / "hard_intent_query_bank_v1.jsonl"
    )
    reference_path = (
        research_root
        / "results"
        / "exhaustive_unsupported_18"
        / "18_original40_replication.csv"
    )

    if not query_path.exists():
        raise FileNotFoundError(query_path)
    if not reference_path.exists():
        raise FileNotFoundError(reference_path)

    query_rows = load_jsonl(query_path)
    query_by_id = {
        str(r["query_id"]): r["intent"]
        for r in query_rows
    }

    ref = pd.read_csv(reference_path, low_memory=False)

    required_ref_cols = {
        "dataset",
        "base_query_id",
        "recorded_copula",
        "recorded_cfm",
    }
    missing = required_ref_cols - set(ref.columns)
    if missing:
        raise KeyError(
            f"{reference_path} is missing columns: {sorted(missing)}"
        )

    print("=" * 88)
    print("PUBLIC ORIGINAL-40 FIDELITY AUDIT")
    print("=" * 88)
    print("reference:", reference_path)
    print("query bank:", query_path)
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
                f"{ds}: expected 40 frozen intents, "
                f"found {counts.get(ds, 0)}"
            )

    result_rows = []

    for ds in DATASETS:
        verifier, copula, cfm = build_public_pipeline(
            dataset=ds,
            data_root=args.data_root,
            research_root=args.research_root,
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
                copula.generate_candidates(
                    intent,
                    budget=args.budget,
                ),
                verifier,
                intent,
            )

            cfm_result = first_verified(
                cfm.generate_candidates(
                    intent,
                    budget=args.budget,
                ),
                verifier,
                intent,
            )

            recorded_cop = as_bool(row.recorded_copula)
            recorded_cfm = as_bool(row.recorded_cfm)

            result_rows.append({
                "dataset": ds,
                "base_query_id": qid,
                "intent_json": json.dumps(
                    intent,
                    sort_keys=True,
                    ensure_ascii=False,
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

        # Free the GPU model before constructing the next domain.
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

    for ds, g in result.groupby("dataset"):
        print(
            f"{ds}: N={len(g)}, "
            f"Copula={int(g['copula_match'].sum())}/{len(g)} "
            f"({100*g['copula_match'].mean():.1f}%), "
            f"CFM={int(g['cfm_match'].sum())}/{len(g)} "
            f"({100*g['cfm_match'].mean():.1f}%), "
            f"case-exact={int(g['case_exact'].sum())}/{len(g)} "
            f"({100*g['case_exact'].mean():.1f}%)"
        )

    print(
        f"OVERALL: intents={len(result)}, "
        f"case-exact={int(result['case_exact'].sum())}/{len(result)}, "
        f"tool-outcome matches="
        f"{int(result['copula_match'].sum()+result['cfm_match'].sum())}"
        f"/{2*len(result)}"
    )
    print("saved:", output)

    mismatches = result[~result["case_exact"]]
    if len(mismatches):
        print("\nMISMATCHES:")
        cols = [
            "dataset",
            "base_query_id",
            "recorded_copula",
            "public_copula",
            "recorded_cfm",
            "public_cfm",
        ]
        print(mismatches[cols].to_string(index=False))
        raise SystemExit(1)

    print("\nPASS: public implementation reproduces all frozen original-40")
    print("      outcomes for both domains (80 intents total).")


if __name__ == "__main__":
    main()
