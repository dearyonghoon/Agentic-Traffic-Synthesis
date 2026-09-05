#!/usr/bin/env python3
"""Audit the prepared public datasets against the frozen paper protocol."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


EXPECTED_GAVIST_WINDOWS = 11068
EXPECTED_GAVIST_SOURCE_FILES = 122

EXPECTED_CESNET_WINDOWS = 280239
EXPECTED_CESNET_SPLITS = {
    "train": 220909,
    "val": 30519,
    "test": 28811,
}

EXPECTED_MAWI_TRACES = {
    "202401011400": "train",
    "202403011400": "train",
    "202405011400": "train",
    "202407011400": "train",
    "202409011400": "val",
    "202411011400": "test",
}


def load_table(base):
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
        f"Neither {base}.parquet nor {base}.csv is readable."
    )


def check(condition, label, details=""):
    prefix = "PASS" if condition else "FAIL"
    print(f"[{prefix}] {label}" + (f" — {details}" if details else ""))
    return bool(condition)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--data-root",
        default="/data/dataset/AgenticTraffic",
    )
    ap.add_argument(
        "--strict-counts",
        action="store_true",
        help="Require exact frozen GAViST/CESNET counts.",
    )
    args = ap.parse_args()
    root = Path(args.data_root)

    ok = True

    # GAViST5G
    gav, gav_path = load_table(
        root / "gavist5g" / "processed" / "gavist_60s_windows_v2"
    )
    ok &= check(
        set(gav["application_canonical"].dropna().astype(str).unique())
        == {"Crave", "LOL", "Netflix", "PrimeVideo", "TFT", "VAL", "YouTube"},
        "GAViST seven canonical applications",
    )
    ok &= check(
        set(gav["split_refined"].dropna().astype(str).unique())
        == {"train", "val", "test"},
        "GAViST train/val/test split",
    )
    if args.strict_counts:
        ok &= check(
            len(gav) == EXPECTED_GAVIST_WINDOWS,
            "GAViST frozen window count",
            f"{len(gav)} vs {EXPECTED_GAVIST_WINDOWS}",
        )
        ok &= check(
            gav["relative_file"].nunique() == EXPECTED_GAVIST_SOURCE_FILES,
            "GAViST frozen source-file count",
            f"{gav['relative_file'].nunique()} vs {EXPECTED_GAVIST_SOURCE_FILES}",
        )

    # CESNET
    ces, ces_path = load_table(
        root / "cesnet_ts24" / "processed" / "cesnet_windows_12rows_contiguous_v2"
    )
    ok &= check(
        bool(ces["is_contiguous"].astype(bool).all()),
        "CESNET strict contiguous windows",
    )
    ok &= check(
        set(ces["split_refined"].dropna().astype(str).unique())
        == {"train", "val", "test"},
        "CESNET train/val/test split",
    )
    if args.strict_counts:
        ok &= check(
            len(ces) == EXPECTED_CESNET_WINDOWS,
            "CESNET frozen window count",
            f"{len(ces)} vs {EXPECTED_CESNET_WINDOWS}",
        )
        got_splits = {
            str(k): int(v) for k, v in ces["split_refined"].value_counts().items()
        }
        ok &= check(
            got_splits == EXPECTED_CESNET_SPLITS,
            "CESNET frozen split counts",
            f"{got_splits}",
        )

    # MAWI
    mawi_npz = root / "mawi" / "processed" / "mawi_temporal_30s_stride5_v1.npz"
    mawi_meta = root / "mawi" / "metadata" / "mawi_temporal_preparation_v1.json"
    ok &= check(mawi_npz.exists(), "MAWI temporal NPZ exists")
    ok &= check(mawi_meta.exists(), "MAWI preparation metadata exists")

    if mawi_npz.exists():
        obj = np.load(mawi_npz, allow_pickle=False)
        n = len(obj["trace_id"])
        ok &= check(
            obj["bytes_seq"].shape == (n, 30)
            and obj["packets_seq"].shape == (n, 30),
            "MAWI 30-s sequence shape",
            f"bytes={obj['bytes_seq'].shape}, packets={obj['packets_seq'].shape}",
        )
        trace_split = {}
        for tid, split in zip(obj["trace_id"].astype(str), obj["split"].astype(str)):
            trace_split.setdefault(tid, split)
            if trace_split[tid] != split:
                ok = False
        ok &= check(
            trace_split == EXPECTED_MAWI_TRACES,
            "MAWI frozen trace split",
            f"{trace_split}",
        )

    print("\nPrepared files:")
    print(" GAViST:", gav_path)
    print(" CESNET:", ces_path)
    print(" MAWI  :", mawi_npz)

    if not ok:
        raise SystemExit(1)

    print("\nDATASET SETUP: PASS")


if __name__ == "__main__":
    main()
