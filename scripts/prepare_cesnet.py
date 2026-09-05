#!/usr/bin/env python3
"""
Prepare the frozen CESNET-TimeSeries24 subset used by the paper.

The script downloads the public Zenodo sample archive, verifies the frozen MD5,
extracts strict non-overlapping 12-row windows, rejects windows whose integer
time index is not consecutive, applies the original source-file group split,
and fits train-only semantic thresholds.

To make the split portable while reproducing the original experiment, hashing
uses the original frozen /data/dataset/AgenticTraffic source-file identifier.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import tarfile
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd


SEED = 2026

CESNET_URL = (
    "https://zenodo.org/records/13382427/files/"
    "ip_addresses_sample.tar.gz?download=1"
)
EXPECTED_MD5 = "08451dab2d1eddb29a79467b03289232"

ROWS_PER_WINDOW = 12
STRIDE_ROWS = 12

FROZEN_DATA_ROOT = Path("/data/dataset/AgenticTraffic")
FROZEN_CESNET_RAW = FROZEN_DATA_ROOT / "cesnet_ts24" / "raw"

CORE_COLS = [
    "n_flows",
    "n_packets",
    "n_bytes",
    "n_dest_asn",
    "n_dest_ports",
    "n_dest_ip",
    "tcp_udp_ratio_packets",
    "tcp_udp_ratio_bytes",
    "dir_ratio_packets",
    "dir_ratio_bytes",
    "avg_duration",
    "avg_ttl",
]

THRESHOLD_FEATURES = [
    "n_bytes_mean",
    "n_packets_mean",
    "n_flows_mean",
    "n_bytes_cv",
    "n_packets_cv",
    "avg_duration_mean",
]


def md5sum(path, chunk_size=8 * 1024 * 1024):
    h = hashlib.md5()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def download_with_progress(url, dst):
    dst = Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)

    def report(block_num, block_size, total_size):
        downloaded = block_num * block_size
        if total_size > 0:
            pct = min(100.0, downloaded * 100 / total_size)
            print(
                f"\rDownloading: {pct:6.2f}% "
                f"({downloaded/1024**2:.1f}/{total_size/1024**2:.1f} MB)",
                end="",
                flush=True,
            )

    urllib.request.urlretrieve(url, dst, reporthook=report)
    print()


def stable_hash(text):
    return hashlib.sha256(str(text).encode("utf-8")).hexdigest()


def unit_hash(text):
    return int(stable_hash(text)[:16], 16) / float(16**16)


def frozen_source_id(relative_file):
    return str(FROZEN_CESNET_RAW / relative_file)


def split_for_relative_file(relative_file):
    u = unit_hash(frozen_source_id(relative_file))
    if u < 0.80:
        return "train"
    if u < 0.90:
        return "val"
    return "test"


def coeff_var(x):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return np.nan
    m = x.mean()
    if m == 0:
        return 0.0
    return float(x.std(ddof=0) / abs(m))


def safe_div(a, b):
    if b is None or not np.isfinite(b) or b == 0:
        return np.nan
    return a / b


def detect_time_column(df):
    if "id_time" in df.columns:
        return "id_time"

    for c in df.columns:
        cl = c.lower()
        if "time" in cl or "timestamp" in cl:
            num = pd.to_numeric(df[c], errors="coerce")
            if num.notna().mean() >= 0.95:
                return c
    return None


def is_consecutive_integer_window(values):
    x = pd.to_numeric(pd.Series(values), errors="coerce").to_numpy(dtype=float)
    if len(x) < 2 or not np.all(np.isfinite(x)):
        return False
    return bool(np.allclose(np.diff(x), 1.0, rtol=0, atol=1e-9))


def extract_windows(path: Path, raw_root: Path):
    try:
        df = pd.read_csv(path, low_memory=False)
    except Exception as e:
        return [], {"file": str(path), "status": "read_error", "error": str(e)}

    cols = [c for c in CORE_COLS if c in df.columns]
    if not cols:
        return [], {"file": str(path), "status": "no_core_columns"}

    time_col = detect_time_column(df)
    if time_col is None:
        return [], {
            "file": str(path),
            "status": "no_time_column",
            "n_rows": int(len(df)),
        }

    numeric = df[cols].apply(pd.to_numeric, errors="coerce")
    time_values = df[time_col]
    rel = str(path.relative_to(raw_root))

    accepted = []
    possible = 0
    rejected_gap = 0

    for start in range(
        0, max(len(df) - ROWS_PER_WINDOW + 1, 0), STRIDE_ROWS
    ):
        end = start + ROWS_PER_WINDOW
        possible += 1

        time_w = time_values.iloc[start:end]
        if not is_consecutive_integer_window(time_w):
            rejected_gap += 1
            continue

        w = numeric.iloc[start:end]
        row = {
            "source_file": str(path),
            "relative_file": rel,
            "segment_start_row": int(start),
            "segment_end_row": int(end),
            "rows_per_segment": ROWS_PER_WINDOW,
            "time_col": time_col,
            "start_time_raw": str(time_values.iloc[start]),
            "end_time_raw": str(time_values.iloc[end - 1]),
            "is_contiguous": True,
        }

        for c in cols:
            vals = w[c].to_numpy(dtype=float)
            finite = vals[np.isfinite(vals)]
            if len(finite) == 0:
                row[f"{c}_mean"] = np.nan
                row[f"{c}_std"] = np.nan
                row[f"{c}_max"] = np.nan
                row[f"{c}_cv"] = np.nan
            else:
                row[f"{c}_mean"] = float(np.nanmean(finite))
                row[f"{c}_std"] = float(np.nanstd(finite))
                row[f"{c}_max"] = float(np.nanmax(finite))
                row[f"{c}_cv"] = coeff_var(finite)

        if "n_bytes" in cols and "n_packets" in cols:
            row["bytes_per_packet"] = safe_div(
                np.nansum(w["n_bytes"].to_numpy(dtype=float)),
                np.nansum(w["n_packets"].to_numpy(dtype=float)),
            )
        if "n_packets" in cols and "n_flows" in cols:
            row["packets_per_flow"] = safe_div(
                np.nansum(w["n_packets"].to_numpy(dtype=float)),
                np.nansum(w["n_flows"].to_numpy(dtype=float)),
            )

        accepted.append(row)

    return accepted, {
        "file": str(path),
        "status": "ok",
        "n_rows": int(len(df)),
        "time_col": time_col,
        "possible_windows": int(possible),
        "accepted_windows": int(len(accepted)),
        "rejected_gap_windows": int(rejected_gap),
        "acceptance_rate": (
            float(len(accepted) / possible) if possible else np.nan
        ),
    }


def fit_thresholds(df):
    train = df[df["split_refined"] == "train"]
    out = {}
    for f in THRESHOLD_FEATURES:
        vals = (
            pd.to_numeric(train[f], errors="coerce")
            .replace([np.inf, -np.inf], np.nan)
            .dropna()
        )
        if len(vals) >= 10:
            out[f] = {
                "q_low": float(vals.quantile(1 / 3)),
                "q_high": float(vals.quantile(2 / 3)),
            }
    return out


def apply_labels(df, thresholds):
    out = df.copy()
    for f, th in thresholds.items():
        x = pd.to_numeric(out[f], errors="coerce")
        label = pd.Series(np.nan, index=out.index, dtype="object")
        label[x <= th["q_low"]] = "low"
        label[(x > th["q_low"]) & (x <= th["q_high"])] = "medium"
        label[x > th["q_high"]] = "high"
        out[f"sem_v2_{f}"] = label
    return out


def save_table(df, base_path):
    try:
        path = Path(base_path).with_suffix(".parquet")
        df.to_parquet(path, index=False)
        return path
    except Exception:
        path = Path(base_path).with_suffix(".csv")
        df.to_csv(path, index=False)
        return path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--data-root",
        default="/data/dataset/AgenticTraffic",
    )
    ap.add_argument(
        "--skip-download",
        action="store_true",
        help="Require the Zenodo archive/raw CSVs to already exist.",
    )
    args = ap.parse_args()

    root = Path(args.data_root) / "cesnet_ts24"
    raw_root = root / "raw"
    processed_root = root / "processed"
    metadata_root = root / "metadata"
    for p in [raw_root, processed_root, metadata_root]:
        p.mkdir(parents=True, exist_ok=True)

    archive = raw_root / "ip_addresses_sample.tar.gz"
    extract_dir = raw_root / "ip_addresses_sample"

    if not archive.exists():
        if args.skip_download:
            print("[SKIP] archive download disabled")
        else:
            print("[DOWNLOAD]", CESNET_URL)
            download_with_progress(CESNET_URL, archive)

    if archive.exists():
        actual_md5 = md5sum(archive)
        if actual_md5 != EXPECTED_MD5:
            raise RuntimeError(
                f"CESNET archive MD5 mismatch: {actual_md5} != {EXPECTED_MD5}"
            )
        print("[MD5 PASS]", actual_md5)

        if not extract_dir.exists() or not any(extract_dir.rglob("*.csv")):
            extract_dir.mkdir(parents=True, exist_ok=True)
            with tarfile.open(archive, "r:gz") as tar:
                tar.extractall(extract_dir)
            print("[EXTRACT]", extract_dir)

    csvs = sorted(raw_root.rglob("*.csv"))
    if not csvs:
        raise FileNotFoundError(f"No CESNET CSV files found under {raw_root}")

    rows, diags = [], []
    for i, path in enumerate(csvs, 1):
        r, d = extract_windows(path, raw_root)
        rows.extend(r)
        diags.append(d)
        if i % 250 == 0 or i == len(csvs):
            print(f"[{i:5d}/{len(csvs)}] accepted_windows={len(rows):,}")

    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError("No CESNET windows were produced.")

    df["split_refined"] = df["relative_file"].astype(str).map(
        split_for_relative_file
    )
    thresholds = fit_thresholds(df)
    df = apply_labels(df, thresholds)

    refined_path = save_table(
        df, processed_root / "cesnet_windows_12rows_contiguous_v2"
    )
    diag_df = pd.DataFrame(diags)
    diag_path = save_table(
        diag_df,
        metadata_root / "cesnet_continuity_diagnostics_v2",
    )

    metadata = {
        "seed": SEED,
        "dataset": "CESNET-TimeSeries24",
        "source_url": CESNET_URL,
        "archive_md5": EXPECTED_MD5,
        "rows_per_window": ROWS_PER_WINDOW,
        "stride_rows": STRIDE_ROWS,
        "window_policy": "strict consecutive integer time index",
        "split_policy": "source-file deterministic group split",
        "split_hash_compatibility_root": str(FROZEN_CESNET_RAW),
        "thresholds_train_only": thresholds,
        "n_source_files": int(df["relative_file"].nunique()),
        "n_windows": int(len(df)),
        "processed_table": str(refined_path),
        "continuity_diagnostics": str(diag_path),
    }
    meta_path = metadata_root / "cesnet_preparation_metadata_v2.json"
    meta_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print("\n[DONE]")
    print("windows:", len(df))
    print("source files:", df["relative_file"].nunique())
    print(df["split_refined"].value_counts())
    print("processed:", refined_path)
    print("metadata:", meta_path)


if __name__ == "__main__":
    main()
