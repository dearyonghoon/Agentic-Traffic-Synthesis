#!/usr/bin/env python3
"""
Prepare the frozen GAViST5G dataset used by the paper.

Pipeline
--------
1. Optionally download the public Kaggle dataset.
2. Build non-overlapping 60-s windows with 1-s aggregation bins.
3. Canonicalize the seven application labels.
4. Apply the frozen application-stratified source-file group split.
5. Fit train-only 1/3 and 2/3 quantile thresholds.
6. Save the refined v2 table and metadata.

The split reproduces the original experiment even when --data-root differs
from /data/dataset/AgenticTraffic by hashing the original frozen source-file
identifier rather than the local absolute path.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
from pathlib import Path

import numpy as np
import pandas as pd


SEED = 2026
WINDOW_SEC = 60
STRIDE_SEC = 60
BIN_SEC = 1
MIN_EVENTS_PER_WINDOW = 5

GAVIST_KAGGLE_SLUG = (
    "ahassanein/aggregated-gaming-and-video-streaming-traffic-for-5g"
)

FROZEN_DATA_ROOT = Path("/data/dataset/AgenticTraffic")
FROZEN_GAVIST_RAW = FROZEN_DATA_ROOT / "gavist5g" / "raw"

CANONICAL_PREFIXES = [
    ("PrimeVideo", ["primevideo", "amazonprime"]),
    ("YouTube", ["youtube"]),
    ("Netflix", ["netflix"]),
    ("Crave", ["crave"]),
    ("LOL", ["lol", "leagueoflegends"]),
    ("TFT", ["tft", "teamfighttactics"]),
    ("VAL", ["valorant", "val"]),
]


def stable_hash(text):
    return hashlib.sha256(str(text).encode("utf-8")).hexdigest()


def frozen_source_id(relative_file: str) -> str:
    return str(FROZEN_GAVIST_RAW / relative_file)


def non_hidden_files(path):
    return [
        p for p in Path(path).rglob("*")
        if p.is_file() and not p.name.startswith(".")
    ]


def maybe_download(raw_root: Path, skip_download: bool):
    existing = non_hidden_files(raw_root)
    if existing:
        print(f"[SKIP] raw data already present: {len(existing)} files")
        return

    if skip_download:
        raise FileNotFoundError(
            f"No GAViST5G files found under {raw_root}. "
            "Download the Kaggle dataset there or rerun without --skip-download."
        )

    try:
        import kagglehub
    except ImportError as e:
        raise RuntimeError(
            "kagglehub is required for automatic download. "
            "Install requirements.txt or use --skip-download after manually "
            "placing the dataset under gavist5g/raw."
        ) from e

    cache_path = Path(kagglehub.dataset_download(GAVIST_KAGGLE_SLUG))
    files = non_hidden_files(cache_path)
    print("[DOWNLOAD] Kaggle cache:", cache_path)
    print("[DOWNLOAD] files:", len(files))

    for src in files:
        rel = src.relative_to(cache_path)
        dst = raw_root / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)

    print("[DONE] copied raw data to", raw_root)


def normalize_token(s):
    return re.sub(r"\s+", " ", str(s).strip())


def parse_gavist_metadata(path: Path, raw_root: Path):
    rel = path.relative_to(raw_root)
    parts = list(rel.parts)
    lower = [p.lower() for p in parts]

    traffic_class = "unknown"
    class_idx = None

    for i, p in enumerate(lower):
        if "online gaming" in p or p == "gaming":
            traffic_class = "gaming"
            class_idx = i
            break
        if "video streaming" in p or p == "video":
            traffic_class = "video"
            class_idx = i
            break

    application = None
    if class_idx is not None and class_idx + 1 < len(parts) - 1:
        application = normalize_token(parts[class_idx + 1])

    if application is None:
        parent = path.parent.name
        if parent.lower() not in {
            "raw", "global", "online gaming", "video streaming"
        }:
            application = normalize_token(parent)

    if application is None:
        stem = re.sub(r"(?i)agg$", "", path.stem)
        m = re.match(r"([A-Za-z]+)", stem)
        application = m.group(1) if m else stem

    return {
        "source_file": str(path),
        "relative_file": str(rel),
        "traffic_class": traffic_class,
        "application": application,
        "filename": path.name,
    }


def robust_numeric(s):
    return pd.to_numeric(s, errors="coerce")


def to_relative_seconds(time_series):
    num = pd.to_numeric(time_series, errors="coerce")
    if num.notna().mean() >= 0.95:
        arr = num.astype(float)
        finite = arr[np.isfinite(arr)]
        if len(finite) == 0:
            return pd.Series(np.nan, index=time_series.index), "numeric_unknown"

        med_abs = np.nanmedian(np.abs(finite))
        diffs = np.diff(np.sort(np.unique(finite)))
        diffs = diffs[np.isfinite(diffs) & (diffs > 0)]
        med_diff = np.nanmedian(diffs) if len(diffs) else np.nan

        scale = 1.0
        scale_name = "seconds"
        if med_abs > 1e14:
            scale = 1e9
            scale_name = "nanoseconds"
        elif med_abs > 1e11:
            scale = 1e3
            scale_name = "milliseconds"
        elif med_abs > 1e9 and np.isfinite(med_diff) and med_diff > 10:
            scale = 1e3
            scale_name = "milliseconds"

        sec = arr / scale
        sec = sec - np.nanmin(sec)
        return sec, f"numeric_{scale_name}"

    dt = pd.to_datetime(time_series, errors="coerce", utc=True)
    if dt.notna().mean() >= 0.80:
        t0 = dt.min()
        return (dt - t0).dt.total_seconds(), "datetime"

    return pd.Series(np.nan, index=time_series.index), "unparsed"


def coeff_var(x):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return np.nan
    m = x.mean()
    if m == 0:
        return 0.0
    return float(x.std(ddof=0) / abs(m))


def peak_to_mean(x):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return np.nan
    m = x.mean()
    if m == 0:
        return 0.0
    return float(x.max() / m)


def protocol_features(protocol_series):
    vals = protocol_series.astype(str).str.upper().str.strip()
    n = max(len(vals), 1)
    out = {}
    for proto in ["TCP", "UDP", "QUIC", "TLS", "HTTP", "HTTPS"]:
        out[f"protocol_{proto.lower()}_ratio"] = float((vals == proto).sum() / n)
    out["protocol_unique_count"] = int(vals.nunique(dropna=True))
    return out


def extract_windows(path: Path, raw_root: Path):
    meta = parse_gavist_metadata(path, raw_root)

    try:
        df = pd.read_csv(path, low_memory=False)
    except Exception as e:
        return [], {"file": str(path), "status": "read_error", "error": str(e)}

    required = {"Time", "Length"}
    missing = required - set(df.columns)
    if missing:
        return [], {
            "file": str(path),
            "status": "missing_columns",
            "missing": sorted(missing),
        }

    rel_sec, time_mode = to_relative_seconds(df["Time"])
    df = df.copy()
    df["_t"] = rel_sec
    df["_length"] = robust_numeric(df["Length"])
    df["_artt"] = (
        robust_numeric(df["ARTT"]) if "ARTT" in df.columns else np.nan
    )

    df = df[np.isfinite(df["_t"]) & np.isfinite(df["_length"])].copy()
    if len(df) == 0:
        return [], {
            "file": str(path),
            "status": "empty_after_clean",
            "time_mode": time_mode,
        }

    tmax = float(df["_t"].max())
    starts = (
        [0.0]
        if tmax < WINDOW_SEC
        else np.arange(0.0, tmax - WINDOW_SEC + 1e-9, STRIDE_SEC)
    )

    rows = []
    for start in starts:
        end = start + WINDOW_SEC
        w = df[(df["_t"] >= start) & (df["_t"] < end)]
        if len(w) < MIN_EVENTS_PER_WINDOW:
            continue

        lengths = w["_length"].to_numpy(dtype=float)
        artt = w["_artt"].to_numpy(dtype=float)
        artt = artt[np.isfinite(artt)]

        bins = np.floor((w["_t"].to_numpy() - start) / BIN_SEC).astype(int)
        n_bins = max(int(np.ceil(WINDOW_SEC / BIN_SEC)), 1)
        length_by_bin = np.zeros(n_bins, dtype=float)
        events_by_bin = np.zeros(n_bins, dtype=float)

        valid_bin = (bins >= 0) & (bins < n_bins)
        np.add.at(length_by_bin, bins[valid_bin], lengths[valid_bin])
        np.add.at(events_by_bin, bins[valid_bin], 1.0)

        total_length = float(np.nansum(lengths))
        event_count = int(len(w))

        row = {
            **meta,
            "segment_start_sec": float(start),
            "segment_end_sec": float(end),
            "duration_sec": float(WINDOW_SEC),
            "event_count": event_count,
            "events_per_sec": event_count / WINDOW_SEC,
            "total_length": total_length,
            "length_per_sec": total_length / WINDOW_SEC,
            "mean_length": float(np.nanmean(lengths)),
            "std_length": float(np.nanstd(lengths)),
            "p50_length": float(np.nanpercentile(lengths, 50)),
            "p95_length": float(np.nanpercentile(lengths, 95)),
            "max_length": float(np.nanmax(lengths)),
            "length_bin_cv": coeff_var(length_by_bin),
            "length_peak_to_mean": peak_to_mean(length_by_bin),
            "event_bin_cv": coeff_var(events_by_bin),
            "event_peak_to_mean": peak_to_mean(events_by_bin),
            "active_bin_ratio": float(np.mean(events_by_bin > 0)),
            "mean_artt": float(np.nanmean(artt)) if len(artt) else np.nan,
            "p95_artt": float(np.nanpercentile(artt, 95)) if len(artt) else np.nan,
            "time_parse_mode": time_mode,
        }

        if "Protocol" in w.columns:
            row.update(protocol_features(w["Protocol"]))

        rows.append(row)

    return rows, {
        "file": str(path),
        "status": "ok",
        "rows_raw": int(len(df)),
        "tmax_sec": tmax,
        "time_mode": time_mode,
        "n_windows": len(rows),
    }


def compact_text(text):
    return re.sub(r"[^a-z0-9]", "", str(text).lower())


def canonical_app_from_row(row):
    texts = [
        row.get("application", ""),
        row.get("filename", ""),
        row.get("relative_file", ""),
        row.get("source_file", ""),
    ]
    for text in texts:
        c = compact_text(text)
        for app, prefixes in CANONICAL_PREFIXES:
            if any(prefix in c for prefix in prefixes):
                return app
    return str(row.get("application", "Unknown"))


def extract_location_from_filename(filename, app):
    stem = re.sub(r"(?i)agg$", "", Path(str(filename)).stem)
    patterns = {
        "PrimeVideo": r"(?i)^prime[ _-]*video",
        "YouTube": r"(?i)^youtube",
        "Netflix": r"(?i)^netflix",
        "Crave": r"(?i)^crave",
        "LOL": r"(?i)^lol",
        "TFT": r"(?i)^tft",
        "VAL": r"(?i)^val(?:orant)?",
    }
    pat = patterns.get(app)
    if pat:
        stem = re.sub(pat, "", stem, count=1)
    stem = re.sub(r"^\d+", "", stem).strip(" _-")
    return stem if stem else None


def allocate_group_counts(n, val_ratio=0.10, test_ratio=0.10):
    if n <= 0:
        return 0, 0, 0
    if n == 1:
        return 1, 0, 0
    if n == 2:
        return 1, 0, 1

    n_val = max(1, int(round(n * val_ratio)))
    n_test = max(1, int(round(n * test_ratio)))
    n_train = n - n_val - n_test

    while n_train < 1:
        if n_val > 1:
            n_val -= 1
        elif n_test > 1:
            n_test -= 1
        else:
            break
        n_train = n - n_val - n_test

    return n_train, n_val, n_test


def build_refined_split(df):
    group_meta = df[
        ["relative_file", "application_canonical"]
    ].drop_duplicates().copy()

    bad = group_meta.groupby("relative_file")["application_canonical"].nunique()
    if (bad > 1).any():
        raise ValueError("At least one source file maps to multiple applications.")

    assignments = {}
    diagnostics = []

    for stratum, g in group_meta.groupby("application_canonical", dropna=False):
        groups = list(g["relative_file"].astype(str))
        groups = sorted(groups, key=lambda rel: stable_hash(frozen_source_id(rel)))
        n_train, n_val, n_test = allocate_group_counts(len(groups))

        train_groups = groups[:n_train]
        val_groups = groups[n_train:n_train+n_val]
        test_groups = groups[n_train+n_val:n_train+n_val+n_test]

        for x in train_groups:
            assignments[x] = "train"
        for x in val_groups:
            assignments[x] = "val"
        for x in test_groups:
            assignments[x] = "test"

        diagnostics.append({
            "application": str(stratum),
            "n_source_files": len(groups),
            "train_files": len(train_groups),
            "val_files": len(val_groups),
            "test_files": len(test_groups),
        })

    return assignments, pd.DataFrame(diagnostics)


def fit_quantile_thresholds(df, features):
    train = df[df["split_refined"] == "train"]
    out = {}
    for f in features:
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
        help="Root containing gavist5g/, cesnet_ts24/, and mawi/.",
    )
    ap.add_argument(
        "--skip-download",
        action="store_true",
        help="Do not use kagglehub; require raw files to already exist.",
    )
    args = ap.parse_args()

    root = Path(args.data_root) / "gavist5g"
    raw_root = root / "raw"
    processed_root = root / "processed"
    metadata_root = root / "metadata"
    for p in [raw_root, processed_root, metadata_root]:
        p.mkdir(parents=True, exist_ok=True)

    maybe_download(raw_root, args.skip_download)

    csvs = sorted(raw_root.rglob("*.csv"))
    if not csvs:
        raise FileNotFoundError(f"No CSV files found under {raw_root}")

    rows, diags = [], []
    for i, path in enumerate(csvs, 1):
        r, d = extract_windows(path, raw_root)
        rows.extend(r)
        diags.append(d)
        if i % 20 == 0 or i == len(csvs):
            print(f"[{i:4d}/{len(csvs)}] windows={len(rows):,}")

    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError("No GAViST5G windows were produced.")

    df["application_canonical"] = df.apply(canonical_app_from_row, axis=1)
    df["location_canonical"] = df.apply(
        lambda r: extract_location_from_filename(
            r.get("filename", ""), r["application_canonical"]
        ),
        axis=1,
    )

    gaming_apps = {"LOL", "TFT", "VAL"}
    video_apps = {"YouTube", "Netflix", "PrimeVideo", "Crave"}
    df["traffic_class_canonical"] = np.where(
        df["application_canonical"].isin(gaming_apps),
        "gaming",
        np.where(
            df["application_canonical"].isin(video_apps),
            "video",
            df.get("traffic_class", "unknown"),
        ),
    )

    assignments, split_diag = build_refined_split(df)
    df["split_refined"] = df["relative_file"].astype(str).map(assignments)

    threshold_features = [
        "length_per_sec",
        "events_per_sec",
        "length_bin_cv",
        "event_bin_cv",
        "mean_artt",
    ]
    thresholds = fit_quantile_thresholds(df, threshold_features)
    df = apply_labels(df, thresholds)

    refined_path = save_table(
        df, processed_root / "gavist_60s_windows_v2"
    )
    diag_path = save_table(
        pd.DataFrame(diags),
        metadata_root / "gavist_processing_diagnostics_v2",
    )
    split_path = save_table(
        split_diag,
        metadata_root / "gavist_split_diagnostics_v2",
    )

    metadata = {
        "seed": SEED,
        "dataset": "GAViST5G",
        "kaggle_slug": GAVIST_KAGGLE_SLUG,
        "window_sec": WINDOW_SEC,
        "stride_sec": STRIDE_SEC,
        "bin_sec": BIN_SEC,
        "min_events_per_window": MIN_EVENTS_PER_WINDOW,
        "split_policy": "application-stratified source-file group split",
        "split_hash_compatibility_root": str(FROZEN_GAVIST_RAW),
        "thresholds_train_only": thresholds,
        "n_source_files": int(df["relative_file"].nunique()),
        "n_windows": int(len(df)),
        "applications": sorted(
            df["application_canonical"].dropna().astype(str).unique().tolist()
        ),
        "processed_table": str(refined_path),
        "processing_diagnostics": str(diag_path),
        "split_diagnostics": str(split_path),
    }
    meta_path = metadata_root / "gavist_preparation_metadata_v2.json"
    meta_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print("\n[DONE]")
    print("windows:", len(df))
    print("source files:", df["relative_file"].nunique())
    print(df["split_refined"].value_counts())
    print("processed:", refined_path)
    print("metadata:", meta_path)


if __name__ == "__main__":
    main()
