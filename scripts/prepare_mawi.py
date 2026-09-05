#!/usr/bin/env python3
"""
Prepare the final frozen MAWI temporal dataset used by the paper.

Protocol
--------
- MAWI samplepoint-F, 2024
- Jan/Mar/May/Jul: train
- Sep: validation
- Nov: test
- tshark 1-s bytes/packet statistics
- 30-s sequences with 5-s stride
- semantic metrics derived from each 30-s sequence
- train-only 1/3 and 2/3 log1p quantile thresholds

This script reproduces the final 10c temporal representation, not the earlier
5-s onboarding representation.
"""

from __future__ import annotations

import argparse
import gzip
import json
import re
import shutil
import subprocess
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests


SEED = 2026
YEAR = 2024

TRACE_IDS = [
    "202401011400",
    "202403011400",
    "202405011400",
    "202407011400",
    "202409011400",
    "202411011400",
]
TRACE_SPLIT = {
    "202401011400": "train",
    "202403011400": "train",
    "202405011400": "train",
    "202407011400": "train",
    "202409011400": "val",
    "202411011400": "test",
}

INFO_BASE = f"https://mawi.wide.ad.jp/mawi/samplepoint-F/{YEAR}/"
DATA_BASE = f"https://mawi.nezu.wide.ad.jp/mawi/samplepoint-F/{YEAR}/"

SEQ_LEN = 30
STRIDE = 5
CHUNK_MB = 8
CONNECT_TIMEOUT_SEC = 30
READ_TIMEOUT_SEC = 300

SEMANTIC_FIELDS = {
    "traffic_volume": "traffic_volume_value",
    "packet_intensity": "packet_intensity_value",
    "packet_size": "packet_size_value",
    "byte_burstiness": "byte_burstiness_value",
    "peakiness": "peakiness_value",
}


def fetch_text(url, timeout=60):
    r = requests.get(
        url,
        timeout=timeout,
        headers={"User-Agent": "Mozilla/5.0 (research data audit)"},
    )
    r.raise_for_status()
    return r.text


def get_trace_info(trace_id):
    info_url = INFO_BASE + f"{trace_id}.html"
    pcap_name = f"{trace_id}.pcap.gz"
    pcap_url = DATA_BASE + pcap_name
    html = fetch_text(info_url)

    size_matches = re.findall(
        r"\(([0-9.]+)\s*MB\)", html, flags=re.IGNORECASE
    )
    compressed_mb = max(map(float, size_matches)) if size_matches else np.nan

    return {
        "trace_id": trace_id,
        "split": TRACE_SPLIT[trace_id],
        "info_url": info_url,
        "pcap_url": pcap_url,
        "pcap_name": pcap_name,
        "reported_compressed_mb": compressed_mb,
    }


def probe_remote_file(url):
    try:
        r = requests.head(
            url,
            allow_redirects=True,
            timeout=(CONNECT_TIMEOUT_SEC, 60),
            headers={"User-Agent": "Mozilla/5.0 (research data audit)"},
        )
        if r.status_code < 400:
            length = r.headers.get("Content-Length")
            return int(length) if length is not None else None
    except Exception:
        pass

    r = requests.get(
        url,
        stream=True,
        timeout=(CONNECT_TIMEOUT_SEC, 60),
        headers={
            "Range": "bytes=0-0",
            "User-Agent": "Mozilla/5.0 (research data audit)",
        },
        allow_redirects=True,
    )
    content_range = r.headers.get("Content-Range", "")
    m = re.search(r"/([0-9]+)$", content_range)
    total = int(m.group(1)) if m else None
    r.close()
    return total


def download_resumable(url, output_path, expected_size_bytes=None):
    output_path = Path(output_path)
    part_path = Path(str(output_path) + ".part")

    if output_path.exists():
        local_size = output_path.stat().st_size
        if expected_size_bytes is None or abs(local_size - expected_size_bytes) <= 1024:
            print("[SKIP COMPLETE]", output_path.name)
            return
        output_path.unlink()

    existing = part_path.stat().st_size if part_path.exists() else 0
    headers = {"User-Agent": "Mozilla/5.0 (research data download)"}
    if existing > 0:
        headers["Range"] = f"bytes={existing}-"

    r = requests.get(
        url,
        stream=True,
        timeout=(CONNECT_TIMEOUT_SEC, READ_TIMEOUT_SEC),
        headers=headers,
        allow_redirects=True,
    )

    if existing > 0 and r.status_code == 206:
        mode, downloaded = "ab", existing
    elif r.status_code == 200:
        mode, downloaded = "wb", 0
    else:
        r.raise_for_status()
        mode, downloaded = "wb", 0

    with open(part_path, mode) as f:
        for chunk in r.iter_content(chunk_size=CHUNK_MB * 1024**2):
            if chunk:
                f.write(chunk)
                downloaded += len(chunk)

    r.close()
    final_size = part_path.stat().st_size

    if expected_size_bytes is not None:
        tol = max(1024, int(expected_size_bytes * 1e-6))
        if abs(final_size - expected_size_bytes) > tol:
            raise RuntimeError(
                f"Size mismatch for {output_path.name}: "
                f"{final_size} != {expected_size_bytes}"
            )

    part_path.replace(output_path)
    print("[DONE]", output_path.name, f"{final_size/1024**3:.2f} GB")


def parse_io_stat(text):
    rows = []
    for line in text.splitlines():
        if "<>" not in line or "|" not in line:
            continue
        parts = [p.strip() for p in line.split("|") if p.strip()]
        if len(parts) < 3:
            continue
        m = re.match(r"([0-9.]+)\s*<>\s*([0-9.]+)", parts[0])
        if not m:
            continue
        nums = []
        for tok in parts[1:]:
            q = tok.replace(",", "")
            if re.fullmatch(r"[0-9]+(?:\.[0-9]+)?", q):
                nums.append(float(q))
        if len(nums) >= 2:
            rows.append({
                "interval_start": float(m.group(1)),
                "interval_end": float(m.group(2)),
                "frames": float(nums[0]),
                "bytes": float(nums[1]),
            })
    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError("Could not parse tshark io,stat output.")
    return df


def extract_io1s(trace_id, pcap_path, processed_root, tshark):
    cache = processed_root / f"{trace_id}.io1s.csv"
    tmp = processed_root / f"{trace_id}.io1s.csv.tmp"

    if cache.exists():
        print("[CACHE]", trace_id)
        return pd.read_csv(cache)

    print("[TSHARK START]", trace_id)
    proc = subprocess.run(
        [tshark, "-r", str(pcap_path), "-q", "-z", "io,stat,1"],
        capture_output=True,
        text=True,
    )
    combined = proc.stdout + "\n" + proc.stderr
    if proc.returncode != 0:
        raise RuntimeError(f"tshark failed for {trace_id}:\n{combined[-5000:]}")

    stats = parse_io_stat(combined)
    stats.to_csv(tmp, index=False)
    tmp.replace(cache)
    print("[TSHARK DONE]", trace_id, "rows=", len(stats))
    return stats


def safe_cv(values):
    values = np.asarray(values, dtype=np.float64)
    m = float(values.mean())
    return 0.0 if m <= 1e-12 else float(values.std(ddof=0) / m)


def derived_metrics(bytes_seq, packets_seq):
    b = np.asarray(bytes_seq, dtype=np.float64)
    p = np.asarray(packets_seq, dtype=np.float64)
    mean_b = float(b.mean())
    total_p = float(p.sum())
    return {
        "traffic_volume_value": mean_b,
        "packet_intensity_value": float(p.mean()),
        "packet_size_value": float(b.sum() / total_p) if total_p > 0 else 0.0,
        "byte_burstiness_value": safe_cv(b),
        "peakiness_value": (
            float(np.quantile(b, 0.95) / mean_b) if mean_b > 0 else 0.0
        ),
    }


def build_sequences(one_sec, trace_id, split):
    df = one_sec.copy()
    for col in ["interval_start", "frames", "bytes"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["interval_start", "frames", "bytes"]).copy()
    df["second"] = np.rint(df["interval_start"]).astype(int)
    df = (
        df.sort_values("second")
        .drop_duplicates("second")
        .reset_index(drop=True)
    )

    by_second = df.set_index("second").sort_index()
    sec_set = set(by_second.index.astype(int).tolist())
    min_sec, max_sec = min(sec_set), max(sec_set)
    first_start = ((min_sec + STRIDE - 1) // STRIDE) * STRIDE

    records = []
    for start in range(first_start, max_sec - SEQ_LEN + 2, STRIDE):
        expected = list(range(start, start + SEQ_LEN))
        if not all(s in sec_set for s in expected):
            continue
        g = by_second.loc[expected]
        b = g["bytes"].to_numpy(dtype=np.float32)
        p = g["frames"].to_numpy(dtype=np.float32)
        records.append({
            "dataset": "mawi",
            "trace_id": trace_id,
            "split": split,
            "start_second": start,
            "bytes_seq": b,
            "packets_seq": p,
            **derived_metrics(b, p),
        })
    return records


def level_from_value(value, thresholds):
    z = float(np.log1p(max(float(value), 0.0)))
    t1, t2 = thresholds
    return "low" if z <= t1 else ("medium" if z <= t2 else "high")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--data-root",
        default="/data/dataset/AgenticTraffic",
    )
    ap.add_argument(
        "--skip-download",
        action="store_true",
        help="Require all six .pcap.gz files to already exist.",
    )
    args = ap.parse_args()

    root = Path(args.data_root) / "mawi"
    raw_root = root / "raw"
    processed_root = root / "processed"
    metadata_root = root / "metadata"
    for p in [raw_root, processed_root, metadata_root]:
        p.mkdir(parents=True, exist_ok=True)

    trace_meta = []
    for tid in TRACE_IDS:
        info = get_trace_info(tid)
        expected = probe_remote_file(info["pcap_url"])
        info["content_length_bytes"] = expected
        trace_meta.append(info)

        path = raw_root / info["pcap_name"]
        if not args.skip_download:
            download_resumable(info["pcap_url"], path, expected)
        elif not path.exists():
            raise FileNotFoundError(path)

        with gzip.open(path, "rb") as f:
            if len(f.read(24)) < 24:
                raise RuntimeError(f"Invalid gzip/PCAP prefix: {path}")

    tshark = shutil.which("tshark")
    if tshark is None:
        raise RuntimeError(
            "tshark is required. Install Wireshark/tshark before preprocessing."
        )

    records = []
    for info in trace_meta:
        tid = info["trace_id"]
        one_sec = extract_io1s(
            tid,
            raw_root / info["pcap_name"],
            processed_root,
            tshark,
        )
        seqs = build_sequences(one_sec, tid, info["split"])
        records.extend(seqs)
        print(tid, info["split"], "sequences=", len(seqs))

    seq_df = pd.DataFrame(records)
    if seq_df.empty:
        raise RuntimeError("No MAWI temporal sequences were produced.")

    train_mask = seq_df["split"].eq("train")
    thresholds = {}
    for field, col in SEMANTIC_FIELDS.items():
        z = np.log1p(
            np.clip(seq_df.loc[train_mask, col].to_numpy(dtype=float), 0, None)
        )
        thresholds[field] = tuple(
            float(x) for x in np.quantile(z, [1 / 3, 2 / 3])
        )
        seq_df[f"sem_{field}"] = [
            level_from_value(v, thresholds[field])
            for v in seq_df[col].to_numpy()
        ]

    bytes_matrix = np.stack(seq_df["bytes_seq"].to_numpy()).astype(np.float32)
    packets_matrix = np.stack(seq_df["packets_seq"].to_numpy()).astype(np.float32)

    npz_path = processed_root / "mawi_temporal_30s_stride5_v1.npz"
    np.savez_compressed(
        npz_path,
        bytes_seq=bytes_matrix,
        packets_seq=packets_matrix,
        trace_id=np.asarray(seq_df["trace_id"].astype(str), dtype="U12"),
        split=np.asarray(seq_df["split"].astype(str), dtype="U5"),
        start_second=seq_df["start_second"].to_numpy(dtype=np.int64),
    )

    # Human-readable metadata table; sequence arrays stay in the NPZ.
    table = seq_df.drop(columns=["bytes_seq", "packets_seq"]).copy()
    table_path = processed_root / "mawi_temporal_30s_stride5_metadata_v1.csv"
    table.to_csv(table_path, index=False)

    meta = {
        "seed": SEED,
        "dataset": "MAWI samplepoint-F 2024",
        "year": YEAR,
        "trace_ids": TRACE_IDS,
        "trace_split": TRACE_SPLIT,
        "info_base": INFO_BASE,
        "data_base": DATA_BASE,
        "sequence_length_seconds": SEQ_LEN,
        "stride_seconds": STRIDE,
        "semantic_fields": list(SEMANTIC_FIELDS),
        "semantic_thresholds_log1p": {
            k: list(v) for k, v in thresholds.items()
        },
        "n_sequences": int(len(seq_df)),
        "split_counts": {
            str(k): int(v)
            for k, v in seq_df["split"].value_counts().items()
        },
        "npz_path": str(npz_path),
        "metadata_table": str(table_path),
    }
    meta_path = metadata_root / "mawi_temporal_preparation_v1.json"
    meta_path.write_text(
        json.dumps(meta, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("\n[DONE]")
    print(seq_df.groupby(["trace_id", "split"]).size())
    print("NPZ:", npz_path)
    print("metadata:", meta_path)


if __name__ == "__main__":
    main()
