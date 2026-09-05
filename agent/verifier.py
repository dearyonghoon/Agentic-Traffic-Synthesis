from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import RobustScaler

from .schema import (
    CESNET_KEYS,
    GAVIST_APPS,
    GAVIST_KEYS,
    LEVELS,
    semantic_keys,
    verifier_keys,
)


GAVIST_FIELDS = {
    "traffic_intensity": {
        "label_col": "sem_v2_length_per_sec",
        "continuous_col": "length_per_sec",
    },
    "event_intensity": {
        "label_col": "sem_v2_events_per_sec",
        "continuous_col": "events_per_sec",
    },
    "burstiness": {
        "label_col": "sem_v2_length_bin_cv",
        "continuous_col": "length_bin_cv",
    },
    "artt": {
        "label_col": "sem_v2_mean_artt",
        "continuous_col": "mean_artt",
    },
}

CESNET_FIELDS = {
    "traffic_volume": {
        "label_col": "sem_v2_n_bytes_mean",
        "continuous_col": "n_bytes_mean",
    },
    "packet_intensity": {
        "label_col": "sem_v2_n_packets_mean",
        "continuous_col": "n_packets_mean",
    },
    "flow_intensity": {
        "label_col": "sem_v2_n_flows_mean",
        "continuous_col": "n_flows_mean",
    },
    "burstiness": {
        "label_col": "sem_v2_n_bytes_cv",
        "continuous_col": "n_bytes_cv",
    },
    "packet_burstiness": {
        "label_col": "sem_v2_n_packets_cv",
        "continuous_col": "n_packets_cv",
    },
    "flow_duration": {
        "label_col": "sem_v2_avg_duration_mean",
        "continuous_col": "avg_duration_mean",
    },
}


def field_map(dataset: str):
    if dataset == "gavist5g":
        return GAVIST_FIELDS
    if dataset == "cesnet_ts24":
        return CESNET_FIELDS
    raise ValueError(dataset)


def _log_feature_array(series):
    x = pd.to_numeric(series, errors="coerce").to_numpy(dtype=np.float32)
    return np.log1p(np.clip(x, 0, None)).astype(np.float32)


def make_log_matrix(df: pd.DataFrame, dataset: str):
    fmap = field_map(dataset)
    return np.column_stack([
        _log_feature_array(df[fmap[k]["continuous_col"]])
        for k in semantic_keys(dataset)
    ]).astype(np.float32)


def fit_three_level_thresholds(df, spec):
    z = _log_feature_array(df[spec["continuous_col"]])
    y = df[spec["label_col"]].astype(str).to_numpy()
    vals = {level: z[y == level] for level in LEVELS}

    low_max = float(np.nanmax(vals["low"]))
    med_min = float(np.nanmin(vals["medium"]))
    med_max = float(np.nanmax(vals["medium"]))
    high_min = float(np.nanmin(vals["high"]))

    t1 = (
        (low_max + med_min) / 2
        if low_max <= med_min
        else (float(np.nanmedian(vals["low"])) + float(np.nanmedian(vals["medium"]))) / 2
    )
    t2 = (
        (med_max + high_min) / 2
        if med_max <= high_min
        else (float(np.nanmedian(vals["medium"])) + float(np.nanmedian(vals["high"]))) / 2
    )
    return t1, t2


def _classify(z, thresholds):
    t1, t2 = thresholds
    if z <= t1:
        return "low"
    if z <= t2:
        return "medium"
    return "high"


def _min_dist(point, reference):
    diff = reference - point[None, :]
    return float(np.sqrt(np.sum(diff * diff, axis=1)).min())


class VectorVerifier:
    """
    Frozen CESNET/GAViST verifier:
      semantic bin reconstruction
      AND
      q99 nearest-reference plausibility screen.
    """

    def __init__(self, dataset: str, seed: int = 2026):
        if dataset not in {"cesnet_ts24", "gavist5g"}:
            raise ValueError(dataset)
        self.dataset = dataset
        self.seed = int(seed)

    def fit(self, train_df: pd.DataFrame):
        self.train_df = train_df.reset_index(drop=True).copy()
        fmap = field_map(self.dataset)

        self.thresholds = {
            k: fit_three_level_thresholds(self.train_df, fmap[k])
            for k in semantic_keys(self.dataset)
        }

        x = make_log_matrix(self.train_df, self.dataset)
        self.median = np.nanmedian(x, axis=0).astype(np.float32)
        q25 = np.nanquantile(x, 0.25, axis=0)
        q75 = np.nanquantile(x, 0.75, axis=0)
        self.iqr = np.maximum(q75 - q25, 1e-6).astype(np.float32)

        rng = np.random.default_rng(self.seed + 77)
        self.plausibility = {}

        if self.dataset == "gavist5g":
            groups = [
                (str(app), g.index.to_numpy())
                for app, g in self.train_df.groupby("application_canonical")
            ]
        else:
            groups = [("__global__", np.arange(len(self.train_df)))]

        x_scaled = (x - self.median) / self.iqr

        for group_name, idx in groups:
            idx = idx.copy()
            rng.shuffle(idx)
            n_ref = min(2500, max(1, int(len(idx) * 0.75)))
            ref_idx = idx[:n_ref]
            cal_idx = idx[n_ref:]
            if len(cal_idx) == 0:
                cal_idx = ref_idx
            cal_idx = cal_idx[:min(600, len(cal_idx))]

            ref = x_scaled[ref_idx]
            cal = x_scaled[cal_idx]
            dists = np.asarray([_min_dist(p, ref) for p in cal])
            threshold = float(np.quantile(dists, 0.99))
            self.plausibility[group_name] = {
                "reference": ref.astype(np.float32),
                "threshold": threshold,
            }
        return self

    def _labels(self, z, application=None):
        labels = {
            key: _classify(float(z[i]), self.thresholds[key])
            for i, key in enumerate(semantic_keys(self.dataset))
        }
        if self.dataset == "gavist5g":
            labels["application"] = application
        return labels

    def verify(self, candidate: Dict[str, Any], intent: Dict[str, Any]):
        z = np.asarray(candidate.get("z"), dtype=np.float32)
        if z.shape != (len(semantic_keys(self.dataset)),) or not np.all(np.isfinite(z)):
            return {
                "verified": False,
                "semantic_ok": False,
                "plausibility_ok": False,
                "distance": np.inf,
                "threshold": np.nan,
                "labels": {},
            }

        labels = self._labels(z, intent.get("application"))
        semantic_ok = all(
            str(labels.get(k)) == str(intent.get(k))
            for k in verifier_keys(self.dataset)
        )

        point = (z - self.median) / self.iqr
        group = str(intent.get("application")) if self.dataset == "gavist5g" else "__global__"
        obj = self.plausibility.get(group)
        if obj is None:
            d, threshold, plausibility_ok = np.inf, np.nan, False
        else:
            d = _min_dist(point, obj["reference"])
            threshold = float(obj["threshold"])
            plausibility_ok = bool(d <= threshold)

        return {
            "verified": bool(semantic_ok and plausibility_ok),
            "semantic_ok": bool(semantic_ok),
            "plausibility_ok": bool(plausibility_ok),
            "distance": float(d),
            "threshold": threshold,
            "labels": labels,
        }


def safe_cv(values):
    values = np.asarray(values, dtype=np.float64)
    m = float(values.mean())
    return 0.0 if m <= 1e-12 else float(values.std(ddof=0) / m)


def derived_mawi_metrics(bytes_seq, packets_seq):
    b = np.asarray(bytes_seq, dtype=np.float64)
    p = np.asarray(packets_seq, dtype=np.float64)
    mean_b = float(b.mean())
    total_p = float(p.sum())
    return {
        "traffic_volume_value": mean_b,
        "packet_intensity_value": float(p.mean()),
        "packet_size_value": float(b.sum() / total_p) if total_p > 0 else 0.0,
        "byte_burstiness_value": safe_cv(b),
        "peakiness_value": float(np.quantile(b, 0.95) / mean_b) if mean_b > 0 else 0.0,
    }


MAWI_FIELD_TO_METRIC = {
    "traffic_volume": "traffic_volume_value",
    "packet_intensity": "packet_intensity_value",
    "packet_size": "packet_size_value",
    "byte_burstiness": "byte_burstiness_value",
    "peakiness": "peakiness_value",
}


class MawiTemporalVerifier:
    """Frozen MAWI operational verifier: semantic consistency AND PCA q99 screen."""

    def __init__(
        self,
        seq_len: int = 30,
        seed: int = 2026,
        dev_trace: str = "202409011400",
    ):
        self.seq_len = int(seq_len)
        self.seed = int(seed)
        self.dev_trace = str(dev_trace)

    @staticmethod
    def flatten_log_sequence(bytes_seq, packets_seq):
        return np.concatenate([
            np.log1p(np.clip(np.asarray(bytes_seq, dtype=np.float32), 0, None)),
            np.log1p(np.clip(np.asarray(packets_seq, dtype=np.float32), 0, None)),
        ]).astype(np.float32)

    def fit(self, seq_df: pd.DataFrame):
        train_mask = seq_df["split"].eq("train").to_numpy()
        self.thresholds = {}
        for field, col in MAWI_FIELD_TO_METRIC.items():
            z = np.log1p(np.clip(
                seq_df.loc[train_mask, col].to_numpy(dtype=float), 0, None
            ))
            self.thresholds[field] = tuple(float(x) for x in np.quantile(z, [1/3, 2/3]))

        X_log = np.stack([
            self.flatten_log_sequence(b, p)
            for b, p in zip(seq_df["bytes_seq"], seq_df["packets_seq"])
        ])
        self.robust_scaler = RobustScaler(quantile_range=(25, 75)).fit(X_log[train_mask])
        X = self.robust_scaler.transform(X_log).astype(np.float32)

        X_reference = X[train_mask]
        pca_dim = min(20, X_reference.shape[1], X_reference.shape[0] - 1)
        self.pca = PCA(
            n_components=pca_dim,
            whiten=True,
            random_state=self.seed,
        ).fit(X_reference)

        self.Z_reference = self.pca.transform(X_reference).astype(np.float32)
        dev_mask = seq_df["trace_id"].astype(str).eq(self.dev_trace).to_numpy()
        Z_cal = self.pca.transform(X[dev_mask]).astype(np.float32)

        dists = self._min_dist_batch(Z_cal, self.Z_reference)
        self.plausibility_threshold = float(np.quantile(dists, 0.99))
        return self

    @staticmethod
    def _min_dist_batch(Zq, Zr, chunk=512):
        Zq = np.asarray(Zq, dtype=np.float32)
        out = np.empty(len(Zq), dtype=np.float32)
        for s in range(0, len(Zq), chunk):
            q = Zq[s:s + chunk]
            d2 = (
                (q * q).sum(1, keepdims=True)
                + (Zr * Zr).sum(1)[None, :]
                - 2 * q @ Zr.T
            )
            out[s:s + len(q)] = np.sqrt(np.maximum(d2, 0).min(1))
        return out

    def _level_from_value(self, field, value):
        z = float(np.log1p(max(float(value), 0.0)))
        t1, t2 = self.thresholds[field]
        return "low" if z <= t1 else ("medium" if z <= t2 else "high")

    def verify(self, candidate: Dict[str, Any], intent: Dict[str, Any]):
        x = np.asarray(candidate.get("x"), dtype=np.float32)
        if x.shape != (2 * self.seq_len,) or not np.all(np.isfinite(x)):
            return {
                "verified": False,
                "semantic_ok": False,
                "plausibility_ok": False,
                "physical_ok": False,
                "distance": np.inf,
                "labels": {},
                "metrics": {},
            }

        log_seq = self.robust_scaler.inverse_transform(x.reshape(1, -1))[0]
        raw = np.expm1(log_seq).astype(np.float32)
        b = raw[:self.seq_len]
        p = raw[self.seq_len:2 * self.seq_len]

        physical_ok = bool(
            np.all(log_seq >= 0) and np.all(b >= 0) and np.all(p >= 0)
        )
        if not physical_ok:
            return {
                "verified": False,
                "semantic_ok": False,
                "plausibility_ok": False,
                "physical_ok": False,
                "distance": np.inf,
                "labels": {},
                "metrics": {},
            }

        metrics = derived_mawi_metrics(b, p)
        labels = {
            field: self._level_from_value(field, metrics[col])
            for field, col in MAWI_FIELD_TO_METRIC.items()
        }
        semantic_ok = all(labels[f] == intent[f] for f in MAWI_FIELD_TO_METRIC)

        z = self.pca.transform(x.reshape(1, -1)).astype(np.float32)
        distance = float(self._min_dist_batch(z, self.Z_reference)[0])
        plausibility_ok = bool(distance <= self.plausibility_threshold)

        return {
            "verified": bool(semantic_ok and plausibility_ok),
            "semantic_ok": bool(semantic_ok),
            "plausibility_ok": bool(plausibility_ok),
            "physical_ok": True,
            "distance": distance,
            "threshold": self.plausibility_threshold,
            "labels": labels,
            "metrics": metrics,
        }
