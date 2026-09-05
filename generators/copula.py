from __future__ import annotations

from copy import deepcopy
import hashlib
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
from scipy.stats import norm, rankdata
from sklearn.linear_model import Ridge

from agent.schema import semantic_keys
from agent.verifier import field_map, make_log_matrix


def stable_int(s):
    h = hashlib.md5(str(s).encode("utf-8")).hexdigest()
    return int(h[:8], 16)


def _log_feature_v07(series):
    x = pd.to_numeric(series, errors="coerce").astype(float)
    return np.log1p(np.clip(x, 0, None))


def _make_log_matrix_v07(df, dataset):
    fmap = field_map(dataset)
    return np.column_stack([
        _log_feature_v07(df[fmap[k]["continuous_col"]]).to_numpy()
        for k in semantic_keys(dataset)
    ])


def _intent_key(intent):
    dataset = str(intent["dataset"])

    if dataset == "gavist5g":
        return (
            dataset,
            str(intent["application"]),
            str(intent["traffic_intensity"]),
            str(intent["event_intensity"]),
            str(intent["burstiness"]),
            str(intent["artt"]),
        )

    if dataset == "cesnet_ts24":
        return (
            dataset,
            str(intent["traffic_volume"]),
            str(intent["packet_intensity"]),
            str(intent["flow_intensity"]),
            str(intent["burstiness"]),
            str(intent["packet_burstiness"]),
            str(intent["flow_duration"]),
        )

    raise ValueError(f"Unsupported vector dataset: {dataset}")


def empirical_quantile(pool, u):
    if len(pool) == 0:
        return np.nan
    u = float(np.clip(u, 1e-6, 1 - 1e-6))
    return float(np.quantile(pool, u))


class SemanticCopulaSynthesizer:
    """Frozen CESNET/GAViST copula synthesizer from Experiment 07/18."""

    def __init__(self, dataset: str, seed: int = 2026, min_app_pool: int = 25):
        if dataset not in {"cesnet_ts24", "gavist5g"}:
            raise ValueError(dataset)
        self.dataset = dataset
        self.seed = int(seed)
        self.min_app_pool = int(min_app_pool)

    @staticmethod
    def _log_feature(series):
        x = pd.to_numeric(series, errors="coerce").astype(float)
        return np.log1p(np.clip(x, 0, None))

    def fit(self, train_df: pd.DataFrame):
        self.train_df = train_df.reset_index(drop=True).copy()
        self.keys = semantic_keys(self.dataset)
        self.fields = field_map(self.dataset)

        # Experiment 07 used float64 here. Do not reuse the
        # float32 CFM/verifier matrix helper from Experiment 08b.
        x = _make_log_matrix_v07(self.train_df, self.dataset)
        if len(x) > 30000:
            rng = np.random.default_rng(self.seed)
            idx = rng.choice(len(x), size=30000, replace=False)
            x_corr = x[idx]
        else:
            x_corr = x

        z_cols = []
        for j in range(x_corr.shape[1]):
            s = pd.Series(x_corr[:, j])
            u = s.rank(method="average", pct=True).to_numpy(dtype=float)
            u = np.clip(u, 1e-4, 1 - 1e-4)
            z_cols.append(norm.ppf(u))
        z = np.column_stack(z_cols)
        corr = np.corrcoef(z, rowvar=False)
        corr = (corr + corr.T) / 2
        eigvals = np.linalg.eigvalsh(corr)
        if eigvals.min() < 1e-6:
            corr = corr + np.eye(corr.shape[0]) * (1e-6 - eigvals.min() + 1e-6)
        d = np.sqrt(np.diag(corr))
        self.corr = corr / np.outer(d, d)

        self.global_pools = {}
        self.app_pools = {}
        for key in self.keys:
            label_col = self.fields[key]["label_col"]
            cont_col = self.fields[key]["continuous_col"]
            vals = self._log_feature(self.train_df[cont_col])
            labels = self.train_df[label_col].astype(str)
            for level in ("low", "medium", "high"):
                arr = vals[labels == level].dropna().to_numpy(dtype=float)
                self.global_pools[(key, level)] = np.sort(arr)

            if self.dataset == "gavist5g":
                apps = self.train_df["application_canonical"].astype(str)
                for app in sorted(apps.unique()):
                    for level in ("low", "medium", "high"):
                        mask = (apps == app) & (labels == level)
                        arr = vals[mask].dropna().to_numpy(dtype=float)
                        if len(arr) >= self.min_app_pool:
                            self.app_pools[(app, key, level)] = np.sort(arr)
        return self

    def _synth_one(self, intent, rng):
        z_latent = rng.multivariate_normal(
            mean=np.zeros(len(self.keys)),
            cov=self.corr,
        )
        u = norm.cdf(z_latent)
        out = []
        for i, key in enumerate(self.keys):
            level = intent[key]
            if self.dataset == "gavist5g":
                app = intent["application"]
                pool = self.app_pools.get((app, key, level))
                if pool is None or len(pool) == 0:
                    pool = self.global_pools[(key, level)]
            else:
                pool = self.global_pools[(key, level)]
            out.append(empirical_quantile(pool, u[i]))
        return np.asarray(out, dtype=float)

    def generate_candidates(self, intent: Dict[str, Any], budget: int = 8):
        cache_key = (
            _intent_key(intent),
            int(budget),
        )
        seed = self.seed + stable_int(cache_key)
        rng = np.random.default_rng(seed)
        return [
            {"attempt": attempt, "z": self._synth_one(intent, rng), "tool": "copula"}
            for attempt in range(1, int(budget) + 1)
        ]


def _stable_seed(seed, *parts):
    s = "|".join(str(p) for p in parts) + f"|{seed}"
    return (seed + int(hashlib.md5(s.encode()).hexdigest()[:8], 16)) % (2**31 - 1)


def _empirical_inverse(sorted_values, u):
    a = np.asarray(sorted_values, dtype=np.float32)
    pos = np.clip(float(u), 0, 1) * (len(a) - 1)
    lo = int(np.floor(pos))
    hi = min(lo + 1, len(a) - 1)
    w = pos - lo
    return float((1 - w) * a[lo] + w * a[hi])


class MawiTemporalCopulaSynthesizer:
    """Frozen MAWI temporal conditional copula from Experiment 10c."""

    LEVEL_TO_IDX = {"low": 0, "medium": 1, "high": 2}
    FIELDS = (
        "traffic_volume",
        "packet_intensity",
        "packet_size",
        "byte_burstiness",
        "peakiness",
    )

    def __init__(self, verifier, seed: int = 2026):
        self.verifier = verifier
        self.seed = int(seed)
        self.cond_dim = 3 * len(self.FIELDS)

    def encode_intent(self, intent):
        c = np.zeros(self.cond_dim, dtype=np.float32)
        for j, field in enumerate(self.FIELDS):
            c[3 * j + self.LEVEL_TO_IDX[intent[field]]] = 1.0
        return c

    def fit(self, seq_df: pd.DataFrame):
        X_log = np.stack([
            self.verifier.flatten_log_sequence(b, p)
            for b, p in zip(seq_df["bytes_seq"], seq_df["packets_seq"])
        ])
        train_mask = seq_df["split"].eq("train").to_numpy()
        X = self.verifier.robust_scaler.transform(X_log).astype(np.float32)
        X_train = X[train_mask]

        C = np.stack([
            self.encode_intent({
                f: str(row[f"sem_{f}"])
                for f in self.FIELDS
            })
            for _, row in seq_df.iterrows()
        ]).astype(np.float32)
        C_train = C[train_mask]

        G_train = np.zeros_like(X_train, dtype=np.float32)
        self.sorted_x_dims = []
        for j in range(X_train.shape[1]):
            values = X_train[:, j]
            ranks = rankdata(values, method="average")
            u = np.clip((ranks - 0.5) / len(values), 1e-5, 1 - 1e-5)
            G_train[:, j] = norm.ppf(u).astype(np.float32)
            self.sorted_x_dims.append(np.sort(values.astype(np.float32)))

        self.cond_model = Ridge(alpha=1.0, fit_intercept=True).fit(C_train, G_train)
        G_resid = G_train - self.cond_model.predict(C_train).astype(np.float32)
        cov = np.cov(G_resid, rowvar=False).astype(np.float64)
        cov = (cov + cov.T) / 2
        emin = float(np.linalg.eigvalsh(cov).min())
        if emin < 1e-5:
            cov += np.eye(cov.shape[0]) * (1e-5 - emin + 1e-5)
        self.cov = cov
        return self

    def generate_candidates(self, intent: Dict[str, Any], budget: int = 8):
        combo = tuple(str(intent[f]) for f in self.FIELDS)
        c = self.encode_intent(intent).reshape(1, -1)
        mu = self.cond_model.predict(c)[0].astype(np.float64)
        rng = np.random.default_rng(
            _stable_seed(self.seed, "temporal-copula", *combo)
        )
        G = rng.multivariate_normal(mu, self.cov, size=int(budget), check_valid="ignore")

        rows = []
        for attempt in range(1, int(budget) + 1):
            u = norm.cdf(G[attempt - 1])
            x = np.asarray([
                _empirical_inverse(self.sorted_x_dims[j], u[j])
                for j in range(len(self.sorted_x_dims))
            ], dtype=np.float32)
            rows.append({"attempt": attempt, "x": x, "tool": "copula"})
        return rows
