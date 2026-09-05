from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from agent.schema import GAVIST_APPS, LEVELS, semantic_keys
from agent.verifier import field_map


LEVEL_TO_IDX = {"low": 0, "medium": 1, "high": 2}
APP_TO_IDX = {app: i for i, app in enumerate(GAVIST_APPS)}

HIDDEN_DIM = 256
COND_HIDDEN_DIM = 192
TIME_DIM = 32
N_BLOCKS = 4

GAVIST_TRAIN_STEPS = 6000
CESNET_TRAIN_STEPS = 12000
TRAIN_BATCH_SIZE = 2048
VAL_BATCH_SIZE = 4096
LR = 2e-4
WEIGHT_DECAY = 1e-5
EVAL_EVERY = 250
GRAD_CLIP = 1.0

FLOW_STEPS = 32
MAIN_ATTEMPT_BUDGET = 8
MAX_ATTEMPT_BUDGET = 16
MIN_GROUP_COUNT = 25
MIN_SCALE = 0.10


class FourierTimeEmbedding(nn.Module):
    def __init__(self, dim=32):
        super().__init__()
        if dim % 2 != 0:
            raise ValueError("time embedding dimension must be even")
        half = dim // 2
        freqs = torch.exp(torch.linspace(math.log(1.0), math.log(1000.0), half))
        self.register_buffer("freqs", freqs)

    def forward(self, t):
        phase = 2.0 * math.pi * t * self.freqs[None, :]
        return torch.cat([torch.sin(phase), torch.cos(phase)], dim=-1)


class FiLMBlock(nn.Module):
    def __init__(self, hidden_dim, context_dim):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim)
        self.fc = nn.Linear(hidden_dim, hidden_dim)
        self.film = nn.Linear(context_dim, 2 * hidden_dim)

    def forward(self, h, context):
        residual = h
        x = self.fc(self.norm(h))
        gamma, beta = self.film(context).chunk(2, dim=-1)
        gamma = 0.5 * torch.tanh(gamma)
        x = (1.0 + gamma) * x + beta
        return residual + F.silu(x)


class ResidualFiLMFlow(nn.Module):
    def __init__(
        self,
        x_dim,
        cond_dim,
        hidden_dim=HIDDEN_DIM,
        cond_hidden_dim=COND_HIDDEN_DIM,
        time_dim=TIME_DIM,
        n_blocks=N_BLOCKS,
    ):
        super().__init__()
        self.time_embed = FourierTimeEmbedding(time_dim)
        self.cond_encoder = nn.Sequential(
            nn.Linear(cond_dim, cond_hidden_dim),
            nn.SiLU(),
            nn.Linear(cond_hidden_dim, cond_hidden_dim),
            nn.SiLU(),
        )
        context_dim = cond_hidden_dim + time_dim
        self.input_proj = nn.Linear(x_dim, hidden_dim)
        self.blocks = nn.ModuleList([
            FiLMBlock(hidden_dim, context_dim)
            for _ in range(n_blocks)
        ])
        self.output_norm = nn.LayerNorm(hidden_dim)
        self.output_proj = nn.Linear(hidden_dim, x_dim)

    def forward(self, x, t, cond):
        context = torch.cat(
            [self.cond_encoder(cond), self.time_embed(t)],
            dim=-1,
        )
        h = self.input_proj(x)
        for block in self.blocks:
            h = block(h, context)
        return self.output_proj(self.output_norm(h))


def _log_feature_array(series):
    x = pd.to_numeric(series, errors="coerce").to_numpy(dtype=np.float32)
    return np.log1p(np.clip(x, 0, None)).astype(np.float32)


def _robust_center_scale(values):
    values = np.asarray(values, dtype=np.float32)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        raise ValueError("Empty group for semantic center/scale.")
    center = float(np.median(values))
    q25 = float(np.quantile(values, 0.25))
    q75 = float(np.quantile(values, 0.75))
    return center, max(q75 - q25, MIN_SCALE)


def _stable_seed(base_seed, obj):
    s = json.dumps(obj, sort_keys=True, ensure_ascii=True)
    h = hashlib.md5(s.encode("utf-8")).hexdigest()
    return (base_seed + int(h[:8], 16)) % (2**31 - 1)


class ResidualFiLMCFMSynthesizer:
    """Frozen CESNET/GAViST Residual-FiLM conditional flow synthesizer."""

    def __init__(
        self,
        dataset: str,
        checkpoint_path: str,
        device: str = None,
        seed: int = 2026,
    ):
        if dataset not in {"cesnet_ts24", "gavist5g"}:
            raise ValueError(dataset)
        self.dataset = dataset
        self.checkpoint_path = Path(checkpoint_path)
        self.device = torch.device(
            device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.seed = int(seed)

    def condition_dim(self):
        if self.dataset == "gavist5g":
            return len(GAVIST_APPS) + 3 * len(semantic_keys(self.dataset))
        return 3 * len(semantic_keys(self.dataset))

    def encode_intent_condition(self, intent):
        c = np.zeros(self.condition_dim(), dtype=np.float32)
        offset = 0
        if self.dataset == "gavist5g":
            c[APP_TO_IDX[intent["application"]]] = 1.0
            offset = len(GAVIST_APPS)
        for j, key in enumerate(semantic_keys(self.dataset)):
            c[offset + 3 * j + LEVEL_TO_IDX[intent[key]]] = 1.0
        return c

    def fit_semantic_stats(self, train_df: pd.DataFrame):
        fmap = field_map(self.dataset)
        self.global_stats = {}
        self.app_stats = {}

        for key in semantic_keys(self.dataset):
            label_col = fmap[key]["label_col"]
            cont_col = fmap[key]["continuous_col"]
            z = _log_feature_array(train_df[cont_col])
            labels = train_df[label_col].astype(str).to_numpy()

            for level in LEVELS:
                self.global_stats[(key, level)] = _robust_center_scale(
                    z[labels == level]
                )

            if self.dataset == "gavist5g":
                apps = train_df["application_canonical"].astype(str).to_numpy()
                for app in GAVIST_APPS:
                    for level in LEVELS:
                        mask = (apps == app) & (labels == level)
                        values = z[mask]
                        if len(values) >= MIN_GROUP_COUNT:
                            self.app_stats[(app, key, level)] = _robust_center_scale(values)
        return self

    def get_center_scale(self, intent):
        centers, scales = [], []
        for key in semantic_keys(self.dataset):
            level = intent[key]
            if self.dataset == "gavist5g":
                app = intent["application"]
                stat = self.app_stats.get(
                    (app, key, level),
                    self.global_stats[(key, level)],
                )
            else:
                stat = self.global_stats[(key, level)]
            centers.append(stat[0])
            scales.append(stat[1])
        return (
            np.asarray(centers, dtype=np.float32),
            np.asarray(scales, dtype=np.float32),
        )

    def load(self):
        self.model = ResidualFiLMFlow(
            x_dim=len(semantic_keys(self.dataset)),
            cond_dim=self.condition_dim(),
        ).to(self.device)

        obj = torch.load(self.checkpoint_path, map_location=self.device)
        self.model.load_state_dict(obj["model"])
        self.model.eval()
        return self

    @torch.no_grad()
    def generate_candidates(self, intent: Dict[str, Any], budget: int = 8):
        if not hasattr(self, "model"):
            self.load()

        budget = int(budget)
        cond_np = self.encode_intent_condition(intent)
        cond = torch.from_numpy(
            np.repeat(cond_np[None, :], budget, axis=0)
        ).to(self.device)

        key = (
            self.dataset,
            tuple(sorted(intent.items())),
        )
        g = torch.Generator(device=self.device)
        g.manual_seed(_stable_seed(self.seed, key))

        x_dim = len(semantic_keys(self.dataset))
        x = torch.randn(
            (budget, x_dim),
            generator=g,
            device=self.device,
            dtype=torch.float32,
        )

        dt = 1.0 / FLOW_STEPS
        for i in range(FLOW_STEPS):
            t0 = i / FLOW_STEPS
            t1 = (i + 1) / FLOW_STEPS

            t = torch.full((budget, 1), t0, device=self.device, dtype=x.dtype)
            with torch.autocast(
                device_type=self.device.type,
                dtype=torch.bfloat16,
                enabled=(self.device.type == "cuda"),
            ):
                v0 = self.model(x, t, cond).float()

            x_euler = x + dt * v0
            t_next = torch.full((budget, 1), t1, device=self.device, dtype=x.dtype)
            with torch.autocast(
                device_type=self.device.type,
                dtype=torch.bfloat16,
                enabled=(self.device.type == "cuda"),
            ):
                v1 = self.model(x_euler, t_next, cond).float()

            x = x + 0.5 * dt * (v0 + v1)

        residual = x.detach().cpu().numpy().astype(np.float32)
        center, scale = self.get_center_scale(intent)
        z_batch = center[None, :] + scale[None, :] * residual

        return [
            {
                "attempt": i + 1,
                "z": z_batch[i].astype(np.float32),
                "tool": "cfm",
            }
            for i in range(budget)
        ]
