from __future__ import annotations

import hashlib
import math
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

import torch
import torch.nn as nn
import torch.nn.functional as F


SEQ_LEN = 30
FLOW_STEPS = 32
HIDDEN_DIM = 384
COND_HIDDEN_DIM = 192
TIME_DIM = 32
N_BLOCKS = 4

TRAIN_STEPS = 6000
TRAIN_BATCH_SIZE = 256
LR = 2e-4
WEIGHT_DECAY = 1e-5
EVAL_EVERY = 200
GRAD_CLIP = 1.0

SEMANTIC_LAMBDA = 1.0
SEMANTIC_WARMUP_STEPS = 1000
SEMANTIC_MARGIN_FRACTION = 0.05

FIELDS = (
    "traffic_volume",
    "packet_intensity",
    "packet_size",
    "byte_burstiness",
    "peakiness",
)
LEVEL_TO_IDX = {"low": 0, "medium": 1, "high": 2}
COND_DIM = 3 * len(FIELDS)


class FourierTimeEmbedding(nn.Module):
    def __init__(self, dim=32):
        super().__init__()
        half = dim // 2
        freqs = torch.exp(torch.linspace(math.log(1.0), math.log(1000.0), half))
        self.register_buffer("freqs", freqs)

    def forward(self, t):
        phase = 2 * math.pi * t * self.freqs[None, :]
        return torch.cat([torch.sin(phase), torch.cos(phase)], dim=-1)


class FiLMBlock(nn.Module):
    def __init__(self, hidden_dim, context_dim):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim)
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.film = nn.Linear(context_dim, 2 * hidden_dim)

    def forward(self, h, context):
        x = self.fc1(self.norm(h))
        gamma, beta = self.film(context).chunk(2, dim=-1)
        x = (1 + 0.5 * torch.tanh(gamma)) * x + beta
        return h + self.fc2(F.silu(x))


class ResidualFiLMFlow(nn.Module):
    def __init__(
        self,
        x_dim=2 * SEQ_LEN,
        cond_dim=COND_DIM,
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


def _stable_seed(seed, *parts):
    s = "|".join(str(p) for p in parts) + f"|{seed}"
    return (seed + int(hashlib.md5(s.encode()).hexdigest()[:8], 16)) % (2**31 - 1)


class MawiSemanticCFMSynthesizer:
    """
    Frozen MAWI Semantic-Consistency CFM sampler.

    The training objective is documented by the source experiment; this class
    implements the frozen inference path used by the agent.
    """

    def __init__(
        self,
        verifier,
        checkpoint_path: str,
        device: str = None,
        seed: int = 2026,
    ):
        self.verifier = verifier
        self.checkpoint_path = Path(checkpoint_path)
        self.device = torch.device(
            device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.seed = int(seed)

    def encode_intent(self, intent):
        c = np.zeros(COND_DIM, dtype=np.float32)
        for j, field in enumerate(FIELDS):
            c[3 * j + LEVEL_TO_IDX[intent[field]]] = 1.0
        return c

    def fit_base(self, seq_df: pd.DataFrame):
        X_log = np.stack([
            self.verifier.flatten_log_sequence(b, p)
            for b, p in zip(seq_df["bytes_seq"], seq_df["packets_seq"])
        ])
        X = self.verifier.robust_scaler.transform(X_log).astype(np.float32)

        C = np.stack([
            self.encode_intent({
                f: str(row[f"sem_{f}"])
                for f in FIELDS
            })
            for _, row in seq_df.iterrows()
        ]).astype(np.float32)

        fit_mask = seq_df["trace_id"].astype(str).isin([
            "202401011400",
            "202403011400",
            "202405011400",
            "202407011400",
        ]).to_numpy()

        X_fit = X[fit_mask]
        C_fit = C[fit_mask]

        self.base_model = Ridge(alpha=1.0, fit_intercept=True).fit(C_fit, X_fit)
        base_fit = self.base_model.predict(C_fit).astype(np.float32)
        resid_fit = X_fit - base_fit
        q25 = np.quantile(resid_fit, 0.25, axis=0)
        q75 = np.quantile(resid_fit, 0.75, axis=0)
        self.resid_scale = np.maximum(q75 - q25, 0.10).astype(np.float32)
        return self

    def load(self):
        self.model = ResidualFiLMFlow().to(self.device)
        obj = torch.load(self.checkpoint_path, map_location=self.device)
        self.model.load_state_dict(obj["model"])
        self.model.eval()
        return self

    @torch.no_grad()
    def generate_candidates(self, intent: Dict[str, Any], budget: int = 8):
        if not hasattr(self, "model"):
            self.load()

        budget = int(budget)
        c_np = self.encode_intent(intent)
        cond = torch.from_numpy(
            np.repeat(c_np[None, :], budget, axis=0)
        ).to(self.device)

        combo = tuple(str(intent[f]) for f in FIELDS)
        g = torch.Generator(device=self.device)
        g.manual_seed(
            _stable_seed(self.seed, "10c-semantic-cfm", *combo)
        )

        r = torch.randn(
            (budget, 2 * SEQ_LEN),
            generator=g,
            device=self.device,
            dtype=torch.float32,
        )

        dt = 1.0 / FLOW_STEPS
        for i in range(FLOW_STEPS):
            t0 = i / FLOW_STEPS
            t1 = (i + 1) / FLOW_STEPS

            t = torch.full((budget, 1), t0, device=self.device)
            with torch.autocast(
                device_type=self.device.type,
                dtype=torch.bfloat16,
                enabled=(self.device.type == "cuda"),
            ):
                v0 = self.model(r, t, cond).float()

            r_euler = r + dt * v0
            t_next = torch.full((budget, 1), t1, device=self.device)
            with torch.autocast(
                device_type=self.device.type,
                dtype=torch.bfloat16,
                enabled=(self.device.type == "cuda"),
            ):
                v1 = self.model(r_euler, t_next, cond).float()

            r = r + 0.5 * dt * (v0 + v1)

        residual = r.detach().cpu().numpy().astype(np.float32)
        base = self.base_model.predict(c_np.reshape(1, -1))[0].astype(np.float32)
        X_generated = base[None, :] + residual * self.resid_scale[None, :]

        return [
            {
                "attempt": i + 1,
                "x": X_generated[i].astype(np.float32),
                "tool": "scfm",
            }
            for i in range(budget)
        ]
