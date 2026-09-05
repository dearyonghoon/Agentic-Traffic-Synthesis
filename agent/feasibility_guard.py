from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import json
import math
from typing import Any, Dict, Mapping, Optional, Tuple

import numpy as np

from .schema import LEVELS, MAWI_KEYS


@dataclass
class FeasibilityDecision:
    feasible: bool
    action: str
    reason: str
    details: Dict[str, Any] = field(default_factory=dict)


class FeasibilityGuard:
    """
    Structural feasibility guard.

    For CESNET and GAViST, the frozen public behavior is pass-through because
    no exact executable relation was used there.

    For MAWI, the guard implements the exact identity
        packet_size = traffic_volume / packet_intensity
    in semantic-bin interval form.
    """

    def __init__(self, mawi_thresholds_log1p: Optional[Mapping[str, Tuple[float, float]]] = None):
        self.mawi_thresholds_log1p = (
            {k: tuple(map(float, v)) for k, v in mawi_thresholds_log1p.items()}
            if mawi_thresholds_log1p is not None
            else None
        )

    @classmethod
    def from_mawi_metadata(cls, metadata_path: str):
        meta = json.loads(Path(metadata_path).read_text(encoding="utf-8"))
        return cls(meta["semantic_thresholds_log1p"])

    def _raw_semantic_interval(self, field: str, level: str):
        if self.mawi_thresholds_log1p is None:
            raise RuntimeError(
                "MAWI thresholds are not configured. "
                "Load the frozen 10c metadata via FeasibilityGuard.from_mawi_metadata(...)."
            )

        t1, t2 = self.mawi_thresholds_log1p[field]
        b1 = float(np.expm1(t1))
        b2 = float(np.expm1(t2))

        if level == "low":
            return 0.0, b1
        if level == "medium":
            return b1, b2
        if level == "high":
            return b2, np.inf
        raise ValueError(level)

    @staticmethod
    def _safe_ratio_lower(v_low, p_high):
        if np.isinf(p_high):
            return 0.0
        if p_high <= 0:
            return np.inf
        return float(v_low / p_high)

    @staticmethod
    def _safe_ratio_upper(v_high, p_low):
        if np.isinf(v_high):
            return np.inf
        if p_low <= 0:
            return np.inf
        return float(v_high / p_low)

    def _check_mawi(self, intent: Dict[str, Any]) -> FeasibilityDecision:
        for field in MAWI_KEYS:
            if intent.get(field) not in LEVELS:
                return FeasibilityDecision(
                    feasible=False,
                    action="safe_reject_invalid_intent",
                    reason=f"invalid_or_missing_{field}",
                )

        v_lo, v_hi = self._raw_semantic_interval(
            "traffic_volume", intent["traffic_volume"]
        )
        p_lo, p_hi = self._raw_semantic_interval(
            "packet_intensity", intent["packet_intensity"]
        )
        s_lo, s_hi = self._raw_semantic_interval(
            "packet_size", intent["packet_size"]
        )

        implied_lo = self._safe_ratio_lower(v_lo, p_hi)
        implied_hi = self._safe_ratio_upper(v_hi, p_lo)

        overlap_lo = max(implied_lo, s_lo)
        overlap_hi = min(implied_hi, s_hi)
        consistent = bool(overlap_lo <= overlap_hi)

        details = {
            "requested_packet_size_low": s_lo,
            "requested_packet_size_high": s_hi,
            "implied_packet_size_low": implied_lo,
            "implied_packet_size_high": implied_hi,
            "overlap_low": overlap_lo,
            "overlap_high": overlap_hi,
        }

        if not consistent:
            return FeasibilityDecision(
                feasible=False,
                action="safe_reject_hard_infeasible",
                reason="mawi_packet_size_identity_conflict",
                details=details,
            )

        return FeasibilityDecision(
            feasible=True,
            action="proceed",
            reason="hard_consistent",
            details=details,
        )

    def check(self, intent: Dict[str, Any]) -> FeasibilityDecision:
        ds = intent.get("dataset")
        if ds == "mawi":
            return self._check_mawi(intent)

        if ds in {"cesnet_ts24", "gavist5g"}:
            return FeasibilityDecision(
                feasible=True,
                action="proceed",
                reason="no_exact_structural_guard_for_domain",
            )

        return FeasibilityDecision(
            feasible=False,
            action="safe_reject_invalid_intent",
            reason="invalid_dataset",
        )
