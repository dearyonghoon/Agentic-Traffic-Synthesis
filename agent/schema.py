from __future__ import annotations

from copy import deepcopy
import json
import re
from typing import Any, Dict, Iterable, Optional, Tuple

LEVELS = ("low", "medium", "high")
LEVEL_ORDER = {"low": 0, "medium": 1, "high": 2}

GAVIST_APPS = (
    "Crave",
    "LOL",
    "Netflix",
    "PrimeVideo",
    "TFT",
    "VAL",
    "YouTube",
)

GAVIST_KEYS = (
    "traffic_intensity",
    "event_intensity",
    "burstiness",
    "artt",
)

CESNET_KEYS = (
    "traffic_volume",
    "packet_intensity",
    "flow_intensity",
    "burstiness",
    "packet_burstiness",
    "flow_duration",
)

MAWI_KEYS = (
    "traffic_volume",
    "packet_intensity",
    "packet_size",
    "byte_burstiness",
    "peakiness",
)

ALL_COMPILER_FIELDS = (
    "dataset",
    "application",
    "traffic_intensity",
    "event_intensity",
    "artt",
    "traffic_volume",
    "packet_intensity",
    "flow_intensity",
    "burstiness",
    "packet_burstiness",
    "flow_duration",
    "needs_clarification",
    "ambiguity_reason",
)


def semantic_keys(dataset: str) -> Tuple[str, ...]:
    if dataset == "gavist5g":
        return GAVIST_KEYS
    if dataset == "cesnet_ts24":
        return CESNET_KEYS
    if dataset == "mawi":
        return MAWI_KEYS
    return ()


def verifier_keys(dataset: str) -> Tuple[str, ...]:
    if dataset == "gavist5g":
        return ("application",) + GAVIST_KEYS
    return semantic_keys(dataset)


def normalize_application(x: Any) -> Any:
    if x is None:
        return None

    s = re.sub(r"[^a-z0-9]", "", str(x).lower())
    aliases = {
        "crave": "Crave",
        "lol": "LOL",
        "leagueoflegends": "LOL",
        "netflix": "Netflix",
        "primevideo": "PrimeVideo",
        "amazonprimevideo": "PrimeVideo",
        "tft": "TFT",
        "teamfighttactics": "TFT",
        "val": "VAL",
        "valorant": "VAL",
        "youtube": "YouTube",
    }
    return aliases.get(s, x)


def normalize_level(x: Any) -> Any:
    if x is None:
        return None

    s = str(x).strip().lower()
    aliases = {
        "low": "low",
        "light": "low",
        "small": "low",
        "sparse": "low",
        "short": "low",
        "medium": "medium",
        "moderate": "medium",
        "mid": "medium",
        "normal": "medium",
        "high": "high",
        "heavy": "high",
        "large": "high",
        "dense": "high",
        "long": "high",
    }
    return aliases.get(s, s)


def extract_first_json_object(text: Any) -> Optional[Dict[str, Any]]:
    if text is None:
        return None

    s = str(text).strip()
    s = re.sub(r"^```(?:json)?\s*", "", s, flags=re.IGNORECASE)
    s = re.sub(r"\s*```$", "", s)

    start = s.find("{")
    if start < 0:
        return None

    depth = 0
    in_string = False
    escape = False

    for i in range(start, len(s)):
        ch = s[i]

        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue

        if ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(s[start:i + 1])
                except Exception:
                    return None
    return None


def normalize_compiled_object(obj: Any) -> Optional[Dict[str, Any]]:
    if not isinstance(obj, dict):
        return None

    out = {k: obj.get(k, None) for k in ALL_COMPILER_FIELDS}

    if out["dataset"] is not None:
        ds = str(out["dataset"]).strip().lower()
        aliases = {
            "gavist": "gavist5g",
            "gavist5g": "gavist5g",
            "gavist_5g": "gavist5g",
            "cesnet": "cesnet_ts24",
            "cesnet_ts24": "cesnet_ts24",
            "cesnet-timeseries24": "cesnet_ts24",
        }
        out["dataset"] = aliases.get(ds, ds)

    out["application"] = normalize_application(out["application"])

    for key in set(GAVIST_KEYS + CESNET_KEYS):
        out[key] = normalize_level(out[key])

    val = out["needs_clarification"]
    if isinstance(val, str):
        out["needs_clarification"] = val.strip().lower() == "true"
    else:
        out["needs_clarification"] = bool(val) if val is not None else False

    return out


def schema_valid_compiler_output(obj: Any) -> bool:
    """Frozen 05b compiler schema: CESNET + GAViST only."""
    if not isinstance(obj, dict):
        return False

    ds = obj.get("dataset")
    if ds not in {"gavist5g", "cesnet_ts24"}:
        return False

    for key in semantic_keys(ds):
        value = obj.get(key)
        if value is not None and value not in LEVELS:
            return False

    if ds == "gavist5g":
        app = obj.get("application")
        if app is not None and app not in GAVIST_APPS:
            return False

    if ds == "cesnet_ts24" and obj.get("application") is not None:
        return False

    return True


def execution_guard(pred: Any):
    """
    Deterministic execution gate.

    The frozen natural-language compiler supports CESNET/GAViST.
    The public agent also accepts already-structured MAWI intents.
    """
    if not isinstance(pred, dict):
        return False, "parse_failure", None

    if bool(pred.get("needs_clarification", False)):
        return False, "needs_clarification", None

    ds = pred.get("dataset")
    if ds not in {"gavist5g", "cesnet_ts24", "mawi"}:
        return False, "invalid_dataset", None

    intent = {"dataset": ds}

    if ds == "gavist5g":
        app = pred.get("application")
        if app not in GAVIST_APPS:
            return False, "invalid_or_missing_application", None
        intent["application"] = app

    for key in semantic_keys(ds):
        value = pred.get(key)
        if value not in LEVELS:
            return False, f"invalid_or_missing_{key}", None
        intent[key] = value

    return True, "ready", intent


def intent_signature(intent: Dict[str, Any]) -> Tuple[Any, ...]:
    ds = intent["dataset"]
    return (ds,) + tuple(intent.get(k) for k in verifier_keys(ds))


def intents_exact(a: Any, b: Any) -> bool:
    if not isinstance(a, dict) or not isinstance(b, dict):
        return False
    if a.get("dataset") != b.get("dataset"):
        return False
    keys = verifier_keys(a.get("dataset"))
    return bool(keys) and all(str(a.get(k)) == str(b.get(k)) for k in keys)


def canonical_intent_copy(intent: Dict[str, Any]) -> Dict[str, Any]:
    return deepcopy({
        "dataset": intent["dataset"],
        **(
            {"application": intent["application"]}
            if intent["dataset"] == "gavist5g"
            else {}
        ),
        **{k: intent[k] for k in semantic_keys(intent["dataset"])},
    })
