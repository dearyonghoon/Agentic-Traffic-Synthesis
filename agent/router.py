from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, Optional, Sequence

from .schema import canonical_intent_copy, intents_exact
from .tool_registry import FrozenToolRegistry


@dataclass
class RoutingResult:
    accepted: bool
    action: str
    original_intent: Dict[str, Any]
    final_intent: Dict[str, Any]
    final_tool: Optional[str] = None
    output: Optional[Dict[str, Any]] = None
    tool_calls: int = 0
    failed_tool_calls: int = 0
    route_trace: list = field(default_factory=list)


class IntentPreservingRouter:
    """
    Verifier-driven fallback with the invariant:
        executed_intent == user_intent

    No automatic semantic relaxation is implemented here.
    """

    def __init__(self, registry: FrozenToolRegistry, attempt_budget: int = 8):
        self.registry = registry
        self.attempt_budget = int(attempt_budget)

    def run(
        self,
        intent: Dict[str, Any],
        tools: Mapping[str, Any],
        verifier: Any,
    ) -> RoutingResult:
        original = canonical_intent_copy(intent)
        result = RoutingResult(
            accepted=False,
            action="safe_tool_failure",
            original_intent=deepcopy(original),
            final_intent=deepcopy(original),
        )

        order = self.registry.get_order(original["dataset"])

        for tool_name in order:
            if tool_name not in tools:
                raise KeyError(
                    f"Tool {tool_name!r} required by registry but not provided."
                )

            # Critical invariant: every tool sees the same original intent.
            tool_intent = deepcopy(original)
            assert intents_exact(tool_intent, original)

            generator = tools[tool_name]
            candidates = generator.generate_candidates(
                tool_intent,
                budget=self.attempt_budget,
            )

            result.tool_calls += 1
            result.route_trace.append(tool_name)

            best = None
            for candidate in candidates:
                verification = verifier.verify(candidate, tool_intent)
                item = {
                    **candidate,
                    "verification": verification,
                }

                if best is None:
                    best = item
                else:
                    old_v = best["verification"]
                    new_v = verification
                    old_score = (
                        int(old_v.get("semantic_ok", False)),
                        int(old_v.get("plausibility_ok", False)),
                        -float(old_v.get("distance", float("inf"))),
                    )
                    new_score = (
                        int(new_v.get("semantic_ok", False)),
                        int(new_v.get("plausibility_ok", False)),
                        -float(new_v.get("distance", float("inf"))),
                    )
                    if new_score > old_score:
                        best = item

                if verification.get("verified", False):
                    result.accepted = True
                    result.action = "verified_traffic"
                    result.final_tool = tool_name
                    result.output = item
                    result.final_intent = deepcopy(original)
                    assert intents_exact(result.final_intent, original)
                    return result

            result.failed_tool_calls += 1
            if best is not None:
                result.output = best

        result.final_intent = deepcopy(original)
        assert intents_exact(result.final_intent, original)
        return result
