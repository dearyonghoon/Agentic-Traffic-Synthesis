from __future__ import annotations

from dataclasses import asdict
from typing import Any, Dict, Mapping, Optional, Union

from .schema import execution_guard
from .feasibility_guard import FeasibilityGuard
from .router import IntentPreservingRouter
from .tool_registry import FrozenToolRegistry


class AgenticTrafficSynthesis:
    """
    End-to-end public agent.

    A generator produces traffic, whereas the agent decides whether and how
    traffic generation should be carried out.
    """

    def __init__(
        self,
        *,
        tools_by_dataset: Mapping[str, Mapping[str, Any]],
        verifiers: Mapping[str, Any],
        compiler: Optional[Any] = None,
        feasibility_guard: Optional[FeasibilityGuard] = None,
        registry: Optional[FrozenToolRegistry] = None,
        attempt_budget: int = 8,
    ):
        self.tools_by_dataset = dict(tools_by_dataset)
        self.verifiers = dict(verifiers)
        self.compiler = compiler
        self.feasibility_guard = feasibility_guard or FeasibilityGuard()
        self.registry = registry or FrozenToolRegistry()
        self.router = IntentPreservingRouter(
            self.registry,
            attempt_budget=attempt_budget,
        )

    def run(self, request: Union[str, Dict[str, Any]]):
        if isinstance(request, str):
            if self.compiler is None:
                return {
                    "status": "safe_failure",
                    "action": "compiler_unavailable",
                }
            compiled = self.compiler.compile(request)
        else:
            compiled = dict(request)

        ready, reason, intent = execution_guard(compiled)
        if not ready:
            return {
                "status": "safe_failure",
                "action": reason,
                "compiled_intent": compiled,
            }

        feasibility = self.feasibility_guard.check(intent)
        if not feasibility.feasible:
            return {
                "status": "safe_failure",
                "action": feasibility.action,
                "reason": feasibility.reason,
                "intent": intent,
                "feasibility": asdict(feasibility),
            }

        ds = intent["dataset"]
        if ds not in self.tools_by_dataset:
            return {
                "status": "safe_failure",
                "action": "tools_unavailable",
                "intent": intent,
            }
        if ds not in self.verifiers:
            return {
                "status": "safe_failure",
                "action": "verifier_unavailable",
                "intent": intent,
            }

        routed = self.router.run(
            intent,
            tools=self.tools_by_dataset[ds],
            verifier=self.verifiers[ds],
        )

        return {
            "status": "verified" if routed.accepted else "safe_failure",
            "action": routed.action,
            "intent": routed.original_intent,
            "final_intent": routed.final_intent,
            "final_tool": routed.final_tool,
            "tool_calls": routed.tool_calls,
            "failed_tool_calls": routed.failed_tool_calls,
            "route_trace": routed.route_trace,
            "output": routed.output,
            "feasibility": asdict(feasibility),
        }
