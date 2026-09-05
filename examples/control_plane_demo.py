#!/usr/bin/env python3
"""
Lightweight control-plane demonstration.

This example intentionally uses mock synthesis tools so it can run without
traffic datasets or pretrained checkpoints. It demonstrates the behavior that
is central to the paper:

    Tool A(intent z) -> verifier failure -> Tool B(intent z)

The user intent is never automatically relaxed or rewritten.
"""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import json
import sys

# Allow `python examples/control_plane_demo.py` from the repository root.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from agent.router import IntentPreservingRouter
from agent.tool_registry import FrozenToolRegistry


class DemoTool:
    def __init__(self, name: str, candidate_tag: str):
        self.name = name
        self.candidate_tag = candidate_tag
        self.received_intents = []

    def generate_candidates(self, intent, budget=8):
        self.received_intents.append(deepcopy(intent))
        return [
            {
                "attempt": 1,
                "tool": self.name,
                "candidate_tag": self.candidate_tag,
            }
        ]


class DemoVerifier:
    """
    Reject the first tool and accept the second.

    The decision is intentionally deterministic and independent of any traffic
    model; this demo is about agent control flow, not generation quality.
    """

    def verify(self, candidate, intent):
        accepted = candidate["candidate_tag"] == "pass"
        return {
            "verified": accepted,
            "semantic_ok": accepted,
            "plausibility_ok": accepted,
            "distance": 0.0 if accepted else 1.0,
        }


def main():
    intent = {
        "dataset": "cesnet_ts24",
        "traffic_volume": "high",
        "packet_intensity": "medium",
        "flow_intensity": "low",
        "burstiness": "high",
        "packet_burstiness": "medium",
        "flow_duration": "low",
    }

    # Frozen final CESNET order: CFM -> Copula.
    cfm = DemoTool("cfm", candidate_tag="fail")
    copula = DemoTool("copula", candidate_tag="pass")

    router = IntentPreservingRouter(
        registry=FrozenToolRegistry(),
        attempt_budget=8,
    )

    result = router.run(
        intent,
        tools={
            "cfm": cfm,
            "copula": copula,
        },
        verifier=DemoVerifier(),
    )

    assert result.accepted is True
    assert result.final_intent == intent
    assert cfm.received_intents == [intent]
    assert copula.received_intents == [intent]
    assert result.route_trace == ["cfm", "copula"]

    print("Agentic Traffic Synthesis — control-plane demo")
    print("=" * 56)
    print("Original intent:")
    print(json.dumps(intent, indent=2))
    print()
    print("Frozen route:", " -> ".join(result.route_trace))
    print("Final tool:", result.final_tool)
    print("Verified:", result.accepted)
    print("Intent preserved:", result.final_intent == intent)
    print()
    print("PASS: fallback changed the tool, not the user intent.")


if __name__ == "__main__":
    main()
