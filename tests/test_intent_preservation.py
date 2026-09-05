from copy import deepcopy

from agent.router import IntentPreservingRouter
from agent.tool_registry import FrozenToolRegistry


class RecordingTool:
    def __init__(self, name):
        self.name = name
        self.seen = []

    def generate_candidates(self, intent, budget=8):
        self.seen.append(deepcopy(intent))
        return [{"attempt": 1, "z": [0.0], "tool": self.name}]


class FailingVerifier:
    def verify(self, candidate, intent):
        return {
            "verified": False,
            "semantic_ok": False,
            "plausibility_ok": False,
            "distance": 1.0,
        }


def test_fallback_preserves_original_intent():
    intent = {
        "dataset": "cesnet_ts24",
        "traffic_volume": "high",
        "packet_intensity": "medium",
        "flow_intensity": "low",
        "burstiness": "high",
        "packet_burstiness": "medium",
        "flow_duration": "low",
    }

    cfm = RecordingTool("cfm")
    copula = RecordingTool("copula")

    router = IntentPreservingRouter(FrozenToolRegistry(), attempt_budget=8)
    result = router.run(
        intent,
        tools={"cfm": cfm, "copula": copula},
        verifier=FailingVerifier(),
    )

    assert result.accepted is False
    assert result.final_intent == intent
    assert cfm.seen == [intent]
    assert copula.seen == [intent]
    assert result.route_trace == ["cfm", "copula"]
