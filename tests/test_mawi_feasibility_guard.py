from itertools import product
from pathlib import Path
import json

from agent.agent import AgenticTrafficSynthesis
from agent.feasibility_guard import FeasibilityGuard

LEVELS = ("low", "medium", "high")
FIELDS = (
    "traffic_volume",
    "packet_intensity",
    "packet_size",
    "byte_burstiness",
    "peakiness",
)


def metadata_path():
    return (
        Path(__file__).resolve().parents[1]
        / "configs"
        / "mawi_metadata_v1.json"
    )


def build_guard():
    return FeasibilityGuard.from_mawi_metadata(str(metadata_path()))


def test_mawi_frozen_semantic_space_counts():
    """
    Frozen 10d audit:
    3^5 = 243 MAWI semantic combinations,
    of which 171 are hard-consistent and 72 are hard-infeasible.
    """
    guard = build_guard()

    feasible = 0
    infeasible = 0

    for combo in product(LEVELS, repeat=len(FIELDS)):
        intent = {
            "dataset": "mawi",
            **dict(zip(FIELDS, combo)),
        }
        decision = guard.check(intent)

        if decision.feasible:
            feasible += 1
        else:
            infeasible += 1

    assert feasible == 171
    assert infeasible == 72
    assert feasible + infeasible == 243


def test_hard_infeasible_mawi_intent_is_rejected_before_tools():
    """
    Find one frozen hard-infeasible MAWI intent and confirm that the public
    agent terminates before any synthesis tool is needed.
    """
    guard = build_guard()

    bad_intent = None
    for combo in product(LEVELS, repeat=len(FIELDS)):
        intent = {
            "dataset": "mawi",
            **dict(zip(FIELDS, combo)),
        }
        if not guard.check(intent).feasible:
            bad_intent = intent
            break

    assert bad_intent is not None

    agent = AgenticTrafficSynthesis(
        tools_by_dataset={},
        verifiers={},
        feasibility_guard=guard,
    )

    result = agent.run(bad_intent)

    assert result["status"] == "safe_failure"
    assert result["action"] == "safe_reject_hard_infeasible"
    assert result["intent"] == bad_intent
