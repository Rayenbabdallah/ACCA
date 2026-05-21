"""
test_planner_agent.py - Exploration/exploitation switch and planner tests.

Part of ACCA (Active Causal Compression Agent)
ARC Prize 2026 . Paper Track
"""
from __future__ import annotations

import numpy as np

from src.agent import ACCAAgent, MechanicMemory
from src.hypothesis.goal_inference import ReachPositionGoal
from src.hypothesis.hypothesis_bank import HypothesisBank
from src.hypothesis.mechanic_dsl import ActionIsCondition, CausalRule, Hypothesis, MoveEffect
from src.perception.event_extractor import ActionEnum
from src.perception.frame_parser import ObjectGraph, ObjectNode
from src.planning.planner import Planner


def _node(left: int = 1) -> ObjectNode:
    pixels = frozenset({(1, left), (1, left + 1), (2, left), (2, left + 1)})
    return ObjectNode(
        obj_id=0,
        color=3,
        bbox=(left, 1, left + 1, 2),
        centroid=(left + 0.5, 1.5),
        area=4,
        pixels=pixels,
    )


def _graph(left: int = 1) -> ObjectGraph:
    node = _node(left)
    return ObjectGraph(nodes={0: node}, edges=[], frame_index=0)


def _frame(left: int = 1) -> np.ndarray:
    frame = np.zeros((8, 8), dtype=np.uint8)
    frame[1:3, left : left + 2] = 3
    return frame


def _click_frame() -> np.ndarray:
    frame = np.zeros((8, 8), dtype=np.uint8)
    frame[2, 5] = 14
    frame[6, 1] = 15
    return frame


def _push_target_frame() -> np.ndarray:
    frame = np.zeros((8, 8), dtype=np.uint8)
    frame[3, 1] = 2
    frame[3, 6] = 3
    return frame


def _push_right_hypothesis() -> Hypothesis:
    return Hypothesis([CausalRule([ActionIsCondition(ActionEnum.ACTION2)], [MoveEffect(0, 1, 0)])])


class FixedEIG:
    def __init__(self, action):
        self.action = action
        self.calls = 0

    def select_action(self, state, bank, action_space):
        self.calls += 1
        return self.action


def test_planner_compiles_reach_position_plan():
    planner = Planner(max_depth=5)
    goal = ReachPositionGoal(obj_id=0, target=(3.5, 1.5))

    plan = planner.compile_plan(
        _graph(left=1),
        _push_right_hypothesis(),
        goal,
        [ActionEnum.ACTION2],
    )

    assert plan == [ActionEnum.ACTION2, ActionEnum.ACTION2]


def test_planner_returns_none_when_goal_unreachable():
    planner = Planner(max_depth=2)
    goal = ReachPositionGoal(obj_id=0, target=(6.5, 1.5))

    plan = planner.compile_plan(
        _graph(left=1),
        _push_right_hypothesis(),
        goal,
        [ActionEnum.ACTION2],
    )

    assert plan is None


def test_agent_uses_eig_while_entropy_high():
    agent = ACCAAgent()
    agent.reset({"initial_grid": _frame(1), "action_space": ["ACTION1", "ACTION2"]})
    agent.eig_selector = FixedEIG("ACTION1")

    action = agent.step(_frame(1))

    assert action == "ACTION1"
    assert agent.eig_selector.calls == 1


def test_agent_expands_coordinate_action_for_click_targets():
    agent = ACCAAgent()
    agent.reset({"initial_grid": _click_frame(), "action_space": ["ACTION6"]})
    agent.eig_selector = FixedEIG("ACTION1")

    action = agent.step(_click_frame())

    assert action == "ACTION6 2 5"
    assert agent.eig_selector.calls == 0


def test_agent_pushes_visible_movable_toward_target():
    agent = ACCAAgent()
    agent.reset({"initial_grid": _push_target_frame(), "action_space": ["ACTION1", "ACTION2"]})
    agent.eig_selector = FixedEIG("ACTION1")

    action = agent.step(_push_target_frame())

    assert action == "ACTION2"
    assert agent.eig_selector.calls == 0


def test_memory_generates_chain_extension_programs():
    memory = MechanicMemory()
    memory.store_program("game", ["ACTION1", "ACTION2", "ACTION2"])

    candidates = memory.candidate_programs(
        "game",
        ["ACTION1", "ACTION2", "ACTION3", "ACTION4", "ACTION7"],
    )

    assert candidates[0] == ["ACTION1", "ACTION2", "ACTION3"]
    assert ["ACTION1", "ACTION1", "ACTION7", "ACTION2"] in candidates


def test_agent_reuses_program_memory_with_prefix_continuation():
    memory = MechanicMemory()
    memory.store_program("game", ["ACTION1"])
    frame = np.zeros((8, 8), dtype=np.uint8)
    frame[2:6, 2:6] = 3

    agent = ACCAAgent(memory=memory)
    agent.reset({"game_id": "game", "initial_grid": frame, "action_space": ["ACTION1", "ACTION2"]})
    agent.eig_selector = FixedEIG("ACTION1")

    assert agent.step(frame) == "ACTION1"
    assert agent.step(frame) == "ACTION2"
    assert agent.step(frame) == "ACTION2"
    assert agent.eig_selector.calls == 0


def test_agent_canonicalizes_numeric_action_space():
    agent = ACCAAgent()

    agent.reset({"initial_grid": _frame(1), "action_space": ["1", "GameAction.ACTION6", "0"]})

    assert agent.action_space == ["ACTION1", "ACTION6", "RESET"]


def test_agent_suppresses_long_same_action_bursts():
    agent = ACCAAgent()
    agent.reset({"initial_grid": _frame(1), "action_space": ["ACTION1", "ACTION2"]})
    agent._last_base_action = "ACTION1"
    agent._base_action_streak = 12

    assert agent._base_suppressed("ACTION1")
    assert agent._alternate_simple_action(exclude="ACTION1") == "ACTION2"


def test_effective_simple_action_avoids_suppressed_family():
    agent = ACCAAgent()
    agent.reset({"initial_grid": _frame(1), "action_space": ["ACTION1", "ACTION2"]})
    agent.stats.record("ACTION1", changed=True, novel=True)
    agent.stats.record("ACTION2", changed=False, novel=False)
    agent._last_base_action = "ACTION1"
    agent._base_action_streak = 8

    assert agent._effective_simple_action() == "ACTION2"


def test_heuristic_push_action_passes_through_burst_suppression():
    frame = _push_target_frame()
    agent = ACCAAgent()
    agent.reset({"initial_grid": frame, "action_space": ["ACTION1", "ACTION2"]})
    assert agent.prev_tracked is not None
    agent._last_base_action = "ACTION2"
    agent._base_action_streak = 8

    assert agent._heuristic_action(agent.prev_tracked) == "ACTION1"


def test_coordinate_action_avoids_reclicked_cells():
    agent = ACCAAgent()
    agent.reset({"initial_grid": _click_frame(), "action_space": ["ACTION6"]})
    assert agent.prev_tracked is not None

    first = agent._coordinate_action(agent.prev_tracked)
    second = agent._coordinate_action(agent.prev_tracked)

    assert first == "ACTION6 2 5"
    assert second != first


def test_agent_switches_to_planner_when_entropy_low():
    agent = ACCAAgent()
    agent.reset({"initial_grid": _frame(1), "action_space": [ActionEnum.ACTION2]})
    h = _push_right_hypothesis()
    agent.bank = HypothesisBank(seed_hypotheses=[h])
    agent.goal_inference.goals = [ReachPositionGoal(obj_id=0, target=(2.5, 1.5))]
    agent.eig_selector = FixedEIG(ActionEnum.ACTION1)

    action = agent.step(_frame(1))

    assert action == ActionEnum.ACTION2
    assert agent.eig_selector.calls == 0


def test_agent_falls_back_to_eig_when_no_plan():
    agent = ACCAAgent()
    agent.reset({"initial_grid": _frame(1), "action_space": [ActionEnum.ACTION2]})
    agent.bank = HypothesisBank(seed_hypotheses=[_push_right_hypothesis()])
    agent.goal_inference.goals = [ReachPositionGoal(obj_id=0, target=(7.5, 1.5))]
    agent.planner = Planner(max_depth=1)
    agent.eig_selector = FixedEIG(ActionEnum.ACTION2)

    action = agent.step(_frame(1))

    assert action == ActionEnum.ACTION2
    assert agent.eig_selector.calls == 1


def test_agent_on_level_complete_updates_goal_confidence():
    agent = ACCAAgent()
    agent.reset({"initial_grid": _frame(1), "action_space": [ActionEnum.ACTION2]})
    agent.bank = HypothesisBank(seed_hypotheses=[_push_right_hypothesis()])

    agent.on_level_complete(_frame(2), success=True)

    assert any(goal.confidence > 0.5 for goal in agent.goal_inference.goals)
