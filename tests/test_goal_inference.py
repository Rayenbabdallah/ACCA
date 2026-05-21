"""
test_goal_inference.py - Goal template and confidence update tests.

Part of ACCA (Active Causal Compression Agent)
ARC Prize 2026 . Paper Track
"""
from __future__ import annotations

import pytest

from src.hypothesis.goal_inference import (
    AllSameColorGoal,
    ClearGridGoal,
    ConnectObjectsGoal,
    CountObjectsGoal,
    FillRegionGoal,
    GoalInference,
    MatchPatternGoal,
    MaximizeAreaGoal,
    ReachPositionGoal,
    SortObjectsGoal,
    SymmetryGoal,
)
from src.perception.frame_parser import ObjectGraph, ObjectNode, RelEdge, RelationType


def _node(obj_id: int, color: int, pixels) -> ObjectNode:
    pix = frozenset(pixels)
    rows = [r for r, _ in pix]
    cols = [c for _, c in pix]
    return ObjectNode(
        obj_id=obj_id,
        color=color,
        bbox=(min(cols), min(rows), max(cols), max(rows)),
        centroid=(sum(cols) / len(pix), sum(rows) / len(pix)),
        area=len(pix),
        pixels=pix,
    )


def _state(nodes, edges=None) -> ObjectGraph:
    return ObjectGraph(
        nodes={node.obj_id: node for node in nodes},
        edges=list(edges or []),
        frame_index=0,
    )


def test_reach_position_goal_satisfied():
    state = _state([_node(0, 3, [(2, 4)])])

    assert ReachPositionGoal(obj_id=0, target=(4.0, 2.0)).satisfied(state)
    assert not ReachPositionGoal(obj_id=0, target=(5.0, 2.0)).satisfied(state)


def test_all_same_color_goal_satisfied():
    same = _state([_node(0, 3, [(0, 0)]), _node(1, 3, [(1, 1)])])
    mixed = _state([_node(0, 3, [(0, 0)]), _node(1, 4, [(1, 1)])])

    assert AllSameColorGoal().satisfied(same)
    assert AllSameColorGoal(color=3).satisfied(same)
    assert not AllSameColorGoal().satisfied(mixed)


def test_clear_and_count_goals():
    empty = _state([])
    state = _state([_node(0, 3, [(0, 0)]), _node(1, 3, [(1, 1)])])

    assert ClearGridGoal().satisfied(empty)
    assert CountObjectsGoal(color=3, count=2).satisfied(state)
    assert not CountObjectsGoal(color=3, count=1).satisfied(state)


def test_match_pattern_goal_compares_colored_pixels():
    state = _state([_node(0, 3, [(0, 0), (0, 1)])])
    pattern = MatchPatternGoal.from_state(state)

    assert pattern.satisfied(state)
    assert not pattern.satisfied(_state([_node(0, 3, [(0, 0)])]))


def test_connect_objects_goal_uses_touching_edges():
    state = _state(
        [_node(0, 3, [(0, 0)]), _node(1, 4, [(0, 1)])],
        [RelEdge(0, 1, RelationType.TOUCHING)],
    )

    assert ConnectObjectsGoal(obj_id_a=1, obj_id_b=0).satisfied(state)


def test_fill_region_goal_requires_all_cells_with_color():
    state = _state([_node(0, 5, [(1, 1), (1, 2), (2, 1), (2, 2)])])

    assert FillRegionGoal(region=(1, 1, 2, 2), color=5).satisfied(state)
    assert not FillRegionGoal(region=(1, 1, 2, 3), color=5).satisfied(state)


def test_sort_objects_goal_by_axis():
    sorted_state = _state(
        [
            _node(0, 3, [(0, 0)]),
            _node(1, 4, [(0, 2)]),
            _node(2, 5, [(0, 4)]),
        ]
    )
    unsorted_state = _state(
        [
            _node(0, 3, [(0, 4)]),
            _node(1, 4, [(0, 2)]),
            _node(2, 5, [(0, 0)]),
        ]
    )

    assert SortObjectsGoal(axis="x").satisfied(sorted_state)
    assert not SortObjectsGoal(axis="x").satisfied(unsorted_state)


def test_symmetry_goal_vertical_horizontal_and_rotational():
    vertical = _state([_node(0, 3, [(0, 0), (0, 3)])])
    horizontal = _state([_node(0, 4, [(0, 1), (3, 1)])])
    rotational = _state([_node(0, 5, [(0, 0), (3, 3)])])

    assert SymmetryGoal(axis="vertical", shape=(1, 4)).satisfied(vertical)
    assert SymmetryGoal(axis="horizontal", shape=(4, 2)).satisfied(horizontal)
    assert SymmetryGoal(axis="rotational", shape=(4, 4)).satisfied(rotational)
    assert not SymmetryGoal(axis="vertical", shape=(1, 4)).satisfied(horizontal)


def test_maximize_area_goal():
    state = _state([_node(0, 3, [(0, 0), (0, 1), (1, 0)])])

    assert MaximizeAreaGoal(color=3, target_area=3).satisfied(state)
    assert not MaximizeAreaGoal(color=3, target_area=4).satisfied(state)


def test_update_on_success_boosts_satisfied_goals_and_top_goal():
    state = _state([_node(0, 3, [(2, 4)])])
    inference = GoalInference()

    inference.update_on_success(state)

    reach = [
        goal for goal in inference.goals if goal.name() == "reach_position:0:4:2"
    ][0]
    assert reach.confidence == pytest.approx(0.8)
    assert inference.top_goal().confidence == pytest.approx(0.8)


def test_update_on_step_and_failure_adjust_recently_active_goals():
    state = _state([])
    inference = GoalInference()
    clear = [goal for goal in inference.goals if goal.name() == "clear_grid"][0]

    inference.update_on_step(state)
    assert clear.confidence == pytest.approx(0.55)

    inference.update_on_failure()
    assert clear.confidence == pytest.approx(0.45)


def test_top_k_goals_sorted_by_confidence():
    inference = GoalInference()
    inference.goals[0].confidence = 0.9
    inference.goals[1].confidence = 0.7

    top_two = inference.top_k_goals(2)

    assert [goal.confidence for goal in top_two] == [0.9, 0.7]
