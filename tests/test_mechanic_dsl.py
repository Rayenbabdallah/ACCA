"""
test_mechanic_dsl.py - Tests for executable causal mechanics.

Part of ACCA (Active Causal Compression Agent)
ARC Prize 2026 . Paper Track
"""
from __future__ import annotations

from src.hypothesis.hypothesis_seeds import load_seeds
from src.hypothesis.mechanic_dsl import (
    ActionIsCondition,
    CausalRule,
    DestroyEffect,
    Hypothesis,
    MoveEffect,
    ObjColorCondition,
    ObjCountCondition,
    ObjPropertyCondition,
    ObjRelationCondition,
    RecolorEffect,
    SpawnEffect,
    ToggleEffect,
)
from src.perception.event_extractor import ActionEnum, AtomicEvent, canonical_delta
from src.perception.frame_parser import ObjectGraph, ObjectNode, RelEdge, RelationType


def _node(obj_id: int, color: int, top: int, left: int, size: int = 2) -> ObjectNode:
    pixels = frozenset(
        (r, c) for r in range(top, top + size) for c in range(left, left + size)
    )
    return ObjectNode(
        obj_id=obj_id,
        color=color,
        bbox=(left, top, left + size - 1, top + size - 1),
        centroid=(left + (size - 1) / 2, top + (size - 1) / 2),
        area=size * size,
        pixels=pixels,
    )


def _state() -> ObjectGraph:
    nodes = {
        0: _node(0, 3, 1, 1),
        1: _node(1, 4, 1, 3),
        2: _node(2, 5, 5, 1),
        3: _node(3, 6, 5, 5),
    }
    edges = [RelEdge(0, 1, RelationType.TOUCHING)]
    return ObjectGraph(nodes=nodes, edges=edges, frame_index=0)


def test_conditions_evaluate_expected_values():
    state = _state()

    assert ObjColorCondition(0, 3).evaluate(state, ActionEnum.ACTION1)
    assert ActionIsCondition(ActionEnum.ACTION1).evaluate(state, "ACTION1")
    assert ObjRelationCondition(0, 1, RelationType.TOUCHING).evaluate(state, "ACTION1")
    assert ObjCountCondition(color=3, op="eq", count=1).evaluate(state, "ACTION1")
    assert ObjPropertyCondition(0, "area", "eq", 4).evaluate(state, "ACTION1")
    assert ObjPropertyCondition(0, "centroid_x", "lt", 3).evaluate(state, "ACTION1")


def test_effects_are_non_mutating_and_executable():
    state = _state()

    moved = MoveEffect(0, dx=2, dy=1).apply(state, [])
    recolored = RecolorEffect(0, 7).apply(state, [])
    spawned = SpawnEffect(8, position=(7, 7)).apply(state, [])
    destroyed = DestroyEffect(0).apply(state, [])
    toggled = ToggleEffect(0, 3, 9).apply(state, [])

    assert state.nodes[0].centroid == (1.5, 1.5)
    assert moved.nodes[0].centroid == (3.5, 2.5)
    assert recolored.nodes[0].color == 7
    assert max(spawned.nodes) == 4
    assert 0 not in destroyed.nodes
    assert toggled.nodes[0].color == 9


def test_rule_matches_and_applies_all_effects():
    rule = CausalRule(
        [ActionIsCondition(ActionEnum.ACTION1), ObjColorCondition(0, 3)],
        [MoveEffect(0, 1, 0), RecolorEffect(0, 4)],
    )
    state = _state()

    assert rule.matches(state, ActionEnum.ACTION1)
    out = rule.apply(state)

    assert out.nodes[0].centroid == (2.5, 1.5)
    assert out.nodes[0].color == 4


def test_hypothesis_execute_only_applies_matching_rules():
    state = _state()
    h = Hypothesis(
        [
            CausalRule([ActionIsCondition(ActionEnum.ACTION1)], [MoveEffect(0, 1, 0)]),
            CausalRule([ActionIsCondition(ActionEnum.ACTION2)], [MoveEffect(0, -1, 0)]),
        ]
    )

    out = h.execute(state, ActionEnum.ACTION1)

    assert out.nodes[0].centroid == (2.5, 1.5)


def test_hypothesis_predict_delta():
    state = _state()
    h = Hypothesis(
        [CausalRule([ActionIsCondition(ActionEnum.ACTION1)], [MoveEffect(0, 2, 0)])]
    )

    assert h.predict_delta(state, ActionEnum.ACTION1) == [
        AtomicEvent("MOVED", 0, {"dx": 2, "dy": 0})
    ]


def test_push_right_seed_moves_object_right_by_one():
    state = _state()
    push_right = load_seeds()[5]

    out = push_right.execute(state, ActionEnum.ACTION2)

    assert out.nodes[0].centroid == (2.5, 1.5)


def test_all_seed_hypotheses_execute_and_have_stable_metadata():
    state = _state()
    seeds = load_seeds()

    assert len(seeds) == 20
    for h in seeds:
        out = h.execute(state, ActionEnum.ACTION1)
        assert isinstance(out, ObjectGraph)
        assert h.description_length() > 0
        assert h.fingerprint() == h.fingerprint()


def test_fingerprint_ignores_rule_order():
    h1 = Hypothesis(
        [
            CausalRule([ActionIsCondition(ActionEnum.ACTION1)], [MoveEffect(0, 1, 0)]),
            CausalRule([ActionIsCondition(ActionEnum.ACTION2)], [RecolorEffect(0, 7)]),
        ]
    )
    h2 = Hypothesis(list(reversed(h1.rules)))

    assert h1.fingerprint() == h2.fingerprint()


def test_delta_comparison_is_order_independent_for_multi_effect_rule():
    state = _state()
    h = Hypothesis(
        [
            CausalRule(
                [ActionIsCondition(ActionEnum.ACTION1)],
                [MoveEffect(0, 1, 0), RecolorEffect(0, 7)],
            )
        ]
    )

    expected = [
        AtomicEvent("RECOLORED", 0, {"old_color": 3, "new_color": 7}),
        AtomicEvent("MOVED", 0, {"dx": 1, "dy": 0}),
    ]

    assert canonical_delta(h.predict_delta(state, ActionEnum.ACTION1)) == canonical_delta(expected)
