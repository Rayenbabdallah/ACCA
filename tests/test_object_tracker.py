"""
test_object_tracker.py — Tests for the Hungarian-IOU object tracker.

Covers: IOU primitive edge cases, cost-matrix construction, Hungarian
assignment on rectangular matrices, splits / merges, 5-frame push
persistence (CLAUDE.md completion check), counter monotonicity, edge
rebuilding with new ids, and the stateful update() vs stateless match() APIs.

Part of ACCA (Active Causal Compression Agent)
ARC Prize 2026 · Paper Track
"""
from __future__ import annotations

from typing import Tuple

import numpy as np
import pytest

from src.perception.frame_parser import (
    FrameParser,
    ObjectGraph,
    ObjectNode,
    RelationType,
)
from src.perception.object_tracker import (
    DEFAULT_IOU_THRESHOLD,
    ObjectTracker,
    TrackingResult,
    build_cost_matrix,
    iou,
)


# ----------------------------- helpers --------------------------------------

def _node(obj_id: int, color: int, pixels) -> ObjectNode:
    pix = frozenset(pixels)
    if not pix:
        return ObjectNode(obj_id=obj_id, color=color, bbox=(0, 0, 0, 0),
                          centroid=(0.0, 0.0), area=0, pixels=pix)
    rows = [r for (r, _) in pix]
    cols = [c for (_, c) in pix]
    bbox = (min(cols), min(rows), max(cols), max(rows))
    centroid = (sum(cols) / len(pix), sum(rows) / len(pix))
    return ObjectNode(obj_id=obj_id, color=color, bbox=bbox,
                      centroid=centroid, area=len(pix), pixels=pix)


def _square_node(obj_id: int, color: int, top: int, left: int, size: int) -> ObjectNode:
    return _node(obj_id, color,
                 [(r, c) for r in range(top, top + size) for c in range(left, left + size)])


def _graph(nodes, frame_index: int = 0) -> ObjectGraph:
    return ObjectGraph(nodes={n.obj_id: n for n in nodes}, edges=[], frame_index=frame_index)


def _frame_with_movable_at(col_start: int) -> np.ndarray:
    f = np.zeros((16, 16), dtype=np.uint8)
    f[10:14, col_start:col_start + 4] = 2   # 4x4 MOVABLE_A
    f[2, 2] = 7                              # stationary marker
    return f


# ----------------------------- IOU ------------------------------------------

class TestIou:
    def test_perfect_overlap_is_one(self):
        a = _square_node(0, 3, 0, 0, 3)
        b = _square_node(1, 3, 0, 0, 3)
        assert iou(a, b) == 1.0

    def test_disjoint_pixel_sets_returns_zero(self):
        a = _square_node(0, 3, 0, 0, 3)
        b = _square_node(1, 4, 10, 10, 3)
        assert iou(a, b) == 0.0

    def test_partial_overlap_jaccard(self):
        # 4x4 squares offset by 1 in column: intersect 4x3=12, union 4x5=20
        a = _square_node(0, 3, 10, 5, 4)
        b = _square_node(1, 3, 10, 6, 4)
        assert iou(a, b) == pytest.approx(12 / 20)

    def test_bbox_disjoint_fast_path_returns_zero(self):
        a = _square_node(0, 3, 0, 0, 2)
        b = _square_node(1, 3, 100, 100, 2)
        # Even though pixel sets are also disjoint, the bbox reject should fire.
        assert iou(a, b) == 0.0

    def test_zero_area_returns_zero(self):
        empty = _node(0, 3, [])
        nonempty = _square_node(1, 3, 0, 0, 2)
        assert iou(empty, nonempty) == 0.0
        assert iou(nonempty, empty) == 0.0

    def test_iou_symmetric(self):
        a = _square_node(0, 3, 0, 0, 3)
        b = _square_node(1, 3, 1, 1, 3)
        assert iou(a, b) == pytest.approx(iou(b, a))

    def test_subset_jaccard(self):
        # b is a 2x2 subset of a 4x4 a
        a = _square_node(0, 3, 0, 0, 4)
        b_pixels = [(r, c) for r in range(2) for c in range(2)]
        b = _node(1, 3, b_pixels)
        # intersection 4, union 16 -> 0.25
        assert iou(a, b) == pytest.approx(4 / 16)


# ----------------------------- cost matrix ----------------------------------

class TestCostMatrix:
    def test_shape_and_values(self):
        a = _square_node(0, 3, 0, 0, 2)
        b = _square_node(1, 3, 5, 5, 2)
        c = _square_node(2, 3, 0, 0, 2)
        cost = build_cost_matrix([a, b], [c])
        assert cost.shape == (2, 1)
        assert cost[0, 0] == pytest.approx(0.0)
        assert cost[1, 0] == pytest.approx(1.0)

    def test_empty_inputs_yield_zero_sized_matrix(self):
        cost = build_cost_matrix([], [])
        assert cost.shape == (0, 0)


# ----------------------------- bootstrap / empty edges ----------------------

class TestBootstrap:
    def test_first_frame_spawns_all(self):
        n1 = _square_node(0, 3, 0, 0, 2)
        n2 = _square_node(1, 4, 5, 5, 2)
        t = ObjectTracker()
        result = t.match(None, _graph([n1, n2], frame_index=0))
        assert set(result.tracked.nodes) == {0, 1}
        assert result.matched == {}
        assert set(result.spawned) == {0, 1}
        assert result.destroyed == []
        assert t.next_obj_id == 2

    def test_match_with_empty_prev_is_bootstrap(self):
        t = ObjectTracker()
        empty_prev = _graph([], frame_index=0)
        curr = _graph([_square_node(0, 3, 0, 0, 2)], frame_index=1)
        result = t.match(empty_prev, curr)
        assert result.matched == {}
        assert result.spawned == [0]

    def test_match_with_empty_curr_destroys_all(self):
        t = ObjectTracker(start_obj_id=100)
        prev = _graph([_square_node(0, 3, 0, 0, 2), _square_node(1, 4, 5, 5, 2)], 0)
        curr = _graph([], frame_index=1)
        result = t.match(prev, curr)
        assert result.tracked.nodes == {}
        assert sorted(result.destroyed) == [0, 1]
        assert result.spawned == []
        assert t.next_obj_id == 100  # no allocation

    def test_match_when_both_empty_returns_empty(self):
        t = ObjectTracker()
        result = t.match(_graph([], 0), _graph([], 1))
        assert result.tracked.nodes == {}
        assert result.spawned == [] and result.destroyed == []


# ----------------------------- 1-to-1 matching ------------------------------

class TestOneToOneMatching:
    def test_identical_graphs_preserve_ids(self):
        prev = _graph([_square_node(7, 3, 0, 0, 3), _square_node(11, 4, 8, 8, 3)], 0)
        curr = _graph([_square_node(99, 3, 0, 0, 3), _square_node(100, 4, 8, 8, 3)], 1)
        t = ObjectTracker()
        result = t.match(prev, curr)
        assert set(result.tracked.nodes) == {7, 11}
        assert result.matched == {7: 7, 11: 11}
        assert result.spawned == [] and result.destroyed == []
        assert result.tracked.frame_index == 1

    def test_partial_overlap_above_threshold_matches(self):
        prev = _graph([_square_node(5, 3, 10, 5, 4)], 0)
        curr = _graph([_square_node(0, 3, 10, 6, 4)], 1)  # IOU = 0.6
        t = ObjectTracker(iou_threshold=0.3)
        result = t.match(prev, curr)
        assert result.matched == {5: 5}
        assert result.spawned == []

    def test_below_threshold_becomes_destroy_plus_spawn(self):
        prev = _graph([_square_node(5, 3, 0, 0, 2)], 0)
        curr = _graph([_square_node(0, 3, 8, 8, 2)], 1)  # IOU = 0
        t = ObjectTracker(iou_threshold=0.1, start_obj_id=50)
        result = t.match(prev, curr)
        assert result.destroyed == [5]
        assert result.spawned == [50]
        assert 5 not in result.tracked.nodes
        assert 50 in result.tracked.nodes

    def test_color_change_within_overlap_still_matches_by_pixels(self):
        # IOU is pixel-based; color shift alone shouldn't break matching
        prev = _graph([_square_node(3, 3, 0, 0, 4)], 0)
        curr = _graph([_square_node(0, 7, 0, 0, 4)], 1)  # same pixels, different color
        t = ObjectTracker()
        result = t.match(prev, curr)
        assert result.matched == {3: 3}
        # The surviving node should reflect curr's color
        assert result.tracked.nodes[3].color == 7


# ----------------------------- rectangular matrices -------------------------

class TestRectangularAssignment:
    def test_more_curr_than_prev_extras_spawn(self):
        prev = _graph([_square_node(0, 3, 0, 0, 2)], 0)
        curr = _graph(
            [_square_node(0, 3, 0, 0, 2),  # matches prev
             _square_node(1, 4, 5, 5, 2),  # new
             _square_node(2, 5, 10, 10, 2)],  # new
            frame_index=1,
        )
        t = ObjectTracker(start_obj_id=10)
        result = t.match(prev, curr)
        assert result.matched == {0: 0}
        assert sorted(result.spawned) == [10, 11]
        assert len(result.tracked.nodes) == 3

    def test_more_prev_than_curr_extras_destroyed(self):
        prev = _graph(
            [_square_node(0, 3, 0, 0, 2),
             _square_node(1, 4, 5, 5, 2),
             _square_node(2, 5, 10, 10, 2)],
            0,
        )
        curr = _graph([_square_node(0, 3, 0, 0, 2)], 1)  # only the first survives
        t = ObjectTracker()
        result = t.match(prev, curr)
        assert result.matched == {0: 0}
        assert sorted(result.destroyed) == [1, 2]
        assert result.spawned == []


# ----------------------------- splits / merges ------------------------------

class TestSplitsAndMerges:
    def test_split_keeps_one_id_spawns_other(self):
        # One large prev splits into two smaller curr (one inherits, other spawns)
        prev = _graph([_square_node(5, 3, 0, 0, 6)], 0)
        # Two children inside prev's bbox
        child_a = _square_node(0, 3, 0, 0, 3)
        child_b = _square_node(1, 3, 0, 3, 3)
        curr = _graph([child_a, child_b], frame_index=1)
        t = ObjectTracker(start_obj_id=20)
        result = t.match(prev, curr)
        # One child inherits id 5; the other spawns id 20.
        assert 5 in result.tracked.nodes
        assert len(result.spawned) == 1
        assert 20 in result.spawned
        assert result.destroyed == []

    def test_merge_keeps_one_destroys_other(self):
        # Two prev nodes merge into one curr node
        a = _square_node(0, 3, 0, 0, 3)
        b = _square_node(1, 3, 0, 3, 3)
        prev = _graph([a, b], 0)
        merged = _square_node(99, 3, 0, 0, 6)
        curr = _graph([merged], frame_index=1)
        t = ObjectTracker(start_obj_id=20)
        result = t.match(prev, curr)
        # One prev id survives, the other goes to destroyed
        surviving = set(result.tracked.nodes)
        assert len(surviving) == 1
        survived_id = next(iter(surviving))
        assert survived_id in (0, 1)
        assert sorted(result.destroyed) == [other for other in [0, 1] if other != survived_id]


# ----------------------------- counter & state ------------------------------

class TestCounter:
    def test_counter_never_reuses_ids(self):
        t = ObjectTracker(start_obj_id=0)
        # Bootstrap with 3
        result1 = t.match(None, _graph([_square_node(i, 3, i * 5, 0, 2) for i in range(3)], 0))
        ids_after_1 = set(result1.tracked.nodes)
        # All 3 vanish; 2 new ones spawn — must NOT reuse 0/1/2
        result2 = t.match(result1.tracked, _graph(
            [_square_node(0, 3, 50, 50, 2), _square_node(1, 3, 60, 60, 2)], 1
        ))
        spawned_set = set(result2.spawned)
        assert spawned_set.isdisjoint(ids_after_1)
        assert min(spawned_set) >= max(ids_after_1) + 1

    def test_start_obj_id_respected(self):
        t = ObjectTracker(start_obj_id=1000)
        result = t.match(None, _graph([_square_node(0, 3, 0, 0, 2)], 0))
        assert 1000 in result.tracked.nodes
        assert t.next_obj_id == 1001

    def test_invalid_threshold_raises(self):
        with pytest.raises(ValueError):
            ObjectTracker(iou_threshold=-0.1)
        with pytest.raises(ValueError):
            ObjectTracker(iou_threshold=1.5)


# ----------------------------- stateful update ------------------------------

class TestStatefulUpdate:
    def test_update_uses_internal_state(self):
        t = ObjectTracker()
        g0 = _graph([_square_node(0, 3, 0, 0, 4)], 0)
        g1 = _graph([_square_node(0, 3, 0, 1, 4)], 1)  # shifted, IOU > 0.5
        r0 = t.update(g0)
        r1 = t.update(g1)
        assert r0.spawned == [0] and r0.matched == {}
        assert r1.matched == {0: 0}
        assert t.prev_tracked is not None
        assert set(t.prev_tracked.nodes) == {0}

    def test_reset_clears_history_keeps_counter(self):
        t = ObjectTracker()
        t.update(_graph([_square_node(0, 3, 0, 0, 4)], 0))
        next_before = t.next_obj_id
        t.reset()
        assert t.prev_tracked is None
        # Next update should be a bootstrap with a fresh id, not match to old graph
        r = t.update(_graph([_square_node(0, 3, 0, 0, 4)], 5))
        assert r.spawned == [next_before]


# ----------------------------- edges rebuilt --------------------------------

class TestEdgesRebuilt:
    def test_touching_edge_uses_new_ids(self):
        parser = FrameParser()
        # Two 3x3 touching blocks
        f = np.zeros((16, 16), dtype=np.uint8)
        f[5:8, 5:8] = 3
        f[5:8, 8:11] = 4
        graph = parser.parse(f, frame_index=0)
        assert len(graph.nodes) == 2
        # Should be at least one TOUCHING edge in parsed graph
        assert any(e.relation == RelationType.TOUCHING for e in graph.edges)

        t = ObjectTracker(start_obj_id=100)
        result = t.match(None, graph)
        # Edges in tracked graph should reference the tracker-allocated ids only
        valid_ids = set(result.tracked.nodes)
        for e in result.tracked.edges:
            assert e.src_id in valid_ids
            assert e.dst_id in valid_ids
        # And we should still have a TOUCHING edge
        assert any(e.relation == RelationType.TOUCHING for e in result.tracked.edges)


# ----------------------------- 5-frame push (CLAUDE.md check) ---------------

class TestFiveFramePush:
    def test_4x4_movable_sliding_one_cell_per_frame_keeps_identity(self):
        parser = FrameParser()
        tracker = ObjectTracker()
        ids_per_frame = []
        for t in range(5):
            graph = parser.parse(_frame_with_movable_at(5 + t), frame_index=t)
            result = tracker.update(graph)
            ids_per_frame.append(sorted(result.tracked.nodes))

        # IDs must be identical across all 5 frames
        first = ids_per_frame[0]
        for ids in ids_per_frame[1:]:
            assert ids == first, f"ID set drift: {first} -> {ids}"
        # No spurious spawns/destructions after the bootstrap
        # (re-run with a fresh tracker to inspect events)
        tracker = ObjectTracker()
        events = [tracker.update(parser.parse(_frame_with_movable_at(5 + t), t)) for t in range(5)]
        assert events[0].spawned and not events[0].matched
        for ev in events[1:]:
            assert not ev.spawned
            assert not ev.destroyed

    def test_atomic_large_jump_falls_back_to_destroy_spawn(self):
        """Documents the known-limit: atomic pushes that jump beyond an object's extent
        cannot keep identity under pure IOU. The tracker correctly emits destroy+spawn."""
        parser = FrameParser()
        tracker = ObjectTracker()
        # 3x3 movable jumping 10 columns — IOU between frames is 0.
        f0 = np.zeros((16, 16), dtype=np.uint8); f0[5:8, 2:5] = 2
        f1 = np.zeros((16, 16), dtype=np.uint8); f1[5:8, 12:15] = 2
        tracker.update(parser.parse(f0, 0))
        r1 = tracker.update(parser.parse(f1, 1))
        assert r1.spawned != [] and r1.destroyed != []


# ----------------------------- TrackingResult contract ----------------------

class TestTrackingResultContract:
    def test_matched_and_spawned_sum_to_curr_node_count(self):
        prev = _graph([_square_node(0, 3, 0, 0, 4)], 0)
        curr = _graph(
            [_square_node(0, 3, 0, 1, 4),     # matches
             _square_node(1, 4, 10, 10, 2),   # spawn
             _square_node(2, 5, 14, 14, 2)],  # spawn
            frame_index=1,
        )
        t = ObjectTracker(start_obj_id=50)
        r = t.match(prev, curr)
        assert len(r.matched) + len(r.spawned) == len(r.tracked.nodes)

    def test_destroyed_ids_not_in_tracked(self):
        prev = _graph([_square_node(0, 3, 0, 0, 2), _square_node(1, 4, 8, 8, 2)], 0)
        curr = _graph([_square_node(0, 3, 0, 0, 2)], 1)
        t = ObjectTracker()
        r = t.match(prev, curr)
        for d in r.destroyed:
            assert d not in r.tracked.nodes
