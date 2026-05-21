"""
test_event_extractor.py - Tests for object-level transition event extraction.

Part of ACCA (Active Causal Compression Agent)
ARC Prize 2026 . Paper Track
"""
from __future__ import annotations

import pytest

from src.perception.event_extractor import (
    ActionEnum,
    AtomicEvent,
    EventExtractor,
    canonical_delta,
)
from src.perception.frame_parser import ObjectGraph, ObjectNode
from src.perception.object_tracker import TrackingResult
from src.perception.replay_buffer import ReplayBuffer


def _node(obj_id: int, color: int, pixels) -> ObjectNode:
    pix = frozenset(pixels)
    rows = [r for (r, _) in pix]
    cols = [c for (_, c) in pix]
    return ObjectNode(
        obj_id=obj_id,
        color=color,
        bbox=(min(cols), min(rows), max(cols), max(rows)),
        centroid=(sum(cols) / len(pix), sum(rows) / len(pix)),
        area=len(pix),
        pixels=pix,
    )


def _square_node(obj_id: int, color: int, top: int, left: int, size: int) -> ObjectNode:
    return _node(
        obj_id,
        color,
        [(r, c) for r in range(top, top + size) for c in range(left, left + size)],
    )


def _graph(nodes, frame_index: int = 0) -> ObjectGraph:
    return ObjectGraph(nodes={n.obj_id: n for n in nodes}, edges=[], frame_index=frame_index)


def test_object_moves_right_five_cells():
    pre = _graph([_square_node(3, 2, 4, 4, 2)], 0)
    post = _graph([_square_node(3, 2, 4, 9, 2)], 1)

    obs = EventExtractor().extract(ActionEnum.ACTION1, pre, post, timestamp=1)

    assert obs.delta == [AtomicEvent("MOVED", 3, {"dx": 5, "dy": 0})]
    assert obs.action == ActionEnum.ACTION1
    assert obs.timestamp == 1


def test_object_moves_down_three_cells():
    pre = _graph([_square_node(3, 2, 4, 4, 2)], 0)
    post = _graph([_square_node(3, 2, 7, 4, 2)], 1)

    obs = EventExtractor().extract("ACTION2", pre, post, timestamp=1)

    assert obs.delta == [AtomicEvent("MOVED", 3, {"dx": 0, "dy": 3})]


def test_object_recolored_old_and_new_color():
    pre = _graph([_square_node(1, 3, 0, 0, 2)], 0)
    post = _graph([_square_node(1, 7, 0, 0, 2)], 1)

    obs = EventExtractor().extract(ActionEnum.ACTION5, pre, post, timestamp=1)

    assert obs.delta == [
        AtomicEvent("RECOLORED", 1, {"old_color": 3, "new_color": 7})
    ]


def test_object_destroyed_from_tracking_result():
    pre = _graph([_square_node(4, 3, 0, 0, 2)], 0)
    post_graph = _graph([], 1)
    tracking = TrackingResult(tracked=post_graph, destroyed=[4])

    obs = EventExtractor().extract(ActionEnum.ACTION3, pre, tracking, timestamp=1)

    assert obs.delta == [AtomicEvent("DESTROYED", 4, {"color": 3})]


def test_object_spawned_from_tracking_result():
    pre = _graph([], 0)
    post_graph = _graph([_square_node(9, 6, 3, 4, 2)], 1)
    tracking = TrackingResult(tracked=post_graph, spawned=[9])

    obs = EventExtractor().extract(ActionEnum.ACTION4, pre, tracking, timestamp=1)

    assert obs.delta == [
        AtomicEvent("SPAWNED", 9, {"color": 6, "position": (4.5, 3.5)})
    ]


def test_no_change_has_empty_delta():
    pre = _graph([_square_node(1, 3, 0, 0, 2)], 0)
    post = _graph([_square_node(1, 3, 0, 0, 2)], 1)

    obs = EventExtractor().extract(ActionEnum.ACTION7, pre, post, timestamp=1)

    assert obs.delta == []


def test_multiple_simultaneous_events_order_independent():
    pre = _graph(
        [
            _square_node(1, 3, 0, 0, 2),
            _square_node(2, 4, 6, 6, 2),
        ],
        0,
    )
    post_graph = _graph(
        [
            _square_node(1, 7, 0, 5, 2),
            _square_node(10, 8, 9, 9, 1),
        ],
        1,
    )
    tracking = TrackingResult(tracked=post_graph, spawned=[10], destroyed=[2])

    obs = EventExtractor().extract("ACTION6 9 9", pre, tracking, timestamp=1)

    expected = [
        AtomicEvent("MOVED", 1, {"dx": 5, "dy": 0}),
        AtomicEvent("RECOLORED", 1, {"old_color": 3, "new_color": 7}),
        AtomicEvent("DESTROYED", 2, {"color": 4}),
        AtomicEvent("SPAWNED", 10, {"color": 8, "position": (9.0, 9.0)}),
    ]
    assert obs.action == ActionEnum.ACTION6
    assert canonical_delta(obs.delta) == canonical_delta(expected)


def test_replay_buffer_filters_by_action_and_discards_oldest():
    extractor = EventExtractor()
    pre = _graph([_square_node(1, 3, 0, 0, 2)], 0)
    post = _graph([_square_node(1, 3, 0, 1, 2)], 1)
    obs1 = extractor.extract(ActionEnum.ACTION1, pre, post, timestamp=1)
    obs2 = extractor.extract(ActionEnum.ACTION2, pre, post, timestamp=2)
    obs3 = extractor.extract(ActionEnum.ACTION1, pre, post, timestamp=3)
    buffer = ReplayBuffer(maxlen=2)

    buffer.add(obs1)
    buffer.add(obs2)
    buffer.add(obs3)

    assert buffer.get_all() == [obs2, obs3]
    assert buffer.get_by_action(ActionEnum.ACTION1) == [obs3]
    assert buffer.get_by_action("ACTION2") == [obs2]


def test_replay_buffer_clear_and_invalid_maxlen():
    buffer = ReplayBuffer(maxlen=1)
    assert buffer.get_all() == []
    buffer.clear()
    assert buffer.get_all() == []

    with pytest.raises(ValueError):
        ReplayBuffer(maxlen=0)
