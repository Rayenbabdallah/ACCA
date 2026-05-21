"""
test_arc2_adapter.py - ARC-AGI-2 static bridge tests.

Part of ACCA (Active Causal Compression Agent)
ARC Prize 2026 . Paper Track
"""
from __future__ import annotations

import numpy as np
import pytest

from src.arc2_adapter import (
    align_output_to_input,
    convert_arc2_pair_to_trace,
    convert_arc2_task_to_traces,
)
from src.perception.event_extractor import ActionEnum, AtomicEvent, canonical_delta
from src.perception.frame_parser import FrameParser


def _blank() -> np.ndarray:
    return np.zeros((8, 8), dtype=np.uint8)


def test_convert_pair_recolor_same_object():
    inp = _blank()
    out = _blank()
    inp[2:4, 2:4] = 3
    out[2:4, 2:4] = 7

    trace = convert_arc2_pair_to_trace(inp, out)

    assert trace.observation.action == ActionEnum.ACTION1
    assert trace.observation.delta == [
        AtomicEvent("RECOLORED", 0, {"old_color": 3, "new_color": 7})
    ]


def test_convert_pair_move_without_overlap_keeps_identity():
    inp = _blank()
    out = _blank()
    inp[1:3, 1:3] = 4
    out[5:7, 5:7] = 4

    trace = convert_arc2_pair_to_trace(inp, out)

    assert trace.tracking.spawned == []
    assert trace.tracking.destroyed == []
    assert trace.observation.delta == [AtomicEvent("MOVED", 0, {"dx": 4, "dy": 4})]


def test_convert_pair_spawn_and_destroy():
    inp = _blank()
    out = _blank()
    inp[1:3, 1:3] = 4
    out[5:7, 5:7] = 9

    trace = convert_arc2_pair_to_trace(inp, out)

    assert canonical_delta(trace.observation.delta) == canonical_delta(
        [
            AtomicEvent("DESTROYED", 0, {"color": 4}),
            AtomicEvent("SPAWNED", 1, {"color": 9, "position": (5.5, 5.5)}),
        ]
    )


def test_convert_pair_multiple_objects_with_move_and_recolor():
    inp = _blank()
    out = _blank()
    inp[1:3, 1:3] = 4
    inp[6, 1] = 5
    out[1:3, 4:6] = 7
    out[6, 1] = 5

    trace = convert_arc2_pair_to_trace(inp, out)

    assert canonical_delta(trace.observation.delta) == canonical_delta(
        [
            AtomicEvent("MOVED", 0, {"dx": 3, "dy": 0}),
            AtomicEvent("RECOLORED", 0, {"old_color": 4, "new_color": 7}),
        ]
    )


def test_convert_task_uses_train_examples_only_and_timestamps():
    inp1 = _blank()
    out1 = _blank()
    inp1[1, 1] = 2
    out1[1, 2] = 2
    inp2 = _blank()
    out2 = _blank()
    inp2[3, 3] = 4
    out2[3, 3] = 8
    task = {
        "train": [
            {"input": inp1.tolist(), "output": out1.tolist()},
            {"input": inp2.tolist(), "output": out2.tolist()},
        ],
        "test": [{"input": inp1.tolist()}],
    }

    observations = convert_arc2_task_to_traces(task)

    assert len(observations) == 2
    assert [obs.timestamp for obs in observations] == [0, 1]
    assert all(obs.action == ActionEnum.ACTION1 for obs in observations)


def test_align_output_empty_cases():
    parser = FrameParser()
    empty = parser.parse(_blank(), 0)
    one = _blank()
    one[2, 2] = 3
    graph = parser.parse(one, 1)

    spawned = align_output_to_input(empty, graph, one.shape)
    destroyed = align_output_to_input(graph, empty, one.shape)

    assert spawned.spawned == [0]
    assert destroyed.destroyed == sorted(graph.nodes)


def test_invalid_arc2_grid_rejected():
    with pytest.raises(ValueError):
        convert_arc2_pair_to_trace([[1, 2]], [[1], [2]])
    with pytest.raises(ValueError):
        convert_arc2_pair_to_trace([[[1]]], [[[1]]])
    with pytest.raises(ValueError):
        convert_arc2_pair_to_trace([[16]], [[0]])
    with pytest.raises(ValueError):
        convert_arc2_task_to_traces({"train": {"input": [], "output": []}})
