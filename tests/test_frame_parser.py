"""
test_frame_parser.py — Unit tests + runtime budget for FrameParser.

Part of ACCA (Active Causal Compression Agent)
ARC Prize 2026 · Paper Track
"""
from __future__ import annotations

import time

import numpy as np
import pytest

from src.perception.frame_parser import (
    BACKGROUND_COLOR,
    FrameParser,
    ObjectNode,
    RelationType,
    RelEdge,
    build_edges,
)


def _empty() -> np.ndarray:
    return np.zeros((64, 64), dtype=np.uint8)


def test_single_object_bbox_centroid_area():
    f = _empty()
    f[10:13, 10:13] = 5
    g = FrameParser().parse(f, frame_index=0)
    assert len(g.nodes) == 1
    n = next(iter(g.nodes.values()))
    assert n.color == 5
    assert n.area == 9
    assert n.bbox == (10, 10, 12, 12)
    assert n.centroid == (11.0, 11.0)
    assert g.frame_index == 0


def test_empty_frame_has_no_objects():
    g = FrameParser().parse(_empty(), 0)
    assert g.nodes == {}
    assert g.edges == []


def test_touching_edge_between_adjacent_blocks():
    f = _empty()
    f[10:13, 10:13] = 3
    f[10:13, 13:16] = 4
    g = FrameParser().parse(f, 0)
    assert len(g.nodes) == 2
    a, b = sorted(g.nodes)
    assert RelEdge(a, b, RelationType.TOUCHING) in g.edges


def test_diagonal_neighbors_are_touching():
    f = _empty()
    f[10, 10] = 3
    f[11, 11] = 4
    g = FrameParser().parse(f, 0)
    assert len(g.nodes) == 2
    a, b = sorted(g.nodes)
    assert RelEdge(a, b, RelationType.TOUCHING) in g.edges


def test_non_adjacent_blocks_have_no_edge():
    f = _empty()
    f[10:13, 10:13] = 3
    f[20:23, 20:23] = 4
    g = FrameParser().parse(f, 0)
    assert g.edges == []


def test_overlapping_edge_from_manual_nodes():
    """OVERLAPPING uses pixel-set intersection; doesn't arise from parse() output."""
    a = ObjectNode(
        obj_id=0, color=3, bbox=(0, 0, 2, 2), centroid=(1.0, 1.0), area=4,
        pixels=frozenset({(0, 0), (0, 1), (1, 0), (1, 1)}),
    )
    b = ObjectNode(
        obj_id=1, color=4, bbox=(1, 1, 3, 3), centroid=(2.0, 2.0), area=4,
        pixels=frozenset({(1, 1), (1, 2), (2, 1), (2, 2)}),
    )
    edges = build_edges([a, b])
    assert RelEdge(0, 1, RelationType.OVERLAPPING) in edges


def test_contained_edge_from_manual_nodes():
    """CONTAINED literal pixel-subset; doesn't arise from parse() output."""
    inner = ObjectNode(
        obj_id=0, color=3, bbox=(1, 1, 1, 1), centroid=(1.0, 1.0), area=1,
        pixels=frozenset({(1, 1)}),
    )
    outer_pixels = frozenset((r, c) for r in range(3) for c in range(3))
    outer = ObjectNode(
        obj_id=1, color=4, bbox=(0, 0, 2, 2), centroid=(1.0, 1.0), area=9,
        pixels=outer_pixels,
    )
    edges = build_edges([inner, outer])
    assert RelEdge(0, 1, RelationType.CONTAINED) in edges
    assert RelEdge(0, 1, RelationType.OVERLAPPING) in edges


def test_full_frame_with_five_objects():
    f = _empty()
    f[5:8, 5:8] = 3
    f[5:8, 20:23] = 4
    f[30:33, 5:8] = 5
    f[30:33, 30:33] = 6
    f[50:54, 50:54] = 7
    g = FrameParser().parse(f, 0)
    assert len(g.nodes) == 5
    assert {n.color for n in g.nodes.values()} == {3, 4, 5, 6, 7}
    assert g.edges == []


def test_background_color_ignored():
    f = _empty()
    f[0:10, 0:10] = BACKGROUND_COLOR
    f[20:23, 20:23] = 3
    g = FrameParser().parse(f, 0)
    assert len(g.nodes) == 1


def test_same_color_disconnected_components_separate():
    f = _empty()
    f[5:8, 5:8] = 3
    f[20:23, 20:23] = 3
    g = FrameParser().parse(f, 0)
    assert len(g.nodes) == 2
    assert all(n.color == 3 for n in g.nodes.values())


def test_obj_ids_persist_across_frames():
    parser = FrameParser(start_obj_id=100)
    f1 = _empty(); f1[0:2, 0:2] = 3
    g1 = parser.parse(f1, 0)
    assert set(g1.nodes) == {100}
    f2 = _empty(); f2[10:12, 10:12] = 4; f2[20:22, 20:22] = 5
    g2 = parser.parse(f2, 1)
    assert set(g2.nodes) == {101, 102}
    assert g2.frame_index == 1


def test_runtime_budget_under_50ms():
    rng = np.random.default_rng(42)
    parser = FrameParser()
    times = []
    for _ in range(100):
        f = np.zeros((64, 64), dtype=np.uint8)
        n_objects = int(rng.integers(5, 16))
        for _ in range(n_objects):
            sz = int(rng.integers(1, 5))
            y = int(rng.integers(0, 64 - sz))
            x = int(rng.integers(0, 64 - sz))
            color = int(rng.integers(1, 16))
            f[y:y + sz, x:x + sz] = color
        t0 = time.perf_counter()
        parser.parse(f, 0)
        times.append((time.perf_counter() - t0) * 1000)
    max_ms = max(times)
    mean_ms = sum(times) / len(times)
    assert max_ms < 50.0, f"max={max_ms:.2f}ms mean={mean_ms:.2f}ms exceeds 50ms budget"
