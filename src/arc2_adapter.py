"""
arc2_adapter.py - Convert static ARC-AGI-2 examples into ACCA observations.

ARC-AGI-2 tasks are input/output grid pairs, not interactive episodes. This
adapter treats each training pair as a one-step pseudo-interaction:
input grid --ACTION1--> output grid. The resulting Observation list gives the
hypothesis engine object-level deltas it can score with the same machinery used
for ARC-AGI-3 action traces.

Part of ACCA (Active Causal Compression Agent)
ARC Prize 2026 . Paper Track
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
from scipy.optimize import linear_sum_assignment

from src.perception.event_extractor import ActionEnum, EventExtractor, Observation
from src.perception.frame_parser import FrameParser, ObjectGraph, ObjectNode, build_edges
from src.perception.object_tracker import TrackingResult


GridLike = Sequence[Sequence[int]] | np.ndarray

SHAPE_MISMATCH_COST = 2.0
COLOR_MISMATCH_COST = 0.35
POSITION_WEIGHT = 0.50
AREA_WEIGHT = 0.75
OVERLAP_REWARD = 0.50
MAX_ACCEPT_COST = 0.70


@dataclass(frozen=True)
class Arc2PairTrace:
    """Debug bundle for one converted ARC-AGI-2 input/output example."""

    input_graph: ObjectGraph
    output_graph: ObjectGraph
    tracking: TrackingResult
    observation: Observation


def _as_uint8_grid(grid: GridLike) -> np.ndarray:
    arr = np.asarray(grid, dtype=np.uint8)
    if arr.ndim != 2:
        raise ValueError(f"ARC grid must be 2D, got shape {arr.shape}")
    if arr.size == 0:
        raise ValueError("ARC grid must be non-empty")
    if int(arr.max(initial=0)) > 15:
        raise ValueError("ARC grid values must be in 0..15")
    return arr


def _shape_signature(node: ObjectNode) -> frozenset[tuple[int, int]]:
    min_row = min(r for (r, _) in node.pixels)
    min_col = min(c for (_, c) in node.pixels)
    return frozenset((r - min_row, c - min_col) for (r, c) in node.pixels)


def _bbox_size(node: ObjectNode) -> tuple[int, int]:
    x1, y1, x2, y2 = node.bbox
    return (x2 - x1 + 1, y2 - y1 + 1)


def _overlap(a: ObjectNode, b: ObjectNode) -> float:
    if not a.pixels or not b.pixels:
        return 0.0
    inter = len(a.pixels & b.pixels)
    if inter == 0:
        return 0.0
    return inter / max(a.area, b.area)


def _match_cost(a: ObjectNode, b: ObjectNode, grid_shape: tuple[int, int]) -> float:
    shape_cost = 0.0 if _shape_signature(a) == _shape_signature(b) else SHAPE_MISMATCH_COST
    color_cost = 0.0 if a.color == b.color else COLOR_MISMATCH_COST
    area_cost = AREA_WEIGHT * (abs(a.area - b.area) / max(a.area, b.area))
    height, width = grid_shape
    norm = max(height, width, 1)
    position_cost = POSITION_WEIGHT * (
        abs(a.centroid[0] - b.centroid[0]) + abs(a.centroid[1] - b.centroid[1])
    ) / norm
    overlap_reward = OVERLAP_REWARD * _overlap(a, b)
    return shape_cost + color_cost + area_cost + position_cost - overlap_reward


def _relabel_node(src: ObjectNode, obj_id: int) -> ObjectNode:
    return ObjectNode(
        obj_id=obj_id,
        color=src.color,
        bbox=src.bbox,
        centroid=src.centroid,
        area=src.area,
        pixels=src.pixels,
    )


def _next_free_id(nodes: Iterable[int]) -> int:
    ids = list(nodes)
    return max(ids, default=-1) + 1


def align_output_to_input(
    input_graph: ObjectGraph,
    output_graph: ObjectGraph,
    grid_shape: tuple[int, int],
) -> TrackingResult:
    """Relabel output nodes with input IDs where a static object match is plausible."""
    if not input_graph.nodes:
        next_id = 0
        new_nodes = {}
        spawned = []
        for node in sorted(output_graph.nodes.values(), key=lambda n: n.obj_id):
            new_nodes[next_id] = _relabel_node(node, next_id)
            spawned.append(next_id)
            next_id += 1
        tracked = ObjectGraph(new_nodes, build_edges(new_nodes.values()), output_graph.frame_index)
        return TrackingResult(tracked=tracked, spawned=spawned)

    if not output_graph.nodes:
        tracked = ObjectGraph({}, [], output_graph.frame_index)
        return TrackingResult(tracked=tracked, destroyed=sorted(input_graph.nodes))

    input_nodes = sorted(input_graph.nodes.values(), key=lambda n: n.obj_id)
    output_nodes = sorted(output_graph.nodes.values(), key=lambda n: n.obj_id)
    cost = np.zeros((len(input_nodes), len(output_nodes)), dtype=np.float64)
    for i, src in enumerate(input_nodes):
        for j, dst in enumerate(output_nodes):
            cost[i, j] = _match_cost(src, dst, grid_shape)

    rows, cols = linear_sum_assignment(cost)
    matched_input: set[int] = set()
    matched_output: set[int] = set()
    new_nodes: dict[int, ObjectNode] = {}
    matched: dict[int, int] = {}

    for row, col in zip(rows.tolist(), cols.tolist()):
        if cost[row, col] > MAX_ACCEPT_COST:
            continue
        input_id = input_nodes[row].obj_id
        new_nodes[input_id] = _relabel_node(output_nodes[col], input_id)
        matched[input_id] = input_id
        matched_input.add(row)
        matched_output.add(col)

    next_id = _next_free_id(input_graph.nodes)
    spawned: list[int] = []
    for j, node in enumerate(output_nodes):
        if j in matched_output:
            continue
        new_nodes[next_id] = _relabel_node(node, next_id)
        spawned.append(next_id)
        next_id += 1

    destroyed = [
        input_nodes[i].obj_id for i in range(len(input_nodes)) if i not in matched_input
    ]
    tracked = ObjectGraph(new_nodes, build_edges(new_nodes.values()), output_graph.frame_index)
    return TrackingResult(
        tracked=tracked,
        matched=matched,
        spawned=spawned,
        destroyed=destroyed,
    )


def convert_arc2_pair_to_trace(
    input_grid: GridLike,
    output_grid: GridLike,
    timestamp: int = 0,
) -> Arc2PairTrace:
    """Convert one ARC-AGI-2 input/output pair to a one-step Observation."""
    input_arr = _as_uint8_grid(input_grid)
    output_arr = _as_uint8_grid(output_grid)
    if input_arr.shape != output_arr.shape:
        raise ValueError(
            f"input/output grid shapes must match, got {input_arr.shape} and {output_arr.shape}"
        )

    input_graph = FrameParser(start_obj_id=0).parse(input_arr, frame_index=timestamp)
    output_graph = FrameParser(start_obj_id=0).parse(output_arr, frame_index=timestamp + 1)
    tracking = align_output_to_input(input_graph, output_graph, input_arr.shape)
    observation = EventExtractor().extract(
        ActionEnum.ACTION1,
        input_graph,
        tracking,
        timestamp=timestamp,
    )
    return Arc2PairTrace(
        input_graph=input_graph,
        output_graph=output_graph,
        tracking=tracking,
        observation=observation,
    )


def convert_arc2_task_to_traces(task: Mapping[str, Any]) -> list[Observation]:
    """Convert all ARC-AGI-2 train examples in a task to pseudo-interaction traces."""
    train_examples = task.get("train")
    if not isinstance(train_examples, list):
        raise ValueError("ARC-AGI-2 task must contain a list under key 'train'")

    observations: list[Observation] = []
    for timestamp, example in enumerate(train_examples):
        if not isinstance(example, Mapping):
            raise ValueError(f"train example {timestamp} must be a mapping")
        if "input" not in example or "output" not in example:
            raise ValueError(f"train example {timestamp} must contain input and output")
        trace = convert_arc2_pair_to_trace(
            example["input"],
            example["output"],
            timestamp=timestamp,
        )
        observations.append(trace.observation)
    return observations
