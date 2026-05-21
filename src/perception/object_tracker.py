"""
object_tracker.py — Cross-frame object identity via Hungarian on IOU.

Given two ObjectGraphs (prev, curr) produced by FrameParser on consecutive
frames, the tracker assigns each curr node a *persistent* obj_id such that
objects that didn't move much keep the same ID across frames. Matching is
1-1 via the Hungarian assignment on the cost matrix `1 - IOU(prev_i, curr_j)`;
pairs whose IOU falls below `iou_threshold` (default 0.1) are rejected and
treated as destroy + spawn.

Two APIs:
  match(prev, curr)  — stateless, matches the CLAUDE.md spec verbatim. Used
                       when the caller manages the previous frame externally.
  update(curr)       — stateful convenience: tracks against the last graph
                       this tracker emitted. The typical production path.

Both return a `TrackingResult` with `.tracked` (the relabeled ObjectGraph)
plus `.matched / .spawned / .destroyed` lists for the event extractor.

Known limit: pure-IOU tracking cannot preserve identity across motions
larger than an object's own extent (IOU drops to 0). With atomic pushes
like sy01 L1 where MOVABLE_A jumps many cells, the tracker will (correctly,
per the spec) treat it as destroy + spawn. A richer similarity (color +
size + shape) would close that gap; deferred to future work.

Part of ACCA (Active Causal Compression Agent)
ARC Prize 2026 · Paper Track
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment

from src.perception.frame_parser import ObjectGraph, ObjectNode, build_edges

DEFAULT_IOU_THRESHOLD: float = 0.1


@dataclass
class TrackingResult:
    """Output of one tracker call.

    Fields:
      tracked    — curr's nodes relabeled with persistent obj_ids; edges rebuilt.
      matched    — { prev_obj_id: tracked_obj_id }. tracked_obj_id == prev_obj_id
                   for surviving objects (the carry-over).
      spawned    — obj_ids of objects newly appearing in curr (no prev match).
      destroyed  — obj_ids from prev that didn't survive to curr.
    """
    tracked: ObjectGraph
    matched: Dict[int, int] = field(default_factory=dict)
    spawned: List[int] = field(default_factory=list)
    destroyed: List[int] = field(default_factory=list)


def iou(a: ObjectNode, b: ObjectNode) -> float:
    """Jaccard index on pixel sets. Cheap bbox-reject fast path. Range [0, 1]."""
    if a.area == 0 or b.area == 0:
        return 0.0
    ax1, ay1, ax2, ay2 = a.bbox
    bx1, by1, bx2, by2 = b.bbox
    if ax2 < bx1 or bx2 < ax1 or ay2 < by1 or by2 < ay1:
        return 0.0
    inter = len(a.pixels & b.pixels)
    if inter == 0:
        return 0.0
    union = a.area + b.area - inter
    return inter / union


def build_cost_matrix(prev_nodes: List[ObjectNode], curr_nodes: List[ObjectNode]) -> np.ndarray:
    """Cost = 1 - IOU. Returns an (n_prev, n_curr) float64 array."""
    n, m = len(prev_nodes), len(curr_nodes)
    cost = np.ones((n, m), dtype=np.float64)
    for i, p in enumerate(prev_nodes):
        for j, c in enumerate(curr_nodes):
            cost[i, j] = 1.0 - iou(p, c)
    return cost


def _hungarian(cost: np.ndarray) -> List[Tuple[int, int]]:
    """Wrap scipy.optimize.linear_sum_assignment for rectangular cost matrices."""
    if cost.size == 0:
        return []
    rows, cols = linear_sum_assignment(cost)
    return list(zip(rows.tolist(), cols.tolist()))


def _relabel_node(src: ObjectNode, new_id: int) -> ObjectNode:
    return ObjectNode(
        obj_id=new_id,
        color=src.color,
        bbox=src.bbox,
        centroid=src.centroid,
        area=src.area,
        pixels=src.pixels,
    )


class ObjectTracker:
    """Hungarian-IOU tracker with persistent obj_ids across frames.

    The tracker owns a monotonically-increasing id counter; FrameParser's
    obj_ids are discarded on every call (they're per-frame artifacts).
    """

    def __init__(self, iou_threshold: float = DEFAULT_IOU_THRESHOLD, start_obj_id: int = 0):
        if not 0.0 <= iou_threshold <= 1.0:
            raise ValueError(f"iou_threshold must be in [0, 1], got {iou_threshold}")
        self.iou_threshold = iou_threshold
        self._next_id = start_obj_id
        self._prev_tracked: Optional[ObjectGraph] = None

    @property
    def next_obj_id(self) -> int:
        return self._next_id

    @property
    def prev_tracked(self) -> Optional[ObjectGraph]:
        """The last graph this tracker emitted (None if never updated)."""
        return self._prev_tracked

    def reset(self) -> None:
        """Drop history. Counter continues — call `__init__` again to also reset IDs."""
        self._prev_tracked = None

    def _allocate_id(self) -> int:
        out = self._next_id
        self._next_id += 1
        return out

    def _bootstrap(self, curr: ObjectGraph) -> TrackingResult:
        """First-frame path: every curr node gets a fresh persistent id."""
        new_nodes: Dict[int, ObjectNode] = {}
        spawned: List[int] = []
        for c in sorted(curr.nodes.values(), key=lambda n: n.obj_id):
            new_id = self._allocate_id()
            new_nodes[new_id] = _relabel_node(c, new_id)
            spawned.append(new_id)
        tracked = ObjectGraph(
            nodes=new_nodes,
            edges=build_edges(new_nodes.values()),
            frame_index=curr.frame_index,
        )
        return TrackingResult(tracked=tracked, matched={}, spawned=spawned, destroyed=[])

    def match(self, prev: Optional[ObjectGraph], curr: ObjectGraph) -> TrackingResult:
        """Stateless: match `curr` against `prev` (may be None / empty).

        Spawned/new ids come from the tracker's persistent counter (so calling
        `match` twice with identical inputs will *not* return identical ids —
        each call allocates fresh ones for the curr-only nodes). Use
        `update()` for the stateful convenience path.
        """
        if prev is None or not prev.nodes:
            return self._bootstrap(curr)

        if not curr.nodes:
            empty = ObjectGraph(nodes={}, edges=[], frame_index=curr.frame_index)
            return TrackingResult(
                tracked=empty,
                matched={},
                spawned=[],
                destroyed=sorted(prev.nodes),
            )

        prev_list = sorted(prev.nodes.values(), key=lambda n: n.obj_id)
        curr_list = sorted(curr.nodes.values(), key=lambda n: n.obj_id)

        cost = build_cost_matrix(prev_list, curr_list)
        pairs = _hungarian(cost)

        accept_cost = 1.0 - self.iou_threshold
        new_nodes: Dict[int, ObjectNode] = {}
        matched: Dict[int, int] = {}
        matched_prev_idx: set = set()
        matched_curr_idx: set = set()

        for (i, j) in pairs:
            if cost[i, j] <= accept_cost:
                surviving_id = prev_list[i].obj_id
                new_nodes[surviving_id] = _relabel_node(curr_list[j], surviving_id)
                matched[prev_list[i].obj_id] = surviving_id
                matched_prev_idx.add(i)
                matched_curr_idx.add(j)

        spawned: List[int] = []
        for j, c in enumerate(curr_list):
            if j in matched_curr_idx:
                continue
            new_id = self._allocate_id()
            new_nodes[new_id] = _relabel_node(c, new_id)
            spawned.append(new_id)

        destroyed: List[int] = [
            prev_list[i].obj_id for i in range(len(prev_list)) if i not in matched_prev_idx
        ]

        tracked = ObjectGraph(
            nodes=new_nodes,
            edges=build_edges(new_nodes.values()),
            frame_index=curr.frame_index,
        )
        return TrackingResult(
            tracked=tracked, matched=matched, spawned=spawned, destroyed=destroyed
        )

    def update(self, curr: ObjectGraph) -> TrackingResult:
        """Stateful: match against the last graph this tracker emitted, then store."""
        result = self.match(self._prev_tracked, curr)
        self._prev_tracked = result.tracked
        return result


def _demo_push_sequence() -> None:
    """Visual demo: 5-frame 4x4 movable sliding right by 1 cell each frame."""
    from src.perception.frame_parser import FrameParser

    parser = FrameParser()
    tracker = ObjectTracker()
    print("5-frame push sequence -- IDs should remain stable (movable IOU ~= 0.6):")
    for t in range(5):
        f = np.zeros((16, 16), dtype=np.uint8)
        f[10:14, 5 + t:9 + t] = 2  # MOVABLE_A
        f[2, 2] = 7                # stationary marker
        graph = parser.parse(f, frame_index=t)
        result = tracker.update(graph)
        ids = sorted(result.tracked.nodes.keys())
        print(f"  frame {t}: ids={ids}  matched={result.matched}  spawned={result.spawned}  destroyed={result.destroyed}")


if __name__ == "__main__":
    _demo_push_sequence()
