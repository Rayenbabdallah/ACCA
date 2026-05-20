"""
frame_parser.py — Raw 64x64 ARC-AGI-3 grid -> ObjectGraph.

Per-color connected-component labeling yields object nodes; pairwise spatial
relations (TOUCHING / OVERLAPPING / CONTAINED) yield edges. Target: <50ms on a
64x64 frame.

Edge semantics note: a single-layer per-color labeling guarantees distinct
components have disjoint pixel sets, so the spec's literal pixel-set
OVERLAPPING (`A.pixels & B.pixels`) and CONTAINED (`A.pixels.issubset(B.pixels)`)
do NOT arise from parse() output on standard ARC grids. They are kept in
`build_edges()` for downstream use (tracker, hand-built graphs, multi-layer
parsers). Only TOUCHING fires from parse() output.

Part of ACCA (Active Causal Compression Agent)
ARC Prize 2026 · Paper Track
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, FrozenSet, Iterable, List, Set, Tuple

import numpy as np
from skimage.measure import label as skimage_label

BACKGROUND_COLOR: int = 0


class RelationType(Enum):
    TOUCHING = "touching"
    OVERLAPPING = "overlapping"
    CONTAINED = "contained"


@dataclass
class ObjectNode:
    obj_id: int
    color: int
    bbox: Tuple[int, int, int, int]
    centroid: Tuple[float, float]
    area: int
    pixels: FrozenSet[Tuple[int, int]]


@dataclass
class RelEdge:
    src_id: int
    dst_id: int
    relation: RelationType

    def __hash__(self) -> int:
        return hash((self.src_id, self.dst_id, self.relation))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, RelEdge):
            return NotImplemented
        return (
            self.src_id == other.src_id
            and self.dst_id == other.dst_id
            and self.relation == other.relation
        )


@dataclass
class ObjectGraph:
    nodes: Dict[int, ObjectNode]
    edges: List[RelEdge]
    frame_index: int


def _dilate_pixels(pixels: FrozenSet[Tuple[int, int]]) -> Set[Tuple[int, int]]:
    """8-neighborhood dilation, returning new cells only (excludes input)."""
    out: Set[Tuple[int, int]] = set()
    for (r, c) in pixels:
        out.add((r - 1, c - 1)); out.add((r - 1, c)); out.add((r - 1, c + 1))
        out.add((r, c - 1));                          out.add((r, c + 1))
        out.add((r + 1, c - 1)); out.add((r + 1, c)); out.add((r + 1, c + 1))
    out -= pixels
    return out


def _bboxes_dilated_overlap(bbox_a, bbox_b) -> bool:
    """Cheap pre-filter: do A's and B's bboxes (each dilated by 1) intersect?"""
    ax1, ay1, ax2, ay2 = bbox_a
    bx1, by1, bx2, by2 = bbox_b
    return not (ax2 + 1 < bx1 - 1 or bx2 + 1 < ax1 - 1
                or ay2 + 1 < by1 - 1 or by2 + 1 < ay1 - 1)


def build_edges(nodes: Iterable[ObjectNode]) -> List[RelEdge]:
    """Pairwise relation detection. Symmetric edges (OVERLAPPING, TOUCHING) emit
    with src_id < dst_id. CONTAINED is directional: src ⊆ dst."""
    nodes_sorted = sorted(nodes, key=lambda n: n.obj_id)
    n = len(nodes_sorted)
    edges: List[RelEdge] = []
    for i in range(n):
        a = nodes_sorted[i]
        a_dilated: Set[Tuple[int, int]] | None = None
        for j in range(i + 1, n):
            b = nodes_sorted[j]
            if not _bboxes_dilated_overlap(a.bbox, b.bbox):
                continue
            shared = a.pixels & b.pixels
            if shared:
                edges.append(RelEdge(a.obj_id, b.obj_id, RelationType.OVERLAPPING))
                if a.pixels.issubset(b.pixels):
                    edges.append(RelEdge(a.obj_id, b.obj_id, RelationType.CONTAINED))
                elif b.pixels.issubset(a.pixels):
                    edges.append(RelEdge(b.obj_id, a.obj_id, RelationType.CONTAINED))
            else:
                if a_dilated is None:
                    a_dilated = _dilate_pixels(a.pixels)
                if not a_dilated.isdisjoint(b.pixels):
                    edges.append(RelEdge(a.obj_id, b.obj_id, RelationType.TOUCHING))
    return edges


class FrameParser:
    """Parser with a global obj_id counter so IDs are unique across frames."""

    def __init__(self, start_obj_id: int = 0):
        self._next_id = start_obj_id

    @property
    def next_obj_id(self) -> int:
        return self._next_id

    def parse(self, frame: np.ndarray, frame_index: int) -> ObjectGraph:
        if frame.ndim != 2:
            raise ValueError(f"frame must be 2D, got shape {frame.shape}")
        nodes: Dict[int, ObjectNode] = {}
        unique = np.unique(frame)
        for c_np in unique:
            c = int(c_np)
            if c == BACKGROUND_COLOR:
                continue
            mask = (frame == c)
            labels = skimage_label(mask, connectivity=2)
            n_comp = int(labels.max())
            for k in range(1, n_comp + 1):
                ys, xs = np.where(labels == k)
                if ys.size == 0:
                    continue
                obj_id = self._next_id
                self._next_id += 1
                pixels = frozenset(zip(ys.tolist(), xs.tolist()))
                bbox = (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max()))
                centroid = (float(xs.mean()), float(ys.mean()))
                nodes[obj_id] = ObjectNode(
                    obj_id=obj_id,
                    color=c,
                    bbox=bbox,
                    centroid=centroid,
                    area=int(ys.size),
                    pixels=pixels,
                )
        edges = build_edges(nodes.values())
        return ObjectGraph(nodes=nodes, edges=edges, frame_index=frame_index)


def _profile_self() -> None:
    """Profile a single parse() call on a representative 64x64 frame."""
    import cProfile
    import io
    import pstats

    rng = np.random.default_rng(0)
    frame = np.zeros((64, 64), dtype=np.uint8)
    for _ in range(12):
        sz = int(rng.integers(1, 5))
        y = int(rng.integers(0, 64 - sz))
        x = int(rng.integers(0, 64 - sz))
        color = int(rng.integers(1, 16))
        frame[y:y + sz, x:x + sz] = color

    parser = FrameParser()
    pr = cProfile.Profile()
    pr.enable()
    for _ in range(100):
        parser.parse(frame, 0)
        parser._next_id = 0
    pr.disable()
    s = io.StringIO()
    pstats.Stats(pr, stream=s).sort_stats("cumulative").print_stats(15)
    print(s.getvalue())


if __name__ == "__main__":
    _profile_self()
