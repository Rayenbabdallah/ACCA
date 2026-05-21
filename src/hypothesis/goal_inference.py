"""
goal_inference.py - Candidate goal templates over object graphs.

ARC-AGI-3 never states the win condition. This module maintains a small
posterior-like set of executable goal predicates and updates confidence from
observed terminal states and intermediate states. It uses only object graphs,
does not assume fixed grid dimensions, and performs no network calls.

Part of ACCA (Active Causal Compression Agent)
ARC Prize 2026 . Paper Track
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Literal

from src.perception.frame_parser import ObjectGraph, RelationType

Axis = Literal["x", "y"]
SymmetryAxis = Literal["vertical", "horizontal", "rotational"]


def _clamp(x: float) -> float:
    return max(0.0, min(1.0, x))


def _colored_pixels(state: ObjectGraph) -> dict[tuple[int, int], int]:
    pixels: dict[tuple[int, int], int] = {}
    for node in state.nodes.values():
        for pixel in node.pixels:
            pixels[pixel] = node.color
    return pixels


def _inferred_shape(state: ObjectGraph) -> tuple[int, int]:
    pixels = _colored_pixels(state)
    if not pixels:
        return (0, 0)
    max_row = max(r for r, _ in pixels)
    max_col = max(c for _, c in pixels)
    return (max_row + 1, max_col + 1)


@dataclass
class GoalTemplate(ABC):
    confidence: float = 0.5

    @abstractmethod
    def satisfied(self, state: ObjectGraph) -> bool:
        """Return True if this goal is satisfied by state."""

    @abstractmethod
    def name(self) -> str:
        """Stable human-readable goal identifier."""


@dataclass
class ReachPositionGoal(GoalTemplate):
    obj_id: int = 0
    target: tuple[float, float] = (0.0, 0.0)
    tolerance: float = 0.5

    def satisfied(self, state: ObjectGraph) -> bool:
        node = state.nodes.get(self.obj_id)
        if node is None:
            return False
        return (
            abs(node.centroid[0] - self.target[0]) <= self.tolerance
            and abs(node.centroid[1] - self.target[1]) <= self.tolerance
        )

    def name(self) -> str:
        return f"reach_position:{self.obj_id}:{self.target[0]:g}:{self.target[1]:g}"


@dataclass
class AllSameColorGoal(GoalTemplate):
    color: int | None = None

    def satisfied(self, state: ObjectGraph) -> bool:
        if not state.nodes:
            return False
        colors = {node.color for node in state.nodes.values()}
        if self.color is not None:
            return colors == {self.color}
        return len(colors) == 1

    def name(self) -> str:
        return f"all_same_color:{self.color if self.color is not None else 'any'}"


@dataclass
class ClearGridGoal(GoalTemplate):
    def satisfied(self, state: ObjectGraph) -> bool:
        return len(state.nodes) == 0

    def name(self) -> str:
        return "clear_grid"


@dataclass
class CountObjectsGoal(GoalTemplate):
    color: int = 0
    count: int = 0

    def satisfied(self, state: ObjectGraph) -> bool:
        actual = sum(1 for node in state.nodes.values() if node.color == self.color)
        return actual == self.count

    def name(self) -> str:
        return f"count_objects:{self.color}:{self.count}"


@dataclass
class MatchPatternGoal(GoalTemplate):
    target_pixels: dict[tuple[int, int], int] | None = None

    @classmethod
    def from_state(cls, state: ObjectGraph, confidence: float = 0.5) -> MatchPatternGoal:
        return cls(confidence=confidence, target_pixels=_colored_pixels(state))

    def satisfied(self, state: ObjectGraph) -> bool:
        if self.target_pixels is None:
            return False
        return _colored_pixels(state) == self.target_pixels

    def name(self) -> str:
        n = 0 if self.target_pixels is None else len(self.target_pixels)
        checksum = sum((r + 1) * 31 + (c + 1) * 17 + color for (r, c), color in (
            self.target_pixels or {}
        ).items())
        return f"match_pattern:{n}:{checksum}"


@dataclass
class ConnectObjectsGoal(GoalTemplate):
    obj_id_a: int = 0
    obj_id_b: int = 1

    def satisfied(self, state: ObjectGraph) -> bool:
        pair = {self.obj_id_a, self.obj_id_b}
        return any(
            edge.relation == RelationType.TOUCHING and {edge.src_id, edge.dst_id} == pair
            for edge in state.edges
        )

    def name(self) -> str:
        a, b = sorted((self.obj_id_a, self.obj_id_b))
        return f"connect_objects:{a}:{b}"


@dataclass
class FillRegionGoal(GoalTemplate):
    region: tuple[int, int, int, int] = (0, 0, 0, 0)
    color: int = 1

    def satisfied(self, state: ObjectGraph) -> bool:
        r1, c1, r2, c2 = self.region
        pixels = _colored_pixels(state)
        for row in range(r1, r2 + 1):
            for col in range(c1, c2 + 1):
                if pixels.get((row, col)) != self.color:
                    return False
        return True

    def name(self) -> str:
        return f"fill_region:{self.region}:{self.color}"


@dataclass
class SortObjectsGoal(GoalTemplate):
    axis: Axis = "x"
    obj_ids: tuple[int, ...] | None = None

    def satisfied(self, state: ObjectGraph) -> bool:
        if self.obj_ids is None:
            nodes = sorted(state.nodes.values(), key=lambda node: (node.color, node.obj_id))
        else:
            if any(obj_id not in state.nodes for obj_id in self.obj_ids):
                return False
            nodes = [state.nodes[obj_id] for obj_id in self.obj_ids]
        idx = 0 if self.axis == "x" else 1
        coords = [node.centroid[idx] for node in nodes]
        return coords == sorted(coords)

    def name(self) -> str:
        ids = "all" if self.obj_ids is None else ",".join(str(obj_id) for obj_id in self.obj_ids)
        return f"sort_objects:{self.axis}:{ids}"


@dataclass
class SymmetryGoal(GoalTemplate):
    axis: SymmetryAxis = "vertical"
    shape: tuple[int, int] | None = None

    def satisfied(self, state: ObjectGraph) -> bool:
        pixels = _colored_pixels(state)
        if not pixels:
            return False
        height, width = self.shape or _inferred_shape(state)
        for (row, col), color in pixels.items():
            mirror = self._mirror(row, col, height, width)
            if pixels.get(mirror) != color:
                return False
        return True

    def _mirror(self, row: int, col: int, height: int, width: int) -> tuple[int, int]:
        if self.axis == "vertical":
            return (row, width - 1 - col)
        if self.axis == "horizontal":
            return (height - 1 - row, col)
        return (height - 1 - row, width - 1 - col)

    def name(self) -> str:
        return f"symmetry:{self.axis}:{self.shape}"


@dataclass
class MaximizeAreaGoal(GoalTemplate):
    color: int = 0
    target_area: int = 0

    def satisfied(self, state: ObjectGraph) -> bool:
        area = sum(node.area for node in state.nodes.values() if node.color == self.color)
        return area >= self.target_area

    def name(self) -> str:
        return f"maximize_area:{self.color}:{self.target_area}"


class GoalInference:
    """Maintains and ranks candidate goals by confidence."""

    def __init__(self):
        self.goals: list[GoalTemplate] = self._init_all_goals()
        self._recently_active: set[str] = set()

    def _init_all_goals(self) -> list[GoalTemplate]:
        return [
            ClearGridGoal(confidence=0.5),
            AllSameColorGoal(confidence=0.5),
            SortObjectsGoal(confidence=0.5, axis="x"),
            SortObjectsGoal(confidence=0.5, axis="y"),
        ]

    def update_on_step(self, state: ObjectGraph) -> None:
        self._recently_active.clear()
        for goal in self.goals:
            if goal.satisfied(state):
                goal.confidence = _clamp(goal.confidence + 0.05)
                self._recently_active.add(goal.name())

    def update_on_success(self, terminal_state: ObjectGraph) -> None:
        for goal in self._candidate_goals_from_state(terminal_state):
            self._upsert(goal)
        self._recently_active.clear()
        for goal in self.goals:
            if goal.satisfied(terminal_state):
                goal.confidence = _clamp(goal.confidence + 0.3)
                self._recently_active.add(goal.name())

    def update_on_failure(self) -> None:
        active = self._recently_active or {goal.name() for goal in self.goals}
        for goal in self.goals:
            if goal.name() in active:
                goal.confidence = _clamp(goal.confidence - 0.1)
        self._recently_active.clear()

    def top_goal(self) -> GoalTemplate:
        if not self.goals:
            raise ValueError("no goal templates available")
        return self.top_k_goals(1)[0]

    def top_k_goals(self, k: int = 3) -> list[GoalTemplate]:
        return sorted(self.goals, key=lambda goal: (goal.confidence, goal.name()), reverse=True)[:k]

    def _upsert(self, candidate: GoalTemplate) -> None:
        name = candidate.name()
        for idx, existing in enumerate(self.goals):
            if existing.name() == name:
                existing.confidence = max(existing.confidence, candidate.confidence)
                return
        self.goals.append(candidate)

    def _candidate_goals_from_state(self, state: ObjectGraph) -> list[GoalTemplate]:
        candidates: list[GoalTemplate] = [
            ClearGridGoal(confidence=0.5),
            MatchPatternGoal.from_state(state, confidence=0.5),
        ]
        colors = sorted({node.color for node in state.nodes.values()})
        for color in colors:
            count = sum(1 for node in state.nodes.values() if node.color == color)
            area = sum(node.area for node in state.nodes.values() if node.color == color)
            candidates.append(CountObjectsGoal(confidence=0.5, color=color, count=count))
            candidates.append(MaximizeAreaGoal(confidence=0.5, color=color, target_area=area))

        if len(colors) == 1 and colors:
            candidates.append(AllSameColorGoal(confidence=0.5, color=colors[0]))

        for node in state.nodes.values():
            candidates.append(
                ReachPositionGoal(confidence=0.5, obj_id=node.obj_id, target=node.centroid)
            )
            x1, y1, x2, y2 = node.bbox
            candidates.append(
                FillRegionGoal(confidence=0.5, region=(y1, x1, y2, x2), color=node.color)
            )

        for edge in state.edges:
            if edge.relation == RelationType.TOUCHING:
                candidates.append(
                    ConnectObjectsGoal(
                        confidence=0.5,
                        obj_id_a=edge.src_id,
                        obj_id_b=edge.dst_id,
                    )
                )

        shape = _inferred_shape(state)
        candidates.extend(
            [
                SortObjectsGoal(confidence=0.5, axis="x"),
                SortObjectsGoal(confidence=0.5, axis="y"),
                SymmetryGoal(confidence=0.5, axis="vertical", shape=shape),
                SymmetryGoal(confidence=0.5, axis="horizontal", shape=shape),
                SymmetryGoal(confidence=0.5, axis="rotational", shape=shape),
            ]
        )
        return candidates
