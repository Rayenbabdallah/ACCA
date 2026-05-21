"""
mechanic_dsl.py - Executable typed causal hypothesis language.

The DSL represents candidate ARC mechanics as compact, executable IF/THEN
rules over object graphs. It is intentionally offline and deterministic:
no LLM calls, no network access, and no assumptions about fixed grid size.

Part of ACCA (Active Causal Compression Agent)
ARC Prize 2026 . Paper Track
"""
from __future__ import annotations

import hashlib
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Literal
from uuid import uuid4

from src.perception.event_extractor import ActionEnum, AtomicEvent, EventExtractor
from src.perception.frame_parser import ObjectGraph, ObjectNode, RelationType, build_edges

CountOp = Literal["eq", "gt", "lt", "ge", "le"]
PropertyOp = Literal["eq", "gt", "lt", "ge", "le"]


def _as_action(action: ActionEnum | str) -> ActionEnum:
    if isinstance(action, ActionEnum):
        return action
    return ActionEnum(str(action).split()[0])


def _compare(left: float, op: CountOp | PropertyOp, right: float) -> bool:
    if op == "eq":
        return left == right
    if op == "gt":
        return left > right
    if op == "lt":
        return left < right
    if op == "ge":
        return left >= right
    if op == "le":
        return left <= right
    raise ValueError(f"unsupported comparison op: {op}")


def _copy_node(node: ObjectNode, obj_id: int | None = None) -> ObjectNode:
    return ObjectNode(
        obj_id=node.obj_id if obj_id is None else obj_id,
        color=node.color,
        bbox=node.bbox,
        centroid=node.centroid,
        area=node.area,
        pixels=node.pixels,
    )


def _copy_graph(state: ObjectGraph) -> ObjectGraph:
    nodes = {obj_id: _copy_node(node) for obj_id, node in state.nodes.items()}
    return ObjectGraph(nodes=nodes, edges=list(state.edges), frame_index=state.frame_index)


def _graph_with_nodes(state: ObjectGraph, nodes: dict[int, ObjectNode]) -> ObjectGraph:
    return ObjectGraph(
        nodes=nodes,
        edges=build_edges(nodes.values()),
        frame_index=state.frame_index,
    )


def _move_node(node: ObjectNode, dx: int, dy: int) -> ObjectNode:
    moved_pixels = frozenset((r + dy, c + dx) for (r, c) in node.pixels)
    x1, y1, x2, y2 = node.bbox
    cx, cy = node.centroid
    return ObjectNode(
        obj_id=node.obj_id,
        color=node.color,
        bbox=(x1 + dx, y1 + dy, x2 + dx, y2 + dy),
        centroid=(cx + dx, cy + dy),
        area=node.area,
        pixels=moved_pixels,
    )


def _next_obj_id(state: ObjectGraph) -> int:
    return max(state.nodes, default=-1) + 1


class Condition(ABC):
    @abstractmethod
    def evaluate(self, state: ObjectGraph, action: ActionEnum | str) -> bool:
        """Return True if this condition holds in state under action."""

    @abstractmethod
    def description_length(self) -> int:
        """Number of DSL tokens used by this condition."""

    @abstractmethod
    def describe(self) -> str:
        """Stable textual representation for hashing and debugging."""


@dataclass(frozen=True)
class ObjColorCondition(Condition):
    obj_id: int
    color: int

    def evaluate(self, state: ObjectGraph, action: ActionEnum | str) -> bool:
        node = state.nodes.get(self.obj_id)
        return node is not None and node.color == self.color

    def description_length(self) -> int:
        return 3

    def describe(self) -> str:
        return f"ObjColor({self.obj_id},{self.color})"


@dataclass(frozen=True)
class ActionIsCondition(Condition):
    action: ActionEnum | str

    def evaluate(self, state: ObjectGraph, action: ActionEnum | str) -> bool:
        return _as_action(action) == _as_action(self.action)

    def description_length(self) -> int:
        return 2

    def describe(self) -> str:
        return f"ActionIs({_as_action(self.action).value})"


@dataclass(frozen=True)
class ObjRelationCondition(Condition):
    src_id: int
    dst_id: int
    relation: RelationType

    def evaluate(self, state: ObjectGraph, action: ActionEnum | str) -> bool:
        return any(
            edge.src_id == self.src_id
            and edge.dst_id == self.dst_id
            and edge.relation == self.relation
            for edge in state.edges
        )

    def description_length(self) -> int:
        return 4

    def describe(self) -> str:
        return f"ObjRelation({self.src_id},{self.relation.value},{self.dst_id})"


@dataclass(frozen=True)
class ObjCountCondition(Condition):
    color: int
    op: CountOp
    count: int

    def evaluate(self, state: ObjectGraph, action: ActionEnum | str) -> bool:
        actual = sum(1 for node in state.nodes.values() if node.color == self.color)
        return _compare(actual, self.op, self.count)

    def description_length(self) -> int:
        return 4

    def describe(self) -> str:
        return f"ObjCount(color={self.color},{self.op},{self.count})"


@dataclass(frozen=True)
class ObjPropertyCondition(Condition):
    obj_id: int
    property_name: str
    op: PropertyOp
    value: float

    def evaluate(self, state: ObjectGraph, action: ActionEnum | str) -> bool:
        node = state.nodes.get(self.obj_id)
        if node is None:
            return False
        actual = self._property_value(node)
        return _compare(actual, self.op, self.value)

    def _property_value(self, node: ObjectNode) -> float:
        x1, y1, x2, y2 = node.bbox
        if self.property_name == "area":
            return float(node.area)
        if self.property_name == "bbox_width":
            return float(x2 - x1 + 1)
        if self.property_name == "bbox_height":
            return float(y2 - y1 + 1)
        if self.property_name == "centroid_x":
            return float(node.centroid[0])
        if self.property_name == "centroid_y":
            return float(node.centroid[1])
        if self.property_name == "left_edge":
            return float(x1)
        if self.property_name == "top_edge":
            return float(y1)
        raise ValueError(f"unsupported object property: {self.property_name}")

    def description_length(self) -> int:
        return 5

    def describe(self) -> str:
        return f"ObjProperty({self.obj_id},{self.property_name},{self.op},{self.value:g})"


class Effect(ABC):
    @abstractmethod
    def apply(self, state: ObjectGraph, matching_objs: list[int]) -> ObjectGraph:
        """Apply this effect to a copied graph and return the new graph."""

    @abstractmethod
    def description_length(self) -> int:
        """Number of DSL tokens used by this effect."""

    @abstractmethod
    def describe(self) -> str:
        """Stable textual representation for hashing and debugging."""


@dataclass(frozen=True)
class MoveEffect(Effect):
    obj_id: int
    dx: int
    dy: int

    def apply(self, state: ObjectGraph, matching_objs: list[int]) -> ObjectGraph:
        nodes = {obj_id: _copy_node(node) for obj_id, node in state.nodes.items()}
        if self.obj_id in nodes:
            nodes[self.obj_id] = _move_node(nodes[self.obj_id], self.dx, self.dy)
        return _graph_with_nodes(state, nodes)

    def description_length(self) -> int:
        return 4

    def describe(self) -> str:
        return f"Move({self.obj_id},{self.dx},{self.dy})"


@dataclass(frozen=True)
class RecolorEffect(Effect):
    obj_id: int
    new_color: int

    def apply(self, state: ObjectGraph, matching_objs: list[int]) -> ObjectGraph:
        nodes = {obj_id: _copy_node(node) for obj_id, node in state.nodes.items()}
        node = nodes.get(self.obj_id)
        if node is not None:
            nodes[self.obj_id] = ObjectNode(
                obj_id=node.obj_id,
                color=self.new_color,
                bbox=node.bbox,
                centroid=node.centroid,
                area=node.area,
                pixels=node.pixels,
            )
        return _graph_with_nodes(state, nodes)

    def description_length(self) -> int:
        return 3

    def describe(self) -> str:
        return f"Recolor({self.obj_id},{self.new_color})"


@dataclass(frozen=True)
class SpawnEffect(Effect):
    color: int
    position: tuple[int, int] | None = None
    relative_to_obj_id: int | None = None
    offset: tuple[int, int] = (0, 0)
    obj_id: int | None = None

    def apply(self, state: ObjectGraph, matching_objs: list[int]) -> ObjectGraph:
        nodes = {obj_id: _copy_node(node) for obj_id, node in state.nodes.items()}
        spawn_id = self.obj_id if self.obj_id is not None else _next_obj_id(state)
        if spawn_id in nodes:
            return _graph_with_nodes(state, nodes)
        row, col = self._spawn_position(state)
        node = ObjectNode(
            obj_id=spawn_id,
            color=self.color,
            bbox=(col, row, col, row),
            centroid=(float(col), float(row)),
            area=1,
            pixels=frozenset({(row, col)}),
        )
        nodes[spawn_id] = node
        return _graph_with_nodes(state, nodes)

    def _spawn_position(self, state: ObjectGraph) -> tuple[int, int]:
        if self.position is not None:
            return self.position
        if self.relative_to_obj_id is not None and self.relative_to_obj_id in state.nodes:
            base = state.nodes[self.relative_to_obj_id]
            row = int(round(base.centroid[1])) + self.offset[0]
            col = int(round(base.centroid[0])) + self.offset[1]
            return row, col
        return self.offset

    def description_length(self) -> int:
        return 5

    def describe(self) -> str:
        return (
            f"Spawn(color={self.color},position={self.position},"
            f"relative={self.relative_to_obj_id},offset={self.offset},id={self.obj_id})"
        )


@dataclass(frozen=True)
class DestroyEffect(Effect):
    obj_id: int

    def apply(self, state: ObjectGraph, matching_objs: list[int]) -> ObjectGraph:
        nodes = {obj_id: _copy_node(node) for obj_id, node in state.nodes.items()}
        nodes.pop(self.obj_id, None)
        return _graph_with_nodes(state, nodes)

    def description_length(self) -> int:
        return 2

    def describe(self) -> str:
        return f"Destroy({self.obj_id})"


@dataclass(frozen=True)
class ToggleEffect(Effect):
    obj_id: int
    color_a: int
    color_b: int

    def apply(self, state: ObjectGraph, matching_objs: list[int]) -> ObjectGraph:
        node = state.nodes.get(self.obj_id)
        if node is None or node.color not in (self.color_a, self.color_b):
            return _copy_graph(state)
        next_color = self.color_b if node.color == self.color_a else self.color_a
        return RecolorEffect(self.obj_id, next_color).apply(state, matching_objs)

    def description_length(self) -> int:
        return 4

    def describe(self) -> str:
        return f"Toggle({self.obj_id},{self.color_a},{self.color_b})"


@dataclass
class CausalRule:
    conditions: list[Condition]
    effects: list[Effect]

    def matches(self, state: ObjectGraph, action: ActionEnum | str) -> bool:
        return all(condition.evaluate(state, action) for condition in self.conditions)

    def apply(self, state: ObjectGraph) -> ObjectGraph:
        out = _copy_graph(state)
        matching_objs: list[int] = []
        for effect in self.effects:
            out = effect.apply(out, matching_objs)
        return out

    def description_length(self) -> int:
        return sum(c.description_length() for c in self.conditions) + sum(
            e.description_length() for e in self.effects
        )

    def describe(self) -> str:
        cond = "&".join(c.describe() for c in self.conditions) or "TRUE"
        eff = "&".join(e.describe() for e in self.effects) or "NOOP"
        return f"IF {cond} THEN {eff}"


@dataclass
class Hypothesis:
    rules: list[CausalRule]
    hypothesis_id: str = field(default_factory=lambda: str(uuid4())[:8])

    def execute(self, state: ObjectGraph, action: ActionEnum | str) -> ObjectGraph:
        out = _copy_graph(state)
        for rule in self.rules:
            if rule.matches(out, action):
                out = rule.apply(out)
        return out

    def description_length(self) -> int:
        return sum(rule.description_length() for rule in self.rules)

    def fingerprint(self) -> str:
        payload = "|".join(sorted(rule.describe() for rule in self.rules))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]

    def predict_delta(self, state: ObjectGraph, action: ActionEnum | str) -> list[AtomicEvent]:
        predicted = self.execute(state, action)
        observation = EventExtractor().extract(
            action,
            state,
            predicted,
            timestamp=state.frame_index + 1,
        )
        return observation.delta
