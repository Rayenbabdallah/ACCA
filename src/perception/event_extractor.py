"""
event_extractor.py - Object-level transition deltas from tracked ARC-AGI-3 frames.

Compares consecutive tracked ObjectGraphs plus the action between them and emits
typed AtomicEvents for mechanic induction. Grids are variable-size max 64x64;
this module only uses object coordinates and never assumes a fixed frame size.

Part of ACCA (Active Causal Compression Agent)
ARC Prize 2026 . Paper Track
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Iterable

from src.perception.frame_parser import ObjectGraph, ObjectNode
from src.perception.object_tracker import TrackingResult


class ActionEnum(str, Enum):
    RESET = "RESET"
    ACTION1 = "ACTION1"
    ACTION2 = "ACTION2"
    ACTION3 = "ACTION3"
    ACTION4 = "ACTION4"
    ACTION5 = "ACTION5"
    ACTION6 = "ACTION6"
    ACTION7 = "ACTION7"


@dataclass(frozen=True)
class AtomicEvent:
    event_type: str
    obj_id: int
    params: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Observation:
    action: ActionEnum
    pre_state: ObjectGraph
    post_state: ObjectGraph
    delta: list[AtomicEvent]
    timestamp: int


def _as_action(action: ActionEnum | str) -> ActionEnum:
    if isinstance(action, ActionEnum):
        return action
    return ActionEnum(action)


def _event_sort_key(event: AtomicEvent) -> tuple[str, int, str]:
    return (event.event_type, event.obj_id, repr(sorted(event.params.items())))


def canonical_delta(events: Iterable[AtomicEvent]) -> tuple[AtomicEvent, ...]:
    """Stable order for comparing event lists independent of extraction order."""
    return tuple(sorted(events, key=_event_sort_key))


def _rounded_delta(pre: ObjectNode, post: ObjectNode) -> tuple[int, int]:
    dx = int(round(post.centroid[0] - pre.centroid[0]))
    dy = int(round(post.centroid[1] - pre.centroid[1]))
    return dx, dy


class EventExtractor:
    """Extract MOVED / RECOLORED / SPAWNED / DESTROYED events."""

    def extract(
        self,
        action: ActionEnum | str,
        pre: ObjectGraph,
        post: ObjectGraph | TrackingResult,
        timestamp: int,
    ) -> Observation:
        post_graph = post.tracked if isinstance(post, TrackingResult) else post
        spawned_ids = set(post.spawned) if isinstance(post, TrackingResult) else (
            set(post_graph.nodes) - set(pre.nodes)
        )
        destroyed_ids = set(post.destroyed) if isinstance(post, TrackingResult) else (
            set(pre.nodes) - set(post_graph.nodes)
        )

        events: list[AtomicEvent] = []

        surviving_ids = (set(pre.nodes) & set(post_graph.nodes)) - spawned_ids - destroyed_ids
        for obj_id in sorted(surviving_ids):
            pre_node = pre.nodes[obj_id]
            post_node = post_graph.nodes[obj_id]
            dx, dy = _rounded_delta(pre_node, post_node)
            if abs(post_node.centroid[0] - pre_node.centroid[0]) > 0.5 or (
                abs(post_node.centroid[1] - pre_node.centroid[1]) > 0.5
            ):
                events.append(
                    AtomicEvent("MOVED", obj_id, {"dx": dx, "dy": dy})
                )
            if pre_node.color != post_node.color:
                events.append(
                    AtomicEvent(
                        "RECOLORED",
                        obj_id,
                        {"old_color": pre_node.color, "new_color": post_node.color},
                    )
                )

        for obj_id in sorted(destroyed_ids):
            if obj_id in pre.nodes:
                events.append(
                    AtomicEvent("DESTROYED", obj_id, {"color": pre.nodes[obj_id].color})
                )

        for obj_id in sorted(spawned_ids):
            if obj_id in post_graph.nodes:
                node = post_graph.nodes[obj_id]
                events.append(
                    AtomicEvent(
                        "SPAWNED",
                        obj_id,
                        {"color": node.color, "position": node.centroid},
                    )
                )

        return Observation(
            action=_as_action(action),
            pre_state=pre,
            post_state=post_graph,
            delta=events,
            timestamp=timestamp,
        )
