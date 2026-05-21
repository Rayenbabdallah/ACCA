"""
replay_buffer.py - Bounded storage for ACCA transition observations.

Part of ACCA (Active Causal Compression Agent)
ARC Prize 2026 . Paper Track
"""
from __future__ import annotations

from collections import deque

from src.perception.event_extractor import ActionEnum, Observation


class ReplayBuffer:
    """Stores recent observations for the hypothesis engine."""

    def __init__(self, maxlen: int = 1000):
        if maxlen <= 0:
            raise ValueError(f"maxlen must be positive, got {maxlen}")
        self._items: deque[Observation] = deque(maxlen=maxlen)

    def add(self, obs: Observation) -> None:
        self._items.append(obs)

    def get_all(self) -> list[Observation]:
        return list(self._items)

    def get_by_action(self, action: ActionEnum | str) -> list[Observation]:
        action_enum = action if isinstance(action, ActionEnum) else ActionEnum(action)
        return [obs for obs in self._items if obs.action == action_enum]

    def clear(self) -> None:
        self._items.clear()
