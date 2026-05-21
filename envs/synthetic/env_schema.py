"""
env_schema.py — Synthetic game / level dataclasses + JSON (de)serialization.

v2 schema per CORRECTION_PROMPT_FINAL §P0-T2 and ARC-AGI-3 Tech Report §3.4:
five games (sy01..sy05), each with exactly 6 levels of progressive
multi-mechanic composition. Each `Level` declares which subset of the game's
available mechanics is *active* — the simulator composes them in canonical
order per tick (see verifier.MECHANIC_ORDER).

`goal_state` is included for verification; the agent never sees it.

Part of ACCA (Active Causal Compression Agent)
ARC Prize 2026 · Paper Track
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import List

import numpy as np


@dataclass
class Level:
    level_index: int            # 1-indexed
    description: str
    initial_state: np.ndarray   # (H, W) uint8
    mechanics_active: List[str]
    human_solution: List[str]
    human_action_count: int
    goal_state: np.ndarray

    def to_dict(self) -> dict:
        return {
            "level_index": self.level_index,
            "description": self.description,
            "initial_state": self.initial_state.astype(np.uint8).tolist(),
            "mechanics_active": list(self.mechanics_active),
            "human_solution": list(self.human_solution),
            "human_action_count": self.human_action_count,
            "goal_state": self.goal_state.astype(np.uint8).tolist(),
        }

    @classmethod
    def from_dict(cls, d: dict) -> Level:
        return cls(
            level_index=int(d["level_index"]),
            description=d["description"],
            initial_state=np.asarray(d["initial_state"], dtype=np.uint8),
            mechanics_active=list(d["mechanics_active"]),
            human_solution=list(d["human_solution"]),
            human_action_count=int(d["human_action_count"]),
            goal_state=np.asarray(d["goal_state"], dtype=np.uint8),
        )


@dataclass
class SyntheticGame:
    game_id: str
    name: str
    description: str
    action_space: List[str]
    available_mechanics: List[str]   # universe of mechanics this game composes from
    levels: List[Level]

    @property
    def total_levels(self) -> int:
        return len(self.levels)

    @property
    def human_total_actions(self) -> int:
        return sum(l.human_action_count for l in self.levels)

    def to_dict(self) -> dict:
        return {
            "game_id": self.game_id,
            "name": self.name,
            "description": self.description,
            "action_space": list(self.action_space),
            "available_mechanics": list(self.available_mechanics),
            "total_levels": self.total_levels,
            "human_total_actions": self.human_total_actions,
            "levels": [l.to_dict() for l in self.levels],
        }

    @classmethod
    def from_dict(cls, d: dict) -> SyntheticGame:
        return cls(
            game_id=d["game_id"],
            name=d["name"],
            description=d["description"],
            action_space=list(d["action_space"]),
            available_mechanics=list(d["available_mechanics"]),
            levels=[Level.from_dict(l) for l in d["levels"]],
        )

    def save(self, path: str | Path) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: str | Path) -> SyntheticGame:
        with open(path, "r", encoding="utf-8") as f:
            return cls.from_dict(json.load(f))
