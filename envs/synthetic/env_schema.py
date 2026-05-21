"""
env_schema.py — Synthetic environment dataclass + JSON (de)serialization.

`goal_state` is included alongside the fields named in the task spec because
verification requires an explicit target — see verifier.py.

LEGACY (v1) — single-mechanic single-level. Per CORRECTION_PROMPT_FINAL §P0-T2
and ARC-AGI-3 Tech Report §3.4, the real synthetic suite must be 5 games × 6
levels × multi-mechanic composition. This schema is kept temporarily as scaffolding
for early eval-harness work; a v2 schema (`SyntheticGame` with `levels: List[Level]`)
is the next planned redesign.

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
class SyntheticEnv:
    env_id: str
    mechanic_family: str
    is_adversarial: bool
    initial_state: np.ndarray
    action_space: List[str]
    human_solution: List[str]
    human_action_count: int
    mechanic_description: str
    goal_state: np.ndarray

    def to_dict(self) -> dict:
        return {
            "env_id": self.env_id,
            "mechanic_family": self.mechanic_family,
            "is_adversarial": self.is_adversarial,
            "initial_state": self.initial_state.astype(np.uint8).tolist(),
            "action_space": list(self.action_space),
            "human_solution": list(self.human_solution),
            "human_action_count": self.human_action_count,
            "mechanic_description": self.mechanic_description,
            "goal_state": self.goal_state.astype(np.uint8).tolist(),
        }

    @classmethod
    def from_dict(cls, d: dict) -> SyntheticEnv:
        return cls(
            env_id=d["env_id"],
            mechanic_family=d["mechanic_family"],
            is_adversarial=bool(d["is_adversarial"]),
            initial_state=np.asarray(d["initial_state"], dtype=np.uint8),
            action_space=list(d["action_space"]),
            human_solution=list(d["human_solution"]),
            human_action_count=int(d["human_action_count"]),
            mechanic_description=d["mechanic_description"],
            goal_state=np.asarray(d["goal_state"], dtype=np.uint8),
        )

    def save(self, path: str | Path) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: str | Path) -> SyntheticEnv:
        with open(path, "r", encoding="utf-8") as f:
            return cls.from_dict(json.load(f))
