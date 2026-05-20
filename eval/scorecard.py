"""
scorecard.py — RHAE scoring and scorecard aggregation for ACCA evaluation.

Part of ACCA (Active Causal Compression Agent)
ARC Prize 2026 · Paper Track
"""
from __future__ import annotations

from typing import List, TypedDict


class LevelResult(TypedDict):
    level_id: str
    human_actions: int
    ai_actions: int
    completed: bool


class LevelScore(TypedDict):
    level_id: str
    human_actions: int
    ai_actions: int
    completed: bool
    rhae: float


class Scorecard(TypedDict):
    mean_rhae: float
    per_level: List[LevelScore]
    completion_rate: float


RHAE_CAP: float = 1.15


def rhae(human_actions: int, ai_actions: int) -> float:
    """Relative Human Action Efficiency.

    Returns 0.0 when the agent took no actions (degenerate case — a level
    can never be considered solved without at least one action).
    """
    if ai_actions == 0:
        return 0.0
    raw = (human_actions / ai_actions) ** 2
    return min(raw, RHAE_CAP)


def compute_scorecard(results: List[LevelResult]) -> Scorecard:
    """Aggregate per-level results into mean RHAE and completion rate.

    Incomplete levels score RHAE = 0.0 regardless of action counts: an
    unsolved level provides no efficiency signal.
    """
    if not results:
        return {"mean_rhae": 0.0, "per_level": [], "completion_rate": 0.0}

    per_level: List[LevelScore] = []
    completed_count = 0
    rhae_sum = 0.0

    for r in results:
        score = rhae(r["human_actions"], r["ai_actions"]) if r["completed"] else 0.0
        if r["completed"]:
            completed_count += 1
        rhae_sum += score
        per_level.append(
            {
                "level_id": r["level_id"],
                "human_actions": r["human_actions"],
                "ai_actions": r["ai_actions"],
                "completed": r["completed"],
                "rhae": score,
            }
        )

    n = len(results)
    return {
        "mean_rhae": rhae_sum / n,
        "per_level": per_level,
        "completion_rate": completed_count / n,
    }
