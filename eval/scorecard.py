"""
scorecard.py — ARC-AGI-3 scoring (Kaggle data page formulas, authoritative).

`level_score`  : squared (human/agent) ratio capped at 1.0.
`game_score`   : level-index-weighted mean of level scores across a game's levels;
                 uncompleted levels score 0.
`total_score`  : unweighted mean of game scores across all evaluated games.

`compute_scorecard` consumes per-level records and emits per-game + total.

Part of ACCA (Active Causal Compression Agent)
ARC Prize 2026 · Paper Track
"""
from __future__ import annotations

from typing import Dict, List, TypedDict

LEVEL_SCORE_CAP: float = 1.0


class LevelRecord(TypedDict):
    game_id: str
    level_index: int          # 1-indexed
    total_levels: int         # total levels declared for the game
    human_actions: int
    agent_actions: int
    completed: bool


class LevelOut(TypedDict):
    game_id: str
    level_index: int
    human_actions: int
    agent_actions: int
    completed: bool
    level_score: float


class GameOut(TypedDict):
    game_id: str
    total_levels: int
    levels_completed: int
    game_score: float
    level_scores: List[float]


class Scorecard(TypedDict):
    total_score: float
    completion_rate: float
    per_game: List[GameOut]
    per_level: List[LevelOut]


def level_score(human_actions: int, agent_actions: int) -> float:
    """min(human/agent, 1.0)**2. Returns 0.0 if agent_actions is non-positive."""
    if agent_actions <= 0:
        return 0.0
    ratio = human_actions / agent_actions
    return min(ratio, LEVEL_SCORE_CAP) ** 2


def game_score(level_scores_in_order: List[float], total_levels: int) -> float:
    """Level-index-weighted mean. `level_scores_in_order` lists scores by level index;
    missing tail levels are treated as 0.0 (uncompleted)."""
    if total_levels <= 0:
        return 0.0
    padded = list(level_scores_in_order) + [0.0] * (total_levels - len(level_scores_in_order))
    padded = padded[:total_levels]
    weights = range(1, total_levels + 1)
    return sum(w * s for w, s in zip(weights, padded)) / sum(weights)


def total_score(game_scores: List[float]) -> float:
    """Unweighted mean across the games evaluated."""
    if not game_scores:
        return 0.0
    return sum(game_scores) / len(game_scores)


def compute_scorecard(records: List[LevelRecord]) -> Scorecard:
    """Group records by game_id, compute per-game weighted scores, then total."""
    if not records:
        return {"total_score": 0.0, "completion_rate": 0.0, "per_game": [], "per_level": []}

    per_level: List[LevelOut] = []
    by_game: Dict[str, List[LevelRecord]] = {}
    total_completed = 0

    for r in records:
        score = level_score(r["human_actions"], r["agent_actions"]) if r["completed"] else 0.0
        if r["completed"]:
            total_completed += 1
        per_level.append(
            {
                "game_id": r["game_id"],
                "level_index": r["level_index"],
                "human_actions": r["human_actions"],
                "agent_actions": r["agent_actions"],
                "completed": r["completed"],
                "level_score": score,
            }
        )
        by_game.setdefault(r["game_id"], []).append(r)

    per_game: List[GameOut] = []
    for game_id, recs in by_game.items():
        recs_sorted = sorted(recs, key=lambda x: x["level_index"])
        total_levels = max(r["total_levels"] for r in recs_sorted)
        ordered: List[float] = [0.0] * total_levels
        completed = 0
        for r in recs_sorted:
            idx = r["level_index"] - 1
            if 0 <= idx < total_levels:
                s = level_score(r["human_actions"], r["agent_actions"]) if r["completed"] else 0.0
                ordered[idx] = s
                if r["completed"]:
                    completed += 1
        gscore = game_score(ordered, total_levels)
        per_game.append(
            {
                "game_id": game_id,
                "total_levels": total_levels,
                "levels_completed": completed,
                "game_score": gscore,
                "level_scores": ordered,
            }
        )

    return {
        "total_score": total_score([g["game_score"] for g in per_game]),
        "completion_rate": total_completed / len(records),
        "per_game": per_game,
        "per_level": per_level,
    }
