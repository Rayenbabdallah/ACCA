"""
test_scorecard.py — Tests for ARC-AGI-3 scoring (cap=1.0, level-weighted game score).

Part of ACCA (Active Causal Compression Agent)
ARC Prize 2026 · Paper Track
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from eval.local_eval import DummyAgent, HumanSolutionAgent, run_evaluation
from eval.scorecard import compute_scorecard, game_score, level_score, total_score


def test_level_score_parity():
    assert level_score(10, 10) == 1.0


def test_level_score_agent_slower_than_human():
    assert level_score(10, 20) == 0.25


def test_level_score_capped_when_agent_faster_than_human():
    # cap=1.0 (NOT 1.15): min(10/5, 1.0)^2 = 1.0^2 = 1.0
    assert level_score(10, 5) == 1.0


def test_level_score_zero_agent_actions_returns_zero():
    assert level_score(10, 0) == 0.0


def test_level_score_negative_agent_actions_returns_zero():
    assert level_score(10, -1) == 0.0


def test_game_score_full_completion():
    # 6 levels all perfect: weighted mean = sum(1..6 * 1.0) / sum(1..6) = 21/21 = 1.0
    assert game_score([1.0] * 6, 6) == 1.0


def test_game_score_only_first_level_completed():
    # L1 only (weight 1), rest 0: 1/21
    assert game_score([1.0], 6) == pytest.approx(1 / 21)


def test_game_score_first_three_levels_completed():
    # L1..L3 (weights 1+2+3=6) out of 21
    assert game_score([1.0, 1.0, 1.0], 6) == pytest.approx(6 / 21)


def test_game_score_late_levels_dominate():
    # L6 only weighs 6/21 vs L1's 1/21
    s_l1_only = game_score([1.0, 0.0, 0.0, 0.0, 0.0, 0.0], 6)
    s_l6_only = game_score([0.0, 0.0, 0.0, 0.0, 0.0, 1.0], 6)
    assert s_l6_only == pytest.approx(6 * s_l1_only)


def test_total_score_mean_across_games():
    assert total_score([1.0, 0.5, 0.0]) == pytest.approx(0.5)


def test_total_score_empty():
    assert total_score([]) == 0.0


def test_compute_scorecard_multi_level_game():
    records = [
        {"game_id": "g1", "level_index": 1, "total_levels": 6, "human_actions": 10, "agent_actions": 10, "completed": True},
        {"game_id": "g1", "level_index": 2, "total_levels": 6, "human_actions": 10, "agent_actions": 20, "completed": True},
        {"game_id": "g1", "level_index": 3, "total_levels": 6, "human_actions": 10, "agent_actions": 100, "completed": False},
    ]
    sc = compute_scorecard(records)
    # L1: 1.0, L2: 0.25, L3..L6: 0.0
    # game_score = (1*1.0 + 2*0.25 + 0 + 0 + 0 + 0) / 21 = 1.5/21
    assert sc["per_game"][0]["game_score"] == pytest.approx(1.5 / 21)
    assert sc["per_game"][0]["levels_completed"] == 2
    assert sc["total_score"] == pytest.approx(1.5 / 21)


def test_compute_scorecard_multiple_games_averaged():
    records = [
        {"game_id": "g1", "level_index": 1, "total_levels": 1, "human_actions": 10, "agent_actions": 10, "completed": True},
        {"game_id": "g2", "level_index": 1, "total_levels": 1, "human_actions": 10, "agent_actions": 20, "completed": True},
    ]
    sc = compute_scorecard(records)
    # game_scores = [1.0, 0.25], mean = 0.625
    assert sc["total_score"] == pytest.approx(0.625)


def test_compute_scorecard_empty_returns_zeros():
    sc = compute_scorecard([])
    assert sc == {"total_score": 0.0, "completion_rate": 0.0, "per_game": [], "per_level": []}


def test_dummy_agent_scores_zero(tmp_path: Path):
    env_dir = tmp_path / "envs"
    env_dir.mkdir()
    for i in range(3):
        cfg = {"game_id": f"game_{i}", "human_actions": 10, "total_levels": 6}
        (env_dir / f"game_{i}.json").write_text(json.dumps(cfg))
    sc = run_evaluation(str(env_dir), DummyAgent, max_actions_per_level=20)
    assert sc["total_score"] == 0.0
    assert sc["completion_rate"] == 0.0
    assert all(g["levels_completed"] == 0 for g in sc["per_game"])


def test_human_solution_agent_scores_perfect_on_synthetic_games():
    sc = run_evaluation("envs/synthetic", HumanSolutionAgent, max_actions_per_level=50)
    assert sc["total_score"] == pytest.approx(1.0)
    assert sc["completion_rate"] == pytest.approx(1.0)
    assert len(sc["per_game"]) == 5
    assert all(g["levels_completed"] == 6 for g in sc["per_game"])
