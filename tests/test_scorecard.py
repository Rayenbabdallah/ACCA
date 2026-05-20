"""
test_scorecard.py — Tests for RHAE formula and evaluation harness.

Part of ACCA (Active Causal Compression Agent)
ARC Prize 2026 · Paper Track
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from eval.local_eval import DummyAgent, run_evaluation
from eval.scorecard import compute_scorecard, rhae


def test_rhae_parity():
    assert rhae(10, 10) == 1.0


def test_rhae_agent_worse_than_human():
    assert rhae(10, 20) == 0.25


def test_rhae_capped_when_agent_much_better():
    assert rhae(10, 5) == 1.15


def test_rhae_zero_actions_returns_zero():
    assert rhae(10, 0) == 0.0


def test_compute_scorecard_aggregates_mean_and_completion():
    results = [
        {"level_id": "a", "human_actions": 10, "ai_actions": 10, "completed": True},
        {"level_id": "b", "human_actions": 10, "ai_actions": 20, "completed": True},
        {"level_id": "c", "human_actions": 10, "ai_actions": 100, "completed": False},
    ]
    sc = compute_scorecard(results)
    assert sc["completion_rate"] == pytest.approx(2 / 3)
    # mean over: 1.0 (a) + 0.25 (b) + 0.0 (c, incomplete) = 1.25 / 3
    assert sc["mean_rhae"] == pytest.approx(1.25 / 3)
    assert sc["per_level"][2]["rhae"] == 0.0


def test_compute_scorecard_empty():
    sc = compute_scorecard([])
    assert sc == {"mean_rhae": 0.0, "per_level": [], "completion_rate": 0.0}


def test_dummy_agent_scores_zero(tmp_path: Path):
    env_dir = tmp_path / "envs"
    env_dir.mkdir()
    for i in range(3):
        cfg = {"level_id": f"level_{i}", "human_actions": 10}
        (env_dir / f"level_{i}.json").write_text(json.dumps(cfg))

    sc = run_evaluation(str(env_dir), DummyAgent, max_actions_per_level=20)
    assert sc["mean_rhae"] == 0.0
    assert sc["completion_rate"] == 0.0
    assert len(sc["per_level"]) == 3
    for lvl in sc["per_level"]:
        assert lvl["completed"] is False
        assert lvl["rhae"] == 0.0
