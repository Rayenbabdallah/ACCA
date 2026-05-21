"""
test_arc_agi3_bridge.py - SDK-free tests for the Kaggle adapter.

Part of ACCA (Active Causal Compression Agent)
ARC Prize 2026 . Paper Track
"""
from __future__ import annotations

import numpy as np

from src.arc_agi3_bridge import KaggleACCAAgent, _extract_action_space, _extract_game_id, _extract_grid


def test_extract_grid_from_mapping_frame():
    frame = {"grid": [[0, 1], [2, 3]], "game_id": "g"}

    grid = _extract_grid(frame)

    assert grid.dtype == np.uint8
    assert grid.tolist() == [[0, 1], [2, 3]]


def test_extract_metadata_defaults_and_values():
    frame = {"grid": [[0]], "game_id": "game-a", "action_space": ["RESET", "ACTION1"]}

    assert _extract_game_id(frame) == "game-a"
    assert _extract_action_space(frame) == ["RESET", "ACTION1"]


def test_kaggle_agent_resets_then_returns_actions():
    agent = KaggleACCAAgent()
    frame = {"grid": [[0, 14], [0, 0]], "game_id": "click", "action_space": ["RESET", "ACTION6"]}

    assert agent.choose_action([], frame) == "RESET"
    assert agent.choose_action([frame], frame) == "ACTION6 0 1"


def test_kaggle_agent_is_done_on_terminal_status():
    agent = KaggleACCAAgent()

    assert agent.is_done([], {"grid": [[0]], "state": "WIN"})
    assert not agent.is_done([], {"grid": [[0]], "state": "NOT_FINISHED"})
