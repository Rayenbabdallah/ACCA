"""
test_arc_agi3_bridge.py - SDK-free tests for the Kaggle adapter.

Part of ACCA (Active Causal Compression Agent)
ARC Prize 2026 . Paper Track
"""
from __future__ import annotations

import numpy as np

from src.arc_agi3_bridge import (
    KaggleACCAAgent,
    _extract_action_space,
    _extract_game_id,
    _extract_grid,
    _game_id_of,
    _normalize_action_name,
)


def test_extract_grid_from_mapping_frame():
    frame = {"grid": [[0, 1], [2, 3]], "game_id": "g"}

    grid = _extract_grid(frame)

    assert grid.dtype == np.uint8
    assert grid.tolist() == [[0, 1], [2, 3]]


def test_extract_grid_prefers_frame_over_status_state():
    frame = {"frame": [[1, 2], [3, 4]], "state": "NOT_FINISHED"}

    grid = _extract_grid(frame)

    assert grid.tolist() == [[1, 2], [3, 4]]


def test_extract_grid_reduces_channel_first_stack():
    frame = {"frame": [[[0, 1], [2, 0]], [[3, 0], [0, 4]]]}

    grid = _extract_grid(frame)

    assert grid.tolist() == [[3, 1], [2, 4]]


def test_extract_grid_reduces_multi_plane_stack():
    frame = {"frame": np.eye(5, 2, dtype=np.uint8).reshape(5, 1, 2).repeat(2, axis=1)}

    grid = _extract_grid(frame)

    assert grid.shape == (2, 2)


def test_extract_grid_reduces_deep_channel_first_stack():
    frame = {"frame": np.ones((22, 64, 64), dtype=np.uint8)}

    grid = _extract_grid(frame)

    assert grid.shape == (64, 64)


def test_extract_grid_tolerates_empty_official_frame():
    frame = {"frame": []}

    grid = _extract_grid(frame)

    assert grid.tolist() == [[0]]


def test_extract_metadata_defaults_and_values():
    frame = {"grid": [[0]], "game_id": "game-a", "action_space": ["RESET", "ACTION1"]}

    assert _extract_game_id(frame) == "game-a"
    assert _extract_action_space(frame) == ["RESET", "ACTION1"]


def test_extract_action_space_normalizes_official_values():
    frame = {"grid": [[0]], "available_actions": ["1", "GameAction.ACTION6", "RESET"]}

    assert _extract_action_space(frame) == ["ACTION1", "ACTION6", "RESET"]


def test_normalize_action_name_prefers_enum_name():
    action = type("OfficialAction", (), {"name": "ACTION2", "value": 2})()

    assert _normalize_action_name(action) == "ACTION2"


def test_kaggle_agent_resets_then_returns_actions():
    agent = KaggleACCAAgent()
    frame = {"grid": [[0, 14], [0, 0]], "game_id": "click", "action_space": ["RESET", "ACTION6"]}

    assert agent.choose_action([], frame) == "RESET"
    assert agent.choose_action([frame], frame) == "ACTION6 0 1"


def test_kaggle_agent_is_done_on_terminal_status():
    agent = KaggleACCAAgent()

    assert agent.is_done([], {"grid": [[0]], "state": "WIN"})
    assert not agent.is_done([], {"grid": [[0]], "state": "NOT_FINISHED"})


def test_game_id_of_environment_object():
    env = type("Env", (), {"game_id": "sk48-d8078629"})()

    assert _game_id_of(env) == "sk48-d8078629"
