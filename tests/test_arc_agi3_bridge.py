"""
test_arc_agi3_bridge.py - SDK-free tests for the Kaggle adapter.

Part of ACCA (Active Causal Compression Agent)
ARC Prize 2026 . Paper Track
"""
from __future__ import annotations

import numpy as np

from src.arc_agi3_bridge import (
    BRIDGE_VERSION,
    KaggleACCAAgent,
    _click_targets,
    _extract_action_space,
    _extract_game_id,
    _extract_grid,
    _extract_levels_completed,
    _game_id_of,
    _normalize_action_name,
)


def test_bridge_version_is_visible_for_kaggle_cache_checks():
    assert BRIDGE_VERSION


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


def test_extract_levels_completed_from_frame_and_payload():
    assert _extract_levels_completed({"levels_completed": 2}) == 2
    assert _extract_levels_completed({"payload": {"completed_levels": 3}}) == 3
    assert _extract_levels_completed({"grid": [[0]]}) == 0


def test_extract_action_space_normalizes_official_values():
    frame = {"grid": [[0]], "available_actions": ["1", "GameAction.ACTION6", "RESET"]}

    assert _extract_action_space(frame) == ["ACTION1", "ACTION6", "RESET"]


def test_normalize_action_name_prefers_enum_name():
    action = type("OfficialAction", (), {"name": "ACTION2", "value": 2})()

    assert _normalize_action_name(action) == "ACTION2"


def test_kaggle_agent_acts_immediately_on_live_initial_frame():
    agent = KaggleACCAAgent()
    frame = {"grid": [[0, 14], [0, 0]], "game_id": "click", "action_space": ["RESET", "ACTION6"]}

    assert agent.choose_action([], frame) == "ACTION6 0 1"


def test_kaggle_agent_resets_after_game_over_keeps_agent_alive():
    """After 2026-05-21: GAME_OVER mid-game no longer nulls self.agent.
    The earlier code wiped the hypothesis bank on every GAME_OVER, so cross-level
    memory never had a chance to apply. We now RESET + call agent.on_new_level()
    but keep the agent instance (and its bank) alive."""
    agent = KaggleACCAAgent()
    # First call sets self.agent (game_id-change branch)
    frame = {"grid": [[1]], "game_id": "g", "state": "NOT_FINISHED", "action_space": ["RESET", "ACTION1"]}
    agent.choose_action([], frame)
    first_inner_agent = agent.agent
    assert first_inner_agent is not None

    # Now GAME_OVER mid-game on the SAME game_id — must RESET but keep the agent.
    game_over_frame = {"grid": [[1]], "game_id": "g", "state": "GAME_OVER", "action_space": ["RESET", "ACTION1"]}
    assert agent.choose_action([], game_over_frame) == "RESET"
    assert agent.agent is first_inner_agent


def test_click_targets_include_color_centroids_and_center():
    grid = np.zeros((8, 8), dtype=np.uint8)
    grid[1:3, 5:7] = 4

    targets = _click_targets(grid)

    assert targets[0] == (1, 5)
    assert (4, 4) in targets


def test_kaggle_agent_is_done_on_terminal_status():
    agent = KaggleACCAAgent()

    assert agent.is_done([], {"grid": [[0]], "state": "WIN"})
    assert not agent.is_done([], {"grid": [[0]], "state": "NOT_FINISHED"})


def test_kaggle_agent_respects_max_actions_without_off_by_one(monkeypatch):
    class FakeEnv:
        def __init__(self):
            self.calls = 0
            self.observation_space = {
                "grid": [[0, 1], [0, 0]],
                "game_id": "g",
                "state": "NOT_FINISHED",
                "action_space": ["ACTION6"],
            }

        def step(self, action, data=None, reasoning=None):
            self.calls += 1
            return self.observation_space

    monkeypatch.setenv("ACCA_QUIET", "1")
    env = FakeEnv()
    agent = KaggleACCAAgent(arc_env=env)
    agent.MAX_ACTIONS = 2

    agent.main()

    assert env.calls == 2
    assert agent.action_counter == 2


def test_kaggle_agent_records_level_reward_event(monkeypatch):
    class RewardEnv:
        def __init__(self):
            self.observation_space = {
                "grid": [[1, 0], [0, 0]],
                "game_id": "g",
                "state": "NOT_FINISHED",
                "action_space": ["ACTION1"],
                "levels_completed": 0,
            }

        def step(self, action, data=None, reasoning=None):
            return {
                "grid": [[0, 1], [0, 0]],
                "game_id": "g",
                "state": "NOT_FINISHED",
                "action_space": ["ACTION1"],
                "levels_completed": 1,
            }

    monkeypatch.setenv("ACCA_QUIET", "1")
    agent = KaggleACCAAgent(arc_env=RewardEnv())
    agent.MAX_ACTIONS = 1

    agent.main()

    assert agent._level_reward_events == [(1, 1)]
    assert agent.agent is not None
    assert agent.agent.memory.has_programs("g")


def test_game_id_of_environment_object():
    env = type("Env", (), {"game_id": "sk48-d8078629"})()

    assert _game_id_of(env) == "sk48-d8078629"
