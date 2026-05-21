"""
test_synthetic_games.py — Sanity tests for the v2 synthetic-game suite.

Part of ACCA (Active Causal Compression Agent)
ARC Prize 2026 · Paper Track
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from envs.synthetic.env_schema import Level, SyntheticGame
from envs.synthetic.generate_envs import ALL_BUILDERS
from envs.synthetic.verifier import Episode, simulate, verify_game, verify_level


@pytest.fixture(scope="module")
def games():
    return [b() for b in ALL_BUILDERS]


def test_five_games_six_levels_each(games):
    assert len(games) == 5
    for g in games:
        assert g.total_levels == 6, f"{g.game_id} has {g.total_levels} levels (need 6)"


def test_all_levels_verify(games):
    for g in games:
        assert verify_game(g), f"verify_game failed for {g.game_id}"


def test_initial_and_goal_differ(games):
    for g in games:
        for l in g.levels:
            assert not np.array_equal(l.initial_state, l.goal_state), \
                f"{g.game_id} L{l.level_index}: initial == goal (level is a no-op)"


def test_solution_length_matches_count(games):
    for g in games:
        for l in g.levels:
            assert len(l.human_solution) == l.human_action_count


def test_tutorial_level_uses_subset_of_full_mechanics(games):
    """L1 must use at most the smallest mechanic set — composition grows over levels."""
    for g in games:
        l1 = g.levels[0]
        assert set(l1.mechanics_active).issubset(set(g.available_mechanics))


def test_episode_replay_matches_simulate(games):
    """Episode runner and the simulate() helper agree."""
    for g in games:
        for l in g.levels:
            ep = Episode(l)
            for a in l.human_solution:
                ep.step(a)
            assert np.array_equal(ep.state, simulate(l))


def test_undo_reverts_one_step():
    """ACTION7 when 'undo' is in mechanics_active reverts the last action."""
    # Use the sy02 L6 setup directly
    from envs.synthetic.generate_envs import build_sy02
    g = build_sy02()
    l6 = g.levels[5]
    assert "undo" in l6.mechanics_active

    ep = Episode(l6)
    ep.step("ACTION1")
    state_after_one = ep.state.copy()
    ep.step("ACTION1")
    ep.step("ACTION7")
    assert np.array_equal(ep.state, state_after_one), "undo did not restore state"


def test_persisted_games_match_in_memory(tmp_path: Path, games):
    for g in games:
        p = tmp_path / f"{g.game_id}.json"
        g.save(p)
        loaded = SyntheticGame.load(p)
        assert loaded.game_id == g.game_id
        assert loaded.total_levels == g.total_levels
        for orig, l2 in zip(g.levels, loaded.levels):
            assert np.array_equal(orig.initial_state, l2.initial_state)
            assert np.array_equal(orig.goal_state, l2.goal_state)
            assert orig.human_solution == l2.human_solution


def test_grids_fit_max_64x64(games):
    """ARC-AGI-3 caps grids at 64×64."""
    for g in games:
        for l in g.levels:
            h, w = l.initial_state.shape
            assert h <= 64 and w <= 64, f"{g.game_id} L{l.level_index}: grid {h}x{w} exceeds max 64x64"


def test_color_palette_within_0_to_15(games):
    for g in games:
        for l in g.levels:
            assert int(l.initial_state.max()) <= 15
            assert int(l.initial_state.min()) >= 0
