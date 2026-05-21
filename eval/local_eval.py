"""
local_eval.py - Local evaluation harness for ACCA agents.

Runs agents against synthetic ARC-AGI-3-style games when given v2
`SyntheticGame` JSON files, falling back to the legacy stub format for scoring
unit tests. Counts environment actions only; internal compute remains free.

Part of ACCA (Active Causal Compression Agent)
ARC Prize 2026 . Paper Track
"""
from __future__ import annotations

import argparse
import importlib
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Protocol

import numpy as np

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from envs.synthetic.env_schema import Level, SyntheticGame
from envs.synthetic.verifier import Episode
from eval.scorecard import LevelRecord, Scorecard, compute_scorecard
from src.perception.event_extractor import ActionEnum


class Agent(Protocol):
    def reset(self, env_config: Mapping[str, Any]) -> None: ...

    def act(self, observation: np.ndarray) -> str: ...


def _action_to_string(action: Any) -> str:
    if isinstance(action, ActionEnum):
        return action.value
    return str(action)


class DummyAgent:
    """Baseline that always sends RESET; never completes anything."""

    def reset(self, env_config: Mapping[str, Any]) -> None:
        return None

    def act(self, observation: np.ndarray) -> str:
        return "RESET"


class HumanSolutionAgent:
    """Oracle baseline for local verifier sanity checks only."""

    def __init__(self):
        self._plan: list[str] = []

    def reset(self, env_config: Mapping[str, Any]) -> None:
        self._plan = list(env_config.get("human_solution", []))

    def act(self, observation: np.ndarray) -> str:
        if not self._plan:
            return "RESET"
        return self._plan.pop(0)


class _StubEnv:
    """Minimal env stub for legacy flat JSON tests."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        grid = config.get("initial_grid")
        # Grid dimensions are variable; never assume 64x64.
        self.observation = (
            np.asarray(grid, dtype=np.uint8)
            if grid is not None
            else np.zeros((1, 1), dtype=np.uint8)
        )
        self.completed = False
        self.action_count = 0

    def step(self, action: str) -> tuple[np.ndarray, bool]:
        self.action_count += 1
        return self.observation, self.completed


def _load_game_configs(env_dir: str) -> List[Dict[str, Any]]:
    configs: List[Dict[str, Any]] = []
    for path in sorted(Path(env_dir).glob("*.json")):
        with open(path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        cfg.setdefault("game_id", cfg.get("env_id", path.stem))
        cfg.setdefault("human_actions", 1)
        cfg.setdefault("total_levels", 1)
        configs.append(cfg)
    return configs


def _load_synthetic_games(env_dir: str) -> list[SyntheticGame]:
    games: list[SyntheticGame] = []
    for path in sorted(Path(env_dir).glob("game_*.json")):
        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        if "levels" in raw and "action_space" in raw:
            games.append(SyntheticGame.from_dict(raw))
    return games


def _run_stub_episode(agent: Agent, env: _StubEnv, max_actions: int) -> tuple[int, bool]:
    agent.reset(env.config)
    obs = env.observation
    for _ in range(max_actions):
        action = _action_to_string(agent.act(obs))
        obs, done = env.step(action)
        if done:
            return env.action_count, True
    return env.action_count, env.completed


def _notify_level_complete(agent: Agent, terminal_frame: np.ndarray, success: bool) -> None:
    hook = getattr(agent, "on_level_complete", None)
    if callable(hook):
        hook(terminal_frame, success)


def _run_synthetic_level(
    agent: Agent,
    game: SyntheticGame,
    level: Level,
    max_actions: int,
) -> tuple[int, bool]:
    env = Episode(level)
    reset_payload = {
        "game_id": game.game_id,
        "level_index": level.level_index,
        "total_levels": game.total_levels,
        "action_space": list(game.action_space),
        "initial_grid": level.initial_state,
        "human_solution": list(level.human_solution),
    }
    agent.reset(reset_payload)
    obs = env.state.copy()
    for action_count in range(1, max_actions + 1):
        action = _action_to_string(agent.act(obs.copy()))
        obs = env.step(action)
        if np.array_equal(obs, level.goal_state):
            _notify_level_complete(agent, obs.copy(), True)
            return action_count, True
    _notify_level_complete(agent, obs.copy(), False)
    return max_actions, False


def _run_synthetic_games(
    games: list[SyntheticGame],
    agent_class: type,
    max_actions_per_level: int,
) -> Scorecard:
    records: List[LevelRecord] = []

    for game in games:
        agent = agent_class()
        for level in sorted(game.levels, key=lambda l: l.level_index):
            agent_actions, completed = _run_synthetic_level(
                agent,
                game,
                level,
                max_actions_per_level,
            )
            records.append(
                {
                    "game_id": game.game_id,
                    "level_index": level.level_index,
                    "total_levels": game.total_levels,
                    "human_actions": level.human_action_count,
                    "agent_actions": agent_actions,
                    "completed": completed,
                }
            )

    return compute_scorecard(records)


def _run_stub_configs(
    configs: list[Dict[str, Any]],
    agent_class: type,
    max_actions_per_level: int,
) -> Scorecard:
    records: List[LevelRecord] = []

    for cfg in configs:
        env = _StubEnv(cfg)
        agent = agent_class()
        agent_actions, completed = _run_stub_episode(agent, env, max_actions_per_level)
        records.append(
            {
                "game_id": str(cfg["game_id"]),
                "level_index": int(cfg.get("level_index", 1)),
                "total_levels": int(cfg["total_levels"]),
                "human_actions": int(cfg["human_actions"]),
                "agent_actions": agent_actions,
                "completed": completed,
            }
        )

    return compute_scorecard(records)


def run_evaluation(
    env_dir: str,
    agent_class: type,
    max_actions_per_level: int = 500,
) -> Scorecard:
    """Run agent_class against env_dir and build a Kaggle-formula scorecard."""
    synthetic_games = _load_synthetic_games(env_dir)
    if synthetic_games:
        return _run_synthetic_games(synthetic_games, agent_class, max_actions_per_level)
    return _run_stub_configs(_load_game_configs(env_dir), agent_class, max_actions_per_level)


def _resolve_agent(dotted: str) -> type:
    module_path, _, class_name = dotted.rpartition(".")
    module = importlib.import_module(module_path)
    return getattr(module, class_name)


def main() -> None:
    parser = argparse.ArgumentParser(description="ACCA local evaluation harness")
    parser.add_argument("--env_dir", required=True)
    parser.add_argument("--agent", required=True, help="Dotted path, e.g. src.agent.ACCAAgent")
    parser.add_argument("--output", required=True)
    parser.add_argument("--max_actions", type=int, default=500)
    args = parser.parse_args()

    agent_class = _resolve_agent(args.agent)
    scorecard = run_evaluation(args.env_dir, agent_class, args.max_actions)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(scorecard, f, indent=2)
    print(f"total_score={scorecard['total_score']:.4f}  completion={scorecard['completion_rate']:.2%}")


if __name__ == "__main__":
    main()
