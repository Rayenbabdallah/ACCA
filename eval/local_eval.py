"""
local_eval.py — Local evaluation harness for ACCA agents.

Loads ARC-AGI-3-format environment configs from a directory, runs an agent
against each, counts environment actions (the scarce resource — internal
compute is free), and emits a scorecard JSON.

Part of ACCA (Active Causal Compression Agent)
ARC Prize 2026 · Paper Track
"""
from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path
from typing import Any, Dict, List, Protocol

import numpy as np

from eval.scorecard import LevelResult, Scorecard, compute_scorecard


class Agent(Protocol):
    def reset(self, env_config: Dict[str, Any]) -> None: ...
    def act(self, observation: np.ndarray) -> str: ...


class DummyAgent:
    """Baseline that always sends RESET — should score RHAE = 0.0."""

    def reset(self, env_config: Dict[str, Any]) -> None:
        return None

    def act(self, observation: np.ndarray) -> str:
        return "RESET"


class _StubEnv:
    """Minimal env stub that consumes actions and never reports completion.

    Used when env JSON configs don't ship with an executable simulator.
    Real ARC-AGI-3 evaluation plugs in the official env runner here.
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        grid = config.get("initial_grid")
        self.observation = (
            np.asarray(grid, dtype=np.uint8) if grid is not None else np.zeros((64, 64), dtype=np.uint8)
        )
        self.completed = False
        self.action_count = 0

    def step(self, action: str) -> tuple[np.ndarray, bool]:
        self.action_count += 1
        # A real env would mutate self.observation and set self.completed.
        return self.observation, self.completed


def _load_env_configs(env_dir: str) -> List[Dict[str, Any]]:
    configs: List[Dict[str, Any]] = []
    for path in sorted(Path(env_dir).glob("*.json")):
        with open(path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        cfg.setdefault("level_id", path.stem)
        cfg.setdefault("human_actions", 1)
        configs.append(cfg)
    return configs


def _run_episode(agent: Agent, env: _StubEnv, max_actions: int) -> tuple[int, bool]:
    agent.reset(env.config)
    obs = env.observation
    for _ in range(max_actions):
        action = agent.act(obs)
        obs, done = env.step(action)
        if done:
            return env.action_count, True
    return env.action_count, env.completed


def run_evaluation(
    env_dir: str,
    agent_class: type,
    max_actions_per_level: int = 500,
) -> Scorecard:
    """Run agent_class against every env config in env_dir; return a scorecard."""
    configs = _load_env_configs(env_dir)
    results: List[LevelResult] = []

    for cfg in configs:
        env = _StubEnv(cfg)
        agent = agent_class()
        ai_actions, completed = _run_episode(agent, env, max_actions_per_level)
        results.append(
            {
                "level_id": cfg["level_id"],
                "human_actions": int(cfg["human_actions"]),
                "ai_actions": ai_actions,
                "completed": completed,
            }
        )

    return compute_scorecard(results)


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
    print(f"mean_rhae={scorecard['mean_rhae']:.4f}  completion={scorecard['completion_rate']:.2%}")


if __name__ == "__main__":
    main()
