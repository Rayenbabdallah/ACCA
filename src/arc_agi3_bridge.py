"""
arc_agi3_bridge.py - Kaggle ARC-AGI-3 adapter for ACCAAgent.

This module keeps the competition notebook thin. It installs no packages and
makes no network calls; the notebook is responsible for installing the official
SDK from Kaggle's offline `arc_agi_3_wheels/` input directory.

Part of ACCA (Active Causal Compression Agent)
ARC Prize 2026 . Paper Track
"""
from __future__ import annotations

import glob
import sys
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from src.agent import ACCAAgent, MechanicMemory


def _add_official_agent_paths() -> None:
    """Expose Kaggle's attached ARC-AGI-3-Agents repo to Python imports."""
    patterns = [
        "/kaggle/input/competitions/arc-prize-2026-arc-agi-3/ARC-AGI-3-Agents",
        "/kaggle/input/**/ARC-AGI-3-Agents",
        "/kaggle/input/**/ARC-AGI-3-Agents/*",
        "/kaggle/input/**/ARC-AGI-3-Agents/**/*",
        "/kaggle/input/**/agents",
        "/kaggle/working/**/ARC-AGI-3-Agents",
        "/kaggle/working/**/ARC-AGI-3-Agents/*",
        "/kaggle/working/**/ARC-AGI-3-Agents/**/*",
        "/kaggle/working/**/agents",
    ]
    for pattern in patterns:
        for path in sorted(glob.glob(pattern, recursive=True)):
            root = Path(path)
            if root.name == "agents":
                root = root.parent
            if root.is_dir() and (root / "agents").exists() and str(root) not in sys.path:
                sys.path.insert(0, str(root))
            if root.is_file() and root.name == "__init__.py" and root.parent.name == "agents":
                parent = root.parent.parent
                if str(parent) not in sys.path:
                    sys.path.insert(0, str(parent))


def _agent_path_diagnostics() -> str:
    candidates: list[str] = []
    for pattern in (
        "/kaggle/input/**/agents",
        "/kaggle/input/**/agents/__init__.py",
        "/kaggle/input/**/ARC-AGI-3-Agents",
        "/kaggle/input/**/Swarm.py",
        "/kaggle/input/**/swarm.py",
    ):
        candidates.extend(glob.glob(pattern, recursive=True))
    visible = [p for p in sys.path if "agent" in p.lower() or "arc-agi" in p.lower()]
    return f"agent candidates={candidates[:30]} sys.path={visible[:30]}"


try:  # Local development does not have the Kaggle SDK installed.
    _add_official_agent_paths()
    from agents.agent import Agent as _OfficialAgent
except Exception:  # pragma: no cover - exercised only when SDK is absent.
    _OfficialAgent = object


def _get_value(obj: Any, names: tuple[str, ...]) -> Any:
    for name in names:
        if isinstance(obj, Mapping) and name in obj:
            return obj[name]
        if hasattr(obj, name):
            return getattr(obj, name)
    return None


def _extract_grid(frame: Any) -> np.ndarray:
    grid = _get_value(frame, ("grid", "state", "frame", "observation", "board"))
    if grid is None:
        data = _get_value(frame, ("data", "payload"))
        if data is not None:
            grid = _get_value(data, ("grid", "state", "frame", "observation", "board"))
    if grid is None:
        raise ValueError("could not extract grid from ARC-AGI-3 frame")
    arr = np.asarray(grid, dtype=np.uint8)
    if arr.ndim != 2:
        raise ValueError(f"ARC-AGI-3 grid must be 2D, got shape {arr.shape}")
    return arr


def _extract_action_space(frame: Any) -> list[str]:
    action_space = _get_value(frame, ("action_space", "available_actions", "actions"))
    if action_space is None:
        data = _get_value(frame, ("data", "payload", "game"))
        if data is not None:
            action_space = _get_value(data, ("action_space", "available_actions", "actions"))
    if action_space is None:
        return [
            "RESET",
            "ACTION1",
            "ACTION2",
            "ACTION3",
            "ACTION4",
            "ACTION5",
            "ACTION6",
            "ACTION7",
        ]
    return [str(action) for action in action_space]


def _extract_game_id(frame: Any) -> str:
    value = _get_value(frame, ("game_id", "game", "environment_id", "env_id", "id"))
    if value is None:
        data = _get_value(frame, ("data", "payload"))
        if data is not None:
            value = _get_value(data, ("game_id", "game", "environment_id", "env_id", "id"))
    return "kaggle" if value is None else str(value)


def _extract_status(frame: Any) -> str:
    value = _get_value(frame, ("state", "status", "game_state"))
    if value is None:
        return ""
    if hasattr(value, "name"):
        return str(value.name)
    return str(value)


class KaggleACCAAgent(_OfficialAgent):
    """Official-API wrapper around the SDK-independent ACCAAgent."""

    def __init__(self, *args: Any, memory: MechanicMemory | None = None, **kwargs: Any):
        if _OfficialAgent is not object:
            super().__init__(*args, **kwargs)
        self.memory = memory or MechanicMemory()
        self.agent: ACCAAgent | None = None
        self.game_id: str | None = None
        self.last_action: str | None = None

    def is_done(self, frames: list[Any], latest_frame: Any) -> bool:
        status = _extract_status(latest_frame).upper()
        return status in {"WIN", "GAME_OVER", "DONE", "FINISHED"}

    def choose_action(self, frames: list[Any], latest_frame: Any) -> str:
        grid = _extract_grid(latest_frame)
        game_id = _extract_game_id(latest_frame)
        if self.agent is None or self.game_id != game_id:
            self.game_id = game_id
            self.agent = ACCAAgent(game_id=game_id, memory=self.memory)
            self.agent.reset(
                {
                    "game_id": game_id,
                    "initial_grid": grid,
                    "action_space": _extract_action_space(latest_frame),
                }
            )
            self.last_action = "RESET"
            return "RESET"

        action = self.agent.act(grid)
        self.last_action = str(action)
        return self.last_action


def run_competition() -> None:
    """Run ACCA through the official ARC-AGI-3 Swarm in competition mode."""
    try:
        _add_official_agent_paths()
        try:
            from agents import Swarm
        except ImportError:
            from agents.swarm import Swarm
        from arc_agi_3 import Arcade, OperationMode
    except Exception as exc:  # pragma: no cover - requires Kaggle SDK.
        raise RuntimeError(
            "ARC-AGI-3 SDK is unavailable. Install from Kaggle's "
            "arc_agi_3_wheels/ directory before calling run_competition(). "
            + _agent_path_diagnostics()
        ) from exc

    arcade = Arcade(operation_mode=OperationMode.COMPETITION)
    swarm = Swarm(agent_class=KaggleACCAAgent, arcade=arcade)
    swarm.run()
