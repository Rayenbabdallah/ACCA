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
import importlib
import types
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


def _official_agents_root() -> Path | None:
    candidates = [
        Path("/kaggle/input/competitions/arc-prize-2026-arc-agi-3/ARC-AGI-3-Agents"),
    ]
    candidates.extend(Path(p) for p in glob.glob("/kaggle/input/**/ARC-AGI-3-Agents", recursive=True))
    candidates.extend(Path(p).parent for p in glob.glob("/kaggle/input/**/agents", recursive=True))
    for root in candidates:
        if (root / "agents" / "swarm.py").exists():
            return root
    return None


def _ensure_lightweight_agents_package(root: Path) -> None:
    """Register `agents` as a package without executing its heavy __init__.py."""
    pkg = sys.modules.get("agents")
    if pkg is None or not hasattr(pkg, "__path__"):
        pkg = types.ModuleType("agents")
        pkg.__path__ = [str(root / "agents")]
        sys.modules["agents"] = pkg
    elif str(root / "agents") not in pkg.__path__:
        pkg.__path__.insert(0, str(root / "agents"))


def _import_official_agent_base():
    root = _official_agents_root()
    if root is not None:
        _ensure_lightweight_agents_package(root)
    return importlib.import_module("agents.agent").Agent


def _import_official_swarm():
    root = _official_agents_root()
    if root is not None:
        _ensure_lightweight_agents_package(root)
    return importlib.import_module("agents.swarm").Swarm


def _import_arcade_api():
    errors: list[str] = []
    for module_name in ("arc_agi_3", "arc_agi", "arc_agi.arcade"):
        try:
            module = importlib.import_module(module_name)
            return module.Arcade, module.OperationMode
        except Exception as exc:  # pragma: no cover - depends on Kaggle SDK.
            errors.append(f"{module_name}: {exc}")
    raise ModuleNotFoundError("Could not import Arcade/OperationMode from ARC SDK: " + "; ".join(errors))


def _import_game_action_api():
    module = importlib.import_module("arcengine")
    return module.GameAction, module.GameState


def _competition_mode_available() -> bool:
    import os

    return bool(os.environ.get("ARC_API_KEY") or os.environ.get("ARC_AGI_API_KEY"))


def _public_environment_dir() -> str | None:
    for path in glob.glob("/kaggle/input/**/environment_files", recursive=True):
        if Path(path).is_dir():
            return path
    return None


def _game_id_of(game: Any) -> str:
    value = _get_value(game, ("game_id", "id", "name"))
    if value is not None:
        return str(value)
    return str(game)


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
    _OfficialAgent = _import_official_agent_base()
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


def _to_game_action(action: str):
    try:
        GameAction, _ = _import_game_action_api()
    except Exception:  # Local tests run without arcengine.
        return action
    parts = action.split()
    base = parts[0]
    game_action = getattr(GameAction, base)
    if game_action.is_complex():
        x = int(parts[2]) if len(parts) >= 3 else 0
        y = int(parts[1]) if len(parts) >= 2 else 0
        game_action.set_data({"x": x, "y": y})
        game_action.reasoning = {
            "desired_action": base,
            "my_reason": "ACCA selected coordinate action",
        }
    elif game_action.is_simple():
        game_action.reasoning = f"ACCA selected {base}"
    return game_action


class KaggleACCAAgent(_OfficialAgent):
    """Official-API wrapper around the SDK-independent ACCAAgent."""

    MAX_ACTIONS = 80

    def __init__(self, *args: Any, memory: MechanicMemory | None = None, **kwargs: Any):
        if _OfficialAgent is not object:
            super().__init__(*args, **kwargs)
        else:
            for name, value in kwargs.items():
                setattr(self, name, value)
            self.frames = []
            self.action_counter = 0
        self.memory = memory or MechanicMemory()
        self.agent: ACCAAgent | None = None
        self.game_id: str | None = None
        self.last_action: str | None = None

    def main(self) -> None:
        """Run one ARC-AGI-3 environment without depending on official Swarm."""
        if not hasattr(self, "arc_env") or self.arc_env is None:
            raise RuntimeError("KaggleACCAAgent requires an arc_env before main().")

        latest_frame = self._latest_frame_from_env()
        if not getattr(self, "frames", None):
            self.frames = [latest_frame]

        self.action_counter = int(getattr(self, "action_counter", 0))
        while not self.is_done(self.frames, latest_frame) and self.action_counter <= self.MAX_ACTIONS:
            action = self.choose_action(self.frames, latest_frame)
            frame = self._take_arc_action(action)
            if frame is not None:
                latest_frame = frame
                if hasattr(self, "append_frame"):
                    self.append_frame(frame)
                else:
                    self.frames.append(frame)
            self.action_counter += 1

        if hasattr(self, "cleanup"):
            self.cleanup()

    def _latest_frame_from_env(self) -> Any:
        raw = self.arc_env.observation_space
        if hasattr(self, "_convert_raw_frame_data"):
            return self._convert_raw_frame_data(raw)
        return raw

    def _take_arc_action(self, action: Any) -> Any:
        if hasattr(self, "take_action"):
            return self.take_action(action)
        data = action.action_data.model_dump() if hasattr(action, "action_data") else {}
        raw = self.arc_env.step(
            action,
            data=data,
            reasoning=data.get("reasoning", {}) if isinstance(data, dict) else {},
        )
        if hasattr(self, "_convert_raw_frame_data"):
            return self._convert_raw_frame_data(raw)
        return raw

    def is_done(self, frames: list[Any], latest_frame: Any) -> bool:
        try:
            _, GameState = _import_game_action_api()
            return latest_frame.state is GameState.WIN
        except Exception:
            status = _extract_status(latest_frame).upper()
            return status in {"WIN", "DONE", "FINISHED"}

    def choose_action(self, frames: list[Any], latest_frame: Any):
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
            return _to_game_action("RESET")

        action = self.agent.act(grid)
        self.last_action = str(action)
        return _to_game_action(self.last_action)


def register_acca_agent() -> None:
    root = _official_agents_root()
    if root is not None:
        _ensure_lightweight_agents_package(root)
    pkg = sys.modules.get("agents")
    if pkg is None:
        pkg = types.ModuleType("agents")
        sys.modules["agents"] = pkg
    available = getattr(pkg, "AVAILABLE_AGENTS", {})
    available["acca"] = KaggleACCAAgent
    setattr(pkg, "AVAILABLE_AGENTS", available)


def run_competition() -> None:
    """Run ACCA through the official ARC-AGI-3 Arcade API."""
    try:
        _add_official_agent_paths()
        Arcade, OperationMode = _import_arcade_api()
    except Exception as exc:  # pragma: no cover - requires Kaggle SDK.
        raise RuntimeError(
            "ARC-AGI-3 SDK is unavailable. Install from Kaggle's "
            "arc_agi_3_wheels/ directory before calling run_competition(). "
            + _agent_path_diagnostics()
        ) from exc

    if _competition_mode_available():
        arcade = Arcade(operation_mode=OperationMode.COMPETITION)
    else:
        env_dir = _public_environment_dir()
        if env_dir is None:
            raise RuntimeError(
                "No ARC API key is available for competition mode, and no "
                "environment_files/ directory was found for offline smoke mode."
            )
        arcade = Arcade(operation_mode=OperationMode.OFFLINE, environments_dir=env_dir)
    games = [_game_id_of(game) for game in arcade.get_environments()]
    register_acca_agent()
    tags = ["acca", "competition" if _competition_mode_available() else "offline-smoke"]
    card_id = arcade.open_scorecard(tags=tags)
    try:
        for game_id in games:
            env = arcade.make(game_id, scorecard_id=card_id)
            agent = KaggleACCAAgent(
                card_id=card_id,
                game_id=game_id,
                agent_name="acca",
                ROOT_URL="http://localhost:8001",
                record=True,
                arc_env=env,
                tags=tags,
            )
            agent.main()
    finally:
        scorecard = arcade.close_scorecard(card_id)
        print(scorecard)
