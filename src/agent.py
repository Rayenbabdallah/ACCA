"""
agent.py - Top-level ACCA exploration/exploitation loop.

This is the local, SDK-independent agent facade. The Kaggle notebook will wrap
the same internals behind ARC-AGI-3's `is_done` / `choose_action` interface.

Part of ACCA (Active Causal Compression Agent)
ARC Prize 2026 . Paper Track
"""
from __future__ import annotations

from collections import deque
from typing import Any, Mapping

import numpy as np

from src import config
from src.hypothesis.goal_inference import GoalInference, GoalTemplate
from src.hypothesis.hypothesis_bank import HypothesisBank
from src.hypothesis.hypothesis_seeds import load_seeds
from src.perception.event_extractor import ActionEnum, EventExtractor
from src.perception.frame_parser import FrameParser, ObjectGraph
from src.perception.object_tracker import ObjectTracker
from src.perception.replay_buffer import ReplayBuffer
from src.planning.eig_selector import EIGSelector
from src.planning.planner import Planner, state_fingerprint


def _as_grid(value: Any) -> np.ndarray:
    arr = np.asarray(value, dtype=np.uint8)
    if arr.ndim != 2:
        raise ValueError(f"frame grid must be 2D, got shape {arr.shape}")
    return arr


class MechanicMemory:
    """Temporary no-op memory facade until P4 implements persistent memory."""

    def warm_start(self, bank: HypothesisBank, game_id: str) -> None:
        return None

    def store(self, game_id: str, hypothesis, confidence: float) -> None:
        return None


class ACCAAgent:
    """Integrates perception, hypothesis posterior, EIG, and planning."""

    def __init__(self, game_id: str = "local", memory: MechanicMemory | None = None):
        self.game_id = game_id
        self.memory = memory or MechanicMemory()
        self.bank = HypothesisBank(seed_hypotheses=load_seeds())
        self.memory.warm_start(self.bank, game_id)
        self.goal_inference = GoalInference()
        self.eig_selector = EIGSelector()
        self.planner = Planner()
        self.parser = FrameParser()
        self.tracker = ObjectTracker()
        self.extractor = EventExtractor()
        self.replay_buffer = ReplayBuffer()
        self.actions_taken = 0
        self.prev_tracked: ObjectGraph | None = None
        self.current_frame: np.ndarray | None = None
        self.last_action: ActionEnum | str | None = None
        self.action_space: list[ActionEnum | str] = list(config.ACTION_SPACE)

    def reset(self, initial: np.ndarray | Mapping[str, Any]) -> None:
        frame = self._frame_from_reset_input(initial)
        self.current_frame = frame.copy()
        self.action_space = self._action_space_from_reset_input(initial)
        self.parser = FrameParser()
        self.tracker = ObjectTracker()
        initial_graph = self.parser.parse(frame, frame_index=0)
        initial_result = self.tracker.update(initial_graph)
        self.prev_tracked = initial_result.tracked
        self.bank.clear_observations()
        self.planner.failure_count = 0
        self.planner.current_plan = []
        self.planner.last_predicted_state = None
        self.replay_buffer.clear()
        self.actions_taken = 0
        self.last_action = None

    def act(self, observation: np.ndarray) -> ActionEnum | str:
        return self.step(observation)

    def step(self, frame: np.ndarray) -> ActionEnum | str:
        grid = _as_grid(frame)
        self.current_frame = grid.copy()
        curr_graph = self.parser.parse(grid, frame_index=self.actions_taken + 1)
        tracking = self.tracker.update(curr_graph)
        curr_tracked = tracking.tracked

        if self.prev_tracked is not None and self.last_action is not None:
            obs = self.extractor.extract(
                self.last_action,
                self.prev_tracked,
                tracking,
                timestamp=self.actions_taken,
            )
            self.replay_buffer.add(obs)
            self.bank.update(obs)
            self.goal_inference.update_on_step(curr_tracked)

            if self.planner.last_predicted_state is not None and (
                state_fingerprint(curr_tracked)
                != state_fingerprint(self.planner.last_predicted_state)
            ):
                self.planner.plan_failed()
                if self.planner.failure_count >= config.RECOVERY_TRIGGER:
                    self.bank.clear_observations()

        action = self._select_action(curr_tracked)
        self.prev_tracked = curr_tracked
        self.last_action = action
        self.actions_taken += 1
        return action

    def on_level_complete(self, terminal_frame: np.ndarray, success: bool) -> None:
        terminal_graph = self.parser.parse(_as_grid(terminal_frame), self.actions_taken + 1)
        terminal = self.tracker.update(terminal_graph).tracked
        if success:
            self.goal_inference.update_on_success(terminal)
            self.memory.store(self.game_id, self.bank.map_hypothesis(), self.bank.map_confidence())
        else:
            self.goal_inference.update_on_failure()

    def _select_action(self, state: ObjectGraph) -> ActionEnum | str:
        coordinate_action = self._coordinate_action(state)
        if coordinate_action is not None:
            self.planner.last_predicted_state = None
            return coordinate_action

        push_action = self._push_toward_target_action()
        if push_action is not None:
            self.planner.last_predicted_state = None
            return push_action

        if (
            self.bank.entropy() > config.ENTROPY_THRESHOLD
            and self.actions_taken < config.MAX_EXPLORATION_ACTIONS
        ):
            self.planner.last_predicted_state = None
            return self.eig_selector.select_action(state, self.bank, self.action_space)

        if not self.planner.has_plan():
            goal = self.goal_inference.top_goal()
            plan = self.planner.compile_plan(
                state,
                self.bank.map_hypothesis(),
                goal,
                self.action_space,
            )
            if plan is None or len(plan) == 0:
                self.planner.last_predicted_state = None
                return self.eig_selector.select_action(state, self.bank, self.action_space)
            self.planner.current_plan = plan

        action = self.planner.next_action()
        if action is None:
            self.planner.last_predicted_state = None
            return self.eig_selector.select_action(state, self.bank, self.action_space)
        self.planner.last_predicted_state = self.bank.map_hypothesis().execute(state, action)
        return action

    def _coordinate_action(self, state: ObjectGraph) -> str | None:
        if not any(str(action).split()[0] == ActionEnum.ACTION6.value for action in self.action_space):
            return None

        targets: list[tuple[int, int]] = []
        for node in state.nodes.values():
            if node.color not in (14, 15):
                continue
            for row, col in sorted(node.pixels):
                targets.append((row, col))

        if not targets:
            return None
        row, col = sorted(set(targets))[0]
        return f"{ActionEnum.ACTION6.value} {row} {col}"

    def _push_toward_target_action(self) -> str | None:
        if self.current_frame is None:
            return None

        grid = self.current_frame
        movable = np.argwhere(grid == 2)
        targets = np.argwhere(grid == 3)
        if len(movable) == 0 or len(targets) == 0:
            return None

        directions = {
            ActionEnum.ACTION1.value: (0, -1),
            ActionEnum.ACTION2.value: (0, 1),
            ActionEnum.ACTION3.value: (-1, 0),
            ActionEnum.ACTION4.value: (1, 0),
        }
        target_set = {(int(t[0]), int(t[1])) for t in targets}
        plan = self._plan_push_to_target(grid, target_set)
        if plan:
            return plan[0]

        candidates: list[tuple[int, str]] = []
        for action, (dy, dx) in directions.items():
            if not any(str(a).split()[0] == action for a in self.action_space):
                continue
            pos = self._simulate_push(grid, dy, dx)
            if pos is None:
                continue
            score = min(abs(pos[0] - int(t[0])) + abs(pos[1] - int(t[1])) for t in targets)
            candidates.append((score, action))

        if not candidates:
            return None
        return min(candidates)[1]

    def _plan_push_to_target(
        self,
        grid: np.ndarray,
        targets: set[tuple[int, int]],
    ) -> list[str] | None:
        start = np.argwhere(grid == 2)
        if len(start) == 0:
            return None
        start_pos = (int(start[0][0]), int(start[0][1]))
        directions = {
            ActionEnum.ACTION1.value: (0, -1),
            ActionEnum.ACTION2.value: (0, 1),
            ActionEnum.ACTION3.value: (-1, 0),
            ActionEnum.ACTION4.value: (1, 0),
        }
        actions = [
            (action, delta)
            for action, delta in directions.items()
            if any(str(a).split()[0] == action for a in self.action_space)
        ]
        queue = deque([(start_pos, [])])
        visited = {start_pos}
        while queue:
            pos, plan = queue.popleft()
            if len(plan) >= 8:
                continue
            for action, (dy, dx) in actions:
                nxt = self._slide_from(grid, pos, dy, dx)
                if nxt is None or nxt in visited:
                    continue
                next_plan = plan + [action]
                if nxt in targets:
                    return next_plan
                visited.add(nxt)
                queue.append((nxt, next_plan))
        return None

    def _simulate_push(self, grid: np.ndarray, dy: int, dx: int) -> tuple[int, int] | None:
        pos = np.argwhere(grid == 2)
        if len(pos) == 0:
            return None
        return self._slide_from(grid, (int(pos[0][0]), int(pos[0][1])), dy, dx)

    def _slide_from(
        self,
        grid: np.ndarray,
        start: tuple[int, int],
        dy: int,
        dx: int,
    ) -> tuple[int, int] | None:
        row, col = start
        h, w = grid.shape
        next_row, next_col = row, col
        while True:
            cand_row = next_row + dy
            cand_col = next_col + dx
            if not (0 <= cand_row < h and 0 <= cand_col < w):
                break
            if int(grid[cand_row, cand_col]) in (8, 9):
                break
            next_row, next_col = cand_row, cand_col
        if (next_row, next_col) == (row, col):
            return None
        return next_row, next_col

    def _frame_from_reset_input(self, initial: np.ndarray | Mapping[str, Any]) -> np.ndarray:
        if isinstance(initial, Mapping):
            for key in ("initial_grid", "initial_state", "grid"):
                if key in initial:
                    return _as_grid(initial[key])
            return np.zeros((1, 1), dtype=np.uint8)
        return _as_grid(initial)

    def _action_space_from_reset_input(
        self,
        initial: np.ndarray | Mapping[str, Any],
    ) -> list[ActionEnum | str]:
        if isinstance(initial, Mapping) and "action_space" in initial:
            return list(initial["action_space"])
        return list(config.ACTION_SPACE)
