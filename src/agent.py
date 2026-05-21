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


def _base_action(action: ActionEnum | str) -> str:
    return str(action.value if isinstance(action, ActionEnum) else action).split()[0]


def _canonical_action(action: ActionEnum | str) -> ActionEnum | str:
    if isinstance(action, ActionEnum):
        return action
    text = str(action).split()[0]
    if "." in text:
        text = text.rsplit(".", 1)[-1]
    if text in ActionEnum._value2member_map_:
        return text
    if text.isdigit():
        number = int(text)
        if 1 <= number <= 7:
            return f"ACTION{number}"
        if number == 0:
            return "RESET"
    return text


def _next_simple_action(action: str, action_space: list[ActionEnum | str]) -> str | None:
    simple = [
        candidate
        for candidate in ("ACTION1", "ACTION2", "ACTION3", "ACTION4")
        if any(_base_action(a) == candidate for a in action_space)
    ]
    if action not in simple:
        return simple[0] if simple else None
    idx = simple.index(action) + 1
    return simple[idx] if idx < len(simple) else None


def _dedupe_programs(programs: list[list[str]]) -> list[list[str]]:
    seen: set[tuple[str, ...]] = set()
    out: list[list[str]] = []
    for program in programs:
        key = tuple(program)
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(program)
    return out


class MechanicMemory:
    """Per-game memory for successful action programs and MAP hypotheses."""

    def __init__(self):
        self._programs_by_game: dict[str, list[list[str]]] = {}

    def warm_start(self, bank: HypothesisBank, game_id: str) -> None:
        return None

    def store(self, game_id: str, hypothesis, confidence: float) -> None:
        return None

    def store_program(self, game_id: str, actions: list[str]) -> None:
        program = [_base_action(action) for action in actions if _base_action(action) != "RESET"]
        if not program:
            return
        programs = self._programs_by_game.setdefault(game_id, [])
        if program not in programs:
            programs.append(program)

    def candidate_programs(
        self,
        game_id: str,
        action_space: list[ActionEnum | str],
    ) -> list[list[str]]:
        programs = self._programs_by_game.get(game_id, [])
        if not programs:
            return []

        latest = programs[-1]
        collapsed = list(latest)
        while len(collapsed) >= 2 and collapsed[-1] == collapsed[-2]:
            collapsed.pop()

        candidates: list[list[str]] = []

        next_after_collapsed = _next_simple_action(collapsed[-1], action_space) if collapsed else None
        if next_after_collapsed is not None:
            candidates.append(collapsed + [next_after_collapsed])
            candidates.append(collapsed + [next_after_collapsed, next_after_collapsed])

        if len(collapsed) < len(latest) and any(_base_action(a) == "ACTION7" for a in action_space):
            next_after_first = _next_simple_action(latest[0], action_space)
            if next_after_first is not None:
                candidates.append([latest[0], latest[0], "ACTION7", next_after_first])

        candidates.append(list(latest))
        candidates.append(list(latest) + [latest[-1]])

        next_after_latest = _next_simple_action(latest[-1], action_space)
        if next_after_latest is not None:
            candidates.append(list(latest) + [next_after_latest])

        for prior in reversed(programs[:-1]):
            candidates.append(list(prior))

        return _dedupe_programs(candidates)


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
        self.level_actions: list[str] = []
        self.current_attempt: list[str] = []
        self.program_candidates: list[list[str]] = []
        self.program_index = 0
        self.program_pos = 0

    def reset(self, initial: np.ndarray | Mapping[str, Any]) -> None:
        if isinstance(initial, Mapping) and "game_id" in initial:
            self.game_id = str(initial["game_id"])
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
        self.level_actions = []
        self.current_attempt = []
        self.program_candidates = self.memory.candidate_programs(self.game_id, self.action_space)
        self.program_index = 0
        self.program_pos = 0

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
        action_text = str(action.value if isinstance(action, ActionEnum) else action)
        self.level_actions.append(action_text)
        if _base_action(action_text) == "RESET":
            self.current_attempt = []
        else:
            self.current_attempt.append(action_text)
        self.prev_tracked = curr_tracked
        self.last_action = action
        self.actions_taken += 1
        return action

    def on_level_complete(self, terminal_frame: np.ndarray, success: bool) -> None:
        terminal_graph = self.parser.parse(_as_grid(terminal_frame), self.actions_taken + 1)
        terminal = self.tracker.update(terminal_graph).tracked
        if success:
            self.goal_inference.update_on_success(terminal)
            self.memory.store_program(self.game_id, self.current_attempt)
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

        program_action = self._program_action()
        if program_action is not None:
            self.planner.last_predicted_state = None
            return program_action

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

    def _program_action(self) -> str | None:
        if not self._looks_like_toggle_program_world() or not self.program_candidates:
            return None

        while self.program_index < len(self.program_candidates):
            candidate = self.program_candidates[self.program_index]
            if self.program_pos < len(candidate):
                action = candidate[self.program_pos]
                self.program_pos += 1
                return action

            next_index = self.program_index + 1
            if next_index >= len(self.program_candidates):
                return None
            next_candidate = self.program_candidates[next_index]
            if self._is_prefix(self.current_attempt, next_candidate):
                self.program_index = next_index
                self.program_pos = len(self.current_attempt)
                continue
            self.program_index = next_index
            self.program_pos = 0
            return "RESET"
        return None

    def _looks_like_toggle_program_world(self) -> bool:
        if self.current_frame is None:
            return False
        colors = {int(c) for c in np.unique(self.current_frame) if int(c) != 0}
        if not colors or not colors.issubset({3, 4, 5}):
            return False
        if int(np.count_nonzero(self.current_frame)) < 16:
            return False
        return not ({2, 8, 9, 14, 15} & colors)

    def _is_prefix(self, attempt: list[str], candidate: list[str]) -> bool:
        bases = [_base_action(action) for action in attempt]
        return len(bases) <= len(candidate) and bases == candidate[: len(bases)]

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
            return [_canonical_action(action) for action in initial["action_space"]]
        return list(config.ACTION_SPACE)
