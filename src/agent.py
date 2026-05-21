"""
agent.py - Top-level ACCA exploration/exploitation loop.

This is the local, SDK-independent agent facade. The Kaggle notebook will wrap
the same internals behind ARC-AGI-3's `is_done` / `choose_action` interface.

Part of ACCA (Active Causal Compression Agent)
ARC Prize 2026 . Paper Track
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, Tuple

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


@dataclass
class ExplorationStats:
    """Per-game lifetime effectiveness tracking. Persists across levels — the
    whole point is that what we learn on Level 1 informs Level 2.

    Each outcome is one of three buckets:
      - NOVEL change (best): the resulting grid is one we've never seen this game
      - REPEAT change (mediocre): the grid changed but we've seen this state before
      - NO change (worst): action was a no-op

    Score = (1.0 * novels + 0.3 * repeats) / calls + 1/(calls+1). Unseen items
    return 2.0 — forced exploration first. The novel/repeat distinction is what
    stops the agent locking onto a state-changing-but-cycling action like
    "cursor left" pressed 100× while the cursor wraps around forever.
    """
    cell_calls: Dict[Tuple[int, int], int] = field(default_factory=dict)
    cell_changes: Dict[Tuple[int, int], int] = field(default_factory=dict)
    cell_novel: Dict[Tuple[int, int], int] = field(default_factory=dict)
    action_calls: Dict[str, int] = field(default_factory=dict)
    action_changes: Dict[str, int] = field(default_factory=dict)
    action_novel: Dict[str, int] = field(default_factory=dict)

    # Aggressive penalty for cycling: a state-changing action whose changes are
    # all REPEATS scores ~0.1, well below any novel-producing action (~1.0) and
    # well below untried actions (2.0). This is the main lever stopping the
    # observed lock-in to ACTION1 on virtual-keyboard games.
    NOVEL_WEIGHT: float = 1.0
    REPEAT_WEIGHT: float = 0.1

    def _score(self, calls: int, changes: int, novels: int) -> float:
        if calls == 0:
            return 2.0
        repeats = max(0, changes - novels)
        reward = self.NOVEL_WEIGHT * novels + self.REPEAT_WEIGHT * repeats
        return reward / calls + 1.0 / (calls + 1)

    def cell_score(self, cell: Tuple[int, int]) -> float:
        return self._score(
            self.cell_calls.get(cell, 0),
            self.cell_changes.get(cell, 0),
            self.cell_novel.get(cell, 0),
        )

    def action_score(self, action: str) -> float:
        return self._score(
            self.action_calls.get(action, 0),
            self.action_changes.get(action, 0),
            self.action_novel.get(action, 0),
        )

    def record(self, action_str: str, changed: bool, novel: bool) -> None:
        parts = action_str.split()
        if not parts:
            return
        base = parts[0]
        self.action_calls[base] = self.action_calls.get(base, 0) + 1
        if changed:
            self.action_changes[base] = self.action_changes.get(base, 0) + 1
        if novel:
            self.action_novel[base] = self.action_novel.get(base, 0) + 1
        if base == "ACTION6" and len(parts) >= 3:
            try:
                cell = (int(parts[1]), int(parts[2]))
            except ValueError:
                return
            self.cell_calls[cell] = self.cell_calls.get(cell, 0) + 1
            if changed:
                self.cell_changes[cell] = self.cell_changes.get(cell, 0) + 1
            if novel:
                self.cell_novel[cell] = self.cell_novel.get(cell, 0) + 1


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

    def has_programs(self, game_id: str) -> bool:
        return bool(self._programs_by_game.get(game_id))


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
        # Lifetime exploration stats — persist across levels. Built up by
        # observing whether each action changed the grid.
        self.stats = ExplorationStats()
        # Sequence learning: for each grid state we've ever been in, remember
        # which actions led to a NOVEL state. On revisits to the same state,
        # replay those known-good actions round-robin. Chains automatically:
        # X --A--> Y (novel) and Y --B--> Z (novel) means seeing X→A→Y→B→Z.
        self._novel_transitions: Dict[int, set] = {}
        self._play_idx_per_state: Dict[int, int] = {}
        self._current_state_hash: int | None = None
        self._last_base_action: str | None = None
        self._base_action_streak = 0

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
        self._tried_clicks: set[tuple[int, int]] = set()
        # Click-feedback memory: cells that produced zero grid change last time
        # they were clicked. Cleared on level transition.
        self._useless_clicks: set[tuple[int, int]] = set()
        self._last_click: tuple[int, int] | None = None
        self._last_grid_hash: int | None = hash(frame.tobytes())
        # Per-level set of grid hashes ever seen. Used to distinguish NOVEL
        # state changes from REPEAT cycling. Reset on level transition.
        self._state_hashes_seen: set[int] = set()
        # Stuck detection: count steps since the last novel-state outcome.
        # If we hit STUCK_THRESHOLD we forget action stats this game and
        # re-explore — the alternative is grinding on cyclic actions forever.
        self._steps_since_novelty: int = 0
        # Sequence learning state: cleared on new GAME (per-game knowledge).
        # NOT cleared on level transition — post-RESET to initial state still
        # benefits from transitions we learned in prior attempts.
        self._novel_transitions = {}
        self._play_idx_per_state = {}
        self._current_state_hash = None
        self._last_base_action = None
        self._base_action_streak = 0

    def on_new_level(self) -> None:
        """Bridge calls this when a level transition is detected (grid shape change
        or major content shift). Clears per-level state but keeps the cross-level
        hypothesis bank, ExplorationStats, and MechanicMemory intact.

        Reloads `program_candidates` so sequences saved by `on_level_complete`
        during *this* game become available for the next level inside the
        same game."""
        self._tried_clicks.clear()
        self._useless_clicks.clear()
        self._last_click = None
        self._last_grid_hash = None
        self._state_hashes_seen.clear()
        self._steps_since_novelty = 0
        self.current_attempt = []
        self.planner.current_plan = []
        self.planner.last_predicted_state = None
        self.planner.failure_count = 0
        self.program_candidates = self.memory.candidate_programs(
            self.game_id, self.action_space
        )
        self.program_index = 0
        self.program_pos = 0
        self._last_base_action = None
        self._base_action_streak = 0

    def act(self, observation: np.ndarray) -> ActionEnum | str:
        return self.step(observation)

    def set_action_space(self, action_space: list[ActionEnum | str]) -> None:
        """Update currently available actions from the ARC-AGI-3 frame.

        ARC-AGI-3 can expose different actions across states. Keeping the reset
        action space forever made the inner policy stale on games where ACTION6
        dominates only part of the state machine.
        """
        current = [_canonical_action(action) for action in action_space]
        if current == self.action_space:
            return
        self.action_space = current
        self.program_candidates = self.memory.candidate_programs(
            self.game_id, self.action_space
        )
        self.program_index = 0
        self.program_pos = 0

    def step(self, frame: np.ndarray) -> ActionEnum | str:
        grid = _as_grid(frame)
        self.current_frame = grid.copy()

        # Effectiveness feedback: did the previous action change the grid, and
        # if so, was the resulting state a NOVEL one for this level?
        grid_hash = hash(grid.tobytes())
        changed = (
            self._last_grid_hash is not None and self._last_grid_hash != grid_hash
        )
        # Novel = grid changed AND we've never been in this state on this level.
        novel = changed and grid_hash not in self._state_hashes_seen
        self._state_hashes_seen.add(grid_hash)

        # Stuck detection: if we've gone STUCK_THRESHOLD steps without ever
        # producing a novel state, the current strategy is grinding. Wipe the
        # exploration stats and forget useless-this-level so the agent
        # re-explores from scratch — better than infinite cycling.
        STUCK_THRESHOLD = 30
        if novel:
            self._steps_since_novelty = 0
        else:
            self._steps_since_novelty += 1
            if self._steps_since_novelty >= STUCK_THRESHOLD:
                self._useless_clicks.clear()
                self._steps_since_novelty = 0
        # Click-feedback: cells that didn't change the grid are useless this level.
        if self._last_click is not None and self._last_grid_hash is not None and not changed:
            self._useless_clicks.add(self._last_click)
        # Lifetime stats: record (action, changed, novel) for every action
        # including cross-level. The novel flag is what stops the agent from
        # locking onto state-changing-but-cycling actions like "cursor left"
        # pressed 100x while wrapping around.
        if self.last_action is not None and self._last_grid_hash is not None:
            last_text = str(
                self.last_action.value
                if isinstance(self.last_action, ActionEnum)
                else self.last_action
            )
            self.stats.record(last_text, changed, novel)
            # Sequence learning: if the last action produced a novel state,
            # remember it as a known-good action from the previous state.
            if novel:
                self._novel_transitions.setdefault(self._last_grid_hash, set()).add(last_text)
        self._last_grid_hash = grid_hash
        self._last_click = None
        # Stash the current state hash so _state_conditioned_action can read it.
        self._current_state_hash = grid_hash

        curr_graph = self.parser.parse(grid, frame_index=self.actions_taken + 1)
        tracking = self.tracker.update(curr_graph)
        curr_tracked = tracking.tracked

        # Cheap heuristic check FIRST: if push/program/click will fire, skip the
        # entire expensive bank/extractor/goal_inference/planner pipeline.
        # On click games this is the difference between ~5ms and ~5s per step.
        heuristic_action = self._heuristic_action(curr_tracked)
        if heuristic_action is not None:
            return self._finish_step(curr_tracked, heuristic_action)

        # Full pipeline only when EIG/planner fallback is needed.
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
        return self._finish_step(curr_tracked, action)

    def _heuristic_action(self, state: ObjectGraph) -> ActionEnum | str | None:
        """Try the cheap, specific policies — sequence learning, push, program,
        click — in priority order. Returns None if none apply, in which case
        the caller falls back to the full EIG/planner pipeline."""
        # Sequence learning: if we've previously found an action that produced
        # a novel state from this exact grid, replay it. This is the highest
        # form of evidence we have — actual observation, not a heuristic guess.
        learned_action = self._state_conditioned_action()
        if learned_action is not None and not self._base_suppressed(_base_action(learned_action)):
            return self._desuppress_action(learned_action)
        scheduled_action = self._scheduled_exploration_action(state)
        if scheduled_action is not None:
            return self._desuppress_action(scheduled_action)
        push_action = self._push_toward_target_action()
        if push_action is not None:
            return self._desuppress_action(push_action)
        program_action = self._program_action()
        if program_action is not None:
            return self._desuppress_action(program_action)
        coordinate_action = self._coordinate_action(state)
        if coordinate_action is not None:
            return coordinate_action
        effective_action = self._effective_simple_action()
        if effective_action is not None:
            return self._desuppress_action(effective_action)
        return None

    def _state_conditioned_action(self) -> str | None:
        """Look up known-good actions for the current grid state. Round-robin
        through them on subsequent visits so multi-option states aren't stuck
        on one branch.

        Returns None if we've never seen this state, or have seen it but
        nothing was ever novel from it.
        """
        # A novel frame is not the same thing as progress. On keyboard games the
        # agent can create hundreds of novel cursor positions without completing
        # a level. Only trust state replay after a successful program has been
        # stored for this game.
        if not self.memory.has_programs(self.game_id):
            return None
        h = self._current_state_hash
        if h is None:
            return None
        good = self._novel_transitions.get(h)
        if not good:
            return None
        actions = sorted(good)  # deterministic ordering for round-robin
        idx = self._play_idx_per_state.get(h, 0) % len(actions)
        self._play_idx_per_state[h] = idx + 1
        return actions[idx]

    def _scheduled_exploration_action(self, state: ObjectGraph) -> str | None:
        """Broad macro schedule for unsolved multi-action games.

        Before any level has been completed for a game, per-action effectiveness
        can prefer "movement" actions that generate many new frames without
        solving. For rich keyboard/click action spaces, force a repeating macro
        sweep so the budget samples every family instead of collapsing back to
        ACTION1.
        """
        if self.memory.has_programs(self.game_id):
            return None
        bases = sorted({
            _base_action(action)
            for action in self.action_space
            if _base_action(action) != "RESET"
        })
        if len(bases) <= 2 and ActionEnum.ACTION6.value not in bases:
            return None

        simple = [a for a in bases if a != ActionEnum.ACTION6.value]
        click_slots = 4 if ActionEnum.ACTION6.value in bases else 0
        simple_slots = len(simple) * 6
        period = simple_slots + click_slots
        if period <= 0:
            return None

        pos = self.actions_taken % period
        if click_slots and pos >= simple_slots:
            return self._coordinate_action(state)
        if not simple:
            return None
        return simple[min(len(simple) - 1, pos // 6)]

    def _finish_step(self, curr_tracked: ObjectGraph, action) -> ActionEnum | str:
        action_text = str(action.value if isinstance(action, ActionEnum) else action)
        self.level_actions.append(action_text)
        base = _base_action(action_text)
        if base == self._last_base_action:
            self._base_action_streak += 1
        else:
            self._last_base_action = base
            self._base_action_streak = 1
        if base == "RESET":
            self.current_attempt = []
        else:
            self.current_attempt.append(action_text)
        # Track which click we just emitted so the next step can detect uselessness.
        if base == "ACTION6":
            parts = action_text.split()
            if len(parts) >= 3:
                try:
                    self._last_click = (int(parts[1]), int(parts[2]))
                except ValueError:
                    self._last_click = None
        self.prev_tracked = curr_tracked
        self.last_action = action
        self.actions_taken += 1
        return action

    def record_external_action(self, action: ActionEnum | str) -> None:
        """Record an action chosen by the Kaggle bridge before ACCA takes over.

        The bridge emits a short probe sequence for bootstrapping. Those actions
        still count as part of the attempt; if a probe solves a level, the
        resulting program must be available to MechanicMemory.
        """
        if self.prev_tracked is None:
            action_text = str(action.value if isinstance(action, ActionEnum) else action)
            self.level_actions.append(action_text)
            if _base_action(action_text) != "RESET":
                self.current_attempt.append(action_text)
            self.last_action = action
            self.actions_taken += 1
            return
        self._finish_step(self.prev_tracked, action)

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
        # Order matters: actual learned evidence first (state-conditioned),
        # then specific game-shape heuristics, then generic click, then EIG.
        learned_action = self._state_conditioned_action()
        if learned_action is not None and not self._base_suppressed(_base_action(learned_action)):
            self.planner.last_predicted_state = None
            return self._desuppress_action(learned_action)

        scheduled_action = self._scheduled_exploration_action(state)
        if scheduled_action is not None:
            self.planner.last_predicted_state = None
            return self._desuppress_action(scheduled_action)

        push_action = self._push_toward_target_action()
        if push_action is not None:
            self.planner.last_predicted_state = None
            return self._desuppress_action(push_action)

        program_action = self._program_action()
        if program_action is not None:
            self.planner.last_predicted_state = None
            return self._desuppress_action(program_action)

        coordinate_action = self._coordinate_action(state)
        if coordinate_action is not None:
            self.planner.last_predicted_state = None
            return coordinate_action

        # Effectiveness-based simple-action selection — when we have any
        # evidence of which simple actions change state, prefer the one with
        # the highest score over an untrained EIG selector. On real games the
        # seeded hypothesis bank rarely matches, so EIG without this signal
        # was effectively cycling.
        effective_action = self._effective_simple_action()
        if effective_action is not None:
            self.planner.last_predicted_state = None
            return self._desuppress_action(effective_action)

        if (
            self.bank.entropy() > config.ENTROPY_THRESHOLD
            and self.actions_taken < config.MAX_EXPLORATION_ACTIONS
        ):
            self.planner.last_predicted_state = None
            return self._desuppress_action(
                self.eig_selector.select_action(state, self.bank, self.action_space)
            )

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
                return self._desuppress_action(
                    self.eig_selector.select_action(state, self.bank, self.action_space)
                )
            self.planner.current_plan = plan

        action = self.planner.next_action()
        if action is None:
            self.planner.last_predicted_state = None
            return self._desuppress_action(
                self.eig_selector.select_action(state, self.bank, self.action_space)
            )
        self.planner.last_predicted_state = self.bank.map_hypothesis().execute(state, action)
        return self._desuppress_action(action)

    def _coordinate_action(self, state: ObjectGraph) -> str | None:
        """Propose one ACTION6 click on a non-background cluster we haven't tried
        yet this level. Color-agnostic — every real ARC-AGI-3 click game uses
        colors outside our synthetic-env palette, so the prior color-14/15 filter
        never fired in production."""
        if not any(_base_action(action) == ActionEnum.ACTION6.value for action in self.action_space):
            return None
        if self._base_suppressed(ActionEnum.ACTION6.value):
            return None

        candidates = self._click_candidates(state)
        if not candidates:
            return None

        # Filter out cells that produced no grid change earlier this level.
        # If everything in the candidate set is "useless this level", reset
        # uselessness — the state may have shifted enough that those cells
        # are worth retrying.
        eligible = [c for c in candidates if c not in self._useless_clicks]
        if not eligible:
            self._useless_clicks.clear()
            eligible = list(candidates)
        untried = [c for c in eligible if c not in self._tried_clicks]
        if untried:
            eligible = untried
        elif self._tried_clicks:
            self._tried_clicks.clear()

        # Rank by lifetime effectiveness. Unseen cells score 2.0 (forced to
        # try once); proven-effective cells score 0..~1.4; thrashed-useless
        # cells score near zero. Python's sort is stable — ties preserve the
        # candidate-generation order from _click_candidates (centroids first,
        # then small-cluster pixels, then fallback positions), which is the
        # most informative ordering when there's no learned signal yet.
        eligible.sort(key=lambda c: -self.stats.cell_score(c))
        row, col = eligible[0]
        self._tried_clicks.add((row, col))
        return f"{ActionEnum.ACTION6.value} {row} {col}"

    def _effective_simple_action(self) -> str | None:
        """Pick the simple action (ACTION1..5, ACTION7) with the highest
        effectiveness score, but only when we have *some* evidence — at least
        one call recorded for at least one simple action. Otherwise return
        None so the caller falls back to EIG / planner."""
        simple = [
            _base_action(a) for a in self.action_space
            if _base_action(a) not in ("RESET", "ACTION6")
        ]
        simple = [a for a in simple if a]
        if not simple:
            return None
        if not any(self.stats.action_calls.get(a, 0) > 0 for a in simple):
            return None
        unsuppressed = [a for a in simple if not self._base_suppressed(a)]
        if unsuppressed:
            simple = unsuppressed
        return max(simple, key=self.stats.action_score)

    def _base_suppressed(self, base: str) -> bool:
        """Temporarily suppress an action family after a long same-action burst.

        Kaggle logs showed whole environments consumed by ACTION1 or ACTION6.
        A cap forces the policy to sample another available family before it
        spends the remaining budget on the same local optimum.
        """
        if base != self._last_base_action:
            return False
        limit = 24 if base == ActionEnum.ACTION6.value else 8
        if self._base_action_streak < limit:
            return False
        return self._alternate_simple_action(exclude=base) is not None

    def _alternate_simple_action(self, exclude: str | None = None) -> str | None:
        candidates = [
            _base_action(action)
            for action in self.action_space
            if _base_action(action) not in ("RESET", ActionEnum.ACTION6.value, exclude)
        ]
        candidates = sorted(set(a for a in candidates if a))
        if not candidates:
            return None
        return min(candidates, key=lambda a: self.stats.action_calls.get(a, 0))

    def _desuppress_action(self, action: ActionEnum | str) -> ActionEnum | str:
        action_text = str(action.value if isinstance(action, ActionEnum) else action)
        base = _base_action(action_text)
        if not self._base_suppressed(base):
            return action
        alternate = self._alternate_simple_action(exclude=base)
        return action if alternate is None else alternate

    def _click_candidates(self, state: ObjectGraph) -> list[tuple[int, int]]:
        """Click candidates: every non-bg cluster's centroid, plus individual
        pixels of small (<=16-px) clusters, plus a fixed grid-center/quadrant
        fallback set. Deduplicated, order-preserving.

        NOTE: ObjectNode.centroid is `(col_mean, row_mean)` (i.e., (x, y) in
        image-axis convention). ACTION6 wants `(row, col)` format — so we
        unpack as cx, cy and emit (cy, cx). Reversing this was the actual
        production bug behind score=0 on click games — clicks landed at
        transposed coordinates that hit empty cells.
        """
        candidates: list[tuple[int, int]] = []
        for node in state.nodes.values():
            if node.color == 0:
                continue
            cx, cy = node.centroid
            candidates.append((int(round(cy)), int(round(cx))))
        for node in state.nodes.values():
            if node.color == 0 or node.area > 16:
                continue
            for (r, c) in sorted(node.pixels):
                candidates.append((int(r), int(c)))
        if self.current_frame is not None:
            h, w = self.current_frame.shape
            candidates.extend([
                (h // 2, w // 2),
                (h // 4, w // 4),
                (h // 4, (3 * w) // 4),
                ((3 * h) // 4, w // 4),
                ((3 * h) // 4, (3 * w) // 4),
            ])
            # Coarse whole-board sweep. Production click games have 64x64
            # boards and may hide the useful target away from visible object
            # centroids; this gives ACTION6-only games broad coverage within
            # the 300-action cap instead of repeatedly clicking the same few
            # clusters.
            rows = np.linspace(0, max(0, h - 1), num=min(12, h), dtype=int)
            cols = np.linspace(0, max(0, w - 1), num=min(12, w), dtype=int)
            for r in rows:
                for c in cols:
                    candidates.append((int(r), int(c)))
        seen: set[tuple[int, int]] = set()
        out: list[tuple[int, int]] = []
        for c in candidates:
            if c in seen:
                continue
            seen.add(c)
            out.append(c)
        return out

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
