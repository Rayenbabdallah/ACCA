"""
verifier.py — Synthetic-game runner + per-level verification.

`Episode` is the deterministic single-level runner that composes the active
mechanics in MECHANIC_ORDER each tick. It also owns the history stack used by
the `undo` meta-action (ACTION7 — only when `"undo"` is in mechanics_active).

`verify_level` checks: solution length matches count, initial != goal,
simulating the human_solution reaches the recorded goal_state.
`verify_game` requires exactly 6 levels (Tech Report §3.4).

Part of ACCA (Active Causal Compression Agent)
ARC Prize 2026 · Paper Track
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from envs.synthetic.env_schema import Level, SyntheticGame
from envs.synthetic.mechanics import MECHANIC_FNS, MECHANIC_ORDER


def _parse_action(action_str: str) -> Tuple[str, Tuple[int, ...]]:
    parts = action_str.split()
    return parts[0], tuple(int(p) for p in parts[1:])


class Episode:
    """Deterministic single-level runner. Owns the history stack for undo."""

    def __init__(self, level: Level):
        self.level = level
        self.state: np.ndarray = level.initial_state.copy()
        self.history: List[np.ndarray] = [self.state.copy()]

    def step(self, action_str: str) -> np.ndarray:
        base, args = _parse_action(action_str)

        if base == "RESET":
            self.state = self.level.initial_state.copy()
            self.history = [self.state.copy()]
            return self.state

        if base == "ACTION7" and "undo" in self.level.mechanics_active:
            if len(self.history) >= 2:
                self.history.pop()
                self.state = self.history[-1].copy()
            return self.state

        new = self.state.copy()
        for m in MECHANIC_ORDER:
            if m in self.level.mechanics_active:
                new = MECHANIC_FNS[m](new, base, prev=self.state, args=args)
        self.state = new
        self.history.append(self.state.copy())
        return self.state


def simulate(level: Level) -> np.ndarray:
    ep = Episode(level)
    for a in level.human_solution:
        ep.step(a)
    return ep.state


def verify_level(level: Level) -> bool:
    if len(level.human_solution) != level.human_action_count:
        return False
    if np.array_equal(level.initial_state, level.goal_state):
        return False
    final = simulate(level)
    return np.array_equal(final, level.goal_state)


def verify_game(game: SyntheticGame) -> bool:
    if game.total_levels != 6:
        return False
    return all(verify_level(l) for l in game.levels)


def main() -> None:
    base = Path(__file__).parent
    files = sorted(base.glob("game_*.json"))
    if not files:
        print(f"No game_*.json files found in {base}")
        sys.exit(1)
    all_ok = True
    total_levels = 0
    for p in files:
        game = SyntheticGame.load(p)
        ok = verify_game(game)
        tag = "OK  " if ok else "FAIL"
        print(f"  [{tag}] {game.game_id:6s} {game.name:32s} levels={game.total_levels} human_total={game.human_total_actions}")
        if not ok:
            for l in game.levels:
                if not verify_level(l):
                    reason = (
                        "len mismatch" if len(l.human_solution) != l.human_action_count
                        else "init == goal" if np.array_equal(l.initial_state, l.goal_state)
                        else "sim != goal"
                    )
                    print(f"      FAIL L{l.level_index}: {reason}  ({l.description})")
        all_ok = all_ok and ok
        total_levels += game.total_levels
    if not all_ok:
        print("\nVerification FAILED.")
        sys.exit(1)
    print(f"\nAll {len(files)} games verified ({total_levels} levels total).")


if __name__ == "__main__":
    main()
