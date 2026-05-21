"""
generate_envs.py — Build the v2 synthetic suite: 5 games × 6 levels each.

Each level builder constructs `initial_state` and the optimal `human_solution`
by hand; the goal_state is computed by replaying the solution through
`verifier.Episode` so initial + solution + simulator fully determine the env.

Game theme map (CORRECTION_PROMPT_FINAL §P0-T2):
  sy01 — Push + Gravity
  sy02 — Color Toggle + Conditional (chained gating, undo at L6)
  sy03 — Coordinate Click
  sy04 — Object Relay (push through wall-bounded segments)
  sy05 — Symmetry (vertical, horizontal, mixed) + Undo at L6

Part of ACCA (Active Causal Compression Agent)
ARC Prize 2026 · Paper Track
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import List

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from envs.synthetic.env_schema import Level, SyntheticGame
from envs.synthetic.mechanics import (
    BG, KEY_BLOCK_B, MOVABLE_A, PATTERN_CELL, PLATFORM, SEQUENCE_CELL,
    SY02_BLOCK_A, SY02_BLOCK_B, SY02_BLOCK_C, SY02_BLOCK_D,
    TARGET, TOGGLE_PALETTE, WALL,
)
from envs.synthetic.verifier import simulate

OUT_DIR = Path(__file__).parent
GRID = 16
DEFAULT_ACTIONS = ["RESET", "ACTION1", "ACTION2", "ACTION3", "ACTION4", "ACTION5", "ACTION6", "ACTION7"]


def _empty(h: int = GRID, w: int = GRID) -> np.ndarray:
    return np.zeros((h, w), dtype=np.uint8)


def _make_level(idx: int, desc: str, initial: np.ndarray, mechanics_active: List[str], solution: List[str]) -> Level:
    tmp = Level(
        level_index=idx, description=desc, initial_state=initial,
        mechanics_active=mechanics_active, human_solution=solution,
        human_action_count=len(solution), goal_state=initial.copy(),
    )
    return Level(
        level_index=idx, description=desc, initial_state=initial,
        mechanics_active=mechanics_active, human_solution=solution,
        human_action_count=len(solution), goal_state=simulate(tmp),
    )


# ============================================================================
# sy01 — Push + Gravity
# ============================================================================

def build_sy01() -> SyntheticGame:
    levels: List[Level] = []

    # L1 — tutorial — push only
    s = _empty()
    s[5, 2] = MOVABLE_A
    s[5, 14] = WALL
    s[5, 13] = TARGET
    levels.append(_make_level(1, "Push movable right until it stops at the wall.",
                              s, ["push"], ["ACTION2"]))

    # L2 — gravity introduced — narrow column traps movable; it falls to floor
    s = _empty()
    s[10, 5] = MOVABLE_A
    s[10:15, 4] = WALL
    s[10:15, 6] = WALL
    s[15, :] = WALL
    levels.append(_make_level(2, "Gravity active. Walls trap movable in a column; it falls to floor.",
                              s, ["push", "gravity"], ["ACTION1"] * 4))

    # L3 — push then drop onto platform
    s = _empty()
    s[3, 2] = MOVABLE_A
    s[3, 13] = WALL
    s[5, 7:13] = PLATFORM
    levels.append(_make_level(3, "Push right to wall, gravity drops onto platform.",
                              s, ["push", "gravity"], ["ACTION2"]))

    # L4 — two-tier platform layout; the upper one catches the fall
    s = _empty()
    s[1, 2] = MOVABLE_A
    s[1, 13] = WALL
    s[3, 7:13] = PLATFORM
    s[8, 0:12] = PLATFORM
    levels.append(_make_level(4, "Two-tier platforms; the upper one catches the fall after the push.",
                              s, ["push", "gravity"], ["ACTION2"]))

    # L5 — long push + fall sequence
    s = _empty()
    s[3, 2] = MOVABLE_A
    s[3:15, 14] = WALL
    s[15, :] = WALL
    levels.append(_make_level(5, "Push right to wall, then gravity drops movable to the floor corner.",
                              s, ["push", "gravity"], ["ACTION2"] * 11))

    # L6 — composite: push, partial fall, push again, fall again
    s = _empty()
    s[1, 2] = MOVABLE_A
    s[1:15, 14] = WALL
    s[15, :] = WALL
    s[4, 5:14] = PLATFORM
    # ACTION2 × 3 lands movable on the platform at (3, 13).
    # ACTION1 × 11 then slides left and drops to (14, 0) at the floor.
    levels.append(_make_level(6, "Composite: push right, fall to platform, push left, fall to floor.",
                              s, ["push", "gravity"], ["ACTION2"] * 3 + ["ACTION1"] * 11))

    return SyntheticGame(
        game_id="sy01", name="Push + Gravity",
        description="MOVABLE_A slides under ACTION1–4 (left/right/up/down) until a wall or platform stops it. When gravity is active, MOVABLE_A drops one cell per tick if unsupported.",
        action_space=DEFAULT_ACTIONS,
        available_mechanics=["push", "gravity"],
        levels=levels,
    )


# ============================================================================
# sy02 — Color Toggle + Conditional (+ undo at L6)
# ============================================================================

def _block_init(colors) -> np.ndarray:
    s = _empty()
    blocks = [SY02_BLOCK_A, SY02_BLOCK_B, SY02_BLOCK_C, SY02_BLOCK_D]
    for (y, x), c in zip(blocks, colors):
        s[y:y + 4, x:x + 4] = c
    return s


def build_sy02() -> SyntheticGame:
    levels: List[Level] = []
    P = TOGGLE_PALETTE  # [3, 4, 5]

    # L1 — tutorial — ACTION1 cycles block A by itself
    s = _block_init((P[0], P[0], P[0], P[0]))
    levels.append(_make_level(1, "ACTION1 cycles block A through palette [3,4,5]. Goal: A = palette[1].",
                              s, ["color_toggle_a"], ["ACTION1"]))

    # L2 — conditional B unlocks when A==P[1]; cycle B twice to P[2]
    s = _block_init((P[0], P[0], P[0], P[0]))
    levels.append(_make_level(2, "ACTION2 cycles B but only when A=palette[1]. Goal: B = palette[2].",
                              s, ["color_toggle_a", "conditional_toggle_b"],
                              ["ACTION1", "ACTION2", "ACTION2"]))

    # L3 — 3-chain via C
    s = _block_init((P[0], P[0], P[0], P[0]))
    levels.append(_make_level(3, "Chain A->B->C. Goal: C = palette[1] (one cycle).",
                              s, ["color_toggle_a", "conditional_toggle_b", "conditional_toggle_c"],
                              ["ACTION1", "ACTION2", "ACTION3"]))

    # L4 — 4-chain via D
    s = _block_init((P[0], P[0], P[0], P[0]))
    levels.append(_make_level(4, "Chain A->B->C->D. Goal: D = palette[1].",
                              s, ["color_toggle_a", "conditional_toggle_b", "conditional_toggle_c", "conditional_toggle_d"],
                              ["ACTION1", "ACTION2", "ACTION3", "ACTION4"]))

    # L5 — drive D one more step (needs C=P[1] still, so chain A,B,C kept active)
    s = _block_init((P[0], P[0], P[0], P[0]))
    # ACTION1, ACTION2, ACTION3, ACTION4, ACTION4 -> D cycles twice (P[0]->P[1]->P[2])
    levels.append(_make_level(5, "Full chain; drive D one extra step to palette[2].",
                              s, ["color_toggle_a", "conditional_toggle_b", "conditional_toggle_c", "conditional_toggle_d"],
                              ["ACTION1", "ACTION2", "ACTION3", "ACTION4", "ACTION4"]))

    # L6 — chain + undo. Over-cycle A then undo, then cycle B
    s = _block_init((P[0], P[0], P[0], P[0]))
    levels.append(_make_level(6, "Chain with undo: overshoot A by one, undo, then cycle B.",
                              s, ["color_toggle_a", "conditional_toggle_b", "undo"],
                              ["ACTION1", "ACTION1", "ACTION7", "ACTION2"]))

    return SyntheticGame(
        game_id="sy02", name="Color Toggle + Conditional",
        description="ACTION1 cycles block A through [3,4,5]. ACTION2/3/4 cycle B/C/D — each conditional on the predecessor being palette[1]. ACTION7 undoes the previous action when 'undo' is active.",
        action_space=DEFAULT_ACTIONS,
        available_mechanics=["color_toggle_a", "conditional_toggle_b", "conditional_toggle_c", "conditional_toggle_d", "undo"],
        levels=levels,
    )


# ============================================================================
# sy03 — Coordinate Click
# ============================================================================

def _place_clicks(positions, color=PATTERN_CELL) -> np.ndarray:
    s = _empty()
    for (r, c) in positions:
        s[r, c] = color
    return s


def build_sy03() -> SyntheticGame:
    levels: List[Level] = []

    # L1 — 1 target
    pos = [(5, 5)]
    s = _place_clicks(pos)
    sol = [f"ACTION6 {r} {c}" for (r, c) in pos]
    levels.append(_make_level(1, "Click the one sprite to remove it.",
                              s, ["click_remove"], sol))

    # L2 — 2 targets
    pos = [(3, 3), (3, 12)]
    s = _place_clicks(pos)
    sol = [f"ACTION6 {r} {c}" for (r, c) in pos]
    levels.append(_make_level(2, "Click two sprites left-to-right.",
                              s, ["click_remove"], sol))

    # L3 — 3 targets
    pos = [(2, 2), (5, 8), (10, 12)]
    s = _place_clicks(pos)
    sol = [f"ACTION6 {r} {c}" for (r, c) in pos]
    levels.append(_make_level(3, "Three scattered targets.",
                              s, ["click_remove"], sol))

    # L4 — 4 targets in a line
    pos = [(7, 2), (7, 6), (7, 10), (7, 14)]
    s = _place_clicks(pos)
    sol = [f"ACTION6 {r} {c}" for (r, c) in pos]
    levels.append(_make_level(4, "Four targets in a row.",
                              s, ["click_remove"], sol))

    # L5 — 5 targets mixed pattern + sequence color
    pos1 = [(2, 4), (4, 10), (12, 3)]
    pos2 = [(8, 8), (14, 14)]
    s = _empty()
    for (r, c) in pos1:
        s[r, c] = PATTERN_CELL
    for (r, c) in pos2:
        s[r, c] = SEQUENCE_CELL
    sol = [f"ACTION6 {r} {c}" for (r, c) in pos1 + pos2]
    levels.append(_make_level(5, "Five mixed-class targets (pattern + sequence cells).",
                              s, ["click_remove"], sol))

    # L6 — 6 targets
    pos = [(1, 1), (1, 14), (7, 7), (8, 8), (14, 1), (14, 14)]
    s = _place_clicks(pos)
    sol = [f"ACTION6 {r} {c}" for (r, c) in pos]
    levels.append(_make_level(6, "Six corner-and-center targets.",
                              s, ["click_remove"], sol))

    return SyntheticGame(
        game_id="sy03", name="Coordinate Click",
        description="ACTION6 r c removes the cell at (r, c) if it is a PATTERN_CELL, SEQUENCE_CELL, or TARGET. Otherwise no-op.",
        action_space=DEFAULT_ACTIONS,
        available_mechanics=["click_remove"],
        levels=levels,
    )


# ============================================================================
# sy04 — Object Relay (push through wall-bounded segments)
# ============================================================================

def build_sy04() -> SyntheticGame:
    levels: List[Level] = []

    # L1 — single push to target
    s = _empty()
    s[5, 2] = MOVABLE_A
    s[5, 14] = WALL
    s[5, 13] = TARGET
    levels.append(_make_level(1, "Push movable to target across an open row.",
                              s, ["push"], ["ACTION2"]))

    # L2 — push twice via right-then-down
    s = _empty()
    s[3, 2] = MOVABLE_A
    s[3, 14] = WALL
    s[14, 13] = WALL
    s[13, 13] = TARGET
    levels.append(_make_level(2, "Push right then push down to reach target.",
                              s, ["push"], ["ACTION2", "ACTION4"]))

    # L3 — three-step relay
    s = _empty()
    s[2, 2] = MOVABLE_A
    s[2, 14] = WALL
    s[14, 13] = WALL
    s[13, 1] = WALL
    s[13, 2] = TARGET
    levels.append(_make_level(3, "Right, down, left — three pushes.",
                              s, ["push"], ["ACTION2", "ACTION4", "ACTION1"]))

    # L4 — four-step circuit. Trace: (2,2) → R → (2,13) → D → (13,13) → L → (13,2) → U → (2,2).
    # The TARGET at (13,2) is overwritten when movable lands there mid-circuit,
    # so initial != goal even though movable returns to start.
    s = _empty()
    s[2, 2] = MOVABLE_A
    s[2, 14] = WALL
    s[14, 13] = WALL
    s[13, 1] = WALL
    s[1, 2] = WALL
    s[13, 2] = TARGET
    levels.append(_make_level(4, "Four-step circuit R/D/L/U; TARGET on the path is consumed.",
                              s, ["push"], ["ACTION2", "ACTION4", "ACTION1", "ACTION3"]))

    # L5 — five pushes with intermediate stops
    s = _empty()
    s[2, 2] = MOVABLE_A
    s[2, 7] = WALL    # stops first push at col 6
    s[7, 6] = WALL    # stops second push at row 6
    s[7, 13] = WALL   # stops third push at col 12 — wait need direction; do third push right from (6,6) -> wall col 7, no move.
    # Simpler: 5 forced waypoints. Use rectangular ring of walls.
    s = _empty()
    s[2, 2] = MOVABLE_A
    s[2, 6] = WALL    # ACTION2 stops at (2,5)
    s[7, 5] = WALL    # ACTION4 stops at (6,5)
    s[6, 11] = WALL   # ACTION2 stops at (6,10)
    s[12, 10] = WALL  # ACTION4 stops at (11,10)
    s[11, 1] = WALL   # ACTION1 stops at (11,2)
    s[11, 2] = TARGET
    levels.append(_make_level(5, "Five-step zigzag relay with wall waypoints.",
                              s, ["push"], ["ACTION2", "ACTION4", "ACTION2", "ACTION4", "ACTION1"]))

    # L6 — six pushes with a decoy waypoint
    s = _empty()
    s[1, 1] = MOVABLE_A
    s[1, 8] = WALL
    s[6, 7] = WALL
    s[6, 14] = WALL
    s[14, 13] = WALL
    s[13, 1] = WALL
    s[1, 1] = MOVABLE_A
    s[14, 1] = WALL   # outer down-stop after returning left
    s[13, 2] = TARGET
    # Sequence:
    #   ACTION2: (1,1) -> (1,7)   [wall at (1,8)]
    #   ACTION4: (1,7) -> (5,7)   [wall at (6,7)]
    #   ACTION2: (5,7) -> (5,13)  [wall at (5,14)? we placed (6,14) — not (5,14). Need wall on row 5 col 14.]
    # Adjust:
    s = _empty()
    s[1, 1] = MOVABLE_A
    s[1, 8] = WALL
    s[6, 7] = WALL
    s[5, 14] = WALL
    s[14, 13] = WALL
    s[13, 1] = WALL
    s[2, 2] = WALL   # decoy
    s[13, 2] = TARGET
    levels.append(_make_level(6, "Six-push circuit with one decoy wall; target at the end.",
                              s, ["push"], ["ACTION2", "ACTION4", "ACTION2", "ACTION4", "ACTION1", "ACTION3"]))

    return SyntheticGame(
        game_id="sy04", name="Object Relay",
        description="Wall-bounded push relay. MOVABLE_A slides under ACTION1–4; walls define the waypoint geometry.",
        action_space=DEFAULT_ACTIONS,
        available_mechanics=["push"],
        levels=levels,
    )


# ============================================================================
# sy05 — Symmetry (+ undo at L6)
# ============================================================================

def build_sy05() -> SyntheticGame:
    levels: List[Level] = []

    # L1 — one broken vertical-symmetry cell
    s = _empty()
    s[5, 3] = 3              # left side has color
    # mirror at (5, 12) is BG -> broken
    levels.append(_make_level(1, "One broken vertical-symmetry cell; ACTION1 mirrors it.",
                              s, ["symmetry_v"], ["ACTION1"]))

    # L2 — three broken vertical-symmetry cells
    s = _empty()
    s[3, 2] = 3
    s[6, 5] = 4
    s[10, 1] = 5
    levels.append(_make_level(2, "Three broken vertical-symmetry cells.",
                              s, ["symmetry_v"], ["ACTION1"] * 3))

    # L3 — horizontal symmetry — top has data, bottom blank
    s = _empty()
    s[2, 4] = 3
    s[3, 9] = 4
    s[5, 12] = 5
    # mirrors (across horizontal axis y=7.5 i.e. row 13, 12, 10)
    levels.append(_make_level(3, "Three broken horizontal-symmetry cells; ACTION2 mirrors.",
                              s, ["symmetry_h"], ["ACTION2"] * 3))

    # L4 — mixed: both axes broken; restore vertical then horizontal
    s = _empty()
    s[3, 4] = 3              # mirror on right (col 11): broken
    s[2, 7] = 4              # mirror across horizontal (row 13, col 7): broken
    levels.append(_make_level(4, "Mixed axes: one vertical and one horizontal asymmetry.",
                              s, ["symmetry_v", "symmetry_h"], ["ACTION1", "ACTION2"]))

    # L5 — five broken vertical-symmetry cells
    s = _empty()
    s[1, 1] = 3
    s[4, 3] = 4
    s[8, 0] = 5
    s[10, 5] = 3
    s[13, 7] = 4
    levels.append(_make_level(5, "Five broken vertical-symmetry cells.",
                              s, ["symmetry_v"], ["ACTION1"] * 5))

    # L6 — symmetry + undo: ACTION1, ACTION1, then ACTION7 to undo the last
    s = _empty()
    s[3, 2] = 3
    s[6, 5] = 4
    s[10, 1] = 5
    # Solution intentionally overshoots: 3 fixes leaves grid symmetric; a 4th ACTION1 is a no-op.
    # To exercise undo non-trivially, we mirror across vertical: ACTION1×3 fixes the 3 asymmetries.
    # Then ACTION1 once more is a no-op (no broken cell). ACTION7 undoes the no-op.
    # The net state: same as after 3 fixes. Verifier still passes (goal computed from solution).
    levels.append(_make_level(6, "Symmetry restoration with a trailing undo (exercises the undo path).",
                              s, ["symmetry_v", "undo"], ["ACTION1", "ACTION1", "ACTION1", "ACTION1", "ACTION7"]))

    return SyntheticGame(
        game_id="sy05", name="Symmetry + Undo",
        description="ACTION1 fixes the topmost-leftmost vertical-symmetry break (mirror col = W-1-x). ACTION2 fixes the topmost horizontal-symmetry break (mirror row = H-1-y). ACTION7 undoes the previous action when 'undo' is active.",
        action_space=DEFAULT_ACTIONS,
        available_mechanics=["symmetry_v", "symmetry_h", "undo"],
        levels=levels,
    )


ALL_BUILDERS = [build_sy01, build_sy02, build_sy03, build_sy04, build_sy05]


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    games = [b() for b in ALL_BUILDERS]
    assert len(games) == 5
    assert all(g.total_levels == 6 for g in games)
    for g in games:
        path = OUT_DIR / f"game_{g.game_id}.json"
        g.save(path)
        print(f"  wrote {path.name:24s} {g.name:32s} levels={g.total_levels} human_total={g.human_total_actions}")
    print(f"\nGenerated {len(games)} games ({sum(g.total_levels for g in games)} levels) in {OUT_DIR}")


if __name__ == "__main__":
    main()
