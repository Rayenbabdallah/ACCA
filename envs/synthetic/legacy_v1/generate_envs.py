"""
generate_envs.py — Build the 15 synthetic ARC-AGI-3 holdout environments.

Each builder constructs the initial grid by hand and lists the optimal action
sequence (the RHAE denominator); the goal state is computed by replaying that
sequence through verifier.step, so initial + solution + simulator together
fully determine the env. The resulting JSON files are immutable holdout data —
do NOT regenerate after Phase 1.

Part of ACCA (Active Causal Compression Agent)
ARC Prize 2026 · Paper Track
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import List

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from envs.synthetic.env_schema import SyntheticEnv
from envs.synthetic.verifier import (
    COLOR_TOGGLE_BLOCKS,
    COND_A_TOP_LEFT,
    COND_B,
    GRID,
    MOVABLE_GRAV,
    MOVABLE_PUSH,
    TARGET_PUSH,
    WALL,
    ATTRACTOR,
    step,
)

OUT_DIR = Path(__file__).parent
DEFAULT_ACTION_SPACE = ["RESET", "ACTION1", "ACTION2", "ACTION3", "ACTION4", "ACTION5", "ACTION6"]


def _empty() -> np.ndarray:
    return np.zeros((GRID, GRID), dtype=np.uint8)


def _make_env(
    env_id: str,
    family: str,
    adv: bool,
    initial: np.ndarray,
    solution: List[str],
    description: str,
) -> SyntheticEnv:
    s = initial.copy()
    for a in solution:
        s = step(s, a, family, adv)
    return SyntheticEnv(
        env_id=env_id,
        mechanic_family=family,
        is_adversarial=adv,
        initial_state=initial,
        action_space=list(DEFAULT_ACTION_SPACE),
        human_solution=solution,
        human_action_count=len(solution),
        mechanic_description=description,
        goal_state=s,
    )


def _color_toggle_initial(colors) -> np.ndarray:
    s = _empty()
    for (y, x), c in zip(COLOR_TOGGLE_BLOCKS, colors):
        s[y:y + 4, x:x + 4] = c
    return s


def build_color_toggle_01() -> SyntheticEnv:
    return _make_env(
        "color_toggle_01", "color_toggle", False,
        _color_toggle_initial((3, 4, 5)),
        ["ACTION1", "ACTION1", "ACTION1", "ACTION2", "ACTION2", "ACTION3"],
        "Three 4x4 blocks; ACTION{n} cycles block n through palette [3,4,5,6]. Goal: all blocks color 6.",
    )


def build_color_toggle_02() -> SyntheticEnv:
    return _make_env(
        "color_toggle_02", "color_toggle", False,
        _color_toggle_initial((3, 4, 6)),
        ["ACTION1", "ACTION1", "ACTION1", "ACTION2", "ACTION2"],
        "Two blocks need cycling to color 6; third already correct. Palette [3,4,5,6].",
    )


def build_push_01() -> SyntheticEnv:
    s = _empty()
    s[10, 10] = MOVABLE_PUSH
    s[10, 53] = WALL
    s[10, 52] = TARGET_PUSH
    return _make_env(
        "push_01", "push", False, s, ["ACTION4"],
        "Movable slides right until adjacent to a wall; target marker at the resting cell.",
    )


def build_push_02() -> SyntheticEnv:
    s = _empty()
    s[5, 5] = MOVABLE_PUSH
    s[5, 51] = WALL
    s[51, 50] = WALL
    s[50, 50] = TARGET_PUSH
    return _make_env(
        "push_02", "push", False, s, ["ACTION4", "ACTION2"],
        "Push right (slides to col 50), then push down (slides to row 50). Goal: reach (50, 50).",
    )


def build_push_03() -> SyntheticEnv:
    s = _empty()
    s[10, 10] = MOVABLE_PUSH
    s[10, 26] = WALL
    s[41, 25] = WALL
    s[40, 51] = WALL
    s[40, 50] = TARGET_PUSH
    return _make_env(
        "push_03", "push", False, s, ["ACTION4", "ACTION2", "ACTION4"],
        "L-shaped path: right to col 25, down to row 40, right to col 50.",
    )


def build_gravity_01() -> SyntheticEnv:
    s = _empty()
    s[32, 32] = ATTRACTOR
    s[28, 20] = MOVABLE_GRAV
    s[30, 30] = MOVABLE_GRAV
    s[35, 40] = MOVABLE_GRAV
    return _make_env(
        "gravity_01", "gravity", False, s, ["ACTION1", "ACTION1", "ACTION1"],
        "Each ACTION1, movables (color 4) step 1 row toward attractor row 32; stop in rows {31,32,33}.",
    )


def build_gravity_02() -> SyntheticEnv:
    s = _empty()
    s[32, 32] = ATTRACTOR
    s[30, 20] = MOVABLE_GRAV
    s[33, 40] = MOVABLE_GRAV
    return _make_env(
        "gravity_02", "gravity", False, s, ["ACTION1"],
        "One gravity tick: (30,20)->(31,20); (33,40) already in stop zone.",
    )


def build_symmetry_01() -> SyntheticEnv:
    s = _empty()
    s[10, 5] = 3
    s[20, 10] = 4
    s[30, 15] = 5
    return _make_env(
        "symmetry_01", "symmetry_restoration", False,
        s, ["ACTION1", "ACTION1", "ACTION1"],
        "Each ACTION1 reflects the topmost-leftmost right-half cell to match state[y, 63-x]. 3 broken cells.",
    )


def build_symmetry_02() -> SyntheticEnv:
    s = _empty()
    s[15, 8] = 6
    s[25, 12] = 7
    return _make_env(
        "symmetry_02", "symmetry_restoration", False,
        s, ["ACTION1", "ACTION1"],
        "Two broken-symmetry cells. ACTION1 restores them one at a time, raster order.",
    )


def build_conditional_01() -> SyntheticEnv:
    s = _empty()
    ay, ax = COND_A_TOP_LEFT
    s[ay:ay + 4, ax:ax + 4] = 3
    s[30, 30] = COND_B
    return _make_env(
        "conditional_01", "conditional", False,
        s, ["ACTION1", "ACTION2", "ACTION2", "ACTION2"],
        "If A is color 4, ACTION2 moves B right; if 3, moves left. ACTION1 toggles A. Goal: B at col 33.",
    )


def build_conditional_02() -> SyntheticEnv:
    s = _empty()
    ay, ax = COND_A_TOP_LEFT
    s[ay:ay + 4, ax:ax + 4] = 3
    s[30, 30] = COND_B
    return _make_env(
        "conditional_02", "conditional", False,
        s, ["ACTION2", "ACTION2", "ACTION2"],
        "A starts at color 3 (move-left branch). Goal: B at col 27.",
    )


def build_color_toggle_adversarial_01() -> SyntheticEnv:
    return _make_env(
        "color_toggle_adversarial_01", "color_toggle", True,
        _color_toggle_initial((3, 4, 5)),
        ["ACTION1", "ACTION2", "ACTION2", "ACTION3", "ACTION3", "ACTION3"],
        "ADVERSARIAL: palette cycle REVERSED [3,6,5,4]. Same initial state as color_toggle_01.",
    )


def build_push_adversarial_01() -> SyntheticEnv:
    s = _empty()
    s[5, 5] = MOVABLE_PUSH
    s[5, 52] = WALL
    s[53, 50] = WALL
    s[51, 50] = TARGET_PUSH
    return _make_env(
        "push_adversarial_01", "push", True,
        s, ["ACTION4", "ACTION2"],
        "ADVERSARIAL: movable stops 1 cell BEFORE the wall (not adjacent).",
    )


def build_gravity_adversarial_01() -> SyntheticEnv:
    s = _empty()
    s[32, 32] = ATTRACTOR
    s[30, 20] = MOVABLE_GRAV
    s[34, 40] = MOVABLE_GRAV
    return _make_env(
        "gravity_adversarial_01", "gravity", True,
        s, ["ACTION1", "ACTION1"],
        "ADVERSARIAL: stop zone is {32} only (must land on exact attractor row).",
    )


def build_symmetry_adversarial_01() -> SyntheticEnv:
    s = _empty()
    s[10, 5] = 3
    s[20, 10] = 4
    return _make_env(
        "symmetry_adversarial_01", "symmetry_restoration", True,
        s, ["ACTION1", "ACTION1"],
        "ADVERSARIAL: mirror column = (64 - x) instead of (63 - x) (off-by-one reflection axis).",
    )


ALL_BUILDERS = [
    build_color_toggle_01,
    build_color_toggle_02,
    build_push_01,
    build_push_02,
    build_push_03,
    build_gravity_01,
    build_gravity_02,
    build_symmetry_01,
    build_symmetry_02,
    build_conditional_01,
    build_conditional_02,
    build_color_toggle_adversarial_01,
    build_push_adversarial_01,
    build_gravity_adversarial_01,
    build_symmetry_adversarial_01,
]


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    envs = [b() for b in ALL_BUILDERS]
    assert len(envs) == 15, f"expected 15 envs, got {len(envs)}"
    n_adv = sum(1 for e in envs if e.is_adversarial)
    for e in envs:
        path = OUT_DIR / f"env_{e.env_id}.json"
        e.save(path)
        adv = "adv" if e.is_adversarial else "   "
        print(f"  wrote {path.name:42s} {e.mechanic_family:22s} {adv}  actions={e.human_action_count}")
    print(f"\nGenerated {len(envs)} envs ({n_adv} adversarial) in {OUT_DIR}")


if __name__ == "__main__":
    main()
