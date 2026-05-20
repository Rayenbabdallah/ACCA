"""
verifier.py — Single source of truth for synthetic env mechanics + verification.

The `step` function defines, per mechanic family, exactly what one ACTION does
to the grid. The generator computes each env's goal_state by replaying the
human_solution through this same simulator, so verification is a deterministic
re-execution against the persisted goal.

Goal-invariant checks per family are layered on top to guard against trivial
self-consistency (initial_state == goal_state, no-op solutions, etc.).

Part of ACCA (Active Causal Compression Agent)
ARC Prize 2026 · Paper Track
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Set

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from envs.synthetic.env_schema import SyntheticEnv

GRID: int = 64
BG: int = 0
WALL: int = 8
MOVABLE_PUSH: int = 2
TARGET_PUSH: int = 3
MOVABLE_GRAV: int = 4
ATTRACTOR: int = 5
COND_B: int = 7

DIR_MAP = {
    "ACTION1": (-1, 0),
    "ACTION2": (1, 0),
    "ACTION3": (0, -1),
    "ACTION4": (0, 1),
}
PALETTE_FWD = [3, 4, 5, 6]
PALETTE_REV = [3, 6, 5, 4]
COLOR_TOGGLE_BLOCKS = [(8, 8), (8, 16), (8, 24)]
COND_A_TOP_LEFT = (10, 10)


def _step_color_toggle(state: np.ndarray, action: str, adv: bool) -> np.ndarray:
    new = state.copy()
    palette = PALETTE_REV if adv else PALETTE_FWD
    idx_map = {"ACTION1": 0, "ACTION2": 1, "ACTION3": 2}
    if action not in idx_map:
        return new
    idx = idx_map[action]
    y, x = COLOR_TOGGLE_BLOCKS[idx]
    cur = int(state[y, x])
    if cur not in palette:
        return new
    nxt = palette[(palette.index(cur) + 1) % len(palette)]
    new[y:y + 4, x:x + 4] = nxt
    return new


def _step_push(state: np.ndarray, action: str, adv: bool) -> np.ndarray:
    new = state.copy()
    if action not in DIR_MAP:
        return new
    dy, dx = DIR_MAP[action]
    pos = np.argwhere(state == MOVABLE_PUSH)
    if len(pos) == 0:
        return new
    oy, ox = int(pos[0][0]), int(pos[0][1])
    cy, cx = oy, ox
    while True:
        ny, nx = cy + dy, cx + dx
        if not (0 <= ny < GRID and 0 <= nx < GRID):
            break
        if state[ny, nx] == WALL:
            break
        cy, cx = ny, nx
    if adv and (cy, cx) != (oy, ox):
        cy -= dy
        cx -= dx
    new[oy, ox] = BG
    new[cy, cx] = MOVABLE_PUSH
    return new


def _step_gravity(state: np.ndarray, action: str, adv: bool) -> np.ndarray:
    new = state.copy()
    if action != "ACTION1":
        return new
    attractor_row = 32
    stop_zone: Set[int] = {32} if adv else {31, 32, 33}
    positions = [tuple(map(int, p)) for p in np.argwhere(state == MOVABLE_GRAV)]
    for (y, x) in positions:
        new[y, x] = BG
    for (y, x) in positions:
        if y in stop_zone:
            ny = y
        else:
            ny = y + (1 if y < attractor_row else -1)
        if new[ny, x] == ATTRACTOR:
            ny = y
        new[ny, x] = MOVABLE_GRAV
    return new


def _find_broken_cell(state: np.ndarray, adv: bool):
    H, W = state.shape
    for y in range(H):
        for x in range(W // 2, W):
            mx = (64 - x) if adv else (63 - x)
            if 0 <= mx < W and state[y, x] != state[y, mx]:
                return y, x
    return None


def _step_symmetry(state: np.ndarray, action: str, adv: bool) -> np.ndarray:
    new = state.copy()
    if action != "ACTION1":
        return new
    bc = _find_broken_cell(state, adv)
    if bc is None:
        return new
    y, x = bc
    mx = (64 - x) if adv else (63 - x)
    if 0 <= mx < state.shape[1]:
        new[y, x] = state[y, mx]
    return new


def _step_conditional(state: np.ndarray, action: str, adv: bool) -> np.ndarray:
    new = state.copy()
    ay, ax = COND_A_TOP_LEFT
    if action == "ACTION1":
        cur = int(state[ay, ax])
        nxt = 4 if cur == 3 else 3
        new[ay:ay + 4, ax:ax + 4] = nxt
        return new
    if action == "ACTION2":
        a_color = int(state[ay, ax])
        flag_on = (a_color == 3) if adv else (a_color == 4)
        b_pos = np.argwhere(state == COND_B)
        if len(b_pos) == 0:
            return new
        by, bx = int(b_pos[0][0]), int(b_pos[0][1])
        nbx = bx + (1 if flag_on else -1)
        if not (0 <= nbx < GRID):
            return new
        new[by, bx] = BG
        new[by, nbx] = COND_B
        return new
    return new


_STEP_DISPATCH = {
    "color_toggle": _step_color_toggle,
    "push": _step_push,
    "gravity": _step_gravity,
    "symmetry_restoration": _step_symmetry,
    "conditional": _step_conditional,
}


def step(state: np.ndarray, action: str, family: str, adv: bool) -> np.ndarray:
    fn = _STEP_DISPATCH.get(family)
    if fn is None:
        return state.copy()
    return fn(state, action, adv)


def simulate(env: SyntheticEnv) -> np.ndarray:
    s = env.initial_state.copy()
    for a in env.human_solution:
        s = step(s, a, env.mechanic_family, env.is_adversarial)
    return s


def _is_symmetric(state: np.ndarray, adv: bool) -> bool:
    H, W = state.shape
    for y in range(H):
        for x in range(W // 2, W):
            mx = (64 - x) if adv else (63 - x)
            if 0 <= mx < W and state[y, x] != state[y, mx]:
                return False
    return True


def _goal_invariant_holds(env: SyntheticEnv, final: np.ndarray) -> bool:
    """Family-specific sanity check beyond `final == goal_state`."""
    f = env.mechanic_family
    if f == "color_toggle":
        for (y, x) in COLOR_TOGGLE_BLOCKS:
            if int(final[y, x]) != 6:
                return False
        return True
    if f == "push":
        return bool(np.any(final == MOVABLE_PUSH))
    if f == "gravity":
        stop_zone = {32} if env.is_adversarial else {31, 32, 33}
        rows = np.argwhere(final == MOVABLE_GRAV)[:, 0]
        return len(rows) > 0 and all(int(r) in stop_zone for r in rows)
    if f == "symmetry_restoration":
        return _is_symmetric(final, env.is_adversarial)
    if f == "conditional":
        return bool(np.any(final == COND_B))
    return True


def verify_env(env: SyntheticEnv) -> bool:
    """Executes the human_solution and checks it achieves the goal state."""
    if len(env.human_solution) != env.human_action_count:
        return False
    if np.array_equal(env.initial_state, env.goal_state):
        return False
    final = simulate(env)
    if not np.array_equal(final, env.goal_state):
        return False
    return _goal_invariant_holds(env, final)


def main() -> None:
    base = Path(__file__).parent
    json_files = sorted(base.glob("env_*.json"))
    if not json_files:
        print("No env_*.json files found in", base)
        sys.exit(1)
    all_ok = True
    for p in json_files:
        env = SyntheticEnv.load(p)
        ok = verify_env(env)
        tag = "OK  " if ok else "FAIL"
        adv = "adv" if env.is_adversarial else "   "
        print(f"  [{tag}] {env.env_id:32s} {env.mechanic_family:22s} {adv}  actions={env.human_action_count}")
        if not ok:
            all_ok = False
    if not all_ok:
        print("\nVerification FAILED for one or more envs.")
        sys.exit(1)
    print(f"\nAll {len(json_files)} synthetic environments verified.")


if __name__ == "__main__":
    main()
