"""
mechanics.py — Composable mechanic primitives for the v2 synthetic games.

Each mechanic is a pure function `(state, action, prev, args) -> new_state`.
The Episode runner (verifier.py) composes them in MECHANIC_ORDER per tick.
`undo` is NOT a mechanic — it's a meta-action handled by the runner.

Color budget (0–15, max 16 palette):
  0  BG
  2  MOVABLE_A         (the sliding/falling object in sy01, sy04)
  3  TARGET            (visible goal marker; overwritten when reached)
  4  KEY_BLOCK_B       (sy02 secondary key block)
  5  KEY_BLOCK_C       (sy02 tertiary)
  6  KEY_BLOCK_D       (sy02 quaternary)
  7  GOAL_OBJECT
  8  WALL              (solid blocker for push, support for gravity)
  9  PLATFORM          (same as WALL for push/gravity; visually distinct)
  10 ZONE_TRANSFORM    (sy04 zone)
  11 ZONE_GATE         (sy04)
  12 LIT_ZONE          (sy02 conditional gating; unused in current levels)
  13 DECOY             (sy03 distractor — reserved, unused in current levels)
  14 PATTERN_CELL      (sy03 click target)
  15 SEQUENCE_CELL     (sy03 click target, second class)

Part of ACCA (Active Causal Compression Agent)
ARC Prize 2026 · Paper Track
"""
from __future__ import annotations

from typing import Callable, Dict, Tuple

import numpy as np

# === Color constants ===
BG = 0
MOVABLE_A = 2
TARGET = 3
KEY_BLOCK_B = 4
KEY_BLOCK_C = 5
KEY_BLOCK_D = 6
GOAL_OBJECT = 7
WALL = 8
PLATFORM = 9
ZONE_TRANSFORM = 10
ZONE_GATE = 11
LIT_ZONE = 12
DECOY = 13
PATTERN_CELL = 14
SEQUENCE_CELL = 15

# sy02 toggle palette — block colors cycle through these
TOGGLE_PALETTE = [3, 4, 5]

# sy02 key-block positions (4x4 blocks on a 16x16 grid)
SY02_BLOCK_A = (3, 3)
SY02_BLOCK_B = (3, 9)
SY02_BLOCK_C = (9, 3)
SY02_BLOCK_D = (9, 9)

PUSH_DIRS: Dict[str, Tuple[int, int]] = {
    "ACTION1": (0, -1),   # left
    "ACTION2": (0, 1),    # right
    "ACTION3": (-1, 0),   # up
    "ACTION4": (1, 0),    # down
}


def _is_solid(c: int) -> bool:
    return c in (WALL, PLATFORM)


# ---------- sy01 mechanics ----------

def push(state: np.ndarray, action: str, prev=None, args: Tuple[int, ...] = ()) -> np.ndarray:
    if action not in PUSH_DIRS:
        return state.copy()
    dy, dx = PUSH_DIRS[action]
    pos = np.argwhere(state == MOVABLE_A)
    if len(pos) == 0:
        return state.copy()
    oy, ox = int(pos[0][0]), int(pos[0][1])
    H, W = state.shape
    cy, cx = oy, ox
    while True:
        ny, nx = cy + dy, cx + dx
        if not (0 <= ny < H and 0 <= nx < W):
            break
        if _is_solid(int(state[ny, nx])):
            break
        cy, cx = ny, nx
    if (cy, cx) == (oy, ox):
        return state.copy()
    new = state.copy()
    new[oy, ox] = BG
    new[cy, cx] = MOVABLE_A
    return new


def gravity(state: np.ndarray, action: str, prev=None, args: Tuple[int, ...] = ()) -> np.ndarray:
    pos = np.argwhere(state == MOVABLE_A)
    if len(pos) == 0:
        return state.copy()
    y, x = int(pos[0][0]), int(pos[0][1])
    H, _ = state.shape
    if y + 1 >= H:
        return state.copy()
    if _is_solid(int(state[y + 1, x])):
        return state.copy()
    new = state.copy()
    new[y, x] = BG
    new[y + 1, x] = MOVABLE_A
    return new


# ---------- sy02 mechanics ----------

def _cycle_block_color(state: np.ndarray, top_left: Tuple[int, int], palette) -> np.ndarray:
    y, x = top_left
    cur = int(state[y, x])
    if cur not in palette:
        return state.copy()
    nxt = palette[(palette.index(cur) + 1) % len(palette)]
    new = state.copy()
    new[y:y + 4, x:x + 4] = nxt
    return new


def color_toggle_a(state, action, prev=None, args=()):
    if action != "ACTION1":
        return state.copy()
    return _cycle_block_color(state, SY02_BLOCK_A, TOGGLE_PALETTE)


def conditional_toggle_b(state, action, prev=None, args=()):
    if action != "ACTION2":
        return state.copy()
    if int(state[SY02_BLOCK_A]) != TOGGLE_PALETTE[1]:
        return state.copy()
    return _cycle_block_color(state, SY02_BLOCK_B, TOGGLE_PALETTE)


def conditional_toggle_c(state, action, prev=None, args=()):
    if action != "ACTION3":
        return state.copy()
    if int(state[SY02_BLOCK_B]) != TOGGLE_PALETTE[1]:
        return state.copy()
    return _cycle_block_color(state, SY02_BLOCK_C, TOGGLE_PALETTE)


def conditional_toggle_d(state, action, prev=None, args=()):
    if action != "ACTION4":
        return state.copy()
    if int(state[SY02_BLOCK_C]) != TOGGLE_PALETTE[1]:
        return state.copy()
    return _cycle_block_color(state, SY02_BLOCK_D, TOGGLE_PALETTE)


# ---------- sy03 mechanics ----------

def click_remove(state, action, prev=None, args=()):
    if action != "ACTION6" or len(args) != 2:
        return state.copy()
    r, c = args
    H, W = state.shape
    if not (0 <= r < H and 0 <= c < W):
        return state.copy()
    if int(state[r, c]) in (PATTERN_CELL, SEQUENCE_CELL, TARGET):
        new = state.copy()
        new[r, c] = BG
        return new
    return state.copy()


# ---------- sy05 mechanics ----------

def _find_broken_v(state):
    H, W = state.shape
    for y in range(H):
        for x in range(W // 2, W):
            mx = W - 1 - x
            if 0 <= mx < W and state[y, x] != state[y, mx]:
                return y, x
    return None


def symmetry_v(state, action, prev=None, args=()):
    if action != "ACTION1":
        return state.copy()
    bc = _find_broken_v(state)
    if bc is None:
        return state.copy()
    y, x = bc
    mx = state.shape[1] - 1 - x
    new = state.copy()
    new[y, x] = state[y, mx]
    return new


def _find_broken_h(state):
    H, W = state.shape
    for y in range(H // 2, H):
        for x in range(W):
            my = H - 1 - y
            if 0 <= my < H and state[y, x] != state[my, x]:
                return y, x
    return None


def symmetry_h(state, action, prev=None, args=()):
    if action != "ACTION2":
        return state.copy()
    bc = _find_broken_h(state)
    if bc is None:
        return state.copy()
    y, x = bc
    my = state.shape[0] - 1 - y
    new = state.copy()
    new[y, x] = state[my, x]
    return new


# ---------- Dispatch ----------

MECHANIC_ORDER = [
    "push",                       # spatial actions first
    "gravity",
    "color_toggle_a",             # then symbolic state changes
    "conditional_toggle_b",
    "conditional_toggle_c",
    "conditional_toggle_d",
    "click_remove",
    "symmetry_v",
    "symmetry_h",
]

MECHANIC_FNS: Dict[str, Callable] = {
    "push": push,
    "gravity": gravity,
    "color_toggle_a": color_toggle_a,
    "conditional_toggle_b": conditional_toggle_b,
    "conditional_toggle_c": conditional_toggle_c,
    "conditional_toggle_d": conditional_toggle_d,
    "click_remove": click_remove,
    "symmetry_v": symmetry_v,
    "symmetry_h": symmetry_h,
}
