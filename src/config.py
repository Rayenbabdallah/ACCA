"""
config.py - ACCA hyperparameters and constants.

Part of ACCA (Active Causal Compression Agent)
ARC Prize 2026 . Paper Track
"""
from __future__ import annotations

HYPOTHESIS_BANK_SIZE = 50
MDL_LAMBDA = 0.1
EIG_N_SAMPLES = 100
ENTROPY_THRESHOLD = 0.5
MIN_SUPPORT = 3
MEMORY_CONFIDENCE_GATE = 0.85
MAX_EXPLORATION_ACTIONS = 50
RECOVERY_TRIGGER = 3
ACTION_SPACE = [
    "RESET",
    "ACTION1",
    "ACTION2",
    "ACTION3",
    "ACTION4",
    "ACTION5",
    "ACTION6",
    "ACTION7",
]
