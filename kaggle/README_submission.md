# ACCA ARC-AGI-3 Milestone 1 Submission

This directory contains the P3-T3 Kaggle packaging path.

## Files

- `submission_v1.ipynb` - Kaggle notebook entrypoint.
- `../src/arc_agi3_bridge.py` - adapter from Kaggle's official `is_done` / `choose_action` API to local `ACCAAgent`.

## Kaggle Setup

Attach the official ARC-AGI-3 competition dataset so the notebook can install the SDK from:

```text
/kaggle/input/arc-prize-2026-arc-agi-3/arc_agi_3_wheels/*.whl
```

Attach this repository as a Kaggle dataset or include the `src/` package in the notebook environment. The notebook adds common Kaggle source paths to `sys.path`.

## Constraints

- No internet calls at runtime.
- SDK installation uses only provided offline wheels.
- Competition execution uses `OperationMode.COMPETITION` and `Swarm`.
- Submit only when ready; Paper Track is one submission per team.
