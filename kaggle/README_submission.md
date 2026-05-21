# ACCA ARC-AGI-3 Milestone 1 Submission

This directory contains the P3-T3 Kaggle packaging path.

## Files

- `submission_v1.ipynb` - Kaggle notebook entrypoint.
- `../src/arc_agi3_bridge.py` - adapter from Kaggle's official Arcade/Agent API to local `ACCAAgent`.

## Kaggle Setup

Attach the official ARC-AGI-3 competition dataset so the notebook can install the SDK from:

```text
/kaggle/input/arc-prize-2026-arc-agi-3/arc_agi_3_wheels/*.whl
```

Attach this repository as a Kaggle dataset or upload `acca_code.zip` as an input. The notebook extracts `acca_code.zip` into `/kaggle/working/acca_code` and adds the repo root to `sys.path`.

The bootstrap cell must print:

```text
Loaded bridge version: 2026-05-21-reward-v3
Loaded bridge file: /kaggle/working/acca_code/src/arc_agi3_bridge.py
```

If those lines do not appear, restart the Kaggle kernel and re-run the notebook from the first cell. Old `src.*` modules can remain cached in a live kernel, so do not trust a run that lacks the bridge-version banner.

Expected zip layout:

```text
acca_code.zip
  src/
    agent.py
    arc_agi3_bridge.py
    ...
```

## Constraints

- No internet calls at runtime.
- SDK installation uses only provided offline wheels.
- Draft sessions without `ARC_API_KEY` run `OperationMode.OFFLINE` against `environment_files`.
- Competition execution uses the official Arcade API directly, avoiding the sample `Swarm` online constructor.
- Submit only when ready; Paper Track is one submission per team.
