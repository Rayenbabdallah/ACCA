"""
hypothesis_seeds.py - Hand-coded starter mechanics for the ACCA DSL.

Seeds are compact priors for the hypothesis bank. They are not fitted to the
private ARC-AGI-3 games and make no network calls.

Part of ACCA (Active Causal Compression Agent)
ARC Prize 2026 . Paper Track
"""
from __future__ import annotations

from src.hypothesis.mechanic_dsl import (
    ActionIsCondition,
    CausalRule,
    Hypothesis,
    MoveEffect,
    ObjColorCondition,
    ObjCountCondition,
    ObjPropertyCondition,
    RecolorEffect,
    SpawnEffect,
    ToggleEffect,
)
from src.perception.event_extractor import ActionEnum


def _h(rule: CausalRule) -> Hypothesis:
    return Hypothesis([rule])


def _action_move(action: ActionEnum, dx: int, dy: int) -> Hypothesis:
    return _h(CausalRule([ActionIsCondition(action)], [MoveEffect(0, dx, dy)]))


def load_seeds() -> list[Hypothesis]:
    """Return 20 executable starter hypotheses covering synthetic mechanic families."""
    seeds: list[Hypothesis] = []

    # Color-toggle variants.
    seeds.extend(
        [
            _h(CausalRule([ActionIsCondition(ActionEnum.ACTION1)], [ToggleEffect(0, 3, 4)])),
            _h(CausalRule([ActionIsCondition(ActionEnum.ACTION2)], [ToggleEffect(0, 3, 4)])),
            _h(CausalRule([ActionIsCondition(ActionEnum.ACTION3)], [ToggleEffect(0, 4, 5)])),
            _h(CausalRule([ActionIsCondition(ActionEnum.ACTION4)], [ToggleEffect(0, 4, 5)])),
        ]
    )

    # Push / spatial motion variants.
    seeds.extend(
        [
            _action_move(ActionEnum.ACTION1, dx=-1, dy=0),
            _action_move(ActionEnum.ACTION2, dx=1, dy=0),
            _action_move(ActionEnum.ACTION3, dx=0, dy=-1),
            _action_move(ActionEnum.ACTION4, dx=0, dy=1),
            _h(
                CausalRule(
                    [
                        ActionIsCondition(ActionEnum.ACTION5),
                        ObjPropertyCondition(0, "centroid_x", "lt", 8),
                    ],
                    [MoveEffect(0, dx=1, dy=0)],
                )
            ),
            _h(
                CausalRule(
                    [
                        ActionIsCondition(ActionEnum.ACTION5),
                        ObjPropertyCondition(0, "centroid_x", "ge", 8),
                    ],
                    [MoveEffect(0, dx=-1, dy=0)],
                )
            ),
        ]
    )

    # Gravity-like variants.
    seeds.extend(
        [
            _h(CausalRule([ObjCountCondition(color=2, op="gt", count=0)], [MoveEffect(0, 0, 1)])),
            _h(CausalRule([ObjCountCondition(color=2, op="gt", count=0)], [MoveEffect(0, 0, -1)])),
            _h(CausalRule([ActionIsCondition(ActionEnum.ACTION4)], [MoveEffect(0, 0, 2)])),
            _h(CausalRule([ActionIsCondition(ActionEnum.ACTION3)], [MoveEffect(0, 0, -2)])),
        ]
    )

    # Symmetry / pattern completion variants.
    seeds.extend(
        [
            _h(
                CausalRule(
                    [ActionIsCondition(ActionEnum.ACTION1)],
                    [SpawnEffect(3, relative_to_obj_id=0, offset=(0, 4))],
                )
            ),
            _h(
                CausalRule(
                    [ActionIsCondition(ActionEnum.ACTION2)],
                    [SpawnEffect(3, relative_to_obj_id=0, offset=(4, 0))],
                )
            ),
            _h(
                CausalRule(
                    [ActionIsCondition(ActionEnum.ACTION3)],
                    [SpawnEffect(3, relative_to_obj_id=0, offset=(4, 4))],
                )
            ),
        ]
    )

    # Conditional variants.
    seeds.extend(
        [
            _h(
                CausalRule(
                    [ActionIsCondition(ActionEnum.ACTION2), ObjColorCondition(0, 4)],
                    [RecolorEffect(1, 5)],
                )
            ),
            _h(
                CausalRule(
                    [ActionIsCondition(ActionEnum.ACTION3), ObjColorCondition(1, 5)],
                    [RecolorEffect(2, 6)],
                )
            ),
            _h(
                CausalRule(
                    [ActionIsCondition(ActionEnum.ACTION4), ObjColorCondition(2, 6)],
                    [RecolorEffect(3, 7)],
                )
            ),
        ]
    )

    assert len(seeds) == 20
    return seeds
