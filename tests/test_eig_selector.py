"""
test_eig_selector.py - Expected Information Gain action selector tests.

Part of ACCA (Active Causal Compression Agent)
ARC Prize 2026 . Paper Track
"""
from __future__ import annotations

import math
import time

import pytest

from src.hypothesis.hypothesis_bank import HypothesisBank
from src.hypothesis.mechanic_dsl import ActionIsCondition, CausalRule, Hypothesis, MoveEffect
from src.perception.event_extractor import ActionEnum
from src.perception.frame_parser import ObjectGraph, ObjectNode
from src.planning.eig_selector import EIGSelector


def _state() -> ObjectGraph:
    pixels = frozenset({(1, 1), (1, 2), (2, 1), (2, 2)})
    node = ObjectNode(
        obj_id=0,
        color=3,
        bbox=(1, 1, 2, 2),
        centroid=(1.5, 1.5),
        area=4,
        pixels=pixels,
    )
    return ObjectGraph(nodes={0: node}, edges=[], frame_index=0)


def _move_on(action: ActionEnum, dx: int = 1, dy: int = 0) -> Hypothesis:
    return Hypothesis([CausalRule([ActionIsCondition(action)], [MoveEffect(0, dx, dy)])])


def test_score_actions_prefers_action_that_splits_hypotheses():
    bank = HypothesisBank(
        seed_hypotheses=[
            _move_on(ActionEnum.ACTION1),
            _move_on(ActionEnum.ACTION2),
        ]
    )
    selector = EIGSelector(n_samples=10)

    scores = selector.score_actions(
        _state(),
        bank,
        [ActionEnum.ACTION3, ActionEnum.ACTION1],
    )

    assert scores[ActionEnum.ACTION1] == pytest.approx(math.log(2))
    assert scores[ActionEnum.ACTION3] == pytest.approx(0.0)
    assert scores[ActionEnum.ACTION1] > scores[ActionEnum.ACTION3]


def test_select_action_returns_best_available_action_preserving_input_type():
    bank = HypothesisBank(
        seed_hypotheses=[
            _move_on(ActionEnum.ACTION1),
            _move_on(ActionEnum.ACTION2),
        ]
    )
    selector = EIGSelector(n_samples=10)

    action = selector.select_action(_state(), bank, ["ACTION3", "ACTION1"])

    assert action == "ACTION1"


def test_single_hypothesis_has_zero_eig():
    bank = HypothesisBank(seed_hypotheses=[_move_on(ActionEnum.ACTION1)])
    selector = EIGSelector(n_samples=10)

    scores = selector.score_actions(_state(), bank, [ActionEnum.ACTION1, ActionEnum.ACTION2])

    assert scores == {ActionEnum.ACTION1: 0.0, ActionEnum.ACTION2: 0.0}


def test_empty_bank_scores_zero_and_empty_action_space_raises():
    bank = HypothesisBank(seed_hypotheses=[_move_on(ActionEnum.ACTION1)])
    bank.hypotheses = []
    selector = EIGSelector(n_samples=10)

    assert selector.score_actions(_state(), bank, [ActionEnum.ACTION1]) == {
        ActionEnum.ACTION1: 0.0
    }
    with pytest.raises(ValueError):
        selector.select_action(_state(), bank, [])


def test_non_uniform_prior_changes_eig():
    h1 = _move_on(ActionEnum.ACTION1)
    h2 = _move_on(ActionEnum.ACTION2)
    bank = HypothesisBank(seed_hypotheses=[h1, h2])
    bank.scores = {h1.fingerprint(): 2.0, h2.fingerprint(): 0.0}
    selector = EIGSelector(n_samples=10)

    scores = selector.score_actions(_state(), bank, [ActionEnum.ACTION1, ActionEnum.ACTION3])

    assert 0.0 < scores[ActionEnum.ACTION1] < math.log(2)
    assert scores[ActionEnum.ACTION3] == pytest.approx(0.0)


def test_runtime_under_200ms_for_small_bank():
    bank = HypothesisBank(
        seed_hypotheses=[
            _move_on(ActionEnum.ACTION1),
            _move_on(ActionEnum.ACTION2),
            _move_on(ActionEnum.ACTION3),
            _move_on(ActionEnum.ACTION4),
        ]
    )
    selector = EIGSelector(n_samples=100)
    start = time.perf_counter()

    selector.select_action(
        _state(),
        bank,
        [ActionEnum.ACTION1, ActionEnum.ACTION2, ActionEnum.ACTION3, ActionEnum.ACTION4],
    )
    elapsed_ms = (time.perf_counter() - start) * 1000

    assert elapsed_ms < 200.0
