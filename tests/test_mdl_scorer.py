"""
test_mdl_scorer.py - MDL scorer and hypothesis bank tests.

Part of ACCA (Active Causal Compression Agent)
ARC Prize 2026 . Paper Track
"""
from __future__ import annotations

import math

import pytest

from src.hypothesis.hypothesis_bank import HypothesisBank
from src.hypothesis.mdl_scorer import MDLScorer
from src.hypothesis.mechanic_dsl import (
    ActionIsCondition,
    CausalRule,
    Hypothesis,
    MoveEffect,
    RecolorEffect,
)
from src.perception.event_extractor import ActionEnum, EventExtractor
from src.perception.frame_parser import ObjectGraph, ObjectNode


def _node(obj_id: int, color: int, top: int, left: int, size: int = 2) -> ObjectNode:
    pixels = frozenset(
        (r, c) for r in range(top, top + size) for c in range(left, left + size)
    )
    return ObjectNode(
        obj_id=obj_id,
        color=color,
        bbox=(left, top, left + size - 1, top + size - 1),
        centroid=(left + (size - 1) / 2, top + (size - 1) / 2),
        area=size * size,
        pixels=pixels,
    )


def _graph(left: int = 1, color: int = 3, frame_index: int = 0) -> ObjectGraph:
    node = _node(0, color, top=1, left=left)
    return ObjectGraph(nodes={0: node}, edges=[], frame_index=frame_index)


def _move_obs() -> list:
    extractor = EventExtractor()
    observations = []
    for t in range(3):
        pre = _graph(left=1 + t, frame_index=t)
        post = _graph(left=2 + t, frame_index=t + 1)
        observations.append(extractor.extract(ActionEnum.ACTION1, pre, post, timestamp=t))
    return observations


def _move_hypothesis(extra_rule: bool = False) -> Hypothesis:
    rules = [CausalRule([ActionIsCondition(ActionEnum.ACTION1)], [MoveEffect(0, 1, 0)])]
    if extra_rule:
        rules.append(CausalRule([ActionIsCondition(ActionEnum.ACTION7)], [RecolorEffect(0, 9)]))
    return Hypothesis(rules)


def test_perfect_hypothesis_scores_above_wrong_hypothesis():
    observations = _move_obs()
    perfect = _move_hypothesis()
    wrong = Hypothesis(
        [CausalRule([ActionIsCondition(ActionEnum.ACTION1)], [MoveEffect(0, -1, 0)])]
    )
    scorer = MDLScorer(lambda_mdl=0.1)

    assert scorer.score(perfect, observations) > scorer.score(wrong, observations)


def test_shorter_hypothesis_wins_when_accuracy_equal():
    observations = _move_obs()
    short = _move_hypothesis(extra_rule=False)
    long = _move_hypothesis(extra_rule=True)
    scorer = MDLScorer(lambda_mdl=0.1)

    assert scorer.log_likelihood(short, observations) == scorer.log_likelihood(long, observations)
    assert scorer.score(short, observations) > scorer.score(long, observations)


def test_minimum_support_returns_negative_infinity():
    scorer = MDLScorer()
    assert scorer.log_likelihood(_move_hypothesis(), []) == -math.inf
    assert scorer.score(_move_hypothesis(), []) == -math.inf


def test_rank_sorts_descending():
    observations = _move_obs()
    perfect = _move_hypothesis()
    wrong = Hypothesis(
        [CausalRule([ActionIsCondition(ActionEnum.ACTION1)], [MoveEffect(0, -1, 0)])]
    )
    ranked = MDLScorer(lambda_mdl=0.1).rank([wrong, perfect], observations)

    assert ranked[0][0].fingerprint() == perfect.fingerprint()


def test_entropy_single_hypothesis_is_zero():
    bank = HypothesisBank(seed_hypotheses=[_move_hypothesis()])
    assert bank.entropy() == 0.0


def test_entropy_uniform_scores_is_log_k():
    hypotheses = [
        _move_hypothesis(),
        Hypothesis([CausalRule([ActionIsCondition(ActionEnum.ACTION2)], [MoveEffect(0, 1, 0)])]),
        Hypothesis([CausalRule([ActionIsCondition(ActionEnum.ACTION3)], [MoveEffect(0, 1, 0)])]),
    ]
    bank = HypothesisBank(seed_hypotheses=hypotheses)
    bank.scores = {h.fingerprint(): 0.0 for h in bank.hypotheses}

    assert bank.entropy() == pytest.approx(math.log(3))


def test_bank_update_ranks_and_trims_mutations():
    perfect = _move_hypothesis()
    wrong = Hypothesis(
        [CausalRule([ActionIsCondition(ActionEnum.ACTION1)], [MoveEffect(0, -1, 0)])]
    )
    bank = HypothesisBank(seed_hypotheses=[wrong, perfect])

    for obs in _move_obs():
        bank.update(obs)

    assert bank.map_hypothesis().fingerprint() == perfect.fingerprint()
    assert 0.0 <= bank.map_confidence() <= 1.0
    assert len(bank.hypotheses) <= 50


def test_bank_sample_returns_weighted_pairs_and_mutations():
    bank = HypothesisBank(seed_hypotheses=[_move_hypothesis()])

    samples = bank.sample(5)
    mutations = bank.propose_mutations(bank.hypotheses[0])

    assert len(samples) == 5
    assert all(0.0 <= prob <= 1.0 for _, prob in samples)
    assert mutations
    assert len(mutations) <= 10
