"""
eig_selector.py - Expected Information Gain action selection.

Scores available environment actions by how much their predicted outcomes
would reduce posterior uncertainty over the current hypothesis bank. This is
internal compute only; no environment actions are taken here and no network
access is used.

Part of ACCA (Active Causal Compression Agent)
ARC Prize 2026 . Paper Track
"""
from __future__ import annotations

import math
from typing import Hashable

from src import config
from src.hypothesis.hypothesis_bank import HypothesisBank
from src.hypothesis.mechanic_dsl import Hypothesis
from src.perception.event_extractor import ActionEnum, AtomicEvent, canonical_delta
from src.perception.frame_parser import ObjectGraph


def _softmax(scores: list[float]) -> list[float]:
    if not scores:
        return []
    finite = [score for score in scores if math.isfinite(score)]
    if not finite:
        return [1.0 / len(scores)] * len(scores)
    max_score = max(finite)
    weights = [math.exp(score - max_score) if math.isfinite(score) else 0.0 for score in scores]
    total = sum(weights)
    if total <= 0:
        return [1.0 / len(scores)] * len(scores)
    return [weight / total for weight in weights]


def _entropy(probs: list[float]) -> float:
    return -sum(prob * math.log(prob) for prob in probs if prob > 0)


def _outcome_key(events: list[AtomicEvent]) -> tuple[Hashable, ...]:
    out = []
    for event in canonical_delta(events):
        params = tuple(sorted(event.params.items()))
        out.append((event.event_type, event.obj_id, params))
    return tuple(out)


class EIGSelector:
    """Selects the action with maximum expected posterior entropy reduction."""

    def __init__(self, n_samples: int | None = None):
        self.n_samples = config.EIG_N_SAMPLES if n_samples is None else n_samples

    def select_action(
        self,
        state: ObjectGraph,
        bank: HypothesisBank,
        action_space: list[ActionEnum | str],
    ) -> ActionEnum | str:
        if not action_space:
            raise ValueError("action_space must not be empty")
        scores = self.score_actions(state, bank, action_space)
        return max(action_space, key=lambda action: scores[action])

    def score_actions(
        self,
        state: ObjectGraph,
        bank: HypothesisBank,
        action_space: list[ActionEnum | str],
    ) -> dict[ActionEnum | str, float]:
        weighted = self._weighted_hypotheses(bank)
        if not weighted:
            return {action: 0.0 for action in action_space}
        prior_probs = [prob for _, prob in weighted]
        prior_entropy = _entropy(prior_probs)
        return {
            action: self._action_eig(state, weighted, action, prior_entropy)
            for action in action_space
        }

    def _weighted_hypotheses(self, bank: HypothesisBank) -> list[tuple[Hypothesis, float]]:
        if not bank.hypotheses or self.n_samples <= 0:
            return []
        if len(bank.hypotheses) <= self.n_samples:
            scores = [bank.scores.get(h.fingerprint(), 0.0) for h in bank.hypotheses]
            probs = _softmax(scores)
            return list(zip(bank.hypotheses, probs))

        sampled = bank.sample(self.n_samples)
        if not sampled:
            return []
        # Monte Carlo approximation: sampled hypotheses each carry equal mass.
        mass = 1.0 / len(sampled)
        return [(h, mass) for h, _ in sampled]

    def _action_eig(
        self,
        state: ObjectGraph,
        weighted: list[tuple[Hypothesis, float]],
        action: ActionEnum | str,
        prior_entropy: float,
    ) -> float:
        outcome_groups: dict[tuple[Hashable, ...], list[float]] = {}
        for hypothesis, prior_prob in weighted:
            outcome = _outcome_key(hypothesis.predict_delta(state, action))
            outcome_groups.setdefault(outcome, []).append(prior_prob)

        expected_posterior_entropy = 0.0
        for weights in outcome_groups.values():
            outcome_prob = sum(weights)
            posterior_probs = [weight / outcome_prob for weight in weights]
            expected_posterior_entropy += outcome_prob * _entropy(posterior_probs)
        return max(0.0, prior_entropy - expected_posterior_entropy)
