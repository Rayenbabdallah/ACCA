"""
mdl_scorer.py - MDL prior plus event-delta likelihood for hypotheses.

Part of ACCA (Active Causal Compression Agent)
ARC Prize 2026 . Paper Track
"""
from __future__ import annotations

import math

from src import config
from src.hypothesis.mechanic_dsl import Hypothesis
from src.perception.event_extractor import Observation, canonical_delta


class MDLScorer:
    """Scores executable hypotheses under a compactness prior."""

    def __init__(self, lambda_mdl: float | None = None):
        self.lambda_mdl = config.MDL_LAMBDA if lambda_mdl is None else lambda_mdl

    def log_prior(self, h: Hypothesis) -> float:
        return -self.lambda_mdl * h.description_length()

    def log_likelihood(self, h: Hypothesis, observations: list[Observation]) -> float:
        if len(observations) < config.MIN_SUPPORT:
            return -math.inf
        correct = 0
        total = len(observations)
        for obs in observations:
            predicted = canonical_delta(h.predict_delta(obs.pre_state, obs.action))
            observed = canonical_delta(obs.delta)
            if predicted == observed:
                correct += 1
        return math.log((correct + 1) / (total + 2))

    def score(self, h: Hypothesis, observations: list[Observation]) -> float:
        likelihood = self.log_likelihood(h, observations)
        if likelihood == -math.inf:
            return -math.inf
        return self.log_prior(h) + likelihood

    def rank(
        self,
        hypotheses: list[Hypothesis],
        observations: list[Observation],
    ) -> list[tuple[Hypothesis, float]]:
        scored = [(h, self.score(h, observations)) for h in hypotheses]
        return sorted(scored, key=lambda item: item[1], reverse=True)
