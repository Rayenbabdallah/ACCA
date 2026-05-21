"""
hypothesis_bank.py - Posterior-ranked hypothesis store with simple mutations.

Part of ACCA (Active Causal Compression Agent)
ARC Prize 2026 . Paper Track
"""
from __future__ import annotations

import math
import random
from dataclasses import replace

from src import config
from src.hypothesis.hypothesis_seeds import load_seeds
from src.hypothesis.mdl_scorer import MDLScorer
from src.hypothesis.mechanic_dsl import (
    ActionIsCondition,
    CausalRule,
    DestroyEffect,
    Effect,
    Hypothesis,
    MoveEffect,
    ObjColorCondition,
    RecolorEffect,
    SpawnEffect,
    ToggleEffect,
)
from src.perception.event_extractor import ActionEnum, Observation


def _softmax(scores: list[float]) -> list[float]:
    if not scores:
        return []
    finite = [score for score in scores if math.isfinite(score)]
    if not finite:
        return [1 / len(scores)] * len(scores)
    max_score = max(finite)
    weights = [math.exp(score - max_score) if math.isfinite(score) else 0.0 for score in scores]
    total = sum(weights)
    if total <= 0:
        return [1 / len(scores)] * len(scores)
    return [weight / total for weight in weights]


def _dedupe(hypotheses: list[Hypothesis]) -> list[Hypothesis]:
    seen: set[str] = set()
    out: list[Hypothesis] = []
    for h in hypotheses:
        fp = h.fingerprint()
        if fp in seen:
            continue
        seen.add(fp)
        out.append(h)
    return out


def _seed_rule_library() -> list[CausalRule]:
    return [rule for hypothesis in load_seeds() for rule in hypothesis.rules]


class HypothesisBank:
    """Maintains a top-K bank ranked by MDL posterior score."""

    def __init__(self, seed_hypotheses: list[Hypothesis] | None = None):
        self.hypotheses: list[Hypothesis] = _dedupe(seed_hypotheses or load_seeds())
        self.scores: dict[str, float] = {h.fingerprint(): 0.0 for h in self.hypotheses}
        self.scorer = MDLScorer()
        self.observations: list[Observation] = []
        self._rule_library = _seed_rule_library()

    def clear_observations(self) -> None:
        self.observations.clear()
        self.scores = {h.fingerprint(): 0.0 for h in self.hypotheses}

    def update(self, observation: Observation) -> None:
        self.observations.append(observation)
        ranked = self.scorer.rank(self.hypotheses, self.observations)
        self.scores = {h.fingerprint(): score for h, score in ranked}

        candidates = [h for h, _ in ranked[:5]]
        mutations: list[Hypothesis] = []
        for h in candidates:
            mutations.extend(self.propose_mutations(h))

        combined = _dedupe(self.hypotheses + mutations)
        ranked_combined = self.scorer.rank(combined, self.observations)
        trimmed = ranked_combined[: config.HYPOTHESIS_BANK_SIZE]
        self.hypotheses = [h for h, _ in trimmed]
        self.scores = {h.fingerprint(): score for h, score in trimmed}

    def entropy(self) -> float:
        if len(self.hypotheses) <= 1:
            return 0.0
        probs = _softmax([self.scores.get(h.fingerprint(), 0.0) for h in self.hypotheses])
        return -sum(p * math.log(p) for p in probs if p > 0)

    def map_hypothesis(self) -> Hypothesis:
        if not self.hypotheses:
            raise ValueError("hypothesis bank is empty")
        return max(self.hypotheses, key=lambda h: self.scores.get(h.fingerprint(), 0.0))

    def map_confidence(self) -> float:
        if not self.hypotheses:
            return 0.0
        scores = [self.scores.get(h.fingerprint(), 0.0) for h in self.hypotheses]
        probs = _softmax(scores)
        best_idx = max(range(len(scores)), key=lambda i: scores[i])
        return probs[best_idx]

    def sample(self, n: int) -> list[tuple[Hypothesis, float]]:
        if n <= 0 or not self.hypotheses:
            return []
        scores = [self.scores.get(h.fingerprint(), 0.0) for h in self.hypotheses]
        probs = _softmax(scores)
        choices = random.choices(self.hypotheses, weights=probs, k=n)
        prob_by_fp = {h.fingerprint(): p for h, p in zip(self.hypotheses, probs)}
        return [(h, prob_by_fp[h.fingerprint()]) for h in choices]

    def propose_mutations(self, h: Hypothesis) -> list[Hypothesis]:
        mutations: list[Hypothesis] = []

        if h.rules:
            for idx in range(min(len(h.rules), 3)):
                rules = [rule for j, rule in enumerate(h.rules) if j != idx]
                if rules:
                    mutations.append(Hypothesis(rules))

        for rule in self._rule_library[:4]:
            mutations.append(Hypothesis(h.rules + [rule]))

        for idx, rule in enumerate(h.rules[:3]):
            for changed_rule in self._mutate_rule(rule):
                rules = list(h.rules)
                rules[idx] = changed_rule
                mutations.append(Hypothesis(rules))

        return _dedupe(mutations)[:10]

    def _mutate_rule(self, rule: CausalRule) -> list[CausalRule]:
        out: list[CausalRule] = []
        for idx, condition in enumerate(rule.conditions):
            changed = self._mutate_condition(condition)
            if changed is None:
                continue
            conditions = list(rule.conditions)
            conditions[idx] = changed
            out.append(CausalRule(conditions, list(rule.effects)))

        for idx, effect in enumerate(rule.effects):
            changed = self._mutate_effect(effect)
            if changed is None:
                continue
            effects = list(rule.effects)
            effects[idx] = changed
            out.append(CausalRule(list(rule.conditions), effects))
        return out

    def _mutate_condition(self, condition) -> object | None:
        if isinstance(condition, ActionIsCondition):
            action = condition.action
            current_idx = list(ActionEnum).index(ActionEnum(action))
            return ActionIsCondition(list(ActionEnum)[(current_idx + 1) % len(ActionEnum)])
        if isinstance(condition, ObjColorCondition):
            return replace(condition, color=(condition.color % 15) + 1)
        return None

    def _mutate_effect(self, effect: Effect) -> Effect | None:
        if isinstance(effect, MoveEffect):
            return MoveEffect(effect.obj_id, -effect.dx, -effect.dy)
        if isinstance(effect, RecolorEffect):
            return replace(effect, new_color=(effect.new_color % 15) + 1)
        if isinstance(effect, ToggleEffect):
            return ToggleEffect(effect.obj_id, effect.color_b, effect.color_a)
        if isinstance(effect, DestroyEffect):
            return SpawnEffect(color=1, position=(0, 0), obj_id=effect.obj_id)
        return None
