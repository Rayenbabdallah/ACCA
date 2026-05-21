"""
planner.py - Breadth-first planning over executable ACCA hypotheses.

Part of ACCA (Active Causal Compression Agent)
ARC Prize 2026 . Paper Track
"""
from __future__ import annotations

from collections import deque

from src import config
from src.hypothesis.goal_inference import GoalTemplate
from src.hypothesis.mechanic_dsl import Hypothesis
from src.perception.event_extractor import ActionEnum
from src.perception.frame_parser import ObjectGraph


def state_fingerprint(state: ObjectGraph) -> tuple:
    """Stable object-graph key for visited-state tracking."""
    nodes = []
    for obj_id, node in sorted(state.nodes.items()):
        nodes.append(
            (
                obj_id,
                node.color,
                node.bbox,
                tuple(sorted(node.pixels)),
            )
        )
    return tuple(nodes)


class Planner:
    """BFS planner using the MAP hypothesis as a transition model."""

    def __init__(self, max_depth: int = 12):
        self.max_depth = max_depth
        self.current_plan: list[ActionEnum | str] = []
        self.failure_count = 0
        self.last_predicted_state: ObjectGraph | None = None

    def compile_plan(
        self,
        state: ObjectGraph,
        hypothesis: Hypothesis,
        goal: GoalTemplate,
        action_space: list[ActionEnum | str] | None = None,
    ) -> list[ActionEnum | str] | None:
        actions = list(action_space or config.ACTION_SPACE)
        if goal.satisfied(state):
            return []

        queue = deque([(state, [])])
        visited = {state_fingerprint(state)}

        while queue:
            curr, plan = queue.popleft()
            if len(plan) >= self.max_depth:
                continue
            for action in actions:
                nxt = hypothesis.execute(curr, action)
                fp = state_fingerprint(nxt)
                if fp in visited:
                    continue
                next_plan = plan + [action]
                if goal.satisfied(nxt):
                    return next_plan
                visited.add(fp)
                queue.append((nxt, next_plan))
        return None

    def has_plan(self) -> bool:
        return len(self.current_plan) > 0

    def next_action(self) -> ActionEnum | str | None:
        if not self.current_plan:
            return None
        return self.current_plan.pop(0)

    def plan_failed(self) -> None:
        self.failure_count += 1
        self.current_plan = []
        self.last_predicted_state = None
