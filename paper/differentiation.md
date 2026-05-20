# ACCA vs. Rodionov 2026 — Anchor Differentiation

> The whole paper hangs on these two sentences. Refine but do not weaken.

**Rodionov 2026** synthesizes a *single* Python world model per task via LLM code-generation, refines it through an observe-fit-refactor loop with an MDL-flavored simplicity bias, then hands the resulting program to a scripted controller that plays the inferred plan — making the agent fundamentally **reactive** to whatever the LLM happened to propose.

**ACCA** replaces that pipeline with a **Bayesian posterior over a typed causal DSL** scored by *formal* MDL, and selects every environment action by **Expected Information Gain** over the current hypothesis bank — making the agent **actively probing**: each action is the experiment that maximally collapses model uncertainty, with no LLM in the inner loop and a uniform exploration→exploitation handoff governed by posterior entropy.

---

## Why this framing is defensible across all six Paper Track criteria

| Criterion | ACCA's claim |
|---|---|
| **Novelty** | EIG-driven action selection on top of a typed causal DSL is new for ARC-AGI-3. Rodionov, SOAR, CompressARC, TRM all do **passive** scoring (best-of-many-proposals). |
| **Theory** | EIG has a clean Bayesian-experimental-design grounding (Chaloner & Verdinelli 1995); MDL gives an exact posterior factorization. Rodionov's "refactor toward simplicity" is heuristic. |
| **Universality** | Typed DSL is finite (20 primitives). Same engine runs on ARC-AGI-2 induction tasks by collapsing the action space to {∅}. Rodionov's Python code is per-game. |
| **Progress** | EIG turns each environment action into a measured uncertainty reduction — the metric is interpretable as bits-per-action, which directly addresses the "skill-acquisition efficiency" target Chollet 2019 set for ARC. |
| **Completeness** | Ablation table: NO_EIG, NO_MDL, NO_MEMORY, NO_OBJECTS. Each row maps to a paper claim. |
| **Accuracy** | Demonstrated on Kaggle leaderboard via the same ACCA binary that produced the paper's tables. |
