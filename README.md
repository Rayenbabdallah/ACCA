# ACCA — Active Causal Compression Agent
### ARC Prize 2026 · Paper Track · ARC-AGI-3

> **One-line thesis:** An agent that acts like a scientist with a minimal experimental budget — choosing actions that maximally compress its causal world model, then compiling that model into an efficient plan.

---

## What This Project Is

ACCA is a research system and Kaggle submission targeting **1st place in the ARC Prize 2026 Paper Track** ($50K) and a competitive leaderboard score on the **ARC-AGI-3** interactive benchmark.

ARC-AGI-3 presents an agent with novel, instruction-free interactive environments rendered as 64×64 16-color grids. The agent must discover mechanics, infer goals, and complete levels using as few environment actions as possible. Scoring is **Relative Human Action Efficiency (RHAE)**: `score = (human_actions / AI_actions)²` per level — action waste is penalized quadratically.

The Paper Track judges 6 equal criteria: **Accuracy, Universality, Progress, Theory, Completeness, Novelty**. In 2025, the 1st-place paper (TRM, 8% accuracy) beat the highest-scoring Kaggle entry (NVARC, 24% accuracy) because Theory + Novelty + Universality dominate. ACCA is designed to score high on all six.

---

## The Core Idea

Current frontier LLMs score below 1% on ARC-AGI-3. The official ARC Prize failure analysis (May 2026) identifies three root causes:

1. **False world models** — agents act on incorrect beliefs about mechanics
2. **Wrong abstraction transfer** — spurious analogies from training data
3. **No cross-level learning** — every level starts from scratch

ACCA attacks all three directly with five integrated components:

```
Frame stream + action outcomes
        ↓
[1] Object & Relation Extractor     → structured graph G=(V,E)
        ↓
[2] Typed Causal Hypothesis Engine  → posterior P(h | observations)
        ↓
[3] MDL Posterior Scorer            → favors shortest causal explanation
        ↓
[4] EIG Action Selector             → minimizes posterior entropy per action
        ↓
[5] Cross-Level Mechanic Memory     → warm-starts hypothesis bank on next level
        ↓
   Planner Compiler                 → compiles MAP hypothesis → symbolic plan
```

The key formal objective — what makes this theoretically novel — is:

> **Choose the next action that maximally reduces description length and uncertainty of the causal world model per unit environment action cost.**

This is Bayesian experimental design (Chaloner & Verdinelli 1995) applied to an interactive grid environment, grounded in Chollet's ARI theory (Kolmogorov complexity / algorithmic information theory as the measure of intelligence).

---

## Why This Wins the Paper Track

| Rubric Criterion | ACCA's Answer |
|---|---|
| **Accuracy** | EIG directly optimizes RHAE. Every internal compute cycle spent saves environment actions. |
| **Theory** | Formal derivation: MDL prior + Bayesian posterior + EIG = principled Bayesian experimental design. Answers *why* it works, not just how. |
| **Novelty** | No published system combines typed causal DSL + EIG exploration + cross-level memory as a unified interactive agent. |
| **Universality** | Same mechanism tested on ARC-AGI-2 via static bridge. MDL/Bayesian framing extends to any novel-mechanic environment. |
| **Progress** | Directly addresses the 3 official ARC-AGI-3 failure modes. Clearest path toward the 85% ARC goal. |
| **Completeness** | Full ablation table (5 conditions), ARC-AGI-2 universality experiment, reproducibility repo, competition-mode results. |

---

## How ACCA Differs from Prior Work

| Prior Work | What They Do | ACCA's Differentiation |
|---|---|---|
| **Rodionov 2026** (arXiv:2605.05138, 32.58% RHAE) | GPT-5 coding agent with informal "refactoring as MDL proxy" heuristics | ACCA uses a **formal** Bayesian posterior over a typed DSL with mathematically grounded MDL prior and principled EIG exploration — not coding-agent heuristics |
| **TRM** (arXiv:2510.04871, 2025 paper 1st) | Tiny recursive model for static ARC prediction | Static predictor with no action interface. ACCA borrows the recursive refinement insight and derives it as approximate Bayesian inference |
| **CompressARC** (arXiv:2512.06104, 2025 paper 3rd) | MDL via gradient descent on a VAE for static ARC-AGI-2 | Static MDL. ACCA makes MDL **action-selective**: the agent actively chooses interventions to minimize description length, not just gradient-descend on a fixed puzzle |
| **StochasticGoose / search-only agents** | Discover mechanics by chance via random/heuristic search | EIG exploration discovers mechanics with the **minimum possible** environment actions, directly optimizing RHAE's quadratic penalty |
| **SOAR** (2025 paper 2nd) | Self-improving evolutionary program synthesis | Static program search on ARC-AGI-1/2. No interactive mechanic discovery, no cross-level memory |

---

## Positioning Statement

> *Static ARC winners proved that intelligence on ARC requires adaptation. ARC-AGI-3 proves that adaptation without explicit world-model formation still fails. Therefore the next dominant step is action-efficient causal world-model induction.*

---

## Competition Context

| Item | Detail |
|---|---|
| **Competition** | ARC Prize 2026 — Paper Track + ARC-AGI-3 Track |
| **Code deadline** | November 2, 2026 (11:59 PM UTC) |
| **Paper deadline** | November 8, 2026 (11:59 PM UTC) — treat this as hard stop |
| **Results** | December 4, 2026 |
| **Paper prize** | $50K (1st), $20K (2nd), $5K (3rd) + $375K outstanding papers pool |
| **Tie-breaker** | Earlier submission wins — submit early drafts strategically |
| **Open source** | Required for prize eligibility — MIT license, all authored code |
| **No internet** | Kaggle evaluation has no internet access — no API calls at eval time |
| **Compute budget** | ~$50 for 120 evaluation tasks (~$0.42/task) |
| **Paper word limit** | ≤1,500 words for Kaggle writeup (arXiv version can be full-length) |
| **Milestones** | June 30 (M1), September 30 (M2) — submit early for milestone prizes |

---

## Roadmap Overview

```
Phase 0  │ May 20–31    │ Competition-mode replica + synthetic red-team suite
Phase 1  │ Jun 1–12     │ Object & state abstraction (frame parser, event deltas)
Phase 2  │ Jun 1–22     │ Causal hypothesis engine + MDL posterior scorer
Phase 3  │ Jun 8–22     │ EIG exploration policy + planner → MILESTONE 1 (Jun 30)
Phase 4  │ Jul 1–20     │ Cross-level mechanic memory + planner compiler
Phase 5  │ Jul 25–Aug 20│ Full ablation sweep + ARC-AGI-2 universality → MILESTONE 2 (Sep 30)
Phase 6  │ Oct 1–Nov 8  │ Final hardening + paper writing → FINAL SUBMISSION
```

**Decision gate — July 15:** If EIG is not cleanly beating random exploration on the synthetic red-team suite, pivot to fallback strategy (TRM + CompressARC unification paper on ARC-AGI-2). Don't wait beyond this date to make the call.

---

## Fallback Strategy

If the full causal-compression stack underperforms by mid-July:

**Fallback:** Derive TRM's recursive refinement as approximate Bayesian inference under an MDL prior. Build a unified TRM+CompressARC hybrid. Submit on ARC-AGI-2.

- Lower accuracy ceiling (~10–15% ARC-AGI-2)
- Higher Theory score (clean derivation nobody has published)
- 3-month build instead of 6
- Both TRM and CompressARC codebases are fully open-source

Do not switch paradigms entirely. Narrow the ambition while preserving the theoretical thesis.

---

## Repository Structure

```
arc-acca/
├── README.md                    ← This file
├── CLAUDE.md                    ← Tech stack, architecture, dev conventions
├── PLAN.md                      ← Detailed step-by-step build plan
├── PROMPTS.md                   ← Claude Code session prompts
├── REPRODUCIBILITY.md           ← One-command reproduction guide (written last)
├── LICENSE                      ← MIT
│
├── src/
│   ├── perception/
│   │   ├── frame_parser.py      ← 64×64 frame → object graph G=(V,E)
│   │   ├── object_tracker.py    ← Hungarian matching across frames
│   │   └── event_extractor.py   ← (action, pre, post) → symbolic delta
│   │
│   ├── hypothesis/
│   │   ├── mechanic_dsl.py      ← Typed causal hypothesis language
│   │   ├── hypothesis_bank.py   ← Top-K posterior hypothesis store
│   │   ├── mdl_scorer.py        ← MDL prior + likelihood → posterior
│   │   └── goal_inference.py    ← Goal hypothesis set + update
│   │
│   ├── planning/
│   │   ├── eig_selector.py      ← Expected Information Gain action selector
│   │   ├── planner.py           ← BFS on predicted state transitions
│   │   ├── policy_compiler.py   ← MAP hypothesis → symbolic program
│   │   └── mpc_controller.py    ← Model predictive control loop
│   │
│   ├── memory/
│   │   ├── mechanic_memory.py   ← Cross-level mechanic store
│   │   ├── memory_index.py      ← Fingerprint indexing + retrieval
│   │   └── mechanic_transfer.py ← Cross-game structural similarity transfer
│   │
│   ├── agent.py                 ← Top-level agent loop
│   └── config.py                ← Hyperparameters + constants
│
├── envs/
│   ├── synthetic/               ← Red-team holdout environments (never train on)
│   └── arc_agi3_bridge.py       ← ARC-AGI-3 competition-mode interface
│
├── eval/
│   ├── local_eval.py            ← Competition-mode local scorer
│   ├── scorecard.py             ← RHAE computation
│   └── ablation_runner.py       ← 5-condition ablation sweep
│
├── experiments/
│   ├── ablations/               ← Results CSVs, plots
│   └── universality/            ← ARC-AGI-2 universality bridge results
│
├── paper/
│   ├── kaggle_writeup.md        ← ≤1,500 word competition writeup
│   ├── arxiv_paper.tex          ← Full arXiv paper
│   └── figures/                 ← All paper figures
│
├── kaggle/
│   └── submission.ipynb         ← Final Kaggle notebook
│
├── tests/
│   ├── test_frame_parser.py
│   ├── test_mdl_scorer.py
│   ├── test_eig_selector.py
│   └── test_mechanic_memory.py
│
└── requirements.txt
```

---

## Key Papers to Cite

### Must-cite (official sources)
- ARC Prize 2026 overview (dates, open-source rules, compute constraints)
- ARC-AGI-3 technical report arXiv:2603.24621 (benchmark design, RHAE, failure modes)
- ARC Prize 2025 technical report arXiv:2601.10904 (context, past winners)
- Chollet, "On the Measure of Intelligence" arXiv:1911.01547 (theoretical grounding)

### Direct comparison class
- Rodionov 2026 arXiv:2605.05138 (Executable World Models — closest competitor)
- TRM arXiv:2510.04871 (2025 paper 1st — recursive refinement)
- SOAR arXiv:2507.14172 (2025 paper 2nd — evolutionary synthesis)
- CompressARC arXiv:2512.06104 (2025 paper 3rd — MDL ancestor)
- Li et al., "Combining Induction and Transduction" (2024 paper winner)
- Akyürek et al., "Surprising Effectiveness of TTT"

### Theoretical foundations
- Chaloner & Verdinelli 1995 (Bayesian experimental design / EIG)
- Rissanen 1978 (MDL / Minimum Description Length)
- Friston 2017 (Active inference / Free Energy Principle — cite as related work)
- Ellis et al. 2021 DreamCoder (library learning inspiration)

---

## Success Metrics

| Metric | Target | Minimum |
|---|---|---|
| Mean RHAE on synthetic red-team suite | >20% | >10% |
| Mean RHAE on Kaggle public demos | >32% (beat Rodionov) | >15% |
| Ablation delta (full vs no EIG) | >10% RHAE | >5% |
| ARC-AGI-2 coverage | >40% task types | >20% |
| Paper Theory score (self-assessed) | 5/5 | 4/5 |
| Kaggle writeup word count | ≤1,500 | ≤1,500 |
