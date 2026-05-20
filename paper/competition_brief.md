# ARC Prize 2026 — Competition Brief

> Extracted from `arcprize.org/competitions/2026/{paper,arc-agi-3}` on 2026-05-20.
> Re-fetch before final submission — some details ("hardware limits TBA") are still pending.

---

## Paper Track ($450K)

**Prize structure**
- **Top Paper (guaranteed $75K):** 1st $50K · 2nd $20K · 3rd $5K
- **Outstanding Paper Pool: $375K** — distributed at host discretion to papers scoring **>4.5** in any category

**Judging — six equally-weighted criteria, scored 0–5 each:**

| Criterion | Question |
|---|---|
| Accuracy | How accurate is the submission on the leaderboard? |
| Universality | How general is the approach beyond the competition? |
| Progress | How much does the paper increase the field's chance of reaching 85% on ARC-AGI? |
| Theory | How well does the paper describe **why** the approach works? |
| Completeness | How thoroughly does the paper cover the submission? |
| Novelty | How novel is the approach relative to existing public research? |

**Writeup requirements**
- Required sections: Abstract · Introduction · Prior Work · Approach · Results · Conclusion
- Length: not explicitly capped on the rules page; CLAUDE.md applies a self-imposed **≤1,500 word** limit
- Style guidance (verbatim): *"Shorter and clearer is always better. No filler, no unnecessary equations."*
- Must link to a Kaggle submission in **either** the ARC-AGI-2 or ARC-AGI-3 track. *Code submission need not achieve a high score for the paper to be eligible.*

---

## ARC-AGI-3 Track ($700K Grand + $135K side prizes)

**Prize structure**
- **Grand Prize: $700K** for the first agent scoring **100%** on ARC-AGI-3 evaluation
- **Top Score Award: $75K** distributed 1st–5th place
- **Milestone #1 (2026-06-30): $37.5K** — 1st $25K · 2nd $10K · 3rd $2.5K
- **Milestone #2 (2026-09-30): $37.5K** — same split
- **Total side prizes: $150K** (top-score + 2 milestones)

**Hard constraints**
- **No internet access** at evaluation time
- All code & methods must be **open-sourced** to be eligible for prizes
- Hardware / compute limits: **TBA at competition launch** (CLAUDE.md assumes $0.42/task × 120 tasks = ~$50)

**RHAE formula** (from CLAUDE.md, confirmed by Rodionov 2026 reporting `mean RHAE 32.58%`):
```
rhae(h, a) = min((h / a) ** 2, 1.15)        # h = human actions, a = AI actions
score      = mean(rhae across all levels)
```

**Reported baselines (as of 2026-03)**
- Frontier AI systems: **below 1%** mean RHAE
- Humans: **100%** by construction (RHAE = 1.0 when AI matches human action count)
- Rodionov 2026 on public 25-game suite: **7 fully solved, 6 >75% RHAE, mean 32.58%**

---

## Key dates (to confirm on Kaggle pages once accessible)

| Date | Event |
|---|---|
| 2026-06-30 | Milestone #1 leaderboard snapshot |
| 2026-09-30 | Milestone #2 leaderboard snapshot |
| TBD | Final submission deadline (likely ~Nov 2026 based on prior years) |
| TBD | Paper Track submission deadline |

---

## What still needs confirming

- [ ] Exact hardware/compute budget per task (Kaggle infra TBA)
- [ ] Final submission and paper deadlines
- [ ] Whether the leaderboard split matches the CLAUDE.md "120 tasks" assumption
- [ ] Whether GPU is provided or CPU-only at eval
- [ ] LLM-API rules: confirmed *implicitly* by "no internet" → cannot call hosted APIs; *unstated* whether a locally-cached small LM is permitted (relevant for SOAR-style methods, not ACCA)
