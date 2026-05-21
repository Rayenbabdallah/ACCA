# ARC Prize 2026 — Competition Brief

> Verified 2026-05-20 from the official Kaggle rules PDF, the Kaggle data page (direct quotes),
> and the ARC-AGI-3 Technical Report (2026-04-22). Supersedes any earlier inference from
> `arcprize.org` summary pages.

---

## Paper Track

**Format & legal**
- **Hackathon format: ONE submission per team only — no re-submissions.**
- Max team size: **8**; team must match the team submitting to ARC-AGI-2 or ARC-AGI-3.
- Tie-breaker: earlier submission wins.
- Sponsor: ARC Prize Foundation. Governing law: California.
- Winner license: **CC-BY 4.0** for the submission (Kaggle requirement).
- **CC0 or MIT-0** required for code to be prize-eligible (ARC Prize Foundation requirement) — BOTH apply.

**Prizes ($450K total)**
- Top Paper (guaranteed $75K): **$50K · $20K · $5K** (1st/2nd/3rd)
- Outstanding Papers Pool: up to **$375K** — divided equally among papers scoring >4.5/5 on rubric, at host discretion (NOT guaranteed)

**Judging — six equally-weighted criteria, 0–5 each**

| Criterion | Question |
|---|---|
| Accuracy | How accurate is the submission on the leaderboard? |
| Universality | How general is the approach beyond the competition? |
| Progress | How much does the paper increase the field's chance of reaching 85% on ARC-AGI? |
| Theory | How well does the paper describe **why** the approach works? |
| Completeness | How thoroughly does the paper cover the submission? |
| Novelty | How novel is the approach relative to existing public research? |

**Writeup**
- Required sections: Abstract · Introduction · Prior Work · Approach · Results · Conclusion
- Style: *"Shorter and clearer is always better. No filler, no unnecessary equations."*
- CLAUDE.md applies a self-imposed ≤1,500 word limit.
- Paper must link to a Kaggle submission in ARC-AGI-2 or ARC-AGI-3. Code score need NOT be high.

**Deadlines**
- Code: **2026-11-02**
- Paper: **2026-11-08**

---

## ARC-AGI-3 Track

**Datasets**

| Set | Count | Access | Purpose |
|---|---|---|---|
| Public Demo | 25 | Kaggle `environment_files/` | Format demo, dev only |
| Competition — Public LB | 55 | Private, never seen | Public Leaderboard score |
| Competition — Private LB | 55 | Private, never seen | **Final ranking** |

**Total competition evaluation: 110 private games.** Private set is intentionally OOD from the public 25 — do not over-fit on public mechanics.

**Hard constraints**
- **No internet** at evaluation. SDK installed from `arc_agi_3_wheels/` (provided on the data page).
- All code & methods **open-sourced under CC0 or MIT-0** before receiving private scores.
- No GPT/Claude/etc. API calls during evaluation.
- Third-party compute (Modal/Lambda) IS allowed at Kaggle evaluation time.
- Hardware/compute limits TBA at competition launch.

**Observation space (direct quote, Kaggle data page)**
> *"Each frame includes a grid (max 64×64) with integer cell values 0–15 representing different states/colors, using a (0,0) top-left coordinate system."*

Grids are **max 64×64**, not always. Games use smaller grids. The frame parser MUST read dimensions from `frame.shape` — never hardcode 64.

**Actions (direct quote, Kaggle data page)**
> *"Agents interact with environments using up to 7 actions"*

| Action | Description |
|---|---|
| RESET | Start or restart the game |
| ACTION1–ACTION5 | Simple actions — meaning varies per game |
| ACTION6 | **Complex action requiring (x, y) coordinates** |
| ACTION7 | "Undo" — only in games that support it |

> *"Each game defines which actions are available and what they do. The meaning of actions varies per game — your agent must figure out what each action does through exploration."*

This is the reason EIG-driven exploration is necessary.

**Scoring (Kaggle data page — authoritative; supersedes the Tech Report's 1.15 cap)**
```python
def level_score(human_actions, agent_actions):
    return min(human_actions / agent_actions, 1.0) ** 2          # cap = 1.0

def game_score(level_scores, total_levels):                       # 1-indexed weights
    padded  = level_scores + [0.0] * (total_levels - len(level_scores))
    weights = range(1, total_levels + 1)
    return sum(w * s for w, s in zip(weights, padded)) / sum(weights)

def total_score(game_scores):
    return sum(game_scores) / len(game_scores)                    # mean of 110 games
```

**Key implication — late levels dominate**
Every ARC-AGI-3 game has **minimum 6 levels** (Tech Report §3.4). With weights 1..6 (sum=21):
- L1 only → max 1/21 ≈ **4.8%**
- L1–L3 → 6/21 ≈ **28.6%**
- All 6 → **100%**

L6 (weight 6) is worth 6× L1. Plan for level progression, not just first-level solves.

**Environment design rules (Tech Report §3.4)**
- Minimum 6 levels per environment
- Tutorial level 1 — easy, beatable by random agents occasionally
- Multi-mechanic — single-mechanic is an explicit anti-pattern
- Core Knowledge priors only: objectness, geometry/topology, physics, agentness
- No language, numbers, letters, or cultural symbols
- Random agents cannot beat non-tutorial levels >1 in 10,000 times
- Difficulty through composition of mechanics, not obscurity
- **Agent is never told the win condition** — must be inferred

**Prizes**
- **Grand Prize: $700K** for the first agent scoring 100%
- **Top Score: $75K** (1st–5th)
- Milestone #1 (**2026-06-30**): $25K · $10K · $2.5K
- Milestone #2 (**2026-09-30**): $25K · $10K · $2.5K

**Agent architecture (official)**
```python
from agents.agent import Agent
from agents import Swarm
from arc_agi_3 import Arcade, OperationMode

class ACCAAgent(Agent):
    def is_done(self, frames, latest_frame): ...
    def choose_action(self, frames, latest_frame): ...

arc   = Arcade(operation_mode=OperationMode.COMPETITION)
swarm = Swarm(agent_class=ACCAAgent, arcade=arc)
swarm.run()    # parallel across all 110 games
```

**SDK install in Kaggle (no internet)**
```python
import subprocess, glob
for w in glob.glob('/kaggle/input/arc-prize-2026-arc-agi-3/arc_agi_3_wheels/*.whl'):
    subprocess.run(['pip', 'install', w, '--quiet'])
```
Do **NOT** `pip install arc-agi` — will fail in competition runtime.

---

## What still needs confirming
- [ ] Hardware/compute limits (TBA at competition launch)
- [ ] Exact code & paper submission cutoff times (dates known: 2026-11-02 / 2026-11-08)
