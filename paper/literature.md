# Literature Notes — ACCA Prior Work

> Structured 2-paragraph summaries: *what they do* vs *why ACCA differs*. Cite all of these
> in the Paper Track writeup's Prior Work section (~180 words budget total).

---

## 1. Rodionov 2026 — *Executable Python World Models for ARC-AGI-3* (arXiv:2605.05138)

**What they do.** Per ARC-AGI-3 task, the system prompts an LLM to emit a Python program representing the inferred world model, validates the program against observed transitions, and iteratively refactors it toward simpler abstractions ("MDL-like" but heuristic — no formal scoring). A scripted controller turns the program into actions. Reports **7/25 fully solved on the public game suite, 6 games exceeding 75% RHAE, mean RHAE 32.58%** (Rodionov's RHAE uses the Tech Report's 1.15 cap; the authoritative Kaggle-page cap of 1.0 yields slightly lower numbers). This is the main competing baseline in the ARC-AGI-3 paper-track race.

**Why ACCA differs.** Hypotheses are stored as a *bank of typed DSL programs* with an exact Bayesian posterior under a formal MDL prior, not as a single LLM-emitted Python file. Action selection is *actively probing* via Expected Information Gain — Rodionov's controller is a script over a one-shot synthesized model and cannot trade off exploration against exploitation. We pay zero LLM tokens per environment action, which keeps us tractable on the no-internet Kaggle runtime where Rodionov's LLM dependency would fail outright.

---

## 2. TRM — *Tiny Recursive Model* (Jolicoeur-Martineau 2025, arXiv:2510.04871) · **1st place, ARC Prize 2025 Paper Track**

**What they do.** A 2-layer, 7M-parameter network trained recursively on ~1k examples solves ARC-AGI puzzles by iterating its own output as input. Reports **45% on ARC-AGI-1, 8% on ARC-AGI-2**, beating Deepseek R1 / o3-mini / Gemini 2.5 Pro at <0.01% the parameters. The headline contribution is *sample efficiency*: small data + small model + recursive depth.

**Why ACCA differs.** TRM is a **static input-output predictor** — there is no environment, no action space, no exploration. It cannot run on ARC-AGI-3 as-is: the task is sequential decision-making, not pattern completion. ACCA's universality claim is that the *same hypothesis-bank engine* applies to both regimes (ARC-AGI-3 with a non-trivial action set; ARC-AGI-2 with action set {∅}), which TRM's architecture does not support.

---

## 3. CompressARC — *Pure MDL at Inference Time* (arXiv:2512.06104) · **3rd place, ARC Prize 2025 Paper Track**

**What they do.** A 76K-parameter model trained *only on the target puzzle* — no ARC training data, no pretraining transfer — minimizes the description length of the puzzle representation during inference. Solves **20% of ARC-AGI-1 evaluation** under this extreme zero-prior setting. Closest method to ACCA in spirit: MDL as the scoring principle.

**Why ACCA differs.** CompressARC compresses *a static example*; ACCA compresses *the transition function of an environment*, scored against observed (state, action, next-state) tuples. We also keep a *posterior distribution* over hypotheses, not a single MAP minimizer, because EIG requires entropy over the bank to choose the next action. The shared MDL idea lets us cite CompressARC as the closest static analogue, then claim the interactive extension as novel.

---

## 4. SOAR — *Self-Improving LMs for Evolutionary Program Synthesis* (Pourcel et al., ICML 2025, arXiv:2507.14172)

**What they do.** Alternates an LLM-driven evolutionary search over Python programs with a *hindsight learning* phase that fine-tunes the LLM on (problem, found-solution) pairs harvested from the search itself. Each iteration produces a stronger sampler/refiner LLM. **Solves 52% of the ARC-AGI-1 public test set** at scale.

**Why ACCA differs.** SOAR requires fine-tuning a 32B LLM across generations — incompatible with no-internet Kaggle eval (the SDK is installed from offline wheels; outbound network is blocked). ACCA's bank-mutation operator (`propose_mutations(h)`) is a deterministic, gradient-free symbolic edit on a 20-token DSL; the equivalent of SOAR's "refinement" runs in microseconds with no GPU. SOAR is also a static-task method — same gap to ARC-AGI-3 as TRM and CompressARC.

---

## 5. Li, Ellis et al. 2024 — *Combining Induction and Transduction* (arXiv:2411.02272) · **1st place, ARC Prize 2024 Paper Track**

**What they do.** Trains two neural models on synthetic Python program variations: an *inductive* model that infers a latent transformation function, and a *transductive* model that directly predicts the test output. They find — striking, given identical training data and architecture — that the two models solve **disjoint** subsets of ARC tasks. The ensemble approaches human-level on ARC-AGI-1. Induction wins on precise compositional rules; transduction wins on fuzzy perceptual rules.

**Why ACCA differs.** Both their inductive and transductive paths are *neural*; ACCA's induction path is a *symbolic* DSL with explicit MDL scoring, which makes the posterior interpretable and the action-selection rule (EIG) tractable. Their finding that "different representations solve different problems" justifies our memory module (`mechanic_memory.py`) — storing the MAP hypothesis per game lets us share that memory across levels, which their stateless evaluator cannot.

---

## 6. Akyürek et al. 2024 — *The Surprising Effectiveness of Test-Time Training* (arXiv:2411.07279)

**What they do.** Temporarily updates an 8B LM's weights at inference using a loss derived from the in-context examples (the few-shot training pairs of an ARC task). Reports **53% on ARC-AGI with TTT alone, 61.9% ensembled with program synthesis**. The lesson is that adaptation at inference is worth far more than scale.

**Why ACCA differs.** TTT adapts a *neural model's weights*; ACCA adapts a *symbolic posterior* — strictly cheaper, exactly invertible, and immune to catastrophic forgetting across levels. Same critique as TRM/SOAR: TTT targets static few-shot tasks and has no notion of an environment to act in. We do, however, share the central insight that *adaptation per task* beats *one fixed model* — this is what `mechanic_memory.warm_start()` does in our system.

---

## 7. Chollet 2019 — *On the Measure of Intelligence* (arXiv:1911.01547)

**What they do.** Defines intelligence as **skill-acquisition efficiency** — the rate at which a system learns new competencies, normalized by its priors and training experience. Argues that benchmarks must be grounded in *innate human priors* (Core Knowledge) to fairly compare AI to humans. This paper is the theoretical foundation for the entire ARC benchmark family.

**Why ACCA cites this.** EIG-per-environment-action is a *direct operationalization* of skill-acquisition efficiency: bits of model uncertainty resolved per costly action. The Kaggle level-score (`min(h/a,1.0)²`) further normalizes for the human-prior reference point Chollet specifies. ACCA is, by construction, the agent Chollet's framework predicts: priors over a DSL, posterior updated by observations, decisions selected to maximize information gain per unit experience.

---

## 8. Chaloner & Verdinelli 1995 — *Bayesian Experimental Design* (statistical foundation)

**What to cite (Section 2 only).** EIG as the expected reduction in posterior entropy: `EIG(a) = H(p(h|D)) − E_{y|a}[H(p(h|D,a,y))]`. The decision-theoretic justification: under a 0-1 utility on the unknown parameter, EIG is the unique Bayes-optimal experiment-selection criterion. This is the formal grounding that lets ACCA claim its action policy is principled, not heuristic — addressing the **Theory** Paper Track criterion directly.

---

## Cross-cutting takeaways

| Method | Task type | Scoring | Action selection | Notes |
|---|---|---|---|---|
| Rodionov 2026 | Interactive (ARC-AGI-3) | LLM-heuristic | Scripted controller | Closest competitor |
| TRM 2025 | Static | Neural recursive | N/A | 1st place 2025 paper |
| CompressARC | Static | MDL | N/A | Closest MDL analog |
| SOAR | Static | LLM evolutionary | N/A | Best score, expensive |
| Li & Ellis 2024 | Static | Neural ind+trans | N/A | 1st place 2024 paper |
| Akyürek TTT | Static | Neural+TTT | N/A | Adaptation insight |
| **ACCA (ours)** | **Interactive** | **Bayesian + formal MDL** | **EIG** | — |

The whitespace in the bottom row is the contribution claim.
