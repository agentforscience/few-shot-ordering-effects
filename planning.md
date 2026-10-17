# Planning: Few-Shot Example Ordering Effects on In-Context Learning

## Motivation & Novelty Assessment

### Why This Research Matters
Few-shot prompting is the default way practitioners adapt LLMs, yet the *order* of demonstrations is an
uncontrolled nuisance variable. If ordering can swing accuracy by double digits, reported few-shot numbers
are partly noise and prompt designers need order-robust practices. Knowing *when* (task type, shot count,
model size) ordering matters tells practitioners where to invest in order search or calibration.

### Gap in Existing Work
Note: the resource-finder stage was skipped for this workspace (no `literature_review.md`, `papers/`,
`code/`), so this assessment relies on well-known prior work:
- Lu et al. 2022 ("Fantastically Ordered Prompts") — order sensitivity for GPT-2/GPT-3 on classification;
  large gaps, not reduced by scale, but on old base models and mostly single shot counts.
- Zhao et al. 2021 ("Calibrate Before Use") — recency bias (majority/last label) for GPT-3.
- Liu et al. 2023 ("Lost in the Middle") — U-shaped position effects in long-context retrieval/QA.
These do not jointly measure (a) how order variance scales with shot count, (b) *per-position influence*
quantitatively (first vs last), (c) classification vs generation under one protocol, and (d) a clean
within-family size sweep on modern instruction-tuned models.

### Our Novel Contribution
A single controlled protocol on the Qwen2.5-Instruct family (0.5B→14B) that measures, for sentiment
(SST-2), NLI (RTE, MNLI), and summarization (SAMSum): order variance vs. K∈{1..8}, best–worst order gaps,
and a regression-based per-position influence estimate that directly tests "last example 2–3× more
influential than first" (classification) and "primacy dominates" (generation).

### Experiment Justification
- Exp 1 (shot-count sweep, classification): needed for the "variance rises until 4-shot then plateaus" claim
  and the NLI (>15%) vs sentiment (<5%) gap claim.
- Exp 2 (position-influence regression, re-using Exp 1 prompts): tests recency for classification.
- Exp 3 (SAMSum generation sweep): tests primacy for generation (output length / lexical copying traced
  to each demo position) and generation order variance.
- Exp 4 (model-size sweep, all above across 5 sizes): tests "larger models less order-sensitive".

## Research Question
How does the order of few-shot demonstrations affect ICL performance as a function of shot count, task
type, and model size, and which demo positions dominate the model's output?

## Hypothesis Decomposition
- H1: Across-order SD of accuracy increases from K=1 to K=4, then is flat (K=4,6,8 not different).
- H2: Classification: influence of last demo ≥2× influence of first demo (recency).
- H3: Generation: influence of first demo > influence of last demo (primacy).
- H4: Best–worst order gap >15 points on NLI (RTE/MNLI), <5 points on SST-2.
- H5: Order sensitivity (SD, gap) decreases with parameter count.

## Proposed Methodology

### Approach
Real models, run locally with vLLM on one RTX A6000 (no API keys available on this host).
Models: Qwen2.5-{0.5B,1.5B,3B,7B,14B}-Instruct (same family/data → cleanest size comparison).
Plain-text completion prompts (no chat template), greedy / argmax decoding (deterministic, so all
variance across orders is caused by order).

### Experimental Steps
1. Fixed test sets: SST-2 val (200, balanced), RTE val (200), MNLI-matched val (201, balanced).
2. For each K∈{1,2,3,4,6,8}: sample S=10 demo sets from train (random labels, not forced balanced so
   label positions vary); for each set evaluate P=min(K!,8) distinct permutations on all test items.
3. Classification scoring: next-token log-probs restricted to the label words; prediction = argmax.
4. SAMSum: K∈{1,2,3,4,6}, S=6 sets, P=min(K!,6) orders, N=80 test dialogues; greedy, ≤80 tokens.
5. Repeat for all five models.

### Baselines
Zero-shot (K=0) per task/model; majority-class; across-demo-set variance (to contextualize order variance
against "which examples" variance).

### Evaluation Metrics
- Accuracy (classification), ROUGE-L F1 (generation).
- Order sensitivity per demo set: SD across orders; range (best−worst); and *expected pairwise gap*
  E|acc(π)−acc(π')| (unbiased w.r.t. number of permutations sampled, comparable across K).
- Position influence β_i: regression of the model's label log-prob score on "demo at position i has
  label c" with test-item × class fixed effects; for generation, regression of output length on
  demo-i summary length (test FE) and lexical copying overlap with demo i.

### Statistical Analysis Plan
Bootstrap CIs over demo sets (resampling sets); Mann–Whitney / Welch tests for K comparisons;
cluster-robust (by prompt) SEs for β; Spearman correlation for size trend; α=0.05, Holm correction
within each hypothesis family.

## Expected Outcomes
Support: SD rises K=1→4 then flat; β_last/β_first ∈[2,3]; generation β_first>β_last; NLI gap>15,
SST-2 gap<5; Spearman(size, SD)<0. Any deviation is reported as refutation of that sub-claim.

## Timeline and Milestones
Setup 30 min; code + pilot 45 min; runs ~2–3 h GPU; analysis 45 min; write-up 30 min.

## Potential Challenges
- Shared GPU may become busy → keep models ≤14B, checkpoint per model/task.
- Label tokens missing from top-k logprobs → request top-20 and floor-fill; log coverage.
- Small K!: K=2 only has 2 orders, K=3 has 6 → use pairwise-gap metric that is unbiased in P.
- Test-set sampling noise in accuracy (N=200 → SE≈3.5pt) inflates order SD slightly; report it.

## Success Criteria
All five hypotheses receive a clear supported/refuted verdict with CIs, from real model outputs.
