# Few-Shot Example Ordering Effects on In-Context Learning

This project measures how much the **order** of few-shot demonstrations changes LLM outputs. It covers
shot count (K = 1–8), task type (sentiment, NLI, summarization) and model size (Qwen2.5-Instruct
0.5B → 14B). All results come from real model runs: about 2.36 M scored prompts and generations from
local vLLM inference on an RTX A6000. Full write-up: **[REPORT.md](REPORT.md)**.

## Key findings
- **No "rise to 4-shot, then plateau".** Order variance is already at its plateau level at K = 2 and
  stays flat up to K = 8 for every task. Only 1 of 20 task×model curves shows a significant rise from 2 to 4 shots.
- **The NLI > sentiment ordering gap holds, but the magnitudes are smaller than hypothesized.** The mean best–worst order gap
  (8 orders, K ≥ 4) is 6–13 points on MNLI, 2–7 on RTE and 1–5 on SST-2. The ">15 points on NLI" claim
  holds only for the single worst demo set of the smallest models.
- **No universal recency in classification.** Label-flip interventions find "last ≈ 2–3× first" in
  only 2 of 15 model×task cells (SST-2 at 1.5B, RTE at 7B). On NLI, the 1.5B–3B models are pulled
  toward the *first* demos, and flipping the *last* demo's label pushes them *away* from it.
- **Generation shows recency, not primacy.** For 0.5B–3B, a style marker placed in the last demo
  raises the marker's log-probability more than the same marker in the first demo (first/last
  ≈ 0.64–0.80). 7B and 14B are essentially position-neutral (≈ 1.0).
- **Size helps for classification but not generation.** Order sensitivity falls monotonically with size on SST-2 and
  RTE (Spearman ρ = −1). MNLI and SAMSum turn back up at 14B. On SST-2 at ≥3B, the order "gaps" are
  no larger than item-level noise (split-half reliability ≈ 0).
- **Noise floor.** Re-running identical prompts flips only 0–3% of classification predictions,
  versus 1–18% for order changes, so classification order effects are real. For SAMSum, 30–42% of
  the order-induced ROUGE-L variance also appears between identical runs (bf16 non-determinism) for
  0.5B, 1.5B and 7B. There was no 14B rerun because it did not fit in the shared GPU memory.

## Reproduce
```bash
uv venv -p 3.11 && source .venv/bin/activate && uv sync          # vllm 0.31, torch 2.13
python src/make_design.py && python src/make_design_ext.py         # shared demo sets / permutations
export HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=0
for m in 0.5B 1.5B 3B 7B 14B; do
  python src/run_model.py --model Qwen/Qwen2.5-$m-Instruct           # main sweep (cls + SAMSum)
  python src/run_probe.py --model Qwen/Qwen2.5-$m-Instruct           # probes (14B used --cls_k 8 --gen_k 6)
done
for m in 0.5B 1.5B 3B; do python src/run_model.py --model Qwen/Qwen2.5-$m-Instruct \
  --design design_ext.json --suffix _ext --skip_gen; done            # +10 demo sets (small models)
src/run_noise_floor.sh; src/run_noise_floor_large.sh                # identical-prompt reruns (K=4, sets 0-2)
python src/analyze.py                                               # tables + figures
```
`src/run_when_free.sh` and `src/run_when_free2.sh` are the wrappers used on the shared GPU host. They wait for free GPU memory and retry.
Data: SST-2 / GLUE-RTE / GLUE-MNLI parquet files and SAMSum CSVs in `datasets/` (from the HF Hub).

## Layout
```
planning.md            research plan + motivation
src/tasks.py           data loading, prompt templates
src/make_design*.py    demo-set / permutation design (seeded, shared by all models)
src/run_model.py       main sweep (vLLM, label-restricted log-probs; greedy generation)
src/run_probe.py       causal position probes (label flip; "In short," marker)
src/analyze.py         statistics, tables (results/tables), figures (figures/)
results/raw/           per-model raw outputs (parquet)   results/summary.json  hypothesis summaries
figures/               fig0–fig7 PNGs                     logs/                 run logs
```
