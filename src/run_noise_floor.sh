#!/bin/bash
# Noise-floor reruns: identical prompts (K=4, demo sets 0-2, all 8 orders) re-run per model,
# to measure vLLM run-to-run (batch-dependent bf16) numerical variability.
cd "$(dirname "$0")/.."
SUFFIX=_rerun src/run_when_free2.sh 3 5500 0.11 Qwen/Qwen2.5-0.5B-Instruct --only_k 4 --max_sets 3 --suffix _rerun &
SUFFIX=_rerun src/run_when_free2.sh 0 5000 0.105 Qwen/Qwen2.5-1.5B-Instruct --only_k 4 --max_sets 3 --suffix _rerun &
wait
for spec in "15000 0.3 Qwen/Qwen2.5-3B-Instruct" "23000 0.47 Qwen/Qwen2.5-7B-Instruct" "40000 0.82 Qwen/Qwen2.5-14B-Instruct"; do
  set -- $spec
  SUFFIX=_rerun src/run_when_free2.sh 2 $1 $2 $3 --only_k 4 --max_sets 3 --suffix _rerun
done
