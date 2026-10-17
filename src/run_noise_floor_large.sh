#!/bin/bash
# Resume noise-floor reruns for the two largest models (interrupted in the first session).
cd "$(dirname "$0")/.."
GPU=${GPU:-2}
SUFFIX=_rerun src/run_when_free2.sh $GPU 23000 0.44 Qwen/Qwen2.5-7B-Instruct --only_k 4 --max_sets 3 --suffix _rerun
SUFFIX=_rerun src/run_when_free2.sh $GPU 40000 0.82 Qwen/Qwen2.5-14B-Instruct --only_k 4 --max_sets 3 --suffix _rerun
