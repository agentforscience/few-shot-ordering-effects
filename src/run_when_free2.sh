#!/bin/bash
# Same as run_when_free.sh but env SCRIPT (default run_model.py) and SUFFIX select the runner / log name.
SCRIPT=${SCRIPT:-run_model.py}
# Usage: run_when_free.sh <gpu> <min_free_MiB> <gpu_util> <model> [extra args]
# Waits until the given GPU has enough free memory, then runs the model; retries on failure.
GPU=$1; NEED=$2; UTIL=$3; MODEL=$4; shift 4
TAG=$(basename $MODEL)
cd "$(dirname "$0")/.."
export HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=$GPU
for attempt in $(seq 1 200); do
  FREE=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i $GPU)
  if [ "$FREE" -ge "$NEED" ]; then
    echo "$(date) attempt $attempt free=$FREE launching $MODEL" >> logs/${SCRIPT%.py}_$TAG$SUFFIX.status
    .venv/bin/python src/$SCRIPT --model $MODEL --gpu_util $UTIL "$@" > logs/${SCRIPT%.py}_$TAG$SUFFIX.log 2>&1 && { echo "$(date) OK" >> logs/${SCRIPT%.py}_$TAG$SUFFIX.status; exit 0; }
    echo "$(date) failed: $(grep -E 'Error:' logs/${SCRIPT%.py}_$TAG$SUFFIX.log | tail -1 | cut -c1-200)" >> logs/${SCRIPT%.py}_$TAG$SUFFIX.status
  fi
  sleep 30
done
