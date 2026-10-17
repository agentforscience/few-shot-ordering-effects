#!/bin/bash
# Usage: run_when_free.sh <gpu> <min_free_MiB> <gpu_util> <model> [extra args]
# Waits until the given GPU has enough free memory, then runs the model; retries on failure.
GPU=$1; NEED=$2; UTIL=$3; MODEL=$4; shift 4
TAG=$(basename $MODEL)
cd "$(dirname "$0")/.."
export HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=$GPU
for attempt in $(seq 1 200); do
  FREE=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i $GPU)
  if [ "$FREE" -ge "$NEED" ]; then
    echo "$(date) attempt $attempt free=$FREE launching $MODEL" >> logs/run_$TAG.status
    .venv/bin/python src/run_model.py --model $MODEL --gpu_util $UTIL "$@" > logs/run_$TAG.log 2>&1 && { echo "$(date) OK" >> logs/run_$TAG.status; exit 0; }
    echo "$(date) failed: $(grep -E 'Error:' logs/run_$TAG.log | tail -1 | cut -c1-200)" >> logs/run_$TAG.status
  fi
  sleep 30
done
