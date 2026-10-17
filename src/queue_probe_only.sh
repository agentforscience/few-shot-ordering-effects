#!/bin/bash
# Probe-only follow-up (no extension sets) for the large models; waits for the model's main run.
# The 14B probe additionally waits for the 14B main run so main runs keep GPU priority.
GPU=$1; NEED=$2; UTIL=$3; MODEL=$4
TAG=$(basename $MODEL)
cd "$(dirname "$0")/.."
until grep -q OK logs/run_$TAG.status 2>/dev/null; do sleep 30; done
SCRIPT=run_probe.py src/run_when_free2.sh $GPU $NEED $UTIL $MODEL --max_batched_tokens 2048
