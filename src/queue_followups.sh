#!/bin/bash
# Follow-up runs (probe + extension sets) per model, each started after the model's main run succeeded.
# Usage: queue_followups.sh <gpu> <need_MiB> <util> <model> [probe_args]
GPU=$1; NEED=$2; UTIL=$3; MODEL=$4; PARGS=$5
TAG=$(basename $MODEL)
cd "$(dirname "$0")/.."
until grep -q OK logs/run_$TAG.status 2>/dev/null; do sleep 30; done
SCRIPT=run_probe.py src/run_when_free2.sh $GPU $NEED $UTIL $MODEL $PARGS
SUFFIX=_ext src/run_when_free2.sh $GPU $NEED $UTIL $MODEL --design design_ext.json --suffix _ext --skip_gen
