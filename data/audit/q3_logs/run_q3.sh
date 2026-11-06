#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../../.."
make build-labels > data/audit/q3_logs/build_labels.log 2>&1
for h in 30 44 60; do
  make train TRAIN_TAG=h${h}_q1q2_q3 TRAIN_ARGS="--horizon-days ${h} --keep-work-dir --save-scores" > data/audit/q3_logs/train_h${h}.log 2>&1
done
echo done > data/audit/q3_logs/DONE
