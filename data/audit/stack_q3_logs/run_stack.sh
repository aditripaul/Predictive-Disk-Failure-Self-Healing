#!/usr/bin/env bash
set -uo pipefail
cd "$1"
for pair in 30:fag3o60h 44:ff5ejt7a 60:xjxdieoe; do
  h=${pair%%:*}; d=${pair##*:}
  make experiment-model EXPERIMENT_ARGS="--work-dir data/tmp/train_model_frame_${d} --variants --skip-baseline --two-stage --stage2-kinds lightgbm cox stack --two-stage-out data/audit/stack_q3_h${h}" > data/audit/stack_q3_logs/stack_h${h}.log 2>&1
done
echo done > data/audit/stack_q3_logs/STACK_DONE
