#!/usr/bin/env bash
set -uo pipefail
cd "$(dirname "$0")/../../.."
uv run python scripts/compare_horizons.py --scores 30=data/tmp/lightgbm_scores_h30_h30_q1q2_q3.parquet 44=data/tmp/lightgbm_scores_h44_h44_q1q2_q3.parquet 60=data/tmp/lightgbm_scores_h60_h60_q1q2_q3.parquet > data/audit/q3_logs/compare_horizons.log 2>&1
for pair in 30:fag3o60h 44:ff5ejt7a 60:xjxdieoe; do
  h=${pair%%:*}; d=${pair##*:}
  make experiment-model EXPERIMENT_ARGS="--work-dir data/tmp/train_model_frame_${d} --variants --skip-baseline --two-stage --two-stage-out data/audit/cascade_q3_h${h}" > data/audit/q3_logs/cascade_h${h}.log 2>&1
done
echo done > data/audit/q3_logs/CASCADE_DONE
