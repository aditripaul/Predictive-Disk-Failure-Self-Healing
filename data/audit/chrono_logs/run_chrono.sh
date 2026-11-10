#!/usr/bin/env bash
# The chronological split run of docs/model_status_and_runbook.md section 1.1.9.
# Run from the repository root; each step logs to data/audit/chrono_logs/. Stops at the first failure.
set -euo pipefail
cd "$(dirname "$0")/../../.."
L=data/audit/chrono_logs
stamp() { echo "=== $1 $(date -u '+%Y-%m-%d %H:%M:%S UTC')" >> $L/progress.log; }
rm -f $L/DONE $L/FAILED
trap 'stamp "FAILED at step ${STEP:-?}"; touch $L/FAILED' ERR

STEP=build-features; stamp "start $STEP"
make build-features > $L/build_features.log 2>&1; stamp "done $STEP"
STEP=build-labels; stamp "start $STEP"
make build-labels > $L/build_labels.log 2>&1; stamp "done $STEP"
STEP=check-splits; stamp "start $STEP"
make check-splits > $L/check_splits.log 2>&1; stamp "done $STEP"
cat $L/check_splits.log >> $L/progress.log
grep -q "^OK: train, validation and test are separate" $L/check_splits.log
STEP=train; stamp "start $STEP"
make train TRAIN_TAG=chrono_h30 TRAIN_ARGS="--horizon-days 30 --keep-work-dir --save-scores" > $L/train_h30.log 2>&1; stamp "done $STEP"
WORK=$(grep -a -o "data/tmp/train_model_frame_[a-z0-9]*" $L/train_h30.log | tail -1)
echo "work dir: $WORK" >> $L/progress.log
STEP=stack; stamp "start $STEP"
make experiment-model EXPERIMENT_ARGS="--work-dir $WORK --variants --skip-baseline --two-stage --stage2-kinds lightgbm cox stack --two-stage-out data/audit/stack_chrono_h30" > $L/stack_h30.log 2>&1; stamp "done $STEP"
touch $L/DONE
