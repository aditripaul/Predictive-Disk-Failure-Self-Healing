#!/usr/bin/env bash
# The chronological split run of docs/model_status_and_runbook.md section 1.1.9, end to end:
#
#   make chrono-run                       # this script
#   CHRONO_LOGS=data/tmp/chrono_logs make chrono-run    # logs somewhere else
#
# Rebuilds the gold features and labels under configs/model.yaml `splits` (this REPLACES data/gold/,
# which the drive-holdout design also reads), checks the separation, trains the 30-day LightGBM
# two-stage model, then runs the Stage 2 variants (LightGBM, Cox, stack) on its kept work directory.
# About 80 minutes on the machine of the 2026-11-10 run. Each step logs to $CHRONO_LOGS (default
# data/audit/chrono_logs/); progress.log has the timings. Stops at the first failure and leaves a
# FAILED marker, or a DONE marker when everything ran.
#
# `data/audit/chrono_logs/run_chrono.sh` is the launcher the 2026-11-10 run used; this is the same
# sequence expressed through the Makefile targets, plus the figures.
set -euo pipefail
cd "$(dirname "$0")/.."

L="${CHRONO_LOGS:-data/audit/chrono_logs}"
TAG="${CHRONO_TAG:-chrono_h30}"
OUT="${CHRONO_OUT:-data/audit/stack_chrono_h30}"
mkdir -p "$L"

stamp() { echo "=== $1 $(date -u '+%Y-%m-%d %H:%M:%S UTC')" >> "$L/progress.log"; }
rm -f "$L/DONE" "$L/FAILED"
trap 'stamp "FAILED at step ${STEP:-?}"; touch "$L/FAILED"' ERR

STEP=build-features; stamp "start $STEP"
make build-features > "$L/build_features.log" 2>&1; stamp "done $STEP"

STEP=build-labels; stamp "start $STEP"
make build-labels > "$L/build_labels.log" 2>&1; stamp "done $STEP"

STEP=check-splits; stamp "start $STEP"
make check-splits > "$L/check_splits.log" 2>&1; stamp "done $STEP"
cat "$L/check_splits.log" >> "$L/progress.log"
grep -q "^OK: train, validation and test are separate" "$L/check_splits.log"

STEP=train; stamp "start $STEP"
make chrono-train CHRONO_TAG="$TAG" > "$L/train_h30.log" 2>&1; stamp "done $STEP"
WORK=$(grep -a -o "data/tmp/train_model_frame_[a-z0-9_]*" "$L/train_h30.log" | tail -1)
test -n "$WORK"
echo "work dir: $WORK" >> "$L/progress.log"

STEP=stack; stamp "start $STEP"
make chrono-stack CHRONO_WORK_DIR="$WORK" CHRONO_OUT="$OUT" > "$L/stack_h30.log" 2>&1; stamp "done $STEP"

STEP=plots; stamp "start $STEP"
make plot-chrono-split > "$L/plots.log" 2>&1; stamp "done $STEP"

# Third-party messages print this checkout's absolute path; the logs are committed, so rewrite it as a
# repo-relative path (scripts/scrub_paths.py; the pre-commit hook refuses an absolute path).
python3 scripts/scrub_paths.py --in-place "$L"/*.log

touch "$L/DONE"
