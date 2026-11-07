#!/usr/bin/env bash
# Phase S of docs/agent_knowledge_base.md: freeze the 60-day stacked Stage 2 (seed 0,
# net 0.8) as a scorer, from validation only. Takes the repository root as $1.
set -uo pipefail
cd "$1"
uv run python scripts/freeze_stack_scorer.py --work-dir data/tmp/train_model_frame_xjxdieoe \
  > data/audit/stack_scorer_h60_logs/freeze.log 2>&1
echo "exit=$?" > data/audit/stack_scorer_h60_logs/DONE
