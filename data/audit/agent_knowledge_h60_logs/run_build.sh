#!/usr/bin/env bash
# Phase 1 of docs/agent_knowledge_base.md: build the knowledge store from the frozen stack's
# validation scores (2026-01-01..05-01) and silver presence (to 2026-06-30), run the
# self-checks, coverage and drift. Takes the repository root as $1. No test data is read.
set -uo pipefail
cd "$1"
L=data/audit/agent_knowledge_h60_logs
K="uv run python scripts/build_agent_knowledge.py"
{
  $K build --overwrite &&
  $K check &&
  $K coverage --as-of 2026-03-02 &&
  $K coverage --as-of 2026-05-01 &&
  $K drift --reference 2026-01-01:2026-03-31 --current 2026-04-01:2026-05-01
} > $L/build.log 2>&1
echo "exit=$?" > $L/DONE
