#!/usr/bin/env bash
# Phase 2 of docs/agent_knowledge_base.md: the pre-registered validation screen of
# knowledge rules (section 8, Amendment 1). Validation only; takes the repository root as $1.
set -uo pipefail
cd "$1"
uv run python scripts/screen_agent_knowledge.py > data/audit/agent_knowledge_h60_logs/screen.log 2>&1
echo "exit=$?" > data/audit/agent_knowledge_h60_logs/SCREEN_DONE
