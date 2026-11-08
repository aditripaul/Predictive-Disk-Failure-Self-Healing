#!/usr/bin/env bash
# Phase 0 of docs/agent_knowledge_base.md: point-in-time baseline on validation drives at
# the 60-day horizon, four arms (make agent-pit-validation). Takes the repository root as $1.
set -uo pipefail
cd "$1"
make agent-pit-validation > data/audit/agent_pit_validation_h60_logs/pit.log 2>&1
echo "exit=$?" > data/audit/agent_pit_validation_h60_logs/DONE
