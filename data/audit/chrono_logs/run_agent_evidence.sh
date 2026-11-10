#!/usr/bin/env bash
# make chrono-pit-validation, then make chrono-knowledge (runbook 1.1.9). Takes the repository root as $1.
set -uo pipefail
cd "$1"
L=data/audit/chrono_logs
date -u +%FT%TZ > $L/agent_evidence.start
make chrono-pit-validation > $L/pit.log 2>&1; echo "pit exit=$?" >> $L/agent_evidence.done
make chrono-knowledge > $L/knowledge.log 2>&1; echo "knowledge exit=$?" >> $L/agent_evidence.done
date -u +%FT%TZ >> $L/agent_evidence.done
