"""The knowledge base wired into the MAPE-K loop (docs/agent_knowledge_base.md section 6):
read-only context for Analyze and Plan, decisions recorded by Validate, and no decision
changed while no rule is enabled (self-check k)."""

from __future__ import annotations

import datetime as dt

import polars as pl

from data_contracts.schemas import ActionTier, FeatureMaturity
from src.agent.deps import AgentDependencies, InMemoryActionLedger, default_guardrail_pass_all
from src.agent.fleet import to_candidate_drives
from src.agent.graph import compiled_agent
from src.agent.knowledge import KnowledgeBase
from src.agent.nodes import make_plan_node
from src.config import AgentSettings

THRESHOLDS = {"warn": 0.3, "cordon": 0.6, "migrate": 0.8, "drain": 0.9}
DAY = dt.date(2026, 2, 20)


def _kb() -> KnowledgeBase:
    kb = KnowledgeBase.in_memory(
        run_id="stub",
        thresholds={"warn": 0.3, "cordon": 0.6, "primary": 0.9},
        candidate_threshold=0.2,
        horizon_days=10,
        episode_gap_days=3,
        context_window_days=30,
    )
    for i in range(20):
        day = DAY - dt.timedelta(days=19 - i)
        scored = pl.DataFrame({"drive_id": ["chronic", "quiet"], "p_fail": [0.85, 0.1]})
        kb.ingest_snapshot(day, scored, source="test")
    return kb


def _drive(drive_id: str, p: float, ctx=None) -> dict:
    d = {
        "drive_id": drive_id,
        "p_fail": p,
        "feature_confidence": 0.95,
        "feature_maturity": FeatureMaturity.MATURE.value,
        "stale_telemetry": False,
        "as_of": DAY.isoformat(),
    }
    if ctx is not None:
        d["knowledge_context"] = ctx
    return d


def _state(drive: dict) -> dict:
    return {
        "run_id": "run-1",
        "timestamp": dt.datetime(2026, 2, 20, tzinfo=dt.UTC).isoformat(),
        "prediction_summary": {"drives": [drive]},
    }


def _deps(knowledge=None, drives=None) -> AgentDependencies:
    return AgentDependencies(
        fleet_state_provider=lambda: {"fleet_snapshot_id": "s", "drives": drives or []},
        predictor=(
            (lambda s: {"drives": knowledge.annotate(s["drives"])})
            if knowledge is not None
            else (lambda s: {"drives": s["drives"]})
        ),
        guardrail_evaluator=default_guardrail_pass_all,
        executor=lambda a: {"success": True, **a},
        validator=lambda r: {"quorum_ok": True},
        action_ledger=InMemoryActionLedger(),
        knowledge=knowledge,
    )


def _plan(**kw):
    return make_plan_node(
        _deps(),
        action_thresholds=THRESHOLDS,
        hysteresis_cycles_required=1,
        control_fraction=0.0,
        **kw,
    )


def test_without_a_rule_the_context_changes_no_decision():  # check (k)
    kb = _kb()
    for drive_id, p in (("chronic", 0.85), ("quiet", 0.85), ("chronic", 0.45), ("new", 0.95)):
        bare = _plan()(_state(_drive(drive_id, p)))["proposed_action"]
        (annotated,) = kb.annotate([_drive(drive_id, p)])
        with_kb = _plan()(_state(annotated))["proposed_action"]
        assert {k: v for k, v in with_kb.items() if k != "knowledge_context"} == {
            k: v for k, v in bare.items() if k != "knowledge_context"
        }
        assert bare["knowledge_context"] is None


def test_the_proposal_carries_the_context_and_date():
    (annotated,) = _kb().annotate([_drive("chronic", 0.85)])
    proposal = _plan()(_state(annotated))["proposed_action"]
    assert proposal["as_of"] == DAY.isoformat()
    assert proposal["knowledge_context"]["open_episode_days"] == 19


def test_a_rule_runs_before_hysteresis_and_only_with_history():
    calls = []

    def demote(tier, ctx, params, thresholds):
        calls.append((tier, ctx["open_episode_days"], params, thresholds["cordon"]))
        return ActionTier.WARN

    plan = _plan(knowledge_rule=demote, knowledge_rule_params={"n_days": 7})
    (annotated,) = _kb().annotate([_drive("chronic", 0.85)])
    assert plan(_state(annotated))["proposed_action"]["proposed_action"] == "warn"
    assert calls == [(ActionTier.MIGRATE, 19, {"n_days": 7}, 0.6)]
    # a drive with no context is never passed to the rule
    assert plan(_state(_drive("new", 0.85)))["proposed_action"]["proposed_action"] == "migrate"
    assert len(calls) == 1


def test_validate_records_each_decision_in_the_store():
    kb = _kb()
    (annotated,) = kb.annotate([_drive("chronic", 0.95)])
    deps = _deps(
        knowledge=kb, drives=[{k: v for k, v in annotated.items() if k != "knowledge_context"}]
    )
    plan_kwargs = {
        "action_thresholds": THRESHOLDS,
        "hysteresis_cycles_required": 1,
        "control_fraction": 0.0,
    }
    with compiled_agent(deps, plan_kwargs=plan_kwargs) as app:
        result = app.invoke({}, config={"configurable": {"thread_id": "k-1"}})
    assert result["decision_record_id"]
    rows = kb.db.execute(
        "SELECT decision_id, drive_id, as_of, proposed_tier FROM decisions"
    ).fetchall()
    assert rows == [
        (
            result["decision_record_id"],
            "chronic",
            DAY.isoformat(),
            result["proposed_action"]["proposed_action"],
        )
    ]


def test_candidates_carry_context_columns_only_when_asked():
    scored = pl.DataFrame(
        {
            "drive_id": ["a"],
            "date": [DAY],
            "p_fail": [0.9],
            "feature_confidence": [1.0],
            "feature_maturity": ["MATURE"],
            "drive_model": ["M1"],
            "model_family": ["F1"],
            "power_on_hours": [100.0],
        }
    )
    (plain,), _ = to_candidate_drives(scored, action_thresholds=THRESHOLDS)
    assert set(plain) == {
        "drive_id",
        "p_fail",
        "feature_confidence",
        "feature_maturity",
        "stale_telemetry",
    }
    (rich,), _ = to_candidate_drives(
        scored,
        action_thresholds=THRESHOLDS,
        context_columns=("drive_model", "model_family"),
        include_as_of=True,
    )
    assert rich["drive_model"] == "M1" and rich["as_of"] == DAY.isoformat()


def test_settings_default_to_no_knowledge_base():
    settings = AgentSettings.load()
    assert settings.knowledge_backend == "none" and settings.knowledge_rule == "none"
