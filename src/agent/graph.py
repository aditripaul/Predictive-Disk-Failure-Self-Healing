"""Compiles the cyclic MAPE-K LangGraph workflow (docs/design_goal.md
section 9): Monitor -> Analyze -> Plan -> (Human Review | Execute) -> Validate.

One `.invoke()` call runs a single MAPE-K cycle; a driver loop (Phase 6 exit
criteria) re-invokes with the same `thread_id` for continuous operation, so
"return to Monitor" happens as the next external invocation rather than an
internal edge — this keeps each checkpointed step small and bounded.
"""

from __future__ import annotations

from contextlib import contextmanager

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, StateGraph

from src.agent.deps import AgentDependencies
from src.agent.nodes import (
    make_analyze_node,
    make_execute_node,
    make_monitor_node,
    make_plan_node,
    make_validate_node,
)
from src.agent.state import AgentState


def _route_after_plan(state: AgentState) -> str:
    return "human_review" if state.get("human_review_required") else "execute"


def build_agent_graph(deps: AgentDependencies) -> StateGraph:
    """Returns an uncompiled graph; call `.compile(checkpointer=...)` on it."""
    graph = StateGraph(AgentState)
    graph.add_node("monitor", make_monitor_node(deps))
    graph.add_node("analyze", make_analyze_node(deps))
    graph.add_node("plan", make_plan_node(deps))
    graph.add_node("execute", make_execute_node(deps))
    graph.add_node("validate", make_validate_node(deps))
    # Terminal placeholder: the FastAPI approval queue (Phase 10) resumes the
    # cycle by re-invoking with human_review_required cleared.
    graph.add_node("human_review", lambda state: {})

    graph.set_entry_point("monitor")
    graph.add_edge("monitor", "analyze")
    graph.add_edge("analyze", "plan")
    graph.add_conditional_edges(
        "plan", _route_after_plan, {"execute": "execute", "human_review": "human_review"}
    )
    graph.add_edge("execute", "validate")
    graph.add_edge("validate", END)
    graph.add_edge("human_review", END)

    return graph


@contextmanager
def compiled_agent(deps: AgentDependencies, *, checkpoint_path: str = ":memory:"):
    """Context manager yielding a compiled, checkpointed graph, e.g.:

        with compiled_agent(deps) as app:
            app.invoke({}, config={"configurable": {"thread_id": "fleet-1"}})
    """
    graph = build_agent_graph(deps)
    with SqliteSaver.from_conn_string(checkpoint_path) as checkpointer:
        yield graph.compile(checkpointer=checkpointer)
