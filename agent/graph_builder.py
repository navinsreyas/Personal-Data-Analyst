from __future__ import annotations

import logging
from pathlib import Path
from typing import Literal

from langgraph.graph import END, START, StateGraph
from langgraph.checkpoint.sqlite import SqliteSaver

from .state import AgentState, MAX_CODE_RETRIES, MAX_PLAN_REVISIONS
from .nodes import (
    router_node,
    clarifier_node,
    planner_node,
    reviewer_node,
    supervisor_node,
    codeagent_executor_node,
    explainer_node,
    refuser_node,
    SUPERVISOR_VALID_NODES,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def route_after_router(state: AgentState) -> Literal["clarifier", "planner", "refuser"]:
    decision = state.get("router_decision", "red")
    logger.info(f"Routing after router: decision={decision}")
    if decision == "green":
        return "planner"
    elif decision == "yellow":
        return "clarifier"
    else:
        return "refuser"


def route_after_codeagent_executor(state: AgentState) -> Literal["explainer", "supervisor"]:
    result  = state.get("execution_result") or {}
    success = result.get("success", False)
    logger.info(f"Routing after codeagent_executor: success={success}")
    return "explainer" if success else "supervisor"


def route_after_supervisor(state: AgentState) -> str:
    next_node = state.get("next_node", "refuser")
    if next_node not in SUPERVISOR_VALID_NODES:
        logger.warning(f"Supervisor wrote invalid next_node='{next_node}' — defaulting to refuser")
        return "refuser"
    logger.info(f"Routing after supervisor: next_node={next_node}")
    return next_node


def build_graph() -> StateGraph:
    graph = StateGraph(AgentState)

    graph.add_node("router",             router_node)
    graph.add_node("clarifier",          clarifier_node)
    graph.add_node("planner",            planner_node)
    graph.add_node("reviewer",           reviewer_node)
    graph.add_node("supervisor",         supervisor_node)
    graph.add_node("codeagent_executor", codeagent_executor_node)
    graph.add_node("explainer",          explainer_node)
    graph.add_node("refuser",            refuser_node)

    graph.add_edge(START, "router")

    graph.add_conditional_edges(
        "router",
        route_after_router,
        {"planner": "planner", "clarifier": "clarifier", "refuser": "refuser"},
    )

    graph.add_edge("clarifier", END)
    graph.add_edge("planner",   "reviewer")
    graph.add_edge("reviewer",  "supervisor")

    graph.add_conditional_edges(
        "codeagent_executor",
        route_after_codeagent_executor,
        {"explainer": "explainer", "supervisor": "supervisor"},
    )

    graph.add_conditional_edges(
        "supervisor",
        route_after_supervisor,
        {
            "planner":            "planner",
            "codeagent_executor": "codeagent_executor",
            "explainer":          "explainer",
            "refuser":            "refuser",
        },
    )

    graph.add_edge("explainer", END)
    graph.add_edge("refuser",   END)

    return graph


def compile_graph(checkpointer: bool = True) -> StateGraph:
    graph = build_graph()

    if checkpointer:
        db_path  = Path(__file__).parent / "checkpoints.db"
        memory   = SqliteSaver.from_conn_string(str(db_path))
        compiled = graph.compile(checkpointer=memory)
        logger.info(f"Graph compiled with SQLite checkpointer: {db_path}")
    else:
        compiled = graph.compile()
        logger.info("Graph compiled without checkpointer")

    return compiled
