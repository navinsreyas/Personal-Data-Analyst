from __future__ import annotations

from datetime import datetime
from typing import Annotated, Any, Literal, Optional, TypedDict

from langgraph.graph.message import AnyMessage, add_messages


RouterDecision = Literal["green", "yellow", "red"]

NodeName = Literal[
    "router",
    "clarifier",
    "planner",
    "reviewer",
    "supervisor",
    "codeagent_executor",
    "explainer",
    "refuser",
]


class AnalysisPlan(TypedDict, total=False):
    goal: str
    primary_metric: str
    aggregation_type: str
    dimensions: list[str]
    filters: list[dict[str, Any]]
    sort_by: Optional[str]
    sort_order: Optional[Literal["asc", "desc"]]
    limit: Optional[int]
    visualization: Optional[str]
    rationale: str
    confidence: Optional[str]
    assumptions: list[str]


class ExecutionResult(TypedDict, total=False):
    success: bool
    output: Any
    output_type: Literal["dataframe", "scalar", "list", "chart", "error"]
    chart_path: Optional[str]
    error: Optional[str]
    error_type: Optional[str]
    execution_time_ms: float
    memory_usage_mb: Optional[float]


class ClarificationRequest(TypedDict):
    question: str
    options: Optional[list[str]]
    context: str


class AgentState(TypedDict, total=False):
    messages: Annotated[list[AnyMessage], add_messages]
    user_query: str
    schema_profile: dict
    csv_file_path: str

    analysis_plan: Optional[AnalysisPlan]
    plan_critique: Optional[str]
    plan_approved: Optional[bool]
    python_code: Optional[str]
    execution_result: Optional[ExecutionResult]
    generated_code: Optional[list[str]]
    agent_steps: Optional[int]
    final_response: Optional[str]

    router_decision: Optional[RouterDecision]
    clarification_question: Optional[str]
    clarification_response: Optional[str]
    refusal_reason: Optional[str]

    next_node: Optional[str]
    supervisor_reasoning: Optional[str]
    supervisor_confidence: Optional[str]

    retry_count: int
    plan_revision_count: int
    error_message: Optional[str]
    error_traceback: Optional[str]
    error_history: Optional[list[str]]

    current_node: Optional[NodeName]
    started_at: Optional[str]
    completed_at: Optional[str]
    total_tokens_used: Optional[int]
    execution_path: Optional[list[str]]


def create_initial_state(
    user_query: str,
    schema_profile: dict,
    csv_file_path: str,
    messages: Optional[list[AnyMessage]] = None,
) -> AgentState:
    return AgentState(
        messages=messages or [],
        user_query=user_query,
        schema_profile=schema_profile,
        csv_file_path=csv_file_path,

        analysis_plan=None,
        plan_critique=None,
        plan_approved=None,
        python_code=None,
        execution_result=None,
        generated_code=None,
        agent_steps=None,
        final_response=None,

        router_decision=None,
        clarification_question=None,
        clarification_response=None,
        refusal_reason=None,

        next_node=None,
        supervisor_reasoning=None,
        supervisor_confidence=None,

        retry_count=0,
        plan_revision_count=0,
        error_message=None,
        error_traceback=None,
        error_history=[],

        current_node=None,
        started_at=datetime.now().isoformat(),
        completed_at=None,
        total_tokens_used=0,
        execution_path=[],
    )


def validate_state_for_node(state: AgentState, node: NodeName) -> tuple[bool, str]:
    required_fields: dict[str, list[str]] = {
        "router":             ["user_query", "schema_profile"],
        "clarifier":          ["user_query", "router_decision"],
        "planner":            ["user_query", "schema_profile", "router_decision"],
        "reviewer":           ["analysis_plan"],
        "supervisor":         ["plan_approved"],
        "codeagent_executor": ["analysis_plan", "csv_file_path"],
        "explainer":          ["execution_result", "user_query"],
        "refuser":            ["refusal_reason"],
    }

    missing = [
        field
        for field in required_fields.get(node, [])
        if state.get(field) is None
    ]

    if missing:
        return False, f"Missing required fields for '{node}': {missing}"
    return True, ""


MAX_CODE_RETRIES = 3
MAX_PLAN_REVISIONS = 2
EXECUTION_TIMEOUT_MS = 30_000
