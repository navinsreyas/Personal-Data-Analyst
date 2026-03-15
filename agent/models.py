from __future__ import annotations

import re
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field, field_validator


class RouterOutput(BaseModel):
    classification: Literal["green", "yellow", "red"] = Field(
        ...,
        description="Query classification: green (proceed), yellow (clarify), red (refuse)",
    )

    reasoning: str = Field(
        ...,
        description=(
            "Explanation of why this classification was chosen. "
            "Include specific evidence from the query and schema."
        ),
    )

    identified_metric: Optional[str] = Field(
        default=None,
        description="If GREEN: The primary metric being requested (e.g. 'revenue', 'count')",
    )

    identified_dimensions: Optional[list[str]] = Field(
        default=None,
        description="If GREEN: Dimensions for grouping (e.g. ['region', 'product_category'])",
    )

    identified_filters: Optional[dict[str, Any]] = Field(
        default=None,
        description="If GREEN: Any filter conditions mentioned (e.g. {'status': 'completed'})",
    )

    clarification_question: Optional[str] = Field(
        default=None,
        description=(
            "If YELLOW: A specific question to ask the user for clarification. "
            "Should be targeted and help resolve the ambiguity."
        ),
    )

    ambiguous_terms: Optional[list[str]] = Field(
        default=None,
        description="If YELLOW: List of ambiguous terms that need definition",
    )

    refusal_reason: Optional[str] = Field(
        default=None,
        description="If RED: The specific reason why this query cannot be processed",
    )

    red_zone_category: Optional[Literal[
        "prediction",
        "causality",
        "recommendation",
        "external_data",
        "subjective",
        "modification",
        "other",
    ]] = Field(
        default=None,
        description="If RED: The category of refusal",
    )

    suggested_alternative: Optional[str] = Field(
        default=None,
        description="If RED: A suggested rephrasing that would be answerable",
    )


class FilterCondition(BaseModel):
    column: str = Field(..., description="Column name to filter on")

    operator: Literal[
        "eq", "ne", "gt", "gte", "lt", "lte",
        "in", "not_in", "contains", "between",
    ] = Field(..., description="Comparison operator")

    value: Any = Field(..., description="Value(s) to compare against")


class TimeConfig(BaseModel):
    time_column: str = Field(..., description="Column containing datetime values")

    granularity: Optional[Literal["daily", "weekly", "monthly", "quarterly", "yearly"]] = Field(
        default=None,
        description="Time granularity for grouping",
    )

    rolling_window: Optional[int] = Field(
        default=None,
        description="Rolling window size (in periods)",
    )

    period_comparison: Optional[Literal["mom", "yoy", "wow", "qoq"]] = Field(
        default=None,
        description="Period-over-period comparison type",
    )

    date_range_start: Optional[str] = Field(
        default=None,
        description="Start date filter (ISO format)",
    )

    date_range_end: Optional[str] = Field(
        default=None,
        description="End date filter (ISO format)",
    )


class AnalysisPlan(BaseModel):
    goal: str = Field(
        ...,
        description="Natural language description of what this analysis will answer",
    )

    primary_metric: str = Field(
        ...,
        description=(
            "The main metric to compute. Use 'row_count' for counting rows, "
            "or a column name for aggregating that column."
        ),
    )

    aggregation_type: Literal[
        "sum", "mean", "count", "min", "max",
        "median", "std", "var", "nunique", "percentile",
    ] = Field(..., description="Type of aggregation to apply to the metric")

    percentile_value: Optional[int] = Field(
        default=None,
        description="If aggregation_type is 'percentile', specify which percentile (1-99)",
    )

    dimensions: list[str] = Field(
        default_factory=list,
        description="Columns to group by. Empty list means aggregate the entire dataset.",
    )

    filters: list[FilterCondition] = Field(
        default_factory=list,
        description="Filter conditions to apply before aggregation",
    )

    time_config: Optional[TimeConfig] = Field(
        default=None,
        description="Time-series configuration if temporal analysis is needed",
    )

    sort_by: Optional[str] = Field(
        default=None,
        description="Column or 'metric' to sort results by",
    )

    sort_order: Literal["asc", "desc"] = Field(
        default="desc",
        description="Sort direction",
    )

    limit: Optional[int] = Field(
        default=None,
        description="Maximum number of results to return (for top-N queries)",
    )

    visualization: Literal["none", "bar", "line", "histogram", "scatter", "pie"] = Field(
        default="none",
        description="Recommended visualization type",
    )

    visualization_reason: Optional[str] = Field(
        default=None,
        description="Why this visualization was chosen (or why none)",
    )

    rationale: str = Field(
        ...,
        description="Explanation of the analytical approach and any assumptions made",
    )

    confidence: Literal["high", "medium", "low"] = Field(
        default="high",
        description="Confidence that this plan correctly interprets the user's intent",
    )

    assumptions: list[str] = Field(
        default_factory=list,
        description="List of assumptions made in creating this plan",
    )


class ReviewerOutput(BaseModel):
    approved: bool = Field(
        ...,
        description="Whether the plan is approved for execution",
    )

    issues: list[str] = Field(
        default_factory=list,
        description="List of issues found in the plan (empty if approved)",
    )

    critique: Optional[str] = Field(
        default=None,
        description="Detailed feedback for the planner if not approved",
    )

    severity: Optional[Literal["minor", "major", "critical"]] = Field(
        default=None,
        description="Severity of the issues if not approved",
    )

    suggested_fixes: Optional[list[str]] = Field(
        default=None,
        description="Specific suggestions for revising the plan",
    )


class SupervisorDecision(BaseModel):
    next_node: Literal["planner", "codeagent_executor", "explainer", "refuser"] = Field(
        ...,
        description=(
            "The next node to route to. "
            "'planner': re-create the analysis plan. "
            "'codeagent_executor': execute (or retry) the current plan. "
            "'explainer': format results for the user (only if execution succeeded). "
            "'refuser': end with a polite refusal (when limits are exhausted)."
        ),
    )

    reasoning: str = Field(
        ...,
        description="Explanation of the routing decision, referencing specific state values.",
    )

    confidence: Literal["high", "medium", "low"] = Field(
        default="high",
        description="Confidence in this routing decision.",
    )

    action_summary: str = Field(
        ...,
        description="One-sentence summary of what will happen next (shown in logs).",
    )


class ExplainerOutput(BaseModel):
    summary: str = Field(
        ...,
        description="One-sentence direct answer to the user's original question",
    )

    key_findings: list[str] = Field(
        ...,
        description="2-4 bullet points highlighting the most important insights",
    )

    formatted_results: str = Field(
        ...,
        description="Results formatted as a markdown table or appropriate structure",
    )

    context: Optional[str] = Field(
        default=None,
        description="Additional context, caveats, or data limitations",
    )

    follow_up_questions: Optional[list[str]] = Field(
        default=None,
        description="Suggested follow-up questions the user might want to explore",
    )

    visualization_caption: Optional[str] = Field(
        default=None,
        description="Caption for any generated chart or visualization",
    )

    @field_validator("key_findings", "follow_up_questions", mode="before")
    @classmethod
    def ensure_list(cls, v):
        if v is None:
            return v
        if isinstance(v, list):
            return v
        if isinstance(v, str):
            lines = re.split(r'\n(?=[•\-\*\d])', v.strip())
            result = []
            for line in lines:
                cleaned = re.sub(r'^[\s•\-\*\d\.]+', '', line).strip()
                if cleaned:
                    result.append(cleaned)
            return result if result else [v]
        return [str(v)]
