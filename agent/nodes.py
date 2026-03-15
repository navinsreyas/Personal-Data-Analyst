from __future__ import annotations

import json
import logging
import os
import traceback
from datetime import datetime
from typing import Optional

from dotenv import load_dotenv
load_dotenv()

from langchain_core.messages import HumanMessage, SystemMessage
from smolagents import CodeAgent

from ._shared_model import get_model
from .tools import load_csv_data, calculate_statistics, group_and_aggregate, filter_rows, generate_chart
from .memory import SchemaHasher, get_plan_cache
from .models import (
    AnalysisPlan,
    ExplainerOutput,
    ReviewerOutput,
    RouterOutput,
    SupervisorDecision,
)
from .prompts import (
    EXPLAINER_SYSTEM_PROMPT,
    EXPLAINER_USER_PROMPT,
    PLANNER_SYSTEM_PROMPT,
    PLANNER_USER_PROMPT,
    REVIEWER_SYSTEM_PROMPT,
    REVIEWER_USER_PROMPT,
    ROUTER_SYSTEM_PROMPT,
    ROUTER_USER_PROMPT,
    format_grain_description,
    format_schema_summary,
)
from .state import (
    AgentState,
    MAX_CODE_RETRIES,
    MAX_PLAN_REVISIONS,
    validate_state_for_node,
)

logger = logging.getLogger(__name__)

SUPERVISOR_VALID_NODES = frozenset({"planner", "codeagent_executor", "explainer", "refuser"})


def get_llm(temperature: float = 0, model: Optional[str] = None):
    anthropic_key = os.getenv("ANTHROPIC_API_KEY")
    openai_key    = os.getenv("OPENAI_API_KEY")

    if anthropic_key:
        from langchain_anthropic import ChatAnthropic
        model = model or "claude-sonnet-4-6"
        logger.info(f"Using Anthropic model: {model}")
        return ChatAnthropic(model=model, temperature=temperature, api_key=anthropic_key)
    elif openai_key:
        from langchain_openai import ChatOpenAI
        model = model or "gpt-4o"
        logger.info(f"Using OpenAI model: {model}")
        return ChatOpenAI(model=model, temperature=temperature, api_key=openai_key)
    else:
        raise ValueError("No LLM API key found. Set ANTHROPIC_API_KEY or OPENAI_API_KEY.")


def router_node(state: AgentState) -> dict:
    logger.info("=" * 60)
    logger.info("ROUTER NODE - Classifying user query")
    logger.info(f"Query: {state.get('user_query', 'N/A')}")

    is_valid, error = validate_state_for_node(state, "router")
    if not is_valid:
        logger.error(f"State validation failed: {error}")
        return {
            "router_decision": "red",
            "refusal_reason":  f"Internal error: {error}",
            "current_node":    "router",
            "execution_path":  state.get("execution_path", []) + ["router"],
        }

    try:
        llm            = get_llm(temperature=0)
        structured_llm = llm.with_structured_output(RouterOutput)

        schema_summary = format_schema_summary(state["schema_profile"])
        messages = [
            SystemMessage(content=ROUTER_SYSTEM_PROMPT.format(schema_summary=schema_summary)),
            HumanMessage(content=ROUTER_USER_PROMPT.format(user_query=state["user_query"])),
        ]
        result: RouterOutput = structured_llm.invoke(messages)

        logger.info(f"Classification: {result.classification.upper()}")
        logger.info(f"Reasoning: {result.reasoning}")

        updates = {
            "router_decision": result.classification,
            "current_node":    "router",
            "execution_path":  state.get("execution_path", []) + ["router"],
        }

        if result.classification == "green":
            updates["identified_metric"]     = result.identified_metric
            updates["identified_dimensions"] = result.identified_dimensions
            updates["identified_filters"]    = result.identified_filters
            logger.info(f"Extracted metric: {result.identified_metric}")
            logger.info(f"Extracted dimensions: {result.identified_dimensions}")
        elif result.classification == "yellow":
            updates["clarification_question"] = result.clarification_question
            updates["ambiguous_terms"]         = result.ambiguous_terms
            logger.info(f"Clarification needed: {result.clarification_question}")
        else:
            updates["refusal_reason"]       = result.refusal_reason
            updates["suggested_alternative"] = result.suggested_alternative
            logger.info(f"Refusal reason: {result.refusal_reason}")

        logger.info("=" * 60)
        return updates

    except Exception as e:
        logger.error(f"Router error: {e}")
        logger.error(traceback.format_exc())
        return {
            "router_decision": "red",
            "refusal_reason":  f"Error processing query: {str(e)}",
            "current_node":    "router",
            "execution_path":  state.get("execution_path", []) + ["router"],
        }


def planner_node(state: AgentState) -> dict:
    logger.info("=" * 60)
    logger.info("PLANNER NODE - Generating analysis plan")
    logger.info(f"Query: {state.get('user_query', 'N/A')}")

    is_valid, error = validate_state_for_node(state, "planner")
    if not is_valid:
        logger.error(f"State validation failed: {error}")
        return {
            "analysis_plan":  None,
            "current_node":   "planner",
            "execution_path": state.get("execution_path", []) + ["planner"],
        }

    critique       = state.get("plan_critique")
    revision_count = state.get("plan_revision_count", 0)
    schema_hash    = SchemaHasher.hash_schema(state["schema_profile"])

    if not critique:
        cache        = get_plan_cache()
        cached_plan  = cache.lookup(state["user_query"], schema_hash)
        if cached_plan:
            logger.info("CACHE HIT - Using cached analysis plan")
            logger.info("=" * 60)
            return {
                "analysis_plan":  cached_plan,
                "plan_critique":  None,
                "plan_approved":  None,
                "schema_hash":    schema_hash,
                "cache_hit":      True,
                "current_node":   "planner",
                "execution_path": state.get("execution_path", []) + ["planner"],
            }
        logger.info("Cache miss - generating new plan via LLM")

    context = ""
    if critique:
        context = f"Previous plan was rejected. Reviewer feedback: {critique}\nPlease revise the plan to address these issues."
        logger.info(f"Revising plan (attempt {revision_count + 1}): {critique}")

    try:
        llm            = get_llm(temperature=0)
        structured_llm = llm.with_structured_output(AnalysisPlan)

        schema_summary    = format_schema_summary(state["schema_profile"])
        grain_description = format_grain_description(state["schema_profile"])

        messages = [
            SystemMessage(content=PLANNER_SYSTEM_PROMPT.format(
                schema_summary=schema_summary,
                grain_description=grain_description,
            )),
            HumanMessage(content=PLANNER_USER_PROMPT.format(
                user_query=state["user_query"],
                context=context or "None",
            )),
        ]
        result: AnalysisPlan = structured_llm.invoke(messages)

        logger.info(f"Generated plan:")
        logger.info(f"  Goal: {result.goal}")
        logger.info(f"  Metric: {result.primary_metric} ({result.aggregation_type})")
        logger.info(f"  Dimensions: {result.dimensions}")
        logger.info(f"  Filters: {len(result.filters)} conditions")
        logger.info(f"  Visualization: {result.visualization}")
        logger.info(f"  Confidence: {result.confidence}")
        logger.info("=" * 60)

        return {
            "analysis_plan":  result.model_dump(),
            "plan_critique":  None,
            "plan_approved":  None,
            "schema_hash":    schema_hash,
            "cache_hit":      False,
            "current_node":   "planner",
            "execution_path": state.get("execution_path", []) + ["planner"],
        }

    except Exception as e:
        logger.error(f"Planner error: {e}")
        logger.error(traceback.format_exc())
        return {
            "analysis_plan":  None,
            "error_message":  f"Error generating plan: {str(e)}",
            "current_node":   "planner",
            "execution_path": state.get("execution_path", []) + ["planner"],
        }


def reviewer_node(state: AgentState) -> dict:
    logger.info("=" * 60)
    logger.info("REVIEWER NODE - Validating analysis plan")

    is_valid, error = validate_state_for_node(state, "reviewer")
    if not is_valid:
        logger.error(f"State validation failed: {error}")
        return {
            "plan_approved":  False,
            "plan_critique":  f"Cannot review: {error}",
            "current_node":   "reviewer",
            "execution_path": state.get("execution_path", []) + ["reviewer"],
        }

    plan = state["analysis_plan"]
    logger.info(f"Reviewing plan: {plan.get('goal', 'Unknown')}")

    if state.get("cache_hit"):
        logger.info("Skipping LLM review — plan came from cache (already approved)")
        return {
            "plan_approved":  True,
            "plan_critique":  None,
            "current_node":   "reviewer",
            "execution_path": state.get("execution_path", []) + ["reviewer"],
        }

    try:
        llm            = get_llm(temperature=0)
        structured_llm = llm.with_structured_output(ReviewerOutput)

        schema_summary = format_schema_summary(state["schema_profile"])
        messages = [
            SystemMessage(content=REVIEWER_SYSTEM_PROMPT.format(schema_summary=schema_summary)),
            HumanMessage(content=REVIEWER_USER_PROMPT.format(
                user_query=state["user_query"],
                analysis_plan=json.dumps(plan, indent=2, default=str),
            )),
        ]
        result: ReviewerOutput = structured_llm.invoke(messages)

        if result.approved:
            logger.info("Plan APPROVED")
            schema_hash = state.get("schema_hash")
            if schema_hash and state.get("plan_revision_count", 0) == 0:
                try:
                    cache = get_plan_cache()
                    cache.save(
                        query=state["user_query"],
                        schema_hash=schema_hash,
                        plan=plan,
                        metadata={"approved_at": datetime.now().isoformat()},
                    )
                    logger.info("Plan saved to cache")
                except Exception as cache_error:
                    logger.warning(f"Failed to cache plan: {cache_error}")
        else:
            logger.warning(f"Plan REJECTED: {result.critique}")
            logger.warning(f"Issues: {result.issues}")

        revision_count = state.get("plan_revision_count", 0)
        if not result.approved:
            revision_count += 1

        logger.info("=" * 60)
        return {
            "plan_approved":        result.approved,
            "plan_critique":        result.critique if not result.approved else None,
            "plan_revision_count":  revision_count,
            "current_node":         "reviewer",
            "execution_path":       state.get("execution_path", []) + ["reviewer"],
        }

    except Exception as e:
        logger.error(f"Reviewer error: {e}")
        logger.error(traceback.format_exc())
        return {
            "plan_approved":  True,
            "plan_critique":  None,
            "current_node":   "reviewer",
            "execution_path": state.get("execution_path", []) + ["reviewer"],
        }


def explainer_node(state: AgentState) -> dict:
    logger.info("=" * 60)
    logger.info("EXPLAINER NODE - Generating final response")

    is_valid, error = validate_state_for_node(state, "explainer")
    if not is_valid:
        logger.error(f"State validation failed: {error}")

    result = state.get("execution_result", {})
    query  = state.get("user_query", "")
    plan   = state.get("analysis_plan", {})

    try:
        llm            = get_llm(temperature=0.3)
        structured_llm = llm.with_structured_output(ExplainerOutput)

        results_str = json.dumps(result.get("output"), indent=2, default=str)
        plan_str    = json.dumps(plan, indent=2, default=str)

        messages = [
            SystemMessage(content=EXPLAINER_SYSTEM_PROMPT),
            HumanMessage(content=EXPLAINER_USER_PROMPT.format(
                user_query=query,
                analysis_plan=plan_str,
                results=results_str,
                execution_time=result.get("execution_time_ms", 0),
            )),
        ]
        explanation: ExplainerOutput = structured_llm.invoke(messages)

        response_parts = [f"## {explanation.summary}", "", "### Key Findings"]
        for finding in explanation.key_findings:
            response_parts.append(f"- {finding}")
        response_parts.extend(["", "### Results", explanation.formatted_results])
        if explanation.context:
            response_parts.extend(["", f"*{explanation.context}*"])
        if explanation.follow_up_questions:
            response_parts.extend(["", "### You might also want to know:"])
            for q in explanation.follow_up_questions:
                response_parts.append(f"- {q}")

        final_response = "\n".join(response_parts)
        logger.info("Generated response preview:")
        logger.info(final_response[:300] + "..." if len(final_response) > 300 else final_response)
        logger.info("=" * 60)

        return {
            "final_response":   final_response,
            "completed_at":     datetime.now().isoformat(),
            "current_node":     "explainer",
            "execution_path":   state.get("execution_path", []) + ["explainer"],
            "total_tokens_used": state.get("total_tokens_used", 0),
        }

    except Exception as e:
        logger.error(f"Explainer error: {e}")
        output = result.get("output", "No results")
        fallback = f"## Analysis Results\n\n**Query:** {query}\n\n**Results:**\n```\n{output}\n```\n\n*Analysis completed successfully.*\n"
        return {
            "final_response": fallback,
            "completed_at":   datetime.now().isoformat(),
            "current_node":   "explainer",
            "execution_path": state.get("execution_path", []) + ["explainer"],
        }


def clarifier_node(state: AgentState) -> dict:
    logger.info("=" * 60)
    logger.info("CLARIFIER NODE - Requesting clarification")
    logger.info(f"Original query: {state.get('user_query', 'N/A')}")

    question        = state.get("clarification_question", "Could you provide more details?")
    ambiguous_terms = state.get("ambiguous_terms", [])

    response = (
        f"## Clarification Needed\n\n"
        f"I need a bit more information to answer your question accurately.\n\n"
        f"**Your question:** {state.get('user_query', '')}\n\n"
        f"**What I need to know:** {question}\n"
    )
    if ambiguous_terms:
        response += f"\n**Ambiguous terms:** {', '.join(ambiguous_terms)}"
    response += "\n\nPlease provide more details so I can help you better."

    logger.info("=" * 60)
    return {
        "final_response":       response,
        "clarification_question": question,
        "current_node":         "clarifier",
        "execution_path":       state.get("execution_path", []) + ["clarifier"],
    }


def refuser_node(state: AgentState) -> dict:
    logger.info("=" * 60)
    logger.info("REFUSER NODE - Generating refusal response")

    reason      = state.get("refusal_reason", "This query cannot be processed.")
    query       = state.get("user_query", "")
    alternative = state.get("suggested_alternative", "")

    logger.info(f"Refusal reason: {reason}")

    response = f"""## Unable to Process This Query

**Your question:** {query}

**Why I can't help with this:** {reason}

### What I Can Do Instead

I'm designed for **descriptive analytics** - analyzing what has happened in your data. I can help with:

- **Aggregations**: Sum, count, average, min, max of any numeric column
- **Grouping**: Break down metrics by any categorical column
- **Filtering**: Focus on specific subsets of data
- **Trends**: See how metrics change over time
- **Rankings**: Find top/bottom performers
- **Distributions**: Understand the spread of values

### What I Cannot Do

- **Predict** future values
- **Explain** why something happened (causality)
- **Recommend** what you should do
- **Access** data outside this dataset
"""
    if alternative:
        response += f"\n### Try This Instead\n\n{alternative}"

    logger.info("=" * 60)
    return {
        "final_response": response,
        "completed_at":   datetime.now().isoformat(),
        "current_node":   "refuser",
        "execution_path": state.get("execution_path", []) + ["refuser"],
    }


def codeagent_executor_node(state: AgentState) -> dict:
    logger.info("=" * 60)
    logger.info("CODEAGENT EXECUTOR NODE - Running smolagents CodeAgent")

    csv_path       = state.get("csv_file_path", "")
    schema_profile = state.get("schema_profile", {})
    analysis_plan  = state.get("analysis_plan")
    user_query     = state.get("user_query", "")
    retry_count    = state.get("retry_count", 0)

    if analysis_plan is not None and hasattr(analysis_plan, "model_dump"):
        plan_dict = analysis_plan.model_dump(mode="json")
    elif analysis_plan is not None:
        plan_dict = dict(analysis_plan)
    else:
        plan_dict = {}

    schema_context = ""
    try:
        schema_context = format_schema_summary(schema_profile)
    except Exception:
        cols = schema_profile.get("columns", [])
        if isinstance(cols, list):
            schema_context = "Columns: " + ", ".join(
                c.get("name", "") for c in cols if isinstance(c, dict)
            )

    prompt = f"""You are a data analyst. Analyse the CSV at: '{csv_path}'

DATASET SCHEMA:
{schema_context}

ANALYSIS PLAN:
{json.dumps(plan_dict, indent=2)}

ORIGINAL QUESTION:
{user_query}

Instructions:
1. Call load_csv_data('{csv_path}') first to load the data.
2. Execute every step in the analysis plan using the available tools.
3. For any computation not covered by a tool, write raw Python directly.
4. Store your final computed result in a variable named result_output.
5. Call final_answer(result_output) -- this ends the agent loop.

The result must directly answer: {user_query}
"""

    max_steps = 5 + (retry_count * 2)

    agent = CodeAgent(
        tools=[load_csv_data, calculate_statistics, group_and_aggregate, filter_rows, generate_chart],
        model=get_model(),
        additional_authorized_imports=["pandas", "numpy", "json", "statistics"],
        max_steps=max_steps,
        verbosity_level=0,
    )

    try:
        result_value = agent.run(prompt)

        generated_code = [
            step.code_action
            for step in agent.memory.steps
            if hasattr(step, "code_action") and step.code_action
        ]

        chart_path = None
        if isinstance(result_value, dict) and "chart_path" in result_value:
            chart_path = result_value["chart_path"]
        elif isinstance(result_value, str):
            try:
                parsed = json.loads(result_value)
                if isinstance(parsed, dict) and "chart_path" in parsed:
                    chart_path = parsed["chart_path"]
            except (json.JSONDecodeError, TypeError):
                pass

        import pandas as pd
        if isinstance(result_value, pd.DataFrame):
            output_type   = "dataframe"
            result_output = result_value.to_dict("records")
        elif isinstance(result_value, list):
            output_type   = "list"
            result_output = result_value
        elif isinstance(result_value, (int, float)):
            output_type   = "scalar"
            result_output = result_value
        else:
            output_type   = "scalar"
            result_output = result_value

        logger.info(f"CodeAgent completed in {len(agent.memory.steps)} steps")
        logger.info("=" * 60)

        return {
            "execution_result": {
                "success":           True,
                "output":            result_output,
                "output_type":       output_type,
                "chart_path":        chart_path,
                "error":             None,
                "error_type":        None,
                "execution_time_ms": 0,
            },
            "python_code":     "\n\n# --- next block ---\n\n".join(generated_code),
            "generated_code":  generated_code,
            "agent_steps":     len(agent.memory.steps),
            "error_message":   None,
            "error_traceback": None,
            "current_node":    "codeagent_executor",
            "execution_path":  state.get("execution_path", []) + ["codeagent_executor"],
        }

    except Exception as e:
        tb_str = traceback.format_exc()
        logger.error(f"CodeAgent failed: {e}")
        logger.error(tb_str)
        logger.info("=" * 60)

        return {
            "execution_result": {
                "success":           False,
                "output":            None,
                "output_type":       "error",
                "chart_path":        None,
                "error":             str(e),
                "error_type":        type(e).__name__,
                "execution_time_ms": 0,
            },
            "python_code":     "",
            "generated_code":  [],
            "agent_steps":     0,
            "error_message":   str(e),
            "error_traceback": tb_str,
            "current_node":    "codeagent_executor",
            "execution_path":  state.get("execution_path", []) + ["codeagent_executor"],
        }


def supervisor_node(state: AgentState) -> dict:
    logger.info("=" * 60)
    logger.info("SUPERVISOR NODE - Making intelligent routing decision")

    plan_approved       = state.get("plan_approved")
    plan_critique       = state.get("plan_critique")
    plan_revision_count = state.get("plan_revision_count", 0)
    execution_result    = state.get("execution_result")
    retry_count         = state.get("retry_count", 0)
    error_message       = state.get("error_message")
    error_traceback     = state.get("error_traceback", "")
    error_history       = list(state.get("error_history") or [])
    user_query          = state.get("user_query", "")

    came_from_executor_failure = (
        execution_result is not None
        and not execution_result.get("success", True)
    )

    logger.info(
        f"Context: {'executor failure' if came_from_executor_failure else 'plan review'} | "
        f"plan_approved={plan_approved} | plan_revisions={plan_revision_count}/{MAX_PLAN_REVISIONS} | "
        f"retries={retry_count}/{MAX_CODE_RETRIES}"
    )

    if came_from_executor_failure:
        context_lines = [
            f"Context: Execution just FAILED — deciding how to recover.",
            f"User query: {user_query}",
            f"Error message: {error_message}",
            f"Error type: {execution_result.get('error_type', 'Unknown')}",
            f"Retries used so far: {retry_count} / {MAX_CODE_RETRIES}",
            f"Plan revisions used: {plan_revision_count} / {MAX_PLAN_REVISIONS}",
        ]
        if error_history:
            context_lines.append(f"Error history (last 3): {' | '.join(error_history[-3:])}")
    else:
        context_lines = [
            f"Context: Reviewing plan outcome — deciding what to execute next.",
            f"User query: {user_query}",
            f"Plan approved: {plan_approved}",
            f"Plan revisions used: {plan_revision_count} / {MAX_PLAN_REVISIONS}",
        ]
        if plan_critique:
            context_lines.append(f"Reviewer critique: {plan_critique}")

    context_str = "\n".join(context_lines)

    system_prompt = f"""You are the supervisor of a data analysis pipeline.
Your only job is to decide the next node to route to based on the current state.

Available next nodes:
  "planner"            — Re-generate the analysis plan (use when plan is flawed or needs revision).
  "codeagent_executor" — Run the analysis plan (use when plan is approved, or retry after fixable error).
  "explainer"          — Format results for the user (ONLY if execution already succeeded — rare from here).
  "refuser"            — End with a polite refusal (use when all retries/revisions are exhausted).

Hard rules (follow these exactly):
  1. Plan review context (execution_result is None):
       - If plan_approved=True  → "codeagent_executor"
       - If plan_approved=False AND plan_revisions < {MAX_PLAN_REVISIONS} → "planner"
       - If plan_approved=False AND plan_revisions >= {MAX_PLAN_REVISIONS} → "refuser"
  2. Executor failure context:
       - If retries < {MAX_CODE_RETRIES} AND error looks fixable (not a column/schema mismatch) → "codeagent_executor"
       - If retries < {MAX_CODE_RETRIES} AND error suggests plan is wrong (KeyError, column not found) → "planner"
       - If retries >= {MAX_CODE_RETRIES} → "refuser"
  3. Never return "explainer" unless execution clearly already succeeded.
  4. When you choose "refuser", set action_summary to explain what was exhausted."""

    user_prompt = f"""Current pipeline state:\n\n{context_str}\n\nDecide the next node. Follow the hard rules precisely."""

    try:
        llm            = get_llm(temperature=0)
        structured_llm = llm.with_structured_output(SupervisorDecision)

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt),
        ]
        decision: SupervisorDecision = structured_llm.invoke(messages)

        next_node = decision.next_node
        if next_node not in SUPERVISOR_VALID_NODES:
            logger.warning(f"Supervisor returned invalid node '{next_node}' — defaulting to refuser")
            next_node = "refuser"

        logger.info(f"Supervisor decision: {next_node} (confidence={decision.confidence})")
        logger.info(f"Reasoning: {decision.reasoning}")
        logger.info(f"Action: {decision.action_summary}")
        logger.info("=" * 60)

        updates: dict = {
            "next_node":             next_node,
            "supervisor_reasoning":  decision.reasoning,
            "supervisor_confidence": decision.confidence,
            "current_node":          "supervisor",
            "execution_path":        state.get("execution_path", []) + ["supervisor"],
        }

        if came_from_executor_failure and next_node == "codeagent_executor":
            updates["retry_count"] = retry_count + 1

        if came_from_executor_failure and error_message:
            error_history.append(error_message)
            updates["error_history"] = error_history

        if next_node == "refuser":
            if came_from_executor_failure:
                updates["refusal_reason"] = (
                    f"Analysis failed after {retry_count} retries. Last error: {error_message}"
                )
            else:
                updates["refusal_reason"] = (
                    f"Could not produce an approved plan after {plan_revision_count} revision(s). "
                    f"Reviewer critique: {plan_critique}"
                )

        return updates

    except Exception as e:
        logger.error(f"Supervisor node failed: {e}")
        logger.error(traceback.format_exc())
        logger.info("=" * 60)

        return {
            "next_node":             "refuser",
            "supervisor_reasoning":  f"Supervisor encountered an unexpected error: {str(e)}",
            "supervisor_confidence": "low",
            "refusal_reason":        f"Internal routing error in supervisor: {str(e)}",
            "current_node":          "supervisor",
            "execution_path":        state.get("execution_path", []) + ["supervisor"],
        }
