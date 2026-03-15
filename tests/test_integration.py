"""
Smoke test -- smolagents + supervisor integration
Run from project root: python tests/test_integration.py

Tests: environment, agent package imports, state fields,
       graph compilation, AnalysisPlan serialisation,
       and one live pipeline query.
"""

import json
import os
import sys
from pathlib import Path

# Ensure project root is on sys.path so 'agent' package is importable
# when this script is run directly (e.g. python tests/test_integration.py)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Load .env so ANTHROPIC_API_KEY is available before checks run
from dotenv import load_dotenv
load_dotenv(PROJECT_ROOT / ".env")


def ok(msg):
    print(f"  [OK]   {msg}")


def fail(msg, fix=""):
    print(f"  [FAIL] {msg}")
    if fix:
        print(f"         Fix: {fix}")


print("=" * 60)
print("smolagents + Supervisor Integration Smoke Test")
print("=" * 60)

passed = 0
failed = 0


def check(label, condition, fix=""):
    global passed, failed
    if condition:
        ok(label)
        passed += 1
    else:
        fail(label, fix)
        failed += 1
    return condition


# =============================================================================
# 1. Environment
# =============================================================================
print("\n1. Environment")
check(
    "ANTHROPIC_API_KEY set",
    bool(os.getenv("ANTHROPIC_API_KEY")),
    "export ANTHROPIC_API_KEY=<your-key>",
)

try:
    import smolagents as _sa
    check(f"smolagents installed ({_sa.__version__})", True)
except ImportError:
    check("smolagents installed", False, "pip install smolagents")


# =============================================================================
# 2. Agent package files present
# =============================================================================
print("\n2. Agent package files")
for f in ["agent/_shared_model.py", "agent/tools.py", "agent/nodes.py",
          "agent/graph_builder.py", "agent/state.py", "agent/models.py"]:
    full = PROJECT_ROOT / f
    check(str(f), full.exists(), f"missing at {full}")


# =============================================================================
# 3. Tool imports
# =============================================================================
print("\n3. Tool imports")
try:
    from agent.tools import (
        load_csv_data,
        calculate_statistics,
        group_and_aggregate,
        filter_rows,
    )
    check("All 4 tools import cleanly", True)
except Exception as e:
    check("All 4 tools import cleanly", False, str(e))


# =============================================================================
# 4. Node imports
# =============================================================================
print("\n4. Node imports")
try:
    from agent.nodes import codeagent_executor_node, supervisor_node
    check("codeagent_executor_node imports cleanly", True)
    check("supervisor_node imports cleanly", True)
except Exception as e:
    check("Node imports", False, str(e))


# =============================================================================
# 5. State fields
# =============================================================================
print("\n5. State fields")
try:
    from agent.state import AgentState
    hints = AgentState.__annotations__
    for field in ("generated_code", "agent_steps", "next_node",
                  "supervisor_reasoning", "supervisor_confidence"):
        check(
            f"'{field}' in AgentState",
            field in hints,
            f"add '{field}' to AgentState in agent/state.py",
        )
except Exception as e:
    check("AgentState importable", False, str(e))


# =============================================================================
# 6. Graph compiles
# =============================================================================
print("\n6. Graph compilation")
graph = None
try:
    from agent.graph_builder import compile_graph
    graph = compile_graph(checkpointer=False)
    check("Graph compiles and builds successfully", True)
except Exception as e:
    check("Graph compiles", False, str(e))
    print("\n  Stopping -- graph must compile before live test.")
    sys.exit(1)


# =============================================================================
# 7. AnalysisPlan serialisation (no API call)
# =============================================================================
print("\n7. AnalysisPlan unit test (no LLM call)")

DATA_DIR = PROJECT_ROOT / "data"
CSV_PATH = DATA_DIR / "olist_geolocation_dataset.csv"
if not CSV_PATH.exists():
    # Fallback: first CSV found in data/
    csvs = list(DATA_DIR.glob("*.csv"))
    CSV_PATH = csvs[0] if csvs else None

if CSV_PATH:
    try:
        from agent.models import AnalysisPlan

        dummy_plan = AnalysisPlan(
            goal="Find top 3 states by record count",
            primary_metric="record_count",
            aggregation_type="count",
            dimensions=["geolocation_state"],
            filters=[],
            sort_by="metric",
            sort_order="desc",
            limit=3,
            visualization="none",
            visualization_reason="simple ranking does not need a chart",
            rationale="test",
            confidence="high",
            assumptions=[],
        )
        plan_dict = dummy_plan.model_dump(mode="json")
        check("AnalysisPlan.model_dump(mode='json') works", bool(plan_dict))
        check("plan_dict has 'goal' key", "goal" in plan_dict)
        check(
            "plan_dict has 'dimensions' key",
            "dimensions" in plan_dict and plan_dict["dimensions"] == ["geolocation_state"],
        )
    except Exception as e:
        check("AnalysisPlan unit test", False, str(e))
else:
    print("  Skipping (no CSV found in data/)")


# =============================================================================
# 8. Live pipeline test
# =============================================================================
print("\n8. Live pipeline test (real API calls -- ~15-30s)")

if not CSV_PATH:
    print("  Skipping -- no CSV file found in data/")
else:
    CSV_PATH = str(CSV_PATH)

    # Load schema_profile.json from data/ if available
    schema = {}
    schema_path = DATA_DIR / "schema_profile.json"
    if schema_path.exists():
        with open(schema_path) as f:
            schema = json.load(f)
        print(f"  Schema loaded: {schema.get('file_name', 'unknown')}")
    else:
        print("  schema_profile.json not found in data/ -- using empty schema")

    # Use the CSV path recorded in the schema so CSV and schema always match
    live_csv_path = schema.get("file_path", CSV_PATH)
    if not Path(live_csv_path).exists():
        live_csv_path = CSV_PATH
    print(f"  CSV for live test: {Path(live_csv_path).name}")

    # Pick a query that matches whichever schema was loaded
    schema_columns = [c.get("name", "") for c in schema.get("columns", [])]
    if any("customer_state" in c or "seller_state" in c for c in schema_columns):
        live_query = "What are the top 5 customer states by number of orders?"
    elif any("geolocation_state" in c for c in schema_columns):
        live_query = "What are the top 3 states with the most geolocation records?"
    else:
        live_query = "How many records are in the dataset?"
    print(f"  Query: {live_query}")

    try:
        from agent.state import create_initial_state

        initial_state = create_initial_state(
            user_query=live_query,
            schema_profile=schema,
            csv_file_path=live_csv_path,
        )
    except Exception as e:
        print(f"  Could not build initial state: {e}")
        initial_state = {
            "user_query":          live_query,
            "csv_file_path":       live_csv_path,
            "schema_profile":      schema,
            "retry_count":         0,
            "plan_revision_count": 0,
            "execution_path":      [],
        }

    try:
        result = graph.invoke(initial_state)

        exec_result = result.get("execution_result") or {}
        success     = exec_result.get("success", False)
        agent_steps = result.get("agent_steps", "N/A")
        code_blocks = len(result.get("generated_code") or [])
        final_resp  = result.get("final_response", "")
        exec_path   = " -> ".join(result.get("execution_path") or [])
        supervisor  = result.get("supervisor_reasoning", "N/A")

        check(f"Pipeline ran (path: {exec_path})", True)
        check(
            "execution_result.success is True",
            success,
            "Check codeagent_executor_node -- result['execution_result']['output'] may be None",
        )
        check(
            "final_response populated",
            bool(final_resp),
            "Check explainer_node -- execution_result.output may be None",
        )

        print(f"\n  Code blocks written by agent : {code_blocks}")
        print(f"  Agent steps taken            : {agent_steps}")
        print(f"  Supervisor reasoning         : {str(supervisor)[:120]}")
        if final_resp:
            print("\n  Final response (first 400 chars):")
            preview = str(final_resp)[:400].encode("ascii", errors="replace").decode("ascii")
            print(f"  {preview}")

    except Exception as e:
        check("Pipeline ran without exception", False, str(e))
        import traceback
        traceback.print_exc()


# =============================================================================
# Summary
# =============================================================================
print("\n" + "=" * 60)
print(f"Results: {passed} passed, {failed} failed")
if failed == 0:
    print("STATUS: PASSED")
    print("\nNext steps:")
    print("  1. Run: python run_agent.py")
    print("  2. Monitor agent_steps -- healthy range is 3-6 per query")
    print("  3. Monitor supervisor_reasoning for routing quality")
    print("  4. Once stable, remove generator/executor/debugger legacy nodes")
else:
    print("STATUS: FAILED -- fix the issues above and re-run")
print("=" * 60)

sys.exit(0 if failed == 0 else 1)
