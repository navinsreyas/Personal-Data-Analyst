from __future__ import annotations

import argparse
import json
import os
import sys
import uuid
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
load_dotenv()

from agent.graph_builder import compile_graph
from agent.memory import (
    ConversationMemory,
    SchemaHasher,
    get_conversation_memory,
    get_plan_cache,
)
from scanner import DataScanner
from agent.state import create_initial_state


def load_or_create_schema(csv_path: str) -> tuple[dict, str]:
    csv_path = Path(csv_path).resolve()

    schema_path = csv_path.parent / "schema_profile.json"

    if schema_path.exists():
        with open(schema_path) as f:
            schema = json.load(f)

        if schema.get("file_name") == csv_path.name:
            print(f"Using existing schema profile for: {csv_path.name}")
            return schema, str(csv_path)

    print(f"Scanning CSV file: {csv_path}")
    scanner = DataScanner(csv_path)
    schema_profile = scanner.scan()
    scanner.to_json(schema_path)
    print(f"Schema profile saved to: {schema_path}")

    return schema_profile.model_dump(), str(csv_path)


def run_query(
    graph,
    schema: dict,
    csv_path: str,
    query: str,
    thread_id: str,
    clarification_response: Optional[str] = None
) -> dict:
    config = {"configurable": {"thread_id": thread_id}}

    initial_state = create_initial_state(
        user_query=query,
        schema_profile=schema,
        csv_file_path=csv_path
    )

    if clarification_response:
        initial_state["clarification_response"] = clarification_response

    return graph.invoke(initial_state, config)


def handle_clarification(
    graph,
    state: dict,
    schema: dict,
    csv_path: str,
    thread_id: str,
    memory: ConversationMemory
) -> dict:
    question = state.get("clarification_question", "Could you provide more details?")

    print("\n" + "=" * 60)
    print("CLARIFICATION NEEDED")
    print("=" * 60)
    print(f"\n{question}\n")

    memory.set_pending_clarification(
        thread_id=thread_id,
        clarification_question=question,
        partial_state={
            "user_query": state.get("user_query"),
            "router_decision": state.get("router_decision"),
            "ambiguous_terms": state.get("ambiguous_terms")
        }
    )

    try:
        response = input("Your answer: ").strip()
        if not response:
            print("No response provided. Ending this query.")
            return state

        original_query = state.get("user_query", "")
        refined_query = f"{original_query} (Clarification: {response})"

        print(f"\nRetrying with: {refined_query}")

        return run_query(
            graph=graph,
            schema=schema,
            csv_path=csv_path,
            query=refined_query,
            thread_id=thread_id,
            clarification_response=response
        )

    except KeyboardInterrupt:
        print("\n\nClarification cancelled.")
        return state


def print_result(state: dict, verbose: bool = False) -> None:
    print("\n" + "=" * 70)

    decision = state.get("router_decision", "unknown")
    path = state.get("execution_path", [])

    print(f"Path: {' -> '.join(path)}")
    print(f"Decision: {decision.upper()}")

    if state.get("cache_hit"):
        print("Cache: HIT (used cached plan)")

    if state.get("plan_approved") is not None:
        print(f"Plan Approved: {state['plan_approved']}")

    if state.get("retry_count", 0) > 0:
        print(f"Retries: {state['retry_count']}")

    print("=" * 70)

    response = state.get("final_response")
    if response:
        print("\n" + response)
    else:
        clarification = state.get("clarification_question")
        if clarification:
            print(f"\nClarification needed: {clarification}")
        else:
            print("\nNo response generated.")

    if verbose:
        print("\n--- DEBUG INFO ---")
        if state.get("analysis_plan"):
            print(f"Plan: {json.dumps(state['analysis_plan'], indent=2, default=str)[:500]}...")
        if state.get("execution_result"):
            print(f"Execution: {state['execution_result'].get('success')}")

    print()


def print_welcome(schema: dict, thread_id: str) -> None:
    print("\n" + "=" * 70)
    print("PERSONAL DATA ANALYST AGENT - Interactive Mode")
    print("=" * 70)
    print(f"\nSession ID: {thread_id}")
    print(f"Data: {schema.get('file_name', 'Unknown')}")
    print(f"Rows: {schema.get('row_count', '?'):,} | Columns: {schema.get('column_count', '?')}")

    cache = get_plan_cache()
    stats = cache.get_stats()
    print(f"Cache: {stats['total_entries']} plans, {stats['total_hits']} hits")

    print(f"\nCommands:")
    print("  'quit' or 'exit' - End session")
    print("  'schema' - Show available columns")
    print("  'help' - Show what I can do")
    print("  'cache' - Show cache statistics")
    print("  'threads' - List saved sessions")
    print("-" * 70)


def interactive_mode(
    graph,
    schema: dict,
    csv_path: str,
    thread_id: str
) -> None:
    memory = get_conversation_memory()

    pending = memory.get_pending_clarification(thread_id)
    if pending:
        print(f"\nResuming session with pending clarification:")
        print(f"  Question: {pending['question']}")
        print(f"  Asked at: {pending['asked_at']}")

    print_welcome(schema, thread_id)

    while True:
        try:
            query = input("\nYou: ").strip()

            if not query:
                continue

            if query.lower() in ["quit", "exit", "q"]:
                print("Session saved. Goodbye!")
                break

            if query.lower() == "schema":
                print("\n--- SCHEMA SUMMARY ---")
                print(f"Categorical: {schema.get('categorical_columns', [])}")
                print(f"Numerical: {schema.get('numerical_columns', [])}")
                print(f"Datetime: {schema.get('datetime_columns', [])}")
                if schema.get("grain_hint"):
                    print(f"Grain: {schema['grain_hint']}")
                continue

            if query.lower() == "help":
                print("\n--- WHAT I CAN DO ---")
                print("- Aggregations: 'Total orders by status', 'Average price per category'")
                print("- Rankings: 'Top 10 customers by order count'")
                print("- Trends: 'Monthly order trends', 'Orders over time'")
                print("- Distributions: 'Distribution of order values'")
                print("- Filters: 'Orders where status is delivered'")
                print("\n--- WHAT I CANNOT DO ---")
                print("- Predictions: 'What will sales be next month?'")
                print("- Causality: 'Why did orders drop?'")
                print("- Recommendations: 'What should we do?'")
                continue

            if query.lower() == "cache":
                cache = get_plan_cache()
                stats = cache.get_stats()
                print("\n--- CACHE STATISTICS ---")
                print(f"Total schemas: {stats['total_schemas']}")
                print(f"Total plans: {stats['total_entries']}")
                print(f"Total hits: {stats['total_hits']}")
                print(f"Cache file: {stats['cache_file']}")
                continue

            if query.lower() == "threads":
                threads = memory.list_threads()
                print("\n--- SAVED SESSIONS ---")
                if threads:
                    for t in threads[:10]:
                        status = "[pending]" if t.get("has_pending") else ""
                        print(f"  {t['thread_id']}: {t['turn_count']} turns {status}")
                else:
                    print("  No saved sessions found.")
                continue

            print("\nAnalyzing...")
            state = run_query(graph, schema, csv_path, query, thread_id)

            if state.get("router_decision") == "yellow" and not state.get("final_response"):
                state = handle_clarification(
                    graph=graph,
                    state=state,
                    schema=schema,
                    csv_path=csv_path,
                    thread_id=thread_id,
                    memory=memory
                )

            print_result(state)

            response = state.get("final_response", state.get("clarification_question", ""))
            memory.add_turn(
                thread_id=thread_id,
                user_query=query,
                response=response,
                state_snapshot={
                    "router_decision": state.get("router_decision"),
                    "plan_approved": state.get("plan_approved"),
                    "cache_hit": state.get("cache_hit", False)
                }
            )

        except KeyboardInterrupt:
            print("\n\nSession saved. Goodbye!")
            break
        except Exception as e:
            print(f"\nError: {e}")
            import traceback
            traceback.print_exc()


def main():
    parser = argparse.ArgumentParser(
        description="Personal Data Analyst Agent - Query your data with natural language",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python run_agent.py                              # Interactive mode (new session)
    python run_agent.py "Top 5 products by revenue"  # Single query
    python run_agent.py --thread my-session          # Resume session
    python run_agent.py --csv sales.csv              # Use specific CSV file
    python run_agent.py --csv sales.csv "Show trends"  # CSV + query
        """
    )

    parser.add_argument(
        "query",
        nargs="?",
        default=None,
        help="Query to run (if omitted, enters interactive mode)"
    )

    parser.add_argument(
        "--csv",
        type=str,
        default=None,
        help="Path to CSV file (will scan and create schema if needed)"
    )

    parser.add_argument(
        "--schema",
        type=str,
        default="schema_profile.json",
        help="Path to existing schema profile JSON"
    )

    parser.add_argument(
        "--thread",
        type=str,
        default=None,
        help="Thread ID for session persistence (generates new if omitted)"
    )

    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Show detailed debug information"
    )

    args = parser.parse_args()

    if not os.getenv("ANTHROPIC_API_KEY") and not os.getenv("OPENAI_API_KEY"):
        print("ERROR: No API key found!")
        print("\nSet one of the following environment variables:")
        print("  ANTHROPIC_API_KEY - For Claude models (recommended)")
        print("  OPENAI_API_KEY - For GPT models")
        print("\nExample (Windows):")
        print("  set ANTHROPIC_API_KEY=sk-ant-...")
        print("\nExample (Linux/Mac):")
        print("  export ANTHROPIC_API_KEY=sk-ant-...")
        sys.exit(1)

    if args.csv:
        schema, csv_path = load_or_create_schema(args.csv)
    elif Path(args.schema).exists():
        with open(args.schema) as f:
            schema = json.load(f)
        csv_path = schema.get("file_path", "")
        if not Path(csv_path).exists():
            print(f"ERROR: CSV file not found: {csv_path}")
            print("Use --csv to specify the CSV file path")
            sys.exit(1)
        print(f"Loaded schema: {schema.get('file_name', 'Unknown')}")
    else:
        print("ERROR: No schema found!")
        print("\nOptions:")
        print("  1. Run scanner first: python scanner.py your_data.csv")
        print("  2. Specify CSV file: python run_agent.py --csv your_data.csv")
        sys.exit(1)

    thread_id = args.thread or f"session-{uuid.uuid4().hex[:8]}"

    print("Initializing agent...")
    graph = compile_graph(checkpointer=True)

    if args.query:
        print(f"\nThread: {thread_id}")
        print(f"Query: {args.query}")

        state = run_query(graph, schema, csv_path, args.query, thread_id)

        if state.get("router_decision") == "yellow" and not state.get("final_response"):
            memory = get_conversation_memory()
            state = handle_clarification(
                graph=graph,
                state=state,
                schema=schema,
                csv_path=csv_path,
                thread_id=thread_id,
                memory=memory
            )

        print_result(state, verbose=args.verbose)

        memory = get_conversation_memory()
        response = state.get("final_response", state.get("clarification_question", ""))
        memory.add_turn(
            thread_id=thread_id,
            user_query=args.query,
            response=response
        )
    else:
        interactive_mode(graph, schema, csv_path, thread_id)


if __name__ == "__main__":
    main()
