from __future__ import annotations

import json
import sys
import tempfile
import time as _time
import uuid
from pathlib import Path

import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agent.graph_builder import compile_graph
from agent.state import create_initial_state

st.set_page_config(
    page_title="Personal Data Analyst",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_resource(show_spinner="Compiling analysis pipeline...")
def load_pipeline():
    return compile_graph(checkpointer=True)


def load_schema(csv_path: str) -> dict:
    schema_path = PROJECT_ROOT / "data" / "schema_profile.json"
    if schema_path.exists():
        return json.loads(schema_path.read_text(encoding="utf-8"))
    return {}


_DEFAULT_CSV = str(PROJECT_ROOT / "data" / "olist_master.csv")


def _init_state():
    defaults = {
        "messages":            [],
        "pending_clarification": None,
        "csv_path":            _DEFAULT_CSV,
        "schema":              {},
        "graph":               None,
        "loaded_file_name":    None,
        "loaded_display_name": "olist_master.csv",
        "thread_id":           f"session-{uuid.uuid4().hex[:8]}",
        "total_queries":       0,
        "cache_hits":          0,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

    if not st.session_state.schema:
        st.session_state.schema = load_schema(_DEFAULT_CSV)
    if st.session_state.graph is None:
        st.session_state.graph = load_pipeline()


_init_state()


def _load_dataset_from_path(csv_path: str, display_name: str | None = None):
    st.session_state.csv_path            = csv_path
    st.session_state.schema              = load_schema(csv_path)
    st.session_state.graph               = load_pipeline()
    st.session_state.messages            = []
    st.session_state.pending_clarification = None
    st.session_state.thread_id           = f"session-{uuid.uuid4().hex[:8]}"
    st.session_state.total_queries       = 0
    st.session_state.cache_hits          = 0
    name = display_name or Path(csv_path).name
    st.session_state.loaded_display_name = name
    st.success(f"Loaded: {name}")


with st.sidebar:
    st.title("📊 Personal Data Analyst")
    st.markdown("---")

    st.subheader("Dataset")

    uploaded_file = st.file_uploader(
        "Upload a CSV file",
        type=["csv"],
        help="Drop any CSV file to analyse it",
    )

    with st.expander("Or use a file path"):
        manual_path = st.text_input(
            "Full path to CSV",
            value=_DEFAULT_CSV,
        )
        if st.button("Load from path", use_container_width=True):
            if manual_path and Path(manual_path).exists():
                _load_dataset_from_path(manual_path)
            else:
                st.error("File not found. Check the path.")

    if uploaded_file is not None:
        if uploaded_file.name != st.session_state.get("loaded_file_name"):
            with st.spinner(f"Loading {uploaded_file.name}..."):
                try:
                    suffix = Path(uploaded_file.name).suffix
                    with tempfile.NamedTemporaryFile(
                        delete=False, suffix=suffix
                    ) as tmp:
                        tmp.write(uploaded_file.getvalue())
                        tmp_path = tmp.name

                    _load_dataset_from_path(tmp_path, display_name=uploaded_file.name)
                    st.session_state.loaded_file_name = uploaded_file.name
                    st.rerun()
                except Exception as e:
                    st.error(f"Failed to load: {e}")

    schema = st.session_state.schema
    if schema:
        st.markdown("---")
        display_name = (
            schema.get("file_name")
            or st.session_state.get("loaded_display_name", "Unknown")
        )
        st.write(f"**{display_name}**")
        st.write(
            f"{schema.get('row_count', 0):,} rows · "
            f"{schema.get('column_count', 0)} columns"
        )

        grain = schema.get("grain")
        if grain:
            st.caption(f"Grain: {grain}")

        st.markdown("---")
        st.subheader("Columns")

        dt_cols  = schema.get("datetime_columns", [])
        cat_cols = schema.get("categorical_columns", [])
        num_cols = schema.get("numerical_columns", [])

        if dt_cols:
            st.caption(f"**Datetime** ({len(dt_cols)}): {', '.join(dt_cols)}")
        if cat_cols:
            st.caption(f"**Categorical** ({len(cat_cols)}): {', '.join(cat_cols)}")
        if num_cols:
            st.caption(f"**Numerical** ({len(num_cols)}): {', '.join(num_cols)}")

        st.markdown("---")
        null_pct = schema.get("null_percentage", 0)
        dupe_cnt = schema.get("duplicate_row_count", 0)
        mem_mb   = schema.get("memory_usage_mb", 0)
        st.caption(
            f"Nulls: {null_pct}% · Duplicates: {dupe_cnt:,} · "
            f"Memory: {mem_mb} MB"
        )

    else:
        st.warning(
            "No schema profile found.\n\n"
            "Run: `python scanner.py olist_master.csv --output data/schema_profile.json`"
        )

    st.markdown("---")
    if st.button("Clear chat", use_container_width=True):
        st.session_state.messages = []
        st.session_state.pending_clarification = None
        st.rerun()


def _render_metadata(meta: dict) -> None:
    if not meta:
        return

    path       = meta.get("execution_path") or []
    steps      = meta.get("agent_steps")
    cache_hit  = meta.get("cache_hit", False)
    confidence = meta.get("supervisor_confidence") or ""
    sup_note   = meta.get("supervisor_reasoning") or ""

    pills: list[str] = []
    if path:
        pills.append(f"Path: {' -> '.join(path)}")
    if steps is not None:
        pills.append(f"Agent steps: {steps}")
    if cache_hit:
        pills.append("Cache: **hit**")
    if confidence:
        pills.append(f"Confidence: {confidence}")

    if pills:
        st.caption(" · ".join(pills))

    if sup_note:
        with st.expander("Supervisor reasoning", expanded=False):
            st.caption(sup_note)


def _collect_metadata(result: dict) -> dict:
    exec_path = result.get("execution_path") or []
    cache_hit = "reviewer" not in exec_path and "planner" in exec_path
    return {
        "execution_path":        exec_path,
        "agent_steps":           result.get("agent_steps"),
        "cache_hit":             cache_hit,
        "supervisor_reasoning":  result.get("supervisor_reasoning"),
        "supervisor_confidence": result.get("supervisor_confidence"),
    }


def _handle_result(result: dict, user_query: str) -> None:
    clarify_q  = result.get("clarification_question")
    refusal    = result.get("refusal_reason")
    final_resp = result.get("final_response") or ""
    meta       = _collect_metadata(result)

    exec_result = result.get("execution_result") or {}
    chart_path  = exec_result.get("chart_path")

    if clarify_q and not final_resp:
        st.session_state.pending_clarification = {
            "question":       clarify_q,
            "original_query": user_query,
        }
        return

    if refusal and not final_resp:
        st.session_state.messages.append({
            "role":       "assistant",
            "content":    f"I can't help with that query.\n\n**Reason:** {refusal}",
            "metadata":   meta,
            "chart_path": None,
        })
        return

    if not final_resp:
        error = (
            exec_result.get("error")
            or result.get("error_message")
            or "Unknown error"
        )
        final_resp = (
            f"The analysis ran but did not produce a formatted response.\n\n"
            f"**Error:** `{error}`"
        )

    st.session_state.messages.append({
        "role":       "assistant",
        "content":    final_resp,
        "metadata":   meta,
        "chart_path": chart_path,
    })


NODE_LABELS = {
    "router":             "Classifying question",
    "planner":            "Building analysis plan",
    "reviewer":           "Reviewing plan",
    "supervisor":         "Routing decision",
    "codeagent_executor": "Running analysis",
    "explainer":          "Formatting answer",
    "clarifier":          "Preparing clarification",
    "refuser":            "Preparing response",
}


def _run_query_with_progress(
    query: str, clarification: str | None = None
) -> dict | None:
    import queue as _queue
    import threading

    progress_container = st.empty()

    def render_progress(nodes_done: list, current: str | None = None):
        lines = []
        for n in nodes_done:
            label = NODE_LABELS.get(n, n)
            lines.append(f"✓  {label}")
        if current:
            label = NODE_LABELS.get(current, current)
            lines.append(f"⟳  {label}...")
        progress_container.markdown(
            "\n\n".join(lines) if lines else "Starting..."
        )

    render_progress([], current="router")

    state = create_initial_state(
        user_query=query,
        schema_profile=st.session_state.schema,
        csv_file_path=st.session_state.csv_path,
    )
    if clarification:
        state["clarification_response"] = clarification

    result_queue = _queue.Queue()

    def _run():
        try:
            r = st.session_state.graph.invoke(state)
            result_queue.put(("ok", r))
        except Exception as exc:
            result_queue.put(("err", exc))

    thread = threading.Thread(target=_run, daemon=True)
    thread.start()

    while thread.is_alive():
        _time.sleep(0.5)

    status, payload = result_queue.get()

    if status == "err":
        progress_container.empty()
        st.session_state.messages.append({
            "role":       "assistant",
            "content":    f"Pipeline error: {str(payload)}",
            "metadata":   {},
            "chart_path": None,
        })
        return None

    result = payload

    final_path = result.get("execution_path", [])
    render_progress(final_path)
    _time.sleep(0.4)
    progress_container.empty()

    return result


schema       = st.session_state.schema
display_name = (
    schema.get("file_name")
    or st.session_state.get("loaded_display_name", "olist_master.csv")
)

st.header("Ask your data a question")
st.caption(
    "Powered by Claude · LangGraph · smolagents   —   "
    f"Dataset: **{display_name}** "
    f"({schema.get('row_count', 0):,} rows)"
)

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant":
            _render_metadata(msg.get("metadata") or {})
            chart_path = msg.get("chart_path")
            if chart_path and Path(chart_path).exists():
                st.image(chart_path, use_column_width=True)

if st.session_state.pending_clarification:
    pend = st.session_state.pending_clarification
    with st.chat_message("assistant"):
        st.markdown(f"**I need a bit more information:**\n\n{pend['question']}")

    clarify_answer = st.chat_input("Your clarification...")
    if clarify_answer:
        st.session_state.messages.append({"role": "user", "content": clarify_answer})
        combined_query = (
            f"{pend['original_query']}\n\nAdditional context: {clarify_answer}"
        )
        st.session_state.pending_clarification = None

        result = _run_query_with_progress(combined_query, clarification=clarify_answer)
        if result is not None:
            _handle_result(result, combined_query)
        st.rerun()

else:
    user_input = st.chat_input("Ask a question about your data...")
    if user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        result = _run_query_with_progress(user_input)
        if result is not None:
            _handle_result(result, user_input)
        st.rerun()
