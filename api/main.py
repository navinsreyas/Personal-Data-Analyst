from __future__ import annotations

import json
import logging
import os
import sys
import time
import uuid
from pathlib import Path

from dotenv import load_dotenv

_root = Path(__file__).parent.parent
load_dotenv(_root / ".env")

sys.path.insert(0, str(_root))

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from api.schemas import (
    HealthResponse,
    QueryRequest,
    QueryResponse,
    SchemaResponse,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Personal Data Analyst API",
    description=(
        "Natural language query engine for CSV data. "
        "Built with LangGraph, smolagents, and Claude Sonnet."
    ),
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_SCHEMA_PATH = _root / "data" / "schema_sample.json"
_PIPELINE    = None
_SCHEMA      = {}
_CSV_PATH    = str(_root / "data" / "olist_sample.csv")


def _load_pipeline():
    global _PIPELINE, _SCHEMA
    if _SCHEMA_PATH.exists():
        with open(_SCHEMA_PATH) as f:
            _SCHEMA = json.load(f)
        logger.info(
            f"Schema loaded: {_SCHEMA.get('file_name')} "
            f"({_SCHEMA.get('row_count')} rows)"
        )
    else:
        logger.warning("schema_sample.json not found")

    from agent.graph_builder import compile_graph
    _PIPELINE = compile_graph(checkpointer=False)
    logger.info("Pipeline compiled — API ready")


@app.on_event("startup")
async def startup():
    _load_pipeline()


@app.get("/health", response_model=HealthResponse)
async def health():
    return HealthResponse(
        status  = "ok",
        model   = os.getenv("MODEL_ID", "claude-sonnet-4-6"),
        dataset = _SCHEMA.get("file_name", "unknown"),
        rows    = _SCHEMA.get("row_count", 0),
        columns = _SCHEMA.get("column_count", 0),
    )


@app.get("/schema", response_model=SchemaResponse)
async def schema():
    if not _SCHEMA:
        raise HTTPException(status_code=503, detail="Schema not loaded")
    return SchemaResponse(
        file_name           = _SCHEMA.get("file_name", "unknown"),
        row_count           = _SCHEMA.get("row_count", 0),
        column_count        = _SCHEMA.get("column_count", 0),
        categorical_columns = _SCHEMA.get("categorical_columns", []),
        numerical_columns   = _SCHEMA.get("numerical_columns", []),
        datetime_columns    = _SCHEMA.get("datetime_columns", []),
        grain               = _SCHEMA.get("grain"),
    )


@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    if _PIPELINE is None:
        raise HTTPException(status_code=503, detail="Pipeline not ready")

    thread_id = request.thread_id or f"api-{uuid.uuid4().hex[:8]}"
    logger.info(f"[{thread_id}] {request.question[:80]}")

    start = time.time()
    try:
        from agent.state import create_initial_state

        state = create_initial_state(
            user_query     = request.question,
            schema_profile = _SCHEMA,
            csv_file_path  = _CSV_PATH,
        )

        result   = _PIPELINE.invoke(state)
        duration = round(time.time() - start, 2)

        answer = (
            result.get("final_response")
            or result.get("clarification_question")
            or result.get("refusal_reason")
            or "No response generated"
        )

        logger.info(
            f"[{thread_id}] {duration}s — "
            f"{' -> '.join(result.get('execution_path', []))}"
        )

        return QueryResponse(
            answer          = answer,
            path            = result.get("execution_path", []),
            duration_s      = duration,
            router_decision = result.get("router_decision"),
            plan_approved   = result.get("plan_approved"),
            agent_steps     = result.get("agent_steps"),
            cache_hit       = result.get("cache_hit", False),
            retry_count     = result.get("retry_count", 0),
            status          = "ok",
        )

    except Exception as e:
        duration = round(time.time() - start, 2)
        logger.error(f"[{thread_id}] Error after {duration}s: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Pipeline error: {str(e)}"
        )


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled: {exc}")
    return JSONResponse(
        status_code=500,
        content={"error": str(exc), "status": "error"},
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "api.main:app",
        host   = "0.0.0.0",
        port   = int(os.getenv("PORT", 8000)),
        reload = False,
    )
