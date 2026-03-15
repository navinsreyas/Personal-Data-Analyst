from typing import Optional
from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    question: str = Field(
        ...,
        description="Natural language question about the dataset",
        min_length=3,
        max_length=500,
        examples=["What are the top 5 product categories by revenue?"]
    )
    thread_id: Optional[str] = Field(
        default=None,
        description="Optional session ID for conversation continuity"
    )


class QueryResponse(BaseModel):
    answer:          Optional[str]  = None
    path:            list[str]      = []
    duration_s:      float          = 0.0
    router_decision: Optional[str]  = None
    plan_approved:   Optional[bool] = None
    agent_steps:     Optional[int]  = None
    cache_hit:       bool           = False
    retry_count:     int            = 0
    error:           Optional[str]  = None
    status:          str            = "ok"


class HealthResponse(BaseModel):
    status:  str = "ok"
    model:   str
    dataset: str
    rows:    int
    columns: int


class SchemaResponse(BaseModel):
    file_name:           str
    row_count:           int
    column_count:        int
    categorical_columns: list[str]
    numerical_columns:   list[str]
    datetime_columns:    list[str]
    grain:               Optional[str] = None
