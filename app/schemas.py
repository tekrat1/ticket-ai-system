from __future__ import annotations

from typing import Any, Literal, Optional

from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, examples=["How many tickets are currently open?"])


class QueryResponse(BaseModel):
    question: str
    answer: str
    result_preview: list[dict[str, Any]] = Field(default_factory=list)
    query_spec: dict[str, Any] = Field(default_factory=dict)
    row_count: int = 0


class Anomaly(BaseModel):
    ticket_id: str
    reason: str
    rule: str
    severity: Literal["medium", "high"]
    details: dict[str, Any] = Field(default_factory=dict)


class AnomalyResponse(BaseModel):
    count: int
    anomalies: list[Anomaly]


class HealthResponse(BaseModel):
    status: str
    rows_loaded: int
    llm_provider: str
    llm_configured: bool
