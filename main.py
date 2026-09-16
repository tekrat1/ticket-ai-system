"""
Entry point. Run with:

    uvicorn main:app --reload

or via the single-command starter described in the README:

    ./start.sh
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from app.anomaly import detect_anomalies
from app.data_store import DataStore
from app.llm_client import LLMClient
from app.query_engine import QueryEngine
from app.schemas import (
    AnomalyResponse,
    HealthResponse,
    QueryRequest,
    QueryResponse,
)

load_dotenv()

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
logger = logging.getLogger("ticket_ai_system")

DATA_PATH = Path(os.getenv("DATA_PATH", "data/support_tickets.csv"))

app = FastAPI(
    title="Support Ticket AI System",
    description="NL querying + anomaly detection over a customer support ticket dataset.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # demo/assessment scope; tighten for production
    allow_methods=["*"],
    allow_headers=["*"],
)

store = DataStore(DATA_PATH)
llm = LLMClient()
engine = QueryEngine(store, llm)


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(
        status="ok",
        rows_loaded=len(store.df),
        llm_provider=llm.provider,
        llm_configured=llm.is_configured,
    )


@app.post("/query", response_model=QueryResponse)
def query(req: QueryRequest) -> QueryResponse:
    try:
        result = engine.answer(req.question)
    except Exception as e:  # noqa: BLE001 — surface as a clean 500, never crash the process
        logger.exception("Query failed")
        raise HTTPException(status_code=500, detail=f"Query failed: {e}") from e
    return QueryResponse(**result)


@app.get("/anomalies", response_model=AnomalyResponse)
def anomalies() -> AnomalyResponse:
    try:
        found = detect_anomalies(store)
    except Exception as e:  # noqa: BLE001
        logger.exception("Anomaly detection failed")
        raise HTTPException(status_code=500, detail=f"Anomaly detection failed: {e}") from e
    return AnomalyResponse(count=len(found), anomalies=found)


@app.post("/reload")
def reload_data() -> dict:
    """Convenience endpoint to re-read the CSV without restarting the server."""
    store.reload()
    return {"status": "reloaded", "rows_loaded": len(store.df)}
