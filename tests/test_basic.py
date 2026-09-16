import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.anomaly import detect_anomalies  # noqa: E402
from app.data_store import DataStore  # noqa: E402
from app.llm_client import LLMClient  # noqa: E402
from app.query_engine import QueryEngine  # noqa: E402

DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "support_tickets.csv"


def test_data_loads():
    store = DataStore(DATA_PATH)
    assert len(store.df) > 0
    assert "ticket_id" in store.df.columns


def test_fallback_count_open():
    store = DataStore(DATA_PATH)
    llm = LLMClient()  # LLM_PROVIDER unset -> "none" -> rule-based fallback
    engine = QueryEngine(store, llm)
    result = engine.answer("How many tickets are currently open?")
    assert result["row_count"] == 1
    expected = (store.df["status"] == "Open").sum()
    assert str(int(expected)) in result["answer"]


def test_fallback_unsupported_question():
    store = DataStore(DATA_PATH)
    llm = LLMClient()
    engine = QueryEngine(store, llm)
    result = engine.answer("What is the weather like today?")
    assert "couldn't map" in result["answer"].lower()


def test_anomaly_detection_runs_and_is_explainable():
    store = DataStore(DATA_PATH)
    anomalies = detect_anomalies(store)
    assert isinstance(anomalies, list)
    for a in anomalies:
        assert a["ticket_id"] in set(store.df["ticket_id"])
        assert a["reason"]
        assert a["severity"] in ("medium", "high")


def test_fallback_not_resolved_within_hours():
    """Regression test: this is a literal sample query from the assessment
    brief. It used to incorrectly filter on response_time_hrs (time to
    *first reply*, not resolution) and always returned 0 rows. It must use
    the derived hours_to_resolve column instead."""
    store = DataStore(DATA_PATH)
    llm = LLMClient()
    engine = QueryEngine(store, llm)
    result = engine.answer("Show me all Critical tickets not resolved within 12 hours")
    assert result["query_spec"]["filters"][-1]["column"] == "hours_to_resolve"
    expected = store.df[
        (store.df["priority"] == "Critical") & (store.df["hours_to_resolve"] > 12)
    ]
    assert result["row_count"] == len(expected)
    assert result["row_count"] > 0
