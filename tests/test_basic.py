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
