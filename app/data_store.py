"""
Loads support_tickets.csv into a typed, in-memory pandas DataFrame that the
rest of the system (query engine, anomaly detector, API) reads from.

Kept as a small singleton-style module rather than a database because the
dataset is small (hundreds-thousands of rows) and read-heavy. Swapping this
for SQLite/DuckDB later is a one-file change (see README "Scaling" notes).
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

EXPECTED_COLUMNS = [
    "ticket_id",
    "created_at",
    "category",
    "priority",
    "status",
    "response_time_hrs",
    "resolution_time_hrs",
    "agent_id",
    "customer_rating",
    "issue_summary",
]


class DataStore:
    """Holds the loaded ticket dataset and exposes safe, read-only access."""

    def __init__(self, csv_path: str | Path):
        self.csv_path = Path(csv_path)
        self.df: pd.DataFrame = self._load(self.csv_path)

    def _load(self, path: Path) -> pd.DataFrame:
        if not path.exists():
            raise FileNotFoundError(f"Dataset not found at {path}")

        df = pd.read_csv(path)

        missing = set(EXPECTED_COLUMNS) - set(df.columns)
        if missing:
            raise ValueError(f"CSV is missing expected columns: {missing}")

        # Type coercion — never trust the CSV blindly.
        df["created_at"] = pd.to_datetime(df["created_at"], errors="coerce")
        df["response_time_hrs"] = pd.to_numeric(df["response_time_hrs"], errors="coerce")
        df["resolution_time_hrs"] = pd.to_numeric(df["resolution_time_hrs"], errors="coerce")
        df["customer_rating"] = pd.to_numeric(df["customer_rating"], errors="coerce")

        for col in ["category", "priority", "status", "agent_id"]:
            df[col] = df[col].astype("string").str.strip()

        n_bad_dates = df["created_at"].isna().sum()
        if n_bad_dates:
            logger.warning("%d rows had unparseable created_at values", n_bad_dates)

        logger.info("Loaded %d tickets from %s", len(df), path)
        return df

    def reload(self) -> None:
        """Re-reads the CSV from disk (useful if the file is updated at runtime)."""
        self.df = self._load(self.csv_path)

    # ---- small helpers used across the app so column names live in one place ----

    def categories(self) -> list[str]:
        return sorted(self.df["category"].dropna().unique().tolist())

    def priorities(self) -> list[str]:
        return sorted(self.df["priority"].dropna().unique().tolist())

    def statuses(self) -> list[str]:
        return sorted(self.df["status"].dropna().unique().tolist())

    def agents(self) -> list[str]:
        return sorted(self.df["agent_id"].dropna().unique().tolist())
