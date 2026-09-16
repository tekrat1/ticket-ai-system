"""
Anomaly detection over the ticket dataset.

Two complementary techniques, both cheap and explainable (important: an
evaluator should be able to see *why* something was flagged, not just that
it was):

1. Rule-based: domain rules stated directly in the assessment brief
   (e.g. unresolved high-priority tickets open >24h).
2. Statistical: IQR (interquartile range) outlier detection on
   response/resolution time, computed per-category so a "slow" Technical
   ticket isn't compared against fast Billing tickets.

This module deliberately does NOT use the LLM — anomaly *detection* is a
numeric/rules problem where a statistical method is more reliable and
auditable than an LLM guess. The LLM is reserved for NL query understanding,
where it earns its keep. (See README "Architecture" for this reasoning.)
"""

from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd

from .data_store import DataStore

HIGH_PRIORITY = {"High", "Critical"}
IQR_MULTIPLIER = 1.5


def _iqr_outliers(df: pd.DataFrame, column: str, group_col: str | None = None) -> pd.Series:
    """Returns a boolean mask flagging IQR-based outliers, computed within
    each group (e.g. per category) if group_col is given."""
    def _mask(sub: pd.Series) -> pd.Series:
        valid = sub.dropna()
        if len(valid) < 4:
            return pd.Series(False, index=sub.index)
        q1, q3 = valid.quantile(0.25), valid.quantile(0.75)
        iqr = q3 - q1
        upper = q3 + IQR_MULTIPLIER * iqr
        return sub > upper

    if group_col:
        return df.groupby(group_col)[column].transform(_mask)
    return _mask(df[column])


def detect_anomalies(store: DataStore) -> list[dict]:
    df = store.df
    anomalies: list[dict] = []
    now = pd.Timestamp.now(tz=None)

    # Rule 1: unresolved high-priority tickets open > 24h.
    unresolved_high = df[
        (df["priority"].isin(HIGH_PRIORITY))
        & (df["status"] != "Resolved")
        & df["created_at"].notna()
    ].copy()
    unresolved_high["age_hrs"] = (now - unresolved_high["created_at"]).dt.total_seconds() / 3600
    stale = unresolved_high[unresolved_high["age_hrs"] > 24]
    for _, row in stale.iterrows():
        anomalies.append({
            "ticket_id": row["ticket_id"],
            "reason": f"{row['priority']} priority, still '{row['status']}' after "
                      f"{row['age_hrs']:.1f}h (threshold: 24h)",
            "rule": "unresolved_high_priority_over_24h",
            "severity": "high",
            "details": {"priority": row["priority"], "status": row["status"], "age_hrs": round(row["age_hrs"], 1)},
        })

    # Rule 2: resolution time statistical outliers, per category.
    resolved = df[df["resolution_time_hrs"].notna()].copy()
    resolved["is_outlier"] = _iqr_outliers(resolved, "resolution_time_hrs", group_col="category")
    for _, row in resolved[resolved["is_outlier"]].iterrows():
        anomalies.append({
            "ticket_id": row["ticket_id"],
            "reason": f"Resolution time of {row['resolution_time_hrs']:.1f}h is an outlier "
                      f"for the {row['category']} category (IQR method)",
            "rule": "resolution_time_outlier",
            "severity": "medium",
            "details": {"category": row["category"], "resolution_time_hrs": row["resolution_time_hrs"]},
        })

    # Rule 3: response time statistical outliers, overall.
    responded = df[df["response_time_hrs"].notna()].copy()
    responded["is_outlier"] = _iqr_outliers(responded, "response_time_hrs")
    for _, row in responded[responded["is_outlier"]].iterrows():
        anomalies.append({
            "ticket_id": row["ticket_id"],
            "reason": f"First-response time of {row['response_time_hrs']:.1f}h is an outlier overall (IQR method)",
            "rule": "response_time_outlier",
            "severity": "medium",
            "details": {"response_time_hrs": row["response_time_hrs"]},
        })

    # De-duplicate: same ticket can trip multiple rules — keep highest severity, merge reasons.
    merged: dict[str, dict] = {}
    for a in anomalies:
        tid = a["ticket_id"]
        if tid not in merged:
            merged[tid] = {**a, "reason": [a["reason"]], "rule": [a["rule"]]}
        else:
            merged[tid]["reason"].append(a["reason"])
            merged[tid]["rule"].append(a["rule"])
            if a["severity"] == "high":
                merged[tid]["severity"] = "high"

    out = []
    for tid, a in merged.items():
        out.append({
            "ticket_id": tid,
            "reason": "; ".join(a["reason"]),
            "rule": ",".join(a["rule"]),
            "severity": a["severity"],
            "details": a["details"],
        })

    out.sort(key=lambda a: (a["severity"] != "high", a["ticket_id"]))
    return out
