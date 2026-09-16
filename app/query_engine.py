"""
Natural-language query handling.

Design choice (see README "Architecture" for the full reasoning):
  We do NOT let the LLM generate raw pandas/python code to `exec()`. That is
  a prompt-injection / code-injection risk for very little benefit here.
  Instead the LLM's only job is to translate the question into a small,
  constrained JSON "query spec" (filters + optional groupby + aggregation).
  That spec is then executed by our own whitelisted interpreter. This keeps
  the LLM's blast radius limited to "pick valid columns/operators/values",
  and every filter/aggregation it can produce is one we already vetted.

  If no LLM is configured, or the LLM output fails validation, we fall back
  to a small rule-based parser covering the assessment's sample queries, so
  the system is still fully demoable at zero cost / zero setup.
"""

from __future__ import annotations

import json
import logging
import re

import pandas as pd

from .data_store import DataStore
from .llm_client import LLMClient, LLMError

logger = logging.getLogger(__name__)

ALLOWED_COLUMNS = {
    "ticket_id", "created_at", "category", "priority", "status",
    "response_time_hrs", "resolution_time_hrs", "agent_id",
    "customer_rating", "issue_summary",
}
ALLOWED_OPERATORS = {"==", "!=", ">", "<", ">=", "<=", "contains"}
ALLOWED_AGG_FUNCS = {"count", "mean", "sum", "min", "max", "nunique", "median"}

SYSTEM_PROMPT = """You are a query planner for a customer-support ticket dataset.
You NEVER answer the question directly and you NEVER write code. You only
output a single JSON object describing how to compute the answer from the
dataframe below. No prose, no markdown fences, JSON only.

Columns (name: type — meaning):
- ticket_id: string — unique id
- created_at: datetime — when the ticket was created
- category: string, one of Billing/Technical/General
- priority: string, one of Low/Medium/High/Critical
- status: string, one of Open/Resolved/Escalated
- response_time_hrs: float — hours to first response
- resolution_time_hrs: float, nullable — hours to resolution (null if unresolved)
- agent_id: string — support agent handling the ticket
- customer_rating: int 1-5, nullable — nullable if unresolved
- issue_summary: string — free text

Output schema (all keys optional except "intent"):
{
  "intent": "aggregate" | "list" | "groupby_agg",
  "filters": [{"column": "...", "operator": "==|!=|>|<|>=|<=|contains", "value": ...}],
  "groupby": "column_name or null",
  "agg_column": "column_name or null",
  "agg_func": "count|mean|sum|min|max|nunique|median or null",
  "sort_by": "column_name or 'agg' or null",
  "sort_desc": true|false,
  "limit": integer or null
}

Rules:
- Only use column names exactly as listed above.
- "critical"/"high priority" etc map to the priority column values (Low/Medium/High/Critical).
- "unresolved"/"not resolved"/"open" tickets -> filter status != "Resolved" unless the user clearly means only status == "Open".
- For "average X" -> intent "aggregate" or "groupby_agg", agg_func "mean".
- For "which agent has the ... rating/most tickets" -> intent "groupby_agg", groupby "agent_id".
- For plain counts -> intent "aggregate", agg_func "count".
- For "groupby_agg" intent: ALWAYS set "sort_desc" explicitly (never omit it). Default to
  true (highest values first) unless the question specifically asks for the lowest/least/
  smallest/worst/fewest, in which case set it to false. Even for neutral phrasing like
  "how many tickets does each category have" (no explicit ranking direction), still set
  "sort_desc": true so the biggest group appears first.
- If the question cannot be answered from these columns, return {"intent": "unsupported"}.
- Output ONLY the JSON object, nothing else.

CRITICAL formatting rule for filters: every filter is
{"column": "<name>", "operator": "<op>", "value": <value>} and the operator
is ALWAYS a quoted JSON string value after a colon, even when it is a
comparison symbol like >= or <=. Never drop the colon after "operator" and
never write the operator symbol as if it were JSON syntax.

Example of a CORRECT filters array (note ">=" is quoted like any other string):
{"intent": "groupby_agg", "filters": [
  {"column": "status", "operator": "==", "value": "Resolved"},
  {"column": "created_at", "operator": ">=", "value": "2024-09-01"},
  {"column": "created_at", "operator": "<", "value": "2024-10-01"}
], "groupby": "agent_id", "agg_column": "ticket_id", "agg_func": "count",
"sort_by": "agg", "sort_desc": true, "limit": 1}
"""


class QueryValidationError(Exception):
    pass


def _validate_spec(spec: dict) -> dict:
    if not isinstance(spec, dict) or "intent" not in spec:
        raise QueryValidationError("Missing 'intent'")

    if spec["intent"] == "unsupported":
        return spec

    for f in spec.get("filters", []) or []:
        if f.get("column") not in ALLOWED_COLUMNS:
            raise QueryValidationError(f"Bad filter column: {f.get('column')}")
        if f.get("operator") not in ALLOWED_OPERATORS:
            raise QueryValidationError(f"Bad filter operator: {f.get('operator')}")

    for key in ("groupby", "agg_column", "sort_by"):
        val = spec.get(key)
        if val is not None and val not in ALLOWED_COLUMNS and val != "agg":
            raise QueryValidationError(f"Bad column in '{key}': {val}")

    agg_func = spec.get("agg_func")
    if agg_func is not None and agg_func not in ALLOWED_AGG_FUNCS:
        raise QueryValidationError(f"Bad agg_func: {agg_func}")

    return spec


def _apply_filters(df: pd.DataFrame, filters: list[dict]) -> pd.DataFrame:
    out = df
    for f in filters or []:
        col, op, val = f["column"], f["operator"], f.get("value")
        series = out[col]
        if op == "contains":
            out = out[series.astype("string").str.contains(str(val), case=False, na=False)]
            continue
        # Coerce value type to match column for numeric/datetime comparisons.
        if pd.api.types.is_numeric_dtype(series):
            try:
                val = float(val)
            except (TypeError, ValueError):
                pass
        elif pd.api.types.is_datetime64_any_dtype(series):
            val = pd.to_datetime(val, errors="coerce")
        else:
            val = str(val)

        if op == "==":
            out = out[series == val]
        elif op == "!=":
            out = out[series != val]
        elif op == ">":
            out = out[series > val]
        elif op == "<":
            out = out[series < val]
        elif op == ">=":
            out = out[series >= val]
        elif op == "<=":
            out = out[series <= val]
    return out


def _execute_spec(df: pd.DataFrame, spec: dict) -> tuple[pd.DataFrame, str]:
    """Runs the validated spec against df. Returns (result_df, human_summary)."""
    filtered = _apply_filters(df, spec.get("filters", []))
    intent = spec["intent"]

    if intent == "list":
        limit = spec.get("limit") or 50
        result = filtered.head(limit)
        return result, f"Found {len(filtered)} matching ticket(s)."

    if intent == "aggregate":
        agg_func = spec.get("agg_func") or "count"
        agg_col = spec.get("agg_column")
        if agg_func == "count" or not agg_col:
            value = len(filtered)
        else:
            value = getattr(filtered[agg_col], agg_func)()
        result = pd.DataFrame([{"result": value}])
        return result, f"{agg_func}({agg_col or 'rows'}) = {value}"

    if intent == "groupby_agg":
        groupby = spec.get("groupby")
        agg_col = spec.get("agg_column")
        agg_func = spec.get("agg_func") or "count"
        if not groupby:
            raise QueryValidationError("groupby_agg requires 'groupby'")

        if agg_func == "count" or not agg_col:
            grouped = filtered.groupby(groupby).size().reset_index(name="agg")
        else:
            grouped = (
                filtered.groupby(groupby)[agg_col]
                .agg(agg_func)
                .reset_index()
                .rename(columns={agg_col: "agg"})
            )

        # Default to descending (highest first) even if the LLM/spec omitted
        # sort_desc entirely — an unsorted "top result" would otherwise be
        # phrased as the top when it's really just the first row returned.
        sort_desc = spec.get("sort_desc")
        if sort_desc is None:
            sort_desc = True
        grouped = grouped.sort_values("agg", ascending=not sort_desc)
        limit = spec.get("limit") or 20
        result = grouped.head(limit)
        return result, f"Grouped by {groupby}, {agg_func} computed for {len(grouped)} group(s)."

    raise QueryValidationError(f"Unhandled intent: {intent}")


# ---------------------------------------------------------------------------
# Fallback: rule-based parser used when no LLM is configured, or the LLM call
# / its JSON output fails validation. Covers the sample queries in the brief.
# ---------------------------------------------------------------------------

def _fallback_spec(question: str) -> dict:
    q = question.lower()

    if "open" in q and "how many" in q:
        return {"intent": "aggregate", "filters": [{"column": "status", "operator": "==", "value": "Open"}], "agg_func": "count"}

    if "most tickets" in q or ("agent" in q and "resolved the most" in q):
        return {"intent": "groupby_agg", "filters": [{"column": "status", "operator": "==", "value": "Resolved"}],
                "groupby": "agent_id", "agg_func": "count", "sort_desc": True, "limit": 5}

    if "lowest average" in q and "rating" in q:
        return {"intent": "groupby_agg", "groupby": "agent_id", "agg_column": "customer_rating",
                "agg_func": "mean", "sort_desc": False, "limit": 5}

    m = re.search(r"critical.*not resolved within (\d+)\s*hours?", q)
    if m or ("critical" in q and "not resolved" in q):
        hours = float(m.group(1)) if m else 12
        return {"intent": "list", "filters": [
            {"column": "priority", "operator": "==", "value": "Critical"},
            {"column": "status", "operator": "!=", "value": "Resolved"},
            {"column": "response_time_hrs", "operator": ">", "value": hours},
        ], "limit": 100}

    if "average" in q and "rating" in q and "technical" in q:
        return {"intent": "aggregate", "filters": [{"column": "category", "operator": "==", "value": "Technical"}],
                "agg_column": "customer_rating", "agg_func": "mean"}

    if "average" in q and "rating" in q:
        return {"intent": "aggregate", "agg_column": "customer_rating", "agg_func": "mean"}

    if "unresolved" in q or "not resolved" in q:
        filters = [{"column": "status", "operator": "!=", "value": "Resolved"}]
        for p in ["critical", "high", "medium", "low"]:
            if p in q:
                filters.append({"column": "priority", "operator": "==", "value": p.capitalize()})
                break
        return {"intent": "aggregate", "filters": filters, "agg_func": "count"}

    return {"intent": "unsupported"}


class QueryEngine:
    def __init__(self, store: DataStore, llm: LLMClient):
        self.store = store
        self.llm = llm

    def answer(self, question: str) -> dict:
        spec: dict
        source = "llm"

        if self.llm.is_configured:
            raw = None
            try:
                raw = self.llm.complete(SYSTEM_PROMPT, question)
                cleaned = _strip_code_fences(raw)
                try:
                    spec = _validate_spec(json.loads(cleaned))
                except json.JSONDecodeError as parse_err:
                    repaired = _repair_json(cleaned)
                    if repaired == cleaned:
                        raise  # nothing to repair, propagate original error
                    logger.warning(
                        "LLM JSON failed to parse (%s); attempting repair", parse_err
                    )
                    spec = _validate_spec(json.loads(repaired))
                    logger.info("JSON repair succeeded, avoided fallback")
            except (LLMError, json.JSONDecodeError, QueryValidationError) as e:
                logger.warning("LLM query planning failed (%s); falling back to rules", e)
                if raw is not None:
                    # Only reachable on JSONDecodeError/QueryValidationError —
                    # LLMError means we never got content back at all. Logged
                    # at WARNING (not DEBUG) so it shows up with the default
                    # LOG_LEVEL=INFO without needing a config change.
                    logger.warning("Raw LLM output that failed to parse: %r", raw)
                spec = _fallback_spec(question)
                source = "fallback"
        else:
            spec = _fallback_spec(question)
            source = "fallback (no LLM configured)"

        if spec.get("intent") == "unsupported":
            return {
                "question": question,
                "answer": "I couldn't map this question to the available columns. "
                          "Try asking about status, priority, category, agent, response/resolution "
                          "time, or customer rating.",
                "result_preview": [],
                "query_spec": spec,
                "row_count": 0,
            }

        try:
            result_df, summary = _execute_spec(self.store.df, spec)
        except QueryValidationError as e:
            return {
                "question": question,
                "answer": f"The query planner produced an invalid query ({e}). Please rephrase.",
                "result_preview": [],
                "query_spec": spec,
                "row_count": 0,
            }

        preview = json.loads(result_df.head(20).to_json(orient="records", date_format="iso"))
        answer_text = self._phrase_answer(question, spec, result_df, summary)

        return {
            "question": question,
            "answer": answer_text,
            "result_preview": preview,
            "query_spec": {**spec, "_source": source},
            "row_count": len(result_df),
        }

    @staticmethod
    def _phrase_answer(question: str, spec: dict, result_df: pd.DataFrame, summary: str) -> str:
        """Deterministic templating from the *executed* result — never re-asks the
        LLM to state numbers, so the answer can't drift from the actual data."""
        intent = spec["intent"]

        if intent == "aggregate" and "result" in result_df.columns:
            return f"{summary}."

        if intent == "groupby_agg" and not result_df.empty:
            top = result_df.iloc[0]
            group_col = spec.get("groupby")
            return (
                f"Top result: {group_col}={top[group_col]} with agg value = {top['agg']:.2f}"
                if isinstance(top["agg"], float) else
                f"Top result: {group_col}={top[group_col]} with agg value = {top['agg']}"
            ) + f" (showing {len(result_df)} group(s) total)."

        if intent == "list":
            return summary

        return summary


def _strip_code_fences(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(json)?", "", text).strip()
        text = re.sub(r"```$", "", text).strip()
    return text


# Matches a key like "operator" immediately followed by a comparison symbol
# with no colon in between, e.g.  "operator">="""  or  "operator"<=""
# Captures: key name, operator symbol, optional following ("" or empty).
_MISSING_COLON_OP_RE = re.compile(
    r'"(\w+)"\s*(>=|<=|==|!=|>|<)\s*""?'
)


def _repair_json(text: str) -> str:
    """Best-effort fix for a common LLM malformation where a comparison
    operator (>=, <=, ==, !=, >, <) gets written as bare JSON syntax right
    after a key, instead of as a quoted string value with a colon.

    e.g.  {"column":"created_at","operator">="","value":"2024-09-01"}
    ->    {"column":"created_at","operator":">=","value":"2024-09-01"}

    Only ever called after a plain json.loads() has already failed, and its
    own output is re-validated by json.loads() again by the caller — so a
    bad repair just falls through to the existing rule-based fallback.
    """
    return _MISSING_COLON_OP_RE.sub(r'"\1":"\2"', text)