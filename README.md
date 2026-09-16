# Support Ticket AI System

An AI system built on top of a customer support ticket CSV. It can:
- load the CSV and let you query it,
- answer plain English questions about the tickets,
- flag anomalies (tickets stuck open too long, weird outliers in resolution time),
- and expose all of that through a REST API plus a small Streamlit UI.

## 1. Setup

```bash
git clone <this repo>
cd ticket-ai-system
cp .env.example .env   # optional, see LLM provider section below
./start.sh              # installs deps, starts API + UI, one command
```

- API runs at http://localhost:8000 (docs at `/docs`)
- UI runs at http://localhost:8501

`start.sh` needs bash, so it works on macOS/Linux, or WSL/Git Bash on Windows.
If you're on plain Windows without either, just run the two processes
manually:

```bash
python3 -m venv .venv && source .venv/bin/activate    # Windows: .venv\Scripts\activate
pip install -r requirements.txt
uvicorn main:app --reload          # terminal 1
streamlit run ui/app.py            # terminal 2
```

Tests: `pytest tests/ -v`

### LLM provider

You set this in `.env` with `LLM_PROVIDER`:

| Value | Needs | Notes |
|---|---|---|
| `groq` | `GROQ_API_KEY` from console.groq.com (free, no card needed) | Easiest, takes like 30 seconds to get a key |
| `ollama` | Local Ollama running + `ollama pull llama3.1` | Works fully offline if you'd rather not sign up for anything |
| `none` | nothing | Falls back to a simple rule-based parser, works out of the box with zero setup |

I set the default to `groq` in `.env.example` since the point of this
assessment is showing the LLM actually doing the NL parsing, not the
fallback. But if you just want to see it run without grabbing a key first,
`none` still answers the sample questions fine.

## 2. Architecture

```
CSV --> DataStore (pandas, typed) --> QueryEngine  --> FastAPI --> Streamlit UI
                                    \-> AnomalyDetector /
```

**Why I built it this way:**

`DataStore` (`app/data_store.py`) is just one pandas DataFrame loaded at
startup, with the columns coerced to proper types instead of trusting
whatever's in the CSV. With 500 rows there's no real reason to reach for a
database here.

One thing worth calling out: I added a column that's not in the raw
CSV, `hours_to_resolve`. For resolved tickets it's just
`resolution_time_hrs`. For tickets still open, it's hours since
`created_at`. I needed this because a question like "not resolved within 12
hours" has nothing to filter on otherwise — `resolution_time_hrs` is null
for exactly the tickets that question cares about, since they haven't
resolved yet. Took me a minute to realize why my first pass at that query
was returning zero rows.

For the NL query part (`app/query_engine.py`) — this is probably the
decision I spent the most time on. I didn't want the LLM generating or
running raw pandas/Python, because that's a good way to get a hallucinated
column name or something worse running against real data. Instead the LLM
just outputs a small JSON "query spec" (filters + optional groupby/agg),
and my own code executes that against a whitelist of allowed columns,
operators, and agg functions. So the LLM's job is picking valid options
from a menu, not writing code. The answer text also gets built from the
actual query result afterward, not generated separately by the LLM, so the
numbers it reports can't drift from what's really in the data.

If there's no LLM configured, or the JSON it returns doesn't validate, it
falls back to a small rule-based parser that covers the sample questions
from the brief. Didn't want a misconfigured API key to just break the whole
thing.

Anomaly detection (`app/anomaly.py`) doesn't touch the LLM at all, on
purpose. It's IQR-based outlier detection on response/resolution times per
category, plus a threshold check for high-priority tickets that have sat
open too long. Felt like a case where a plain statistical rule is more
trustworthy than asking a model to eyeball it. Each flagged ticket comes
with a `reason` string and which rule triggered it, so it's at least
explainable.

API (`main.py`) is FastAPI, four endpoints: `/health`, `/query`,
`/anomalies`, and `/reload` (just re-reads the CSV without restarting,
handy while testing). Pydantic schemas validate requests and responses.

UI (`ui/app.py`) is Streamlit, and it talks to the API over plain HTTP
rather than importing the backend code directly — figured it should behave
like any other client hitting the same API contract.

### Tools used
- LLM: Groq free tier (`llama-3.1-8b-instant`), or local Ollama (`llama3.1`) — swappable in `app/llm_client.py`
- pandas for the data
- FastAPI + Pydantic + uvicorn for the API
- Streamlit for the UI

## 3. Example queries

```bash
curl -X POST localhost:8000/query -H "Content-Type: application/json" \
  -d '{"question": "How many tickets are currently open?"}'
```
```json
{"question": "How many tickets are currently open?",
 "answer": "count(rows) = 111.",
 "result_preview": [{"result": 111}],
 "query_spec": {"intent": "aggregate", "filters": [{"column": "status", "operator": "==", "value": "Open"}], "agg_func": "count"},
 "row_count": 1}
```

```bash
curl -X POST localhost:8000/query -H "Content-Type: application/json" \
  -d '{"question": "Which agent resolved the most tickets?"}'
```
```json
{"answer": "Top result: agent_id=AGT-09 with agg value = 37 (showing 5 group(s) total).", ...}
```

```bash
curl -X POST localhost:8000/query -H "Content-Type: application/json" \
  -d '{"question": "Show me all Critical tickets not resolved within 12 hours."}'
```
```json
{"answer": "Found 34 matching ticket(s).", "row_count": 34,
 "query_spec": {"intent": "list", "filters": [
   {"column": "priority", "operator": "==", "value": "Critical"},
   {"column": "hours_to_resolve", "operator": ">", "value": 12.0}
 ], "limit": 100}, ...}
```

```bash
curl localhost:8000/anomalies
```
```json
{"count": 102, "anomalies": [{"ticket_id": "TKT-007", "reason": "High priority, still 'Open' after 23256.3h (threshold: 24h)", "rule": "unresolved_high_priority_over_24h", "severity": "high", ...}, ...]}
```

Tested these against a live server while building this, see `tests/test_basic.py`.

## 4. Known limitations

- The dataset's `created_at` values are all from 2024, but the "unresolved
  for >24h" rule compares against whatever "now" actually is. So basically
  every open ticket ends up flagged, since they're all technically 20,000+
  hours old by this point, not just the ones that went stale recently. I
  noticed this while testing and honestly wasn't sure if the brief meant
  wall-clock time or "recent relative to the dataset," so I left it as-is
  and I'm flagging it here rather than quietly hacking around it. A real
  fix would compare against a configurable "as of" timestamp instead of
  `datetime.now()`.
- The query planner handles single filter/groupby/aggregate questions fine
  but doesn't do multi-step reasoning — if a question needs two chained
  aggregations, it just returns `{"intent": "unsupported"}` with a message
  instead of guessing.
- No auth on the API. Fine for this, not something you'd ship.
- The rule-based fallback only really covers the sample question patterns
  from the brief, it's not a general parser — it's there so the system
  still works with zero setup, not as a real substitute for the LLM path.
- The anomaly thresholds (24h, 1.5x IQR) are just reasonable-sounding
  defaults. No labeled anomaly set was provided to actually tune them
  against.

## 5. What I'd do with more time
- Have the LLM also phrase the fallback parser's answers, right now only
  the LLM path's response text is templated nicely, the fallback path is
  more basic.
- Move the DataFrame into DuckDB once the dataset gets bigger, so filtering
  can happen in SQL instead of pandas, without changing how the LLM's JSON
  spec gets interpreted.
- Some kind of conversation memory, so a follow-up like "and just for
  Billing?" reuses the previous filters instead of starting over.
- Show the anomaly breakdown as a chart in the UI instead of a flat table.