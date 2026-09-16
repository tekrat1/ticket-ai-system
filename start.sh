#!/usr/bin/env bash
# Single-command startup: installs deps (if needed), starts the API,
# then the UI. Ctrl+C stops both.
set -euo pipefail

cd "$(dirname "$0")"

if [ ! -d ".venv" ]; then
  python3 -m venv .venv
fi
source .venv/bin/activate
pip install -q -r requirements.txt

if [ -f ".env" ]; then
  export $(grep -v '^#' .env | xargs -0 2>/dev/null || true)
fi

uvicorn main:app --host 0.0.0.0 --port 8000 &
API_PID=$!

sleep 2
echo "API running at http://localhost:8000 (docs at /docs)"

trap "kill $API_PID" EXIT
streamlit run ui/app.py
