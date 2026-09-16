"""
Minimal UI on top of the REST API. Run with:

    streamlit run ui/app.py

Talks to the FastAPI backend over HTTP (API_BASE_URL env var, defaults to
localhost:8000) rather than importing app/ directly — keeps the UI a true
client of the API, so anything the API can do, an external caller can too.
"""

from __future__ import annotations

import os

import pandas as pd
import requests
import streamlit as st

API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")

st.set_page_config(page_title="Support Ticket AI System", layout="wide")
st.title("🎫 Support Ticket AI System")

with st.sidebar:
    st.subheader("Backend status")
    try:
        health = requests.get(f"{API_BASE_URL}/health", timeout=5).json()
        st.success(f"API reachable — {health['rows_loaded']} tickets loaded")
        st.caption(f"LLM provider: {health['llm_provider']} "
                   f"({'configured' if health['llm_configured'] else 'NOT configured — using rule-based fallback'})")
    except requests.RequestException:
        st.error(f"Cannot reach API at {API_BASE_URL}. Is `uvicorn main:app` running?")

tab_query, tab_anomalies = st.tabs(["💬 Ask a question", "🚨 Anomalies"])

with tab_query:
    st.write("Ask a natural-language question about the ticket data.")
    example_qs = [
        "How many tickets are currently open?",
        "Which agent resolved the most tickets?",
        "Show me all Critical tickets not resolved within 12 hours.",
        "What is the average customer rating for Technical category tickets?",
    ]
    picked = st.selectbox("Or pick an example:", ["(type your own below)"] + example_qs)
    default_text = "" if picked == "(type your own below)" else picked
    question = st.text_input("Your question", value=default_text)

    if st.button("Ask", type="primary") and question.strip():
        with st.spinner("Thinking..."):
            try:
                resp = requests.post(f"{API_BASE_URL}/query", json={"question": question}, timeout=30)
                resp.raise_for_status()
                data = resp.json()
                st.markdown(f"**Answer:** {data['answer']}")
                if data["result_preview"]:
                    st.dataframe(pd.DataFrame(data["result_preview"]), use_container_width=True)
                with st.expander("Query plan (debug)"):
                    st.json(data["query_spec"])
            except requests.RequestException as e:
                st.error(f"Request failed: {e}")

with tab_anomalies:
    st.write("Tickets flagged by rule-based and statistical (IQR) anomaly detection.")
    if st.button("Run anomaly detection"):
        with st.spinner("Scanning..."):
            try:
                resp = requests.get(f"{API_BASE_URL}/anomalies", timeout=30)
                resp.raise_for_status()
                data = resp.json()
                st.metric("Anomalies found", data["count"])
                if data["anomalies"]:
                    df = pd.DataFrame(data["anomalies"])
                    st.dataframe(df[["ticket_id", "severity", "reason", "rule"]], use_container_width=True)
                else:
                    st.info("No anomalies detected.")
            except requests.RequestException as e:
                st.error(f"Request failed: {e}")
