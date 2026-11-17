"""Streamlit interface for configured, read-only trustworthy research outputs.

Separate from the sensor-upload workflow so the research browser works even
when no sensor dataset is loaded or an Ollama server is unavailable.
"""
from __future__ import annotations

import json

import pandas as pd
import requests
import streamlit as st


# Exactly the filterable dimensions supported by backend.research.query_research.
FILTERS = {
    "gam_scores": ("threshold_mg_l", "model"),
    "gam_effects": ("threshold_mg_l",),
    "episode_summary": ("station", "threshold_mg_l"),
    "episode_runs": ("station", "threshold_mg_l"),
    "transitions": ("station",),
    "markov_summary": (),
    "warning_scores": ("horizon_days", "model"),
    "warning_hindcasts": ("station", "horizon_days", "model", "issue_date"),
    "warning_snapshot": ("station", "issue_date"),
}


def render_research_panel(api_base: str) -> None:
    st.caption(
        "Read-only environmental GAM, observed episodes and Markov/warning results. "
        "These are historical research outputs, NOT a live monitoring or alert service."
    )
    try:
        resp = requests.get(f"{api_base}/api/research/catalog", timeout=8)
        resp.raise_for_status()
        catalog = resp.json()
    except (requests.RequestException, ValueError, KeyError) as exc:
        st.warning(f"Research catalog unavailable: {exc}")
        return

    if not catalog.get("enabled", False):
        st.info(
            "Point BAY_RESEARCH_OUTPUT_DIR at the locally generated "
            "multinode-bay-imputation/trustworthy_outputs directory and restart the API."
        )
        return

    available = catalog.get("available", [])
    if available:
        kinds = [item["artifact"] for item in available]
        kind = st.selectbox("Research result", kinds, key="selected_research_artifact")
        info = next(item for item in available if item["artifact"] == kind)
        st.caption(info.get("description", ""))

        if st.session_state.get("research_last_artifact") != kind:
            st.session_state.research_result = None
            st.session_state.research_last_artifact = kind

        supported = FILTERS.get(kind, ())
        filters = {}
        a, b, c = st.columns(3)
        with a:
            if "station" in supported:
                station = st.text_input("Exact station ID (optional)", key="research_station")
                if station.strip():
                    filters["station"] = station.strip()
            if "threshold_mg_l" in supported:
                threshold = st.selectbox("DO threshold", ["All", "2", "4"],
                                         key="research_threshold")
                if threshold != "All":
                    filters["threshold_mg_l"] = float(threshold)
        with b:
            if "horizon_days" in supported:
                horizon = st.selectbox("Historical horizon", ["All", "1", "3", "7"],
                                       key="research_horizon")
                if horizon != "All":
                    filters["horizon_days"] = int(horizon)
            if "model" in supported:
                model = st.text_input("Exact model name (optional)", key="research_model")
                if model.strip():
                    filters["model"] = model.strip()
        with c:
            if "issue_date" in supported:
                date = st.text_input("Historical issue date (YYYY-MM-DD)", key="research_date")
                if date.strip():
                    filters["issue_date"] = date.strip()
            limit = st.slider("Rows per query", 5, 60, 20, 5, key="research_limit")

        if st.button("Query research output", key="query_research"):
            payload = {"artifact": kind, "limit": limit, **filters}
            try:
                result_resp = requests.post(
                    f"{api_base}/api/research/query", json=payload, timeout=30
                )
                if result_resp.ok:
                    st.session_state.research_result = result_resp.json()
                else:
                    st.error(result_resp.json().get("detail", result_resp.text))
                    st.session_state.research_result = None
            except (requests.RequestException, ValueError) as exc:
                st.error(f"Research query failed: {exc}")
                st.session_state.research_result = None

        result = st.session_state.get("research_result")
        if result and result.get("artifact") == kind:
            st.write(
                f"Matched rows: **{result['n_matching']}**; "
                f"showing **{result['n_returned']}**"
            )
            st.dataframe(pd.DataFrame(result["rows"]), use_container_width=True)
            if result.get("truncated"):
                st.caption("The view is truncated; narrow the station/date/model filters.")
            st.caption(result.get("interpretation", ""))
            st.caption(f"Research export SHA-256: {result.get('file_sha256', 'unknown')}")
            st.caption(result.get("validation", ""))
    else:
        st.info("No supported research result files were found under the configured root.")

    missing = catalog.get("missing", [])
    if missing:
        with st.expander("Missing upstream exports", expanded=False):
            st.dataframe(pd.DataFrame(missing), use_container_width=True)

    st.divider()
    st.subheader("Ask Ollama about the research")
    st.caption(
        "This sends a research-only request, even with no sensor CSV loaded. "
        "A tool invocation is required to ground specific numerical claims."
    )
    with st.form("research_only_question"):
        question = st.text_area(
            "Research question",
            placeholder="Compare held-out 3-day Markov and DO-history warning scores.",
            key="research_only_prompt",
        )
        submitted = st.form_submit_button("Ask research model")
    if submitted:
        if not question.strip():
            st.warning("Enter a question first.")
        else:
            answer = ""
            tools = []
            error = None
            status_box = st.empty()
            try:
                with requests.post(
                    f"{api_base}/api/chat",
                    json={"research_only": True, "question": question.strip()},
                    stream=True,
                    # The backend sends a keep-alive every few seconds, so a long silence means a stall.
                    timeout=(10, 120),
                ) as resp:
                    resp.raise_for_status()
                    for line in resp.iter_lines(decode_unicode=True):
                        if not line or not line.startswith("data: "):
                            continue
                        payload = line[6:]
                        if payload == "[DONE]":
                            break
                        event = json.loads(payload)
                        if "status" in event:
                            status_box.caption(f"⏳ {event['status']}…")
                        if "token" in event:
                            answer += str(event["token"])
                        if "tool" in event:
                            tools.append(str(event["tool"]))
                        if "error" in event:
                            error = str(event["error"])
            except (requests.RequestException, ValueError, json.JSONDecodeError) as exc:
                error = str(exc)
            status_box.empty()
            st.session_state.research_chat_result = {
                "question": question.strip(), "answer": answer,
                "tools": tools, "error": error,
            }

    chat = st.session_state.get("research_chat_result")
    if chat:
        st.markdown(chat.get("answer") or "(No text response was returned.)")
        if chat.get("error"):
            st.error(chat["error"])
        if chat.get("tools"):
            with st.expander("Research tool activity", expanded=False):
                st.code("\n".join(chat["tools"]))
        else:
            st.warning(
                "The model did not invoke the research-output tool. "
                "Treat any numerical assertions as unverified until checked "
                "against the source CSV."
            )
