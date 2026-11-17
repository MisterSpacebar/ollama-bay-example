import base64
import json
import os

import pandas as pd
import requests
import streamlit as st

from frontend.research_panel import render_research_panel

API_BASE = os.getenv("API_BASE_URL", "http://localhost:8000")

st.set_page_config(page_title="Ollama Data Chat", layout="wide")
st.title("Ollama Data Chat")

if "dataset_id" not in st.session_state:
    st.session_state.dataset_id = None
    st.session_state.dataset_label = None
    st.session_state.columns = []
    st.session_state.preview = []
    st.session_state.history = []  # list of {role, content}
    st.session_state.dataset_group_id = None
    st.session_state.group_datasets = []  # list of {name, dataset_id, rows, columns}


def set_dataset(data: dict):
    st.session_state.dataset_id = data["dataset_id"]
    st.session_state.dataset_label = data["filename"]
    st.session_state.columns = [c["name"] for c in data["columns"]]
    st.session_state.preview = data["preview"]
    st.session_state.cleaning_report = data.get("cleaning_report", [])
    st.session_state.fcm_summary = data.get("fcm_summary")
    st.session_state.dataset_group_id = None
    st.session_state.group_datasets = []
    st.session_state.history = []
    st.session_state.oxygen_analysis = None


def set_dataset_group(data: dict):
    st.session_state.dataset_group_id = data["dataset_group_id"]
    st.session_state.group_datasets = data["datasets"]
    st.session_state.dataset_id = None
    st.session_state.dataset_label = f"ALL datasets ({len(data['datasets'])})"
    st.session_state.columns = []
    st.session_state.preview = []
    st.session_state.cleaning_report = []
    st.session_state.history = []
    st.session_state.oxygen_analysis = None


with st.sidebar:
    st.header("Data source")

    try:
        health = requests.get(f"{API_BASE}/api/health", timeout=5).json()
        st.success(f"Ollama ok · {health.get('model')}")
    except Exception as e:
        st.error(f"Backend/Ollama unreachable: {e}")

    st.subheader("Load from server data")
    try:
        groups = requests.get(f"{API_BASE}/api/local-datasets", timeout=5).json()["groups"]
    except Exception as e:
        groups = []
        st.warning(f"Could not list local datasets: {e}")

    if groups:
        labels = [f"{g['path']} ({len(g['files'])} file{'s' if len(g['files']) != 1 else ''})" for g in groups]
        choice = st.selectbox("Folder/file", options=range(len(groups)), format_func=lambda i: labels[i])
        clean = st.checkbox("Quarantine out-of-range sensor values", key="clean_local")
        col_load, col_load_all = st.columns(2)
        if col_load.button("Load selected"):
            with st.spinner("Loading..."):
                res = requests.post(
                    f"{API_BASE}/api/local-datasets/load",
                    json={"path": groups[choice]["path"], "clean": clean},
                )
            if res.ok:
                set_dataset(res.json())
                st.rerun()
            else:
                st.error(res.json().get("detail", res.text))
        if col_load_all.button("Load ALL (ask about everything)"):
            with st.spinner("Loading all datasets..."):
                res = requests.post(f"{API_BASE}/api/local-datasets/load-all", json={"clean": clean})
            if res.ok:
                set_dataset_group(res.json())
                st.rerun()
            else:
                st.error(res.json().get("detail", res.text))

    st.subheader("Or upload a file")
    uploaded = st.file_uploader("CSV / Excel / JSON", type=["csv", "xlsx", "xlsm", "json"])
    upload_clean = st.checkbox("Quarantine out-of-range sensor values", key="clean_upload")
    if uploaded is not None and st.button("Upload"):
        with st.spinner("Uploading..."):
            res = requests.post(
                f"{API_BASE}/api/upload",
                files={"file": (uploaded.name, uploaded.getvalue())},
                data={"clean": upload_clean},
            )
        if res.ok:
            set_dataset(res.json())
            st.rerun()
        else:
            st.error(res.json().get("detail", res.text))

if not st.session_state.dataset_id and not st.session_state.dataset_group_id:
    st.info("Load a sensor dataset for charts or observed oxygen analysis. "
            "You can browse the configured research results without loading one.")
    render_research_panel(API_BASE)
    st.stop()

st.caption(f"Loaded: **{st.session_state.dataset_label}**")
if st.session_state.dataset_id:
    with st.expander("Preview (first 50 rows)", expanded=False):
        st.dataframe(st.session_state.preview, use_container_width=True)
else:
    with st.expander("Datasets in this group", expanded=True):
        st.dataframe(st.session_state.group_datasets, use_container_width=True)

if st.session_state.get("cleaning_report"):
    with st.expander("Cleaning report", expanded=False):
        st.dataframe(st.session_state.cleaning_report, use_container_width=True)

# Chart/Train-predict operate on a single dataset; in group mode, let the user pick one.
if st.session_state.dataset_id:
    chart_dataset_id = st.session_state.dataset_id
    all_cols = st.session_state.columns
    fcm_summary = st.session_state.get("fcm_summary")
else:
    names = [d["name"] for d in st.session_state.group_datasets]
    picked = st.selectbox("Dataset for Chart / Train & predict tabs", names, key="picked_group_dataset")
    picked_entry = next(d for d in st.session_state.group_datasets if d["name"] == picked)
    chart_dataset_id = picked_entry["dataset_id"]
    all_cols = picked_entry["columns"]
    fcm_summary = picked_entry.get("fcm_summary")

if fcm_summary and fcm_summary.get("top_causal_edges"):
    with st.expander("Exploratory lagged associations (FCM)", expanded=False):
        st.caption("Noncausal and retrospective; available only for regularly sampled, single-station time series.")
        st.dataframe(fcm_summary["top_causal_edges"], use_container_width=True)

tab_chat, tab_chart, tab_model, tab_oxygen, tab_research = st.tabs(["Chat", "Chart", "Train & predict", "Observed oxygen", "Research outputs"])


def render_tool_lines(tool_lines: list[str]):
    if tool_lines:
        st.markdown("\n".join(f"*[tool] {t}*" for t in tool_lines))


def render_display_event(event: dict):
    """Render a {'chart': {...}} / {'chart_native': {...}} / {'table': {...}} / {'map': {...}} event as a real inline widget."""
    if "chart" in event:
        chart = event["chart"]
        st.image(base64.b64decode(chart["image_base64"]), caption=chart.get("caption"))
    elif "chart_native" in event:
        c = event["chart_native"]
        chart_df = pd.DataFrame(c["rows"])
        if chart_df.empty:
            st.caption("(No data to plot.)")
        else:
            chart_type = c["chart_type"]
            x = c.get("x")
            y = c.get("y") or []
            # color= only makes sense when splitting a SINGLE y series by category; with
            # multiple y columns, Streamlit already draws one series per column on its own.
            color = c.get("group_by") if c.get("group_by") and len(y) == 1 else None
            y_arg = y[0] if len(y) == 1 else y
            widget = {"line": st.line_chart, "scatter": st.scatter_chart, "bar": st.bar_chart}[chart_type]
            widget(chart_df, x=x, y=y_arg, color=color)
        st.caption(c.get("caption"))
    elif "map" in event:
        m = event["map"]
        map_df = pd.DataFrame(m["rows"])
        if not map_df.empty:
            # Column names with literal dots (e.g. 'gps.lat') get misread by the underlying
            # Vega-Lite/deck.gl renderer as nested-field paths, silently breaking the point
            # lookup -- rename to flat, safe names before handing off to st.map.
            label_col = m.get("label_col")
            rename_map = {m["lat_col"]: "latitude", m["lon_col"]: "longitude"}
            if label_col:
                rename_map[label_col] = "label"
            map_df = map_df.rename(columns=rename_map)
            st.map(map_df, latitude="latitude", longitude="longitude")
            if label_col and "label" in map_df.columns:
                st.caption("Locations: " + ", ".join(str(v) for v in map_df["label"].dropna().unique()[:20]))
        if m.get("truncated"):
            st.caption(f"Showing {len(m['rows'])} of {m['total_points']} unique locations.")
    elif "table" in event:
        table = event["table"]
        st.dataframe(table["rows"], use_container_width=True)
        if table.get("truncated"):
            st.caption(f"Showing {len(table['rows'])} of {table['total_rows']} rows.")


def ask_question(question: str):
    st.session_state.history.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        tool_lines: list[str] = []
        display_events: list[dict] = []  # ordered {"chart"/"chart_native"/"table"/"map": {...}} events
        tool_placeholder = st.empty()
        answer = ""
        answer_placeholder = st.empty()
        try:
            with requests.post(
                f"{API_BASE}/api/chat",
                json={
                    "dataset_id": st.session_state.dataset_id,
                    "dataset_group_id": st.session_state.dataset_group_id,
                    "question": question,
                },
                stream=True,
                timeout=120,
            ) as res:
                for line in res.iter_lines(decode_unicode=True):
                    if not line or not line.startswith("data: "):
                        continue
                    payload = line[len("data: "):]
                    if payload == "[DONE]":
                        break
                    msg = json.loads(payload)
                    if "tool" in msg:
                        tool_lines.append(msg["tool"])
                        with tool_placeholder.container():
                            render_tool_lines(tool_lines)
                    elif "chart" in msg or "chart_native" in msg or "table" in msg or "map" in msg:
                        display_events.append(msg)
                        # Charts/tables/maps are real widgets, so render them in their own
                        # fixed spot right after the tool log instead of a placeholder.
                        with tool_placeholder.container():
                            render_tool_lines(tool_lines)
                            for event in display_events:
                                render_display_event(event)
                    elif "token" in msg:
                        answer += msg["token"]
                        answer_placeholder.markdown(answer)
                    elif "error" in msg:
                        answer += f"\n\n[error: {msg['error']}]"
                        answer_placeholder.markdown(answer)
        except Exception as e:
            answer = f"Request failed: {e}"
            answer_placeholder.markdown(answer)

    st.session_state.history.append(
        {"role": "assistant", "content": answer, "tool_lines": tool_lines, "display_events": display_events}
    )


with tab_chart:
    chart_type = st.selectbox("Chart type", ["line", "scatter", "bar", "hist"])
    y_cols = st.multiselect("Y column(s)", all_cols, default=all_cols[:1])
    x_col = st.selectbox("X column (optional)", ["(none)"] + all_cols)
    group_by_col = st.selectbox("Group by (optional, line charts only)", ["(none)"] + all_cols)
    if st.button("Generate chart") and y_cols:
        payload = {
            "dataset_id": chart_dataset_id,
            "chart_type": chart_type,
            "y": y_cols,
            "x": None if x_col == "(none)" else x_col,
            "group_by": None if group_by_col == "(none)" else group_by_col,
        }
        res = requests.post(f"{API_BASE}/api/chart", json=payload, timeout=30)
        if res.ok:
            st.image(res.content)
        else:
            st.error(res.json().get("detail", res.text))

with tab_model:
    st.caption("Fits a basic linear regression (target ~ features) on the full dataset.")
    target_col = st.selectbox("Target column", all_cols, key="target_col")
    feature_cols = st.multiselect("Feature columns", [c for c in all_cols if c != target_col], key="feature_cols")
    if st.button("Train model") and feature_cols:
        res = requests.post(
            f"{API_BASE}/api/train-predict",
            json={"dataset_id": chart_dataset_id, "features": feature_cols, "target": target_col},
            timeout=30,
        )
        if res.ok:
            st.session_state.trained = res.json()
        else:
            st.error(res.json().get("detail", res.text))

    trained = st.session_state.get("trained")
    if trained and trained.get("target") == target_col and trained.get("features") == feature_cols:
        st.write(
            f"Train R²: {trained['train_r_squared']} · Test R²: {trained['test_r_squared']} "
            f"(n={trained['n_rows_used']}, holdout={trained['n_test']})"
        )
        st.json(trained["coefficients"])

        st.subheader("Predict for new values")
        new_row = {}
        cols = st.columns(len(feature_cols)) if feature_cols else []
        for col_widget, feat in zip(cols, feature_cols):
            new_row[feat] = col_widget.number_input(feat, value=0.0, key=f"predict_{feat}")
        if st.button("Predict"):
            res = requests.post(
                f"{API_BASE}/api/train-predict",
                json={
                    "dataset_id": chart_dataset_id,
                    "features": feature_cols,
                    "target": target_col,
                    "new_rows": [new_row],
                },
                timeout=30,
            )
            if res.ok:
                st.success(f"Predicted {target_col}: {res.json()['predictions'][0]}")
            else:
                st.error(res.json().get("detail", res.text))

with tab_research:
    render_research_panel(API_BASE)

with tab_oxygen:
    st.caption(
        "Retrospective, coverage-qualified dissolved oxygen episodes at <2 and <4 mg/L. "
        "No imputation, live warnings, or causal claims. Unknown days are never assumed normal."
    )
    preferred_time = ["datetime", "timestamp", "date_time", "time", "date", "Date"]
    preferred_do = ["odo_mgL", "exo.odo_mgL", "DO (mg/L)", "Dissolved Oxygen (mg/L)", "do_mgL"]
    def default_index(options, preferred):
        return next((options.index(c) for c in preferred if c in options), 0)

    if all_cols:
        time_choice = st.selectbox("Timestamp column", all_cols, index=default_index(all_cols, preferred_time))
        do_choice = st.selectbox("Dissolved oxygen (mg/L) column", all_cols, index=default_index(all_cols, preferred_do))
        station_options = ["(auto)"] + all_cols
        station_choice = st.selectbox("Station/source column", station_options)
        flag_choice = st.selectbox("Observed flag column", ["(auto)"] + all_cols)
        interval_choice = st.selectbox("Sampling interval (minutes)", [5, 10, 15, 30, 60])
        coverage_choice = st.slider("Minimum valid daily coverage", 0.5, 1.0, 0.95, 0.01)
        tz_choice = st.text_input("Local timezone", "America/New_York")
        station_filter = st.text_input("Specific station ID (optional)", "")
        if st.button("Analyze observed oxygen"):
            payload = {
                "dataset_id": chart_dataset_id,
                "timestamp_col": time_choice,
                "do_col": do_choice,
                "station_col": None if station_choice == "(auto)" else station_choice,
                "observed_col": None if flag_choice == "(auto)" else flag_choice,
                "interval_minutes": interval_choice,
                "min_coverage": coverage_choice,
                "timezone": tz_choice,
                "station": station_filter.strip() or None,
            }
            try:
                res = requests.post(f"{API_BASE}/api/hypoxia-audit", json=payload, timeout=60)
                if res.ok:
                    st.session_state.oxygen_analysis = {"dataset_id": chart_dataset_id, "result": res.json()}
                else:
                    st.error(res.json().get("detail", res.text))
            except Exception as exc:
                st.error(f"Audit request failed: {exc}")
        cached = st.session_state.get("oxygen_analysis")
        if cached and cached.get("dataset_id") == chart_dataset_id:
            report = cached["result"]
            st.info(f"{report['status']} | {report['provenance']} | stations: {report['stations_analyzed']}")
            st.subheader("Station / threshold summaries")
            st.dataframe(pd.DataFrame(report["station_summaries"]), use_container_width=True)
            if report.get("summaries_truncated"):
                st.caption("Summary preview truncated; select a single station for narrower results.")
            st.subheader("Recent observed episodes")
            st.dataframe(pd.DataFrame(report["episodes"]), use_container_width=True)
            st.subheader("Latest daily coverage / readings")
            st.dataframe(pd.DataFrame(report["latest_daily_records"]), use_container_width=True)
            st.caption(report["interpretation"])


with tab_chat:
    for msg in st.session_state.history:
        with st.chat_message(msg["role"]):
            render_tool_lines(msg.get("tool_lines", []))
            for event in msg.get("display_events", []):
                render_display_event(event)
            st.markdown(msg["content"])

    if st.button("📊 Give me a general overview of this dataset"):
        ask_question(
            "Give me a general overview of the dataset as a whole, based only on the "
            "actual data loaded (not outside knowledge about other places): what it "
            "covers, notable trends over time, how key numeric columns relate to each "
            "other (use the correlation model), and anything unusual (use the anomalies "
            "model). Ground this in the built-in models rather than guessing."
        )

# st.chat_input only pins to the bottom of the page when called at the top level
# (not nested inside a tab/container), so it's called here instead of inside tab_chat.
question = st.chat_input("Ask about your data...")
if question:
    with tab_chat:
        ask_question(question)
