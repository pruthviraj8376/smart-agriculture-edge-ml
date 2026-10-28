"""
Step 6 - Live dashboard for the Smart Agriculture edge ML system.

Run (from the project folder, with the virtual environment active):
    streamlit run dashboard.py

It reads results/predictions.csv written by the edge node and shows:
  * KPIs (readings processed, accuracy, irrigation actions, latency)
  * soil moisture trends per field with the irrigation threshold
  * model probability over time and the actions the edge node took
  * the latest predictions, rejected sensor readings and model registry details
"""
import time

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

import config
import model_registry

st.set_page_config(page_title="Smart Agriculture - Edge ML Dashboard", page_icon="🌱", layout="wide")


# ---------- data loading ----------
@st.cache_data(show_spinner=False)
def _cached_file_stamp(path_str: str, mtime: float):
    return pd.read_csv(path_str)


def load_predictions() -> pd.DataFrame:
    if not config.PREDICTIONS_PATH.exists():
        return pd.DataFrame()
    # The cache key includes the file's modification time, so new results are picked up automatically
    mtime = config.PREDICTIONS_PATH.stat().st_mtime
    df = _cached_file_stamp(str(config.PREDICTIONS_PATH), mtime)
    if df.empty:
        return df
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    return df


def load_feature_importance() -> pd.DataFrame | None:
    path = config.RESULTS_DIR / "feature_importance.csv"
    return pd.read_csv(path) if path.exists() else None


# ---------- sidebar ----------
st.sidebar.title("⚙️ Controls")
auto_refresh = st.sidebar.toggle("Auto-refresh", value=True)
refresh_sec = st.sidebar.slider("Refresh every (seconds)", 1, 30, 3)
st.sidebar.caption("Start the simulator with `python run_pipeline.py` to generate live data.")

df = load_predictions()

if df.empty:
    st.title("🌱 Smart Agriculture - Edge ML Dashboard")
    st.info("No results yet. Run `python run_pipeline.py` in a terminal, then this page will fill in.")
else:
    devices = sorted(df["device_id"].dropna().unique())
    selected = st.sidebar.multiselect("Fields (devices)", devices, default=devices)
    df = df[df["device_id"].isin(selected)] if selected else df.iloc[0:0]

    ok = df[df["status"] == "OK"].copy()
    rejected = df[df["status"] == "REJECTED"].copy()

    # ---------- header ----------
    prod = model_registry.get_production_info()
    st.title("🌱 Smart Agriculture - Edge ML Dashboard")
    if prod:
        m = prod["metrics"]
        st.caption(
            f"Model **{config.MODEL_NAME} v{prod['version']}** (Random Forest) · "
            f"Accuracy {m['accuracy']:.2%} · F1 {m['f1']:.2f} · ROC-AUC {m['roc_auc']:.2f} · "
            f"Updated {time.strftime('%H:%M:%S')}"
        )

    # ---------- KPIs ----------
    labelled = ok.dropna(subset=["actual_irrigation_needed"])
    live_accuracy = (labelled["predicted_irrigation"] == labelled["actual_irrigation_needed"]).mean() \
        if len(labelled) else float("nan")
    irrigation_on = int((ok["action"] == "IRRIGATION_ON").sum())
    avg_latency = ok["latency_ms"].mean() if len(ok) else float("nan")

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Readings processed", f"{len(ok):,}")
    c2.metric("Rejected (bad sensor data)", f"{len(rejected):,}")
    c3.metric("Live accuracy vs ground truth", "-" if pd.isna(live_accuracy) else f"{live_accuracy:.1%}")
    c4.metric("Irrigation ON actions", f"{irrigation_on:,}")
    c5.metric("Avg inference latency", "-" if pd.isna(avg_latency) else f"{avg_latency:.2f} ms")

    st.divider()

    # ---------- soil moisture trend ----------
    left, right = st.columns([3, 2])
    with left:
        st.subheader("Soil moisture by field")
        if len(ok):
            fig = px.line(ok, x="timestamp", y="soil_moisture", color="device_id",
                          labels={"soil_moisture": "Soil moisture (%)", "timestamp": "Time"})
            fig.add_hline(y=config.IRRIGATION_THRESHOLD, line_dash="dash", line_color="red",
                          annotation_text="Irrigation threshold")
            fig.update_layout(height=360, margin=dict(l=10, r=10, t=10, b=10), legend_title_text="")
            st.plotly_chart(fig)
        else:
            st.write("Waiting for valid readings...")

    with right:
        st.subheader("Edge decisions")
        if len(ok):
            counts = ok["action"].value_counts().reset_index()
            counts.columns = ["action", "count"]
            fig = px.bar(counts, x="action", y="count", color="action",
                         color_discrete_map={"IRRIGATION_ON": "#2e7d32", "IRRIGATION_OFF": "#90a4ae",
                                             "MONITOR": "#f9a825"})
            fig.update_layout(height=360, showlegend=False, margin=dict(l=10, r=10, t=10, b=10))
            st.plotly_chart(fig)

    # ---------- probability ----------
    st.subheader("Model probability that irrigation is needed")
    if len(ok):
        fig = go.Figure()
        for device, grp in ok.groupby("device_id"):
            fig.add_trace(go.Scatter(x=grp["timestamp"], y=grp["probability"], mode="lines", name=device))
        fig.add_hline(y=config.ALERT_PROBABILITY, line_dash="dot", line_color="green",
                      annotation_text="Alert threshold")
        fig.update_layout(height=300, yaxis_title="P(irrigation needed)", margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig)

    # ---------- confusion + feature importance ----------
    c_left, c_right = st.columns(2)
    with c_left:
        st.subheader("Predicted vs actual")
        if len(labelled):
            cm = pd.crosstab(labelled["actual_irrigation_needed"].astype(int).map({0: "No", 1: "Yes"}),
                             labelled["predicted_irrigation"].astype(int).map({0: "No", 1: "Yes"}),
                             rownames=["Actual"], colnames=["Predicted"])
            fig = px.imshow(cm, text_auto=True, color_continuous_scale="Greens", aspect="auto")
            fig.update_layout(height=320, margin=dict(l=10, r=10, t=10, b=10))
            st.plotly_chart(fig)
    with c_right:
        st.subheader("What the model relies on")
        fi = load_feature_importance()
        if fi is not None:
            fig = px.bar(fi, x="importance", y="feature", orientation="h")
            fig.update_layout(height=320, yaxis={"categoryorder": "total ascending"},
                              margin=dict(l=10, r=10, t=10, b=10))
            st.plotly_chart(fig)

    # ---------- tables ----------
    st.subheader("Latest predictions")
    show_cols = ["timestamp", "device_id", "soil_moisture", "air_temp", "humidity",
                 "probability", "action", "latency_ms", "actual_irrigation_needed"]
    st.dataframe(ok[show_cols].tail(20).iloc[::-1], hide_index=True)

    with st.expander(f"Rejected sensor readings ({len(rejected)})"):
        if len(rejected):
            st.dataframe(rejected[["received_at", "device_id", "reason"]].tail(50).iloc[::-1],
                         hide_index=True)
        else:
            st.write("No faulty readings so far.")

    with st.expander("Model registry"):
        versions = model_registry.list_versions()
        if versions:
            reg = pd.DataFrame([{
                "version": v["version"], "stage": v["stage"], "registered_at": v["registered_at"],
                **{k: v["metrics"][k] for k in ("accuracy", "f1", "roc_auc")},
            } for v in versions])
            st.dataframe(reg, hide_index=True)
        else:
            st.write("No model registered yet.")

    st.download_button("⬇️ Download predictions (CSV)", ok.to_csv(index=False).encode("utf-8"),
                       file_name="predictions.csv", mime="text/csv")

# ---------- auto refresh ----------
if auto_refresh:
    time.sleep(refresh_sec)
    st.rerun()
