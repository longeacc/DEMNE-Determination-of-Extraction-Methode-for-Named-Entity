import importlib.util as _il
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from core import BratCorpusParser, MetricsCalculator, compute_routing
from core.metrics import PARAMS

st.set_page_config(page_title="Metrics Dashboard", page_icon="📊", layout="wide")

PRESET_FRUGAL = dict(PARAMS["presets"]["FRUGAL"])
PRESET_QUALITY = dict(PARAMS["presets"]["QUALITY"])

# Extractabilité par COUVERTURE de mots-clés (coverage-F1) — remplace la synonymie
# contextuelle : sépare mieux les classes (cf. optimisation). f1_score = coverage-F1.
_tfidf_path = Path(__file__).resolve().parents[2] / "src" / "demne" / "E_tfidf_coverage.py"
_tfidf_spec = _il.spec_from_file_location("_tfidf_cov", _tfidf_path)
_tfidf_mod = _il.module_from_spec(_tfidf_spec)
_tfidf_spec.loader.exec_module(_tfidf_mod)
_compute_tfidf_for_all = _tfidf_mod.compute_coverage_for_all_entities

if "thresholds" not in st.session_state:
    st.session_state["thresholds"] = PRESET_FRUGAL.copy()
    for m, v in PRESET_FRUGAL.items():
        st.session_state[f"slider_{m}"] = float(v)

with st.sidebar:
    st.subheader("📂 Load a BRAT corpus")

    import tkinter as tk
    from tkinter import filedialog

    def select_folder():
        root = tk.Tk()
        root.withdraw()
        root.wm_attributes("-topmost", 1)
        folder = filedialog.askdirectory(master=root)
        root.destroy()
        return folder

    if st.button("🔍 Browse... (Select a folder)"):
        folder_path = select_folder()
        if folder_path:
            st.session_state["local_path"] = folder_path
            st.rerun()

    local_path = st.text_input(
        "Working folder path (BRAT corpus)",
        value=st.session_state.get("local_path", ""),
        placeholder="/path/to/brat_corpus/",
    )

    if local_path and st.button("Analyze corpus"):
        with st.spinner("Analyzing corpus..."):
            parser = BratCorpusParser()
            documents = parser.parse_directory(local_path)
            st.session_state["corpus"] = documents
            st.session_state["entity_stats"] = parser.get_entity_statistics(documents)
            try:
                tfidf_results = _compute_tfidf_for_all(local_path)
                st.session_state["tfidf_scores"] = {
                    etype: {
                        "tfidf_score": res["tfidf_score"],
                        "f1_score": res.get("f1_score", 0.0),
                    }
                    for etype, res in tfidf_results.items()
                }
            except Exception as _e:
                st.session_state["tfidf_scores"] = {}
                st.warning(f"TFIDF not computed: {_e}")
            st.success(f"✅ {len(documents)} documents loaded")

    st.markdown("---")
    st.subheader("🎚️ ROUTING THRESHOLDS")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🌿 Frugal preset"):
            for k, v in PRESET_FRUGAL.items():
                st.session_state[f"slider_{k}"] = float(v)
            st.session_state["thresholds"] = PRESET_FRUGAL.copy()
    with col2:
        if st.button("⭐ Quality preset"):
            for k, v in PRESET_QUALITY.items():
                st.session_state[f"slider_{k}"] = float(v)
            st.session_state["thresholds"] = PRESET_QUALITY.copy()

    for metric in PRESET_FRUGAL.keys():
        if f"slider_{metric}" not in st.session_state:
            st.session_state[f"slider_{metric}"] = float(PRESET_FRUGAL[metric])

        val = st.slider(metric, 0.0, 1.0, step=0.05, key=f"slider_{metric}")
        st.session_state["thresholds"][metric] = val

    st.markdown("---")
    st.subheader("📊 CORPUS STATISTICS")
    if "corpus" in st.session_state and st.session_state["corpus"]:
        st.write(f"Documents: {len(st.session_state['corpus'])}")
        st.write(
            f"Annotations: {sum(stats['count'] for stats in st.session_state['entity_stats'].values())}"
        )
        st.write(f"Unique entities: {len(st.session_state['entity_stats'])}")

st.title("📊 Metrics Dashboard")

if "corpus" not in st.session_state or not st.session_state["corpus"]:
    st.info("👈 Please load a corpus from the sidebar to get started.")
else:
    entity_metrics = {}
    calculator = MetricsCalculator()
    tfidf_scores: dict[str, float] = st.session_state.get("tfidf_scores", {})
    thresholds = st.session_state["thresholds"]

    for entity_type in st.session_state["entity_stats"].keys():
        metrics = calculator.compute_all_metrics(st.session_state["corpus"], entity_type)
        tfidf_entry = tfidf_scores.get(entity_type)
        if tfidf_entry is not None:
            if isinstance(tfidf_entry, dict):
                metrics["tfidf_score"] = tfidf_entry.get("tfidf_score", 0.0)
                metrics["f1_score"] = tfidf_entry.get("f1_score", 0.0)
            else:
                metrics["tfidf_score"] = float(tfidf_entry)
        routing, justification = compute_routing(metrics, thresholds)

        entity_metrics[entity_type] = {
            **metrics,
            "Routing": routing,
            "Justification": justification,
        }

    st.session_state["entity_metrics"] = entity_metrics

    if entity_metrics:
        df = pd.DataFrame(entity_metrics).T.reset_index()
        df.rename(columns={"index": "Entity"}, inplace=True)
    else:
        df = pd.DataFrame(
            columns=[
                "Entity",
                "Te",
                "He",
                "R",
                "Feas",
                "tfidf_score",
                "f1_score",
                "Routing",
                "Justification",
            ]
        )

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Radar Chart")
        if not df.empty:
            selected_entity = st.selectbox("Select an entity", df["Entity"].tolist())
            if selected_entity:
                entity_data = df[df["Entity"] == selected_entity].iloc[0]
                metrics_radar = ["Te", "He", "R", "Feas"]
                r_vals = entity_data[metrics_radar].tolist()
                r_vals.append(r_vals[0])
                theta_vals = metrics_radar + [metrics_radar[0]]
                fig = go.Figure(data=go.Scatterpolar(r=r_vals, theta=theta_vals, fill="toself"))
                fig.update_layout(
                    polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
                    showlegend=False,
                )
                st.plotly_chart(fig, use_container_width=True, key=f"radar_{selected_entity}")
        else:
            st.info("No data extracted yet.")

    with col2:
        st.subheader("Routing Distribution")
        if not df.empty:
            fig_pie = px.pie(
                df,
                names="Routing",
                color="Routing",
                color_discrete_map={
                    "RULES": "#2E7D32",
                    "TBM": "#F57C00",
                    "LLM": "#C62828",
                },
            )
            st.plotly_chart(fig_pie, use_container_width=True)
        else:
            st.info("No data extracted yet.")

    st.subheader("Metrics Heatmap")
    if not df.empty:
        heatmap_cols = [c for c in ["Te", "He", "R", "Feas"] if c in df.columns]
        heatmap_df = df.set_index("Entity")[heatmap_cols]
        fig_heat = px.imshow(heatmap_df.T, color_continuous_scale="RdYlGn", aspect="auto")
        st.plotly_chart(fig_heat, use_container_width=True)
    else:
        st.info("No entities detected in the corpus.")

    st.subheader("Routing Decision Table")

    def color_routing(cell_val):
        return {
            "RULES": "background-color: #C8E6C9; color: #1B5E20;",
            "TBM": "background-color: #FFE0B2; color: #E65100;",
            "LLM": "background-color: #FFCDD2; color: #B71C1C;",
        }.get(cell_val, "")

    if not df.empty:
        display_cols = ["Entity", "Te", "He", "R", "Feas"]
        if "tfidf_score" in df.columns:
            display_cols.append("tfidf_score")
        if "f1_score" in df.columns:
            display_cols.append("f1_score")
        display_cols += ["Routing", "Justification"]
        display_df = df[[c for c in display_cols if c in df.columns]].copy()
        display_df = display_df.rename(
            columns={"tfidf_score": "TFIDF_recall", "f1_score": "TFIDF_F1"}
        )
        st.dataframe(
            display_df.style.map(color_routing, subset=["Routing"]),
            use_container_width=True,
        )
