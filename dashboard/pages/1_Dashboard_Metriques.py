import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from core import BratCorpusParser, MetricsCalculator, compute_routing

st.set_page_config(page_title="Dashboard Métriques", page_icon="📊", layout="wide")

PRESET_FRUGAL = {"Te": 0.10, "He": 0.85, "R": 0.25, "Feas": 0.20}
PRESET_QUALITY = {"Te": 0.25, "He": 0.55, "R": 0.15, "Feas": 0.70}

if "thresholds" not in st.session_state:
    st.session_state["thresholds"] = PRESET_FRUGAL.copy()
    for m, v in PRESET_FRUGAL.items():
        st.session_state[f"slider_{m}"] = float(v)

with st.sidebar:
    st.subheader("📂 Charger un corpus BRAT")

    import tkinter as tk
    from tkinter import filedialog

    def select_folder():
        root = tk.Tk()
        root.withdraw()
        root.wm_attributes("-topmost", 1)
        folder = filedialog.askdirectory(master=root)
        root.destroy()
        return folder

    if st.button("🔍 Parcourir... (Sélectionner un dossier)"):
        folder_path = select_folder()
        if folder_path:
            st.session_state["local_path"] = folder_path
            st.rerun()

    local_path = st.text_input(
        "Chemin du dossier de travail (Corpus BRAT)",
        value=st.session_state.get("local_path", ""),
        placeholder="/chemin/vers/corpus_brat/",
    )

    if local_path and st.button("Analyser le corpus"):
        with st.spinner("Analyse du corpus en cours..."):
            parser = BratCorpusParser()
            documents = parser.parse_directory(local_path)
            st.session_state["corpus"] = documents
            st.session_state["entity_stats"] = parser.get_entity_statistics(documents)
            st.success(f"✅ {len(documents)} documents chargés")

    st.markdown("---")
    st.subheader("🎚️ SEUILS DE ROUTAGE")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🌿 Preset Frugal"):
            for k, v in PRESET_FRUGAL.items():
                st.session_state[f"slider_{k}"] = float(v)
            st.session_state["thresholds"] = PRESET_FRUGAL.copy()
    with col2:
        if st.button("⭐ Preset Qualité"):
            for k, v in PRESET_QUALITY.items():
                st.session_state[f"slider_{k}"] = float(v)
            st.session_state["thresholds"] = PRESET_QUALITY.copy()

    for metric in PRESET_FRUGAL.keys():
        if f"slider_{metric}" not in st.session_state:
            st.session_state[f"slider_{metric}"] = float(PRESET_FRUGAL[metric])

        val = st.slider(metric, 0.0, 1.0, step=0.05, key=f"slider_{metric}")
        st.session_state["thresholds"][metric] = val

    st.markdown("---")
    st.subheader("📊 STATISTIQUES CORPUS")
    if "corpus" in st.session_state and st.session_state["corpus"]:
        st.write(f"Documents: {len(st.session_state['corpus'])}")
        st.write(
            f"Annotations: {sum(stats['count'] for stats in st.session_state['entity_stats'].values())}"
        )
        st.write(f"Entités uniques: {len(st.session_state['entity_stats'])}")

st.title("📊 Dashboard des Métriques")

if "corpus" not in st.session_state or not st.session_state["corpus"]:
    st.info("👈 Veuillez charger un corpus depuis la barre latérale pour commencer.")
else:
    entity_metrics = {}
    calculator = MetricsCalculator()
    for entity_type in st.session_state["entity_stats"].keys():
        metrics = calculator.compute_all_metrics(st.session_state["corpus"], entity_type)
        routing, justification = compute_routing(metrics, st.session_state["thresholds"])

        entity_metrics[entity_type] = {
            **metrics,
            "Routing": routing,
            "Justification": justification,
        }

    st.session_state["entity_metrics"] = entity_metrics

    # Create DataFrame
    if entity_metrics:
        df = pd.DataFrame(entity_metrics).T.reset_index()
        df.rename(columns={"index": "Entity"}, inplace=True)
    else:
        df = pd.DataFrame(
            columns=["Entity", "Te", "He", "R", "Feas", "Routing", "Justification"]
        )

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Radar Chart")
        if not df.empty:
            selected_entity = st.selectbox("Sélectionner une entité", df["Entity"].tolist())
            if selected_entity:
                entity_data = df[df["Entity"] == selected_entity].iloc[0]
                metrics_radar = ["Te", "He", "R", "Feas"]
                r_vals = entity_data[metrics_radar].tolist()
                # Fermer le polygone du radar chart
                r_vals.append(r_vals[0])
                theta_vals = metrics_radar + [metrics_radar[0]]
                fig = go.Figure(data=go.Scatterpolar(r=r_vals, theta=theta_vals, fill="toself"))
                fig.update_layout(
                    polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
                    showlegend=False,
                )
                st.plotly_chart(fig, use_container_width=True, key=f"radar_{selected_entity}")
        else:
            st.info("Aucune donnée extraite pour le moment.")

    with col2:
        st.subheader("Distribution Routage")
        if not df.empty:
            fig_pie = px.pie(
                df,
                names="Routing",
                color="Routing",
                color_discrete_map={
                    "RÈGLES": "#2E7D32",
                    "TBM": "#F57C00",
                    "LLM": "#C62828",
                },
            )
            st.plotly_chart(fig_pie, use_container_width=True)
        else:
            st.info("Aucune donnée extraite pour le moment.")

    st.subheader("Heatmap Métriques")
    if not df.empty:
        heatmap_df = df.set_index("Entity")[["Te", "He", "R", "Feas"]]
        fig_heat = px.imshow(heatmap_df.T, color_continuous_scale="RdYlGn", aspect="auto")
        st.plotly_chart(fig_heat, use_container_width=True)
    else:
        st.info("Aucune entité détectée dans le corpus pour afficher les graphiques.")

    st.subheader("Tableau de Décision Routage")

    def color_routing(val):
        return {
            "RÈGLES": "background-color: #C8E6C9; color: #1B5E20;",
            "TBM": "background-color: #FFE0B2; color: #E65100;",
            "LLM": "background-color: #FFCDD2; color: #B71C1C;",
        }.get(val, "")

    if not df.empty:
        st.dataframe(df.style.map(color_routing, subset=["Routing"]), use_container_width=True)
