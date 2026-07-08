import importlib.util as _il
import json
from datetime import datetime
from pathlib import Path

import streamlit as st

_tfidf_path = Path(__file__).resolve().parents[2] / "src" / "demne" / "E_tfidf.py"
_tfidf_spec = _il.spec_from_file_location("_tfidf", _tfidf_path)
_tfidf_mod = _il.module_from_spec(_tfidf_spec)
_tfidf_spec.loader.exec_module(_tfidf_mod)
_compute_tfidf_for_all = _tfidf_mod.compute_tfidf_for_all_entities

st.set_page_config(page_title="Analyse Corpus", page_icon="🎯", layout="wide")

st.title("🎯 Analyse Corpus BRAT")

if "corpus" not in st.session_state or not st.session_state["corpus"]:
    st.warning("⚠️ Veuillez d'abord charger un corpus depuis le Dashboard Métriques.")
    st.stop()

st.success(f"✅ Corpus chargé avec succès: {len(st.session_state['corpus'])} documents.")

if "tfidf_scores" not in st.session_state:
    local_path = st.session_state.get("local_path", "")
    if local_path:
        try:
            tfidf_results = _compute_tfidf_for_all(local_path)
            st.session_state["tfidf_scores"] = {
                etype: res["tfidf_score"] for etype, res in tfidf_results.items()
            }
        except Exception as _e:
            st.session_state["tfidf_scores"] = {}
            st.warning(f"TFIDF non calculé : {_e}")
    else:
        st.session_state["tfidf_scores"] = {}

tfidf_scores: dict = st.session_state.get("tfidf_scores", {})

st.markdown("---")

st.subheader("📊 Entités détectées dans le corpus")

selected_entities = []
for entity, data in st.session_state["entity_metrics"].items():
    col1, col2, col3 = st.columns([0.5, 2, 1])
    with col1:
        checked = st.checkbox("", key=f"select_{entity}", value=True)
        if checked:
            selected_entities.append(entity)
    with col2:
        count = st.session_state["entity_stats"][entity]["count"]
        st.write(f"**{entity}** ({count} occurrences)")
    with col3:
        routing = data["Routing"]
        badge_colors = {"RÈGLES": "🟢", "TBM": "🟠", "LLM": "🔴"}
        st.write(f"→ {routing} {badge_colors.get(routing, '')}")

st.session_state["selected_entities"] = selected_entities

st.markdown("---")

st.subheader("📋 Détail par Entité")

for entity, data in st.session_state["entity_metrics"].items():
    count = st.session_state["entity_stats"][entity]["count"]
    with st.expander(f"{entity} ({count} annotations)"):
        col1, col2 = st.columns(2)

        with col1:
            st.write("**Valeurs uniques (top 5):**")
            values = st.session_state["entity_stats"][entity]["value_distribution"]
            for val, v_count in sorted(values.items(), key=lambda x: -x[1])[:5]:
                st.write(f"  • {val}: {v_count}")

        with col2:
            st.write("**Métriques L2:**")
            for metric in ["Te", "He", "R", "Freq", "Feas"]:
                st.write(f"  • {metric}: {data.get(metric, 0.0):.2f}")
            tfidf_val = tfidf_scores.get(entity)
            if tfidf_val is not None:
                st.write(f"  • TFIDF_Extractability: {tfidf_val:.4f}")

        st.write(
            f"**Routage:** {data.get('Routing', 'INCONNU')} (confiance: {data.get('Confidence', 0.0):.2f})"
        )
        st.caption(f"Justification: {data.get('Justification', 'N/A')}")

st.markdown("---")

st.subheader("📤 Export Configuration")

export_config = {
    "corpus_info": {
        "path": st.session_state.get("local_path", "unknown"),
        "documents_count": len(st.session_state["corpus"]),
        "total_annotations": sum(
            stats["count"] for stats in st.session_state["entity_stats"].values()
        ),
    },
    "entities": {
        entity: {
            "count": st.session_state["entity_stats"][entity]["count"],
            "metrics": {
                k: v for k, v in data.items() if k not in ["Routing", "Justification", "Confidence"]
            },
            "routing": data.get("Routing"),
            "confidence": data.get("Confidence"),
        }
        for entity, data in st.session_state["entity_metrics"].items()
    },
    "selected_for_extraction": st.session_state["selected_entities"],
    "thresholds": st.session_state["thresholds"],
    "timestamp": datetime.now().isoformat(),
}

json_str = json.dumps(export_config, indent=2, ensure_ascii=False)

st.code(json_str, language="json")

st.download_button(
    "📥 Télécharger configuration JSON",
    data=json_str,
    file_name=f"demne_config_{datetime.now():%Y%m%d_%H%M}.json",
    mime="application/json",
)
