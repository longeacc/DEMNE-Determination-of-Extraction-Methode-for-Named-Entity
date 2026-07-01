# pylint: disable=broad-exception-caught
import json

import streamlit as st

st.set_page_config(page_title="REST Integration", page_icon="🔧", layout="wide")


def main() -> None:
    st.title("🔧 Configuration et Intégration REST")

    if "entity_stats" not in st.session_state or not st.session_state["entity_stats"]:
        st.warning(
            "⚠️ Veuillez d'abord charger un corpus depuis la page 'Dashboard Métriques' pour identifier les entités cibles."
        )
        return

    corpus_entities = list(st.session_state["entity_stats"].keys())

    st.header("Gestion des Entités")

    col1, col2 = st.columns(2)
    with col1:
        if st.button("Tout sélectionner"):
            st.session_state.selected_entities = corpus_entities.copy()
    with col2:
        if st.button("Tout désélectionner"):
            st.session_state.selected_entities = []

    selected = st.session_state.get("selected_entities", [])

    st.subheader("Entités cibles :")
    checks = {}

    # Checkboxes layout en grille
    cols = st.columns(4)
    for i, entity in enumerate(corpus_entities):
        with cols[i % 4]:
            # Initialiser à True par défaut si non défini
            is_sel = entity in selected if "selected_entities" in st.session_state else True
            checks[entity] = st.checkbox(entity, value=is_sel)

    # Mise à jour list session
    st.session_state.selected_entities = [ent for ent, checked in checks.items() if checked]
    st.info(
        f"{len(st.session_state.selected_entities)}/{len(corpus_entities)} entités sélectionnées."
    )

    st.markdown("---")
    st.header("Synchronisation")

    c1, c2 = st.columns(2)

    with c1:
        st.subheader("Exporter Configuration")
        config_data = {
            "selected_entities": st.session_state.selected_entities,
            "thresholds": st.session_state.get("thresholds", {}),
            "routings": st.session_state.get("routings", {}),
        }
        json_str = json.dumps(config_data, indent=2)
        st.download_button(
            "Télécharger config.json",
            json_str,
            file_name="demne_config.json",
            mime="application/json",
        )

    with c2:
        st.subheader("Importer Configuration")
        uploaded = st.file_uploader("Fichier JSON", type=["json"])
        if uploaded is not None:
            try:
                data = json.load(uploaded)
                st.session_state.selected_entities = data.get("selected_entities", [])
                if "thresholds" in data:
                    st.session_state.thresholds.update(data["thresholds"])
                if "routings" in data:
                    st.session_state.routings.update(data["routings"])
                st.success(
                    "Configuration importée avec succès ! (Rafraîchissez ou allez dans le Dashboard)"
                )
            except Exception as e:
                st.error(f"Erreur de lecture du JSON: {e}")

    st.markdown("---")
    st.subheader("Récapitulatif des Routages")
    if "routings" in st.session_state and st.session_state.routings:
        for ent, rtg in st.session_state.routings.items():
            if ent in st.session_state.selected_entities:
                st.markdown(f"- **{ent}** : {rtg}")
    else:
        st.write("Aucun routage disponible. Rendez-vous dans le Dashboard Metrics pour générer.")


if __name__ == "__main__":
    main()
