# pylint: disable=broad-exception-caught
import json

import streamlit as st

st.set_page_config(page_title="REST Integration", page_icon="🔧", layout="wide")


def main() -> None:
    st.title("🔧 REST Configuration & Integration")

    if "entity_stats" not in st.session_state or not st.session_state["entity_stats"]:
        st.warning(
            "⚠️ Please load a corpus from the 'Metrics Dashboard' page first to identify target entities."
        )
        return

    corpus_entities = list(st.session_state["entity_stats"].keys())

    st.header("Entity Management")

    col1, col2 = st.columns(2)
    with col1:
        if st.button("Select all"):
            st.session_state.selected_entities = corpus_entities.copy()
    with col2:
        if st.button("Deselect all"):
            st.session_state.selected_entities = []

    selected = st.session_state.get("selected_entities", [])

    st.subheader("Target entities:")
    checks = {}

    # Grid checkbox layout
    cols = st.columns(4)
    for i, entity in enumerate(corpus_entities):
        with cols[i % 4]:
            # Default to True if not yet set
            is_sel = entity in selected if "selected_entities" in st.session_state else True
            checks[entity] = st.checkbox(entity, value=is_sel)

    # Update session list
    st.session_state.selected_entities = [ent for ent, checked in checks.items() if checked]
    st.info(f"{len(st.session_state.selected_entities)}/{len(corpus_entities)} entities selected.")

    st.markdown("---")
    st.header("Synchronization")

    c1, c2 = st.columns(2)

    with c1:
        st.subheader("Export Configuration")
        config_data = {
            "selected_entities": st.session_state.selected_entities,
            "thresholds": st.session_state.get("thresholds", {}),
            "routings": st.session_state.get("routings", {}),
        }
        json_str = json.dumps(config_data, indent=2)
        st.download_button(
            "Download config.json",
            json_str,
            file_name="demne_config.json",
            mime="application/json",
        )

    with c2:
        st.subheader("Import Configuration")
        uploaded = st.file_uploader("JSON file", type=["json"])
        if uploaded is not None:
            try:
                data = json.load(uploaded)
                st.session_state.selected_entities = data.get("selected_entities", [])
                if "thresholds" in data:
                    st.session_state.thresholds.update(data["thresholds"])
                if "routings" in data:
                    st.session_state.routings.update(data["routings"])
                st.success("Configuration imported successfully! (Refresh or go to the Dashboard)")
            except Exception as e:
                st.error(f"JSON read error: {e}")

    st.markdown("---")
    st.subheader("Routing Summary")
    if "routings" in st.session_state and st.session_state.routings:
        for ent, rtg in st.session_state.routings.items():
            if ent in st.session_state.selected_entities:
                st.markdown(f"- **{ent}**: {rtg}")
    else:
        st.write("No routing available. Go to the Metrics Dashboard to generate one.")


if __name__ == "__main__":
    main()
