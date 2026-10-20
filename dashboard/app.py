import streamlit as st


def main() -> None:
    st.set_page_config(
        page_title="DEMNE Dashboard",
        page_icon="🧬",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    if "selected_entities" not in st.session_state:
        from core.metrics import default_entities

        st.session_state.selected_entities = default_entities()

    # Initialize DEMO_METRICS in session setup
    if "custom_metrics" not in st.session_state:
        from core.metrics import DEMO_METRICS

        st.session_state.custom_metrics = {k: v.copy() for k, v in DEMO_METRICS.items()}

    if "thresholds" not in st.session_state:
        # Default thresholds = calibrated thresholds of data/demne_params.json
        # (single source shared with CLI and scorers — no drift possible).
        from core.metrics import PARAMS

        st.session_state.thresholds = dict(PARAMS["presets"]["CALIBRATED"])

    if "routings" not in st.session_state:
        st.session_state.routings = {}

    st.title("🧬 DEMNE: NLP onco-biomarker extraction dashboard")
    st.markdown("""
    Welcome to the **DEMNE** analysis interface.
    Select a page from the left navigation bar:
    - 📊 **Metrics Dashboard**: L2 supervision, comparative charts, cost/benefit, AI routing
    - 🖥️ **CLI Console**: Command launcher and monitoring
    - 🔧 **REST Integration**: Sync, configuration, JSON export/import
    - 📓 **Notebook & API**: Launch the Jupyter REST server and interface.
    """)

    st.info(
        "Use the sidebar to navigate between the different interfaces of the analytics pipeline."
    )

    st.markdown("---")

    # Embed README directly in the home page
    try:
        import os

        readme_path = os.path.join(os.path.dirname(__file__), "README.md")
        with open(readme_path, encoding="utf-8", errors="replace") as f:
            st.markdown(f.read())
    except FileNotFoundError:
        st.warning("README.md file not found for display.")


if __name__ == "__main__":
    main()
