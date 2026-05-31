import streamlit as st


def main() -> None:
    st.set_page_config(
        page_title="DuraXell Dashboard",
        page_icon="🧬",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    if "selected_entities" not in st.session_state:
        from core.metrics import ENTITIES

        st.session_state.selected_entities = list(ENTITIES)

    # Initialize DEMO_METRICS in session setup
    if "custom_metrics" not in st.session_state:
        from core.metrics import DEMO_METRICS

        st.session_state.custom_metrics = {k: v.copy() for k, v in DEMO_METRICS.items()}

    if "thresholds" not in st.session_state:
        # Seuils par défaut = optimum du grid search DEMNE (Train=Cantemist-35 / Test=Redjdal+RCP)
        st.session_state.thresholds = {
            "Te": 0.10,
            "He": 0.85,
            "R": 0.25,
            "Freq": 0.30,
            "Yield": 0.60,
            "Feas": 0.20,
            "DomainShift": 0.60,
            "LLM_Necessity": 0.70,
        }

    if "routings" not in st.session_state:
        st.session_state.routings = {}

    st.title("🧬 DuraXell : Dashboard d'extraction NLP onco-biomarqueurs")
    st.markdown("""
    Bienvenue sur l'interface d'analyse de **DuraXell**.
    Sélectionnez une page dans la barre de navigation à gauche :
    - 📊 **Dashboard Métriques** : Supervision L2, graphiques comparatifs, coût/bénéfice, routage d'IA
    - 🖥️ **Console CLI** : Lancement d'instructions et monitoring
    - 🔧 **REST Integration** : Synchronisation, configurations, export/import JSON
    - 📓 **Notebook & API** : Lancement du serveur et de l'interface REST Jupyter.
    """)

    st.info(
        "Utilisez la sidebar pour naviguer dans les différentes interfaces du pipeline analytique."
    )

    st.markdown("---")

    # Intégration du README directement dans la page d'accueil
    try:
        import os

        readme_path = os.path.join(os.path.dirname(__file__), "README.md")
        with open(readme_path, encoding="utf-8", errors="replace") as f:
            st.markdown(f.read())
    except FileNotFoundError:
        st.warning("Fichier README.md introuvable pour affichage.")


if __name__ == "__main__":
    main()
