# pylint: disable=broad-exception-caught
import os
import subprocess
import time

import streamlit as st

st.set_page_config(page_title="Notebook & API Server", page_icon="📓", layout="wide")


def launch_jupyter() -> None:
    """Lance Jupyter Notebook et stocke l'état."""
    if "jupyter_process" not in st.session_state:
        # On définit le répertoire où Jupyter sera exécuté (à la racine)
        work_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

        import json
        import sys

        # Trouver l'exécutable Python du .venv
        venv_python = os.path.join(work_dir, ".venv", "Scripts", "python.exe")
        if not os.path.exists(venv_python):
            venv_python = sys.executable

        # Créer le fichier de configuration jupyter_server_config.json pour autoriser l'IFrame
        config_path = os.path.join(work_dir, "jupyter_server_config.json")
        try:
            with open(config_path, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "ServerApp": {
                            "ip": "127.0.0.1",
                            "port": 8888,
                            "allow_origin": "*",
                            "disable_check_xsrf": True,
                            "tornado_settings": {
                                "headers": {
                                    "Content-Security-Policy": "frame-ancestors * 'self' http://127.0.0.1:* http://localhost:*"
                                }
                            },
                        },
                        "IdentityProvider": {"token": ""},
                    },
                    f,
                )
        except Exception:
            pass

        cmd = [
            venv_python,
            "-m",
            "voila",
            "src/demne/REST_interface/REST.ipynb",
            "--no-browser",
            "--port=8888",
            "--Voila.ip=127.0.0.1",
            '--Voila.tornado_settings={"headers":{"Content-Security-Policy":"frame-ancestors *"}}',
        ]

        st.session_state.jupyter_process = subprocess.Popen(
            cmd, cwd=work_dir, stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
        # On donne un peu plus de temps à Voila pour s'initialiser et executer le notebook en pre-rendering
        time.sleep(8)


def launch_rest_api() -> None:
    """Lance l'API REST de démonstration."""
    if "api_process" not in st.session_state:
        # Lancement depuis le répertoire REST_interface
        root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
        work_dir = os.path.join(root_dir, "src", "demne", "REST_interface")

        import sys

        venv_python = os.path.join(root_dir, ".venv", "Scripts", "python.exe")
        if not os.path.exists(venv_python):
            venv_python = sys.executable

        st.session_state.api_process = subprocess.Popen(
            [venv_python, "demo_rest.py"],
            cwd=work_dir,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        time.sleep(2)


def main() -> None:
    st.title("📓 Serveur de Projet REST & Jupyter")

    if "jupyter_process" in st.session_state:
        # Vérify if process is actually running
        if st.session_state.jupyter_process.poll() is not None:
            del st.session_state["jupyter_process"]

    if "api_process" in st.session_state:
        if st.session_state.api_process.poll() is not None:
            del st.session_state["api_process"]

    st.markdown("""
    Cette interface vous permet d'exécuter localement le projet REST API (DuraXell Pipeline)
    et de lancer le **Jupyter Notebook `REST.ipynb`** pour inspecter et tester l'API directement depuis le dashboard.
    """)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Serveur API Pipeline")
        if st.button("Lancer API REST", type="primary"):
            launch_rest_api()
            st.success("API Lancée en arrière-plan (Port habituellement 5000/8000)")

    with col2:
        st.subheader("Environnement Notebook API")
        if st.button("Lancer l'interface Notebook Interactif", type="primary"):
            launch_jupyter()
            st.success("L'interface Serveur a démarré (Localhost:8888)")

    st.markdown("---")
    st.subheader("Interface REST Ihm Embarquée")

    if "jupyter_process" in st.session_state:
        st.info("L'Interface API est active.")
        st.markdown("Vous pouvez interagir avec le visualiseur REST ci-dessous.")

        # Embed avec st.components.v1.iframe
        st.components.v1.iframe("http://127.0.0.1:8888", height=800, scrolling=True)
    else:
        st.warning(
            "Veuillez lancer l'environnement via le bouton ci-dessus pour afficher l'interface."
        )


if __name__ == "__main__":
    main()
