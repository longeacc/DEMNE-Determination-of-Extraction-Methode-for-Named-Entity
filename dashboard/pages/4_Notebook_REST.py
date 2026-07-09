# pylint: disable=broad-exception-caught
import os
import subprocess
import time

import streamlit as st

st.set_page_config(page_title="Notebook & API Server", page_icon="📓", layout="wide")


def launch_jupyter() -> None:
    """Launch Jupyter Notebook and store state."""
    if "jupyter_process" not in st.session_state:
        # Working directory: repo root
        work_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

        import json
        import sys

        # Locate the .venv Python executable
        venv_python = os.path.join(work_dir, ".venv", "Scripts", "python.exe")
        if not os.path.exists(venv_python):
            venv_python = sys.executable

        # Write jupyter_server_config.json to allow embedding in an IFrame
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
        # Allow extra time for Voila to pre-render the notebook
        time.sleep(8)


def launch_rest_api() -> None:
    """Launch the REST demo API."""
    if "api_process" not in st.session_state:
        # Launch from the REST_interface directory
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
    st.title("📓 REST & Jupyter Project Server")

    if "jupyter_process" in st.session_state:
        if st.session_state.jupyter_process.poll() is not None:
            del st.session_state["jupyter_process"]

    if "api_process" in st.session_state:
        if st.session_state.api_process.poll() is not None:
            del st.session_state["api_process"]

    st.markdown("""
    This interface lets you run the REST API project (DuraXell Pipeline) locally
    and launch the **Jupyter Notebook `REST.ipynb`** to inspect and test the API
    directly from the dashboard.
    """)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Pipeline API Server")
        if st.button("Launch REST API", type="primary"):
            launch_rest_api()
            st.success("API launched in background (port 5000/8000)")

    with col2:
        st.subheader("Notebook API Environment")
        if st.button("Launch Interactive Notebook Interface", type="primary"):
            launch_jupyter()
            st.success("Server started (localhost:8888)")

    st.markdown("---")
    st.subheader("Embedded REST Interface")

    if "jupyter_process" in st.session_state:
        st.info("API interface is active.")
        st.markdown("You can interact with the REST viewer below.")

        st.components.v1.iframe("http://127.0.0.1:8888", height=800, scrolling=True)
    else:
        st.warning("Launch the environment using the button above to display the interface.")


if __name__ == "__main__":
    main()
