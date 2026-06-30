"""Lance REST.ipynb dans une fenêtre WebView2 native (Windows) ou WebKit (Linux).

Sans navigateur autonome. Une seule connexion 127.0.0.1 loopback (jamais
exposée au réseau) sert au protocole Comm d'ipywidgets — strictement local
au poste, comme JupyterLab Desktop, VS Code, Cursor.

Usage :
    python dashboard/notebook_webview.py [--engine notebook|lab|voila] [--port 0]

Quand la fenêtre WebView2 est fermée, le serveur Jupyter/Voila est tué.
"""

from __future__ import annotations

import argparse
import os
import re
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path

import webview

ROOT = Path(__file__).resolve().parent.parent
NB_PATH = ROOT / "src" / "demne" / "REST_interface" / "REST.ipynb"


def _free_port() -> int:
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def _wait_ready(proc: subprocess.Popen, marker_re: str, timeout: float = 30.0) -> str | None:
    """Lit stdout/stderr du process jusqu'à trouver un marqueur (URL ready)."""
    pat = re.compile(marker_re)
    start = time.time()
    captured: list[str] = []
    while time.time() - start < timeout:
        line = proc.stdout.readline() if proc.stdout else ""
        if line:
            captured.append(line)
            print(line, end="", file=sys.stderr)
            m = pat.search(line)
            if m:
                return m.group(0)
        if proc.poll() is not None:
            print("[notebook_webview] le process Jupyter s'est arrêté tôt :", file=sys.stderr)
            print("".join(captured), file=sys.stderr)
            return None
        time.sleep(0.05)
    return None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--engine", choices=["notebook", "lab", "voila"],
                        default="notebook",
                        help="Moteur de rendu (Jupyter Notebook, JupyterLab ou Voilà)")
    parser.add_argument("--port", type=int, default=0,
                        help="0 = port libre choisi automatiquement")
    parser.add_argument("--nb", type=str, default=str(NB_PATH),
                        help="Chemin du .ipynb à ouvrir")
    args = parser.parse_args()

    nb_path = Path(args.nb)
    if not nb_path.exists():
        print(f"Notebook introuvable : {nb_path}", file=sys.stderr)
        return 2

    port = args.port or _free_port()
    engine = args.engine

    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUTF8"] = "1"
    env["PYTHONUNBUFFERED"] = "1"

    # On lance le serveur dans le dossier du notebook pour que les imports
    # `from REST_modules import *` fonctionnent.
    cwd = str(nb_path.parent)

    # NB: on appelle directement `python -m jupyterlab` / `python -m notebook`
    # plutôt que `python -m jupyter lab` car le wrapper jupyter perd stderr
    # silencieusement sur Windows quand il est lancé via subprocess.
    if engine == "voila":
        cmd = [sys.executable, "-u", "-m", "voila", str(nb_path),
               f"--port={port}", "--Voila.ip=127.0.0.1", "--no-browser"]
        url_target = f"http://127.0.0.1:{port}/"
        ready_pat = r"http://127\.0\.0\.1:\d+/?"
    elif engine == "lab":
        cmd = [sys.executable, "-u", "-m", "jupyterlab",
               f"--port={port}", "--ip=127.0.0.1", "--no-browser",
               "--ServerApp.token=", "--ServerApp.password=",
               f"--ServerApp.root_dir={cwd}"]
        url_target = f"http://127.0.0.1:{port}/lab/tree/{nb_path.name}"
        ready_pat = r"http://127\.0\.0\.1:\d+/lab"
    else:
        cmd = [sys.executable, "-u", "-m", "notebook",
               f"--port={port}", "--ip=127.0.0.1", "--no-browser",
               "--ServerApp.token=", "--ServerApp.password=",
               f"--ServerApp.root_dir={cwd}"]
        url_target = f"http://127.0.0.1:{port}/tree/{nb_path.name}"
        ready_pat = r"http://127\.0\.0\.1:\d+/(?:tree|notebooks|lab)"

    print(f"[notebook_webview] lancement {engine} sur {url_target}", file=sys.stderr)
    proc = subprocess.Popen(
        cmd, cwd=cwd, env=env,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        bufsize=1, text=True, encoding="utf-8", errors="replace",
    )

    found = _wait_ready(proc, ready_pat, timeout=40.0)
    if not found:
        print("[notebook_webview] serveur non prêt — abandon.", file=sys.stderr)
        proc.terminate()
        return 3

    # Drain le reste du stdout en arrière-plan (pour ne pas bloquer le pipe)
    def _drain():
        if proc.stdout:
            for _ in proc.stdout:
                pass
    threading.Thread(target=_drain, daemon=True).start()

    # Création de la fenêtre WebView2
    window = webview.create_window(
        f"DuraXELL — {nb_path.name} ({engine})",
        url_target,
        width=1280, height=860,
        text_select=True, confirm_close=False,
    )

    try:
        webview.start()
    finally:
        try:
            proc.terminate()
            proc.wait(timeout=3)
        except Exception:
            try:
                proc.kill()
            except Exception:
                pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
