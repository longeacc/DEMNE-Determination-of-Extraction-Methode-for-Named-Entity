"""DuraXELL — GUI Tkinter native (parité avec dashboard/ Streamlit).

Cible : environnements EDS / Citrix / RDP sans navigateur. Aucune dépendance
externe : uniquement la bibliothèque standard Python (tkinter).

Lancement :
    python main.py gui
"""

from __future__ import annotations

import csv
import importlib.util
import io
import json
import os
import queue
import re
import subprocess
import sys
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

CONFIG_PATH = ROOT / "data" / "decision_config.json"
RESULTS_DIR = ROOT / "Results"
DEFAULT_PORT = 8888

# Presets = data/demne_params.json (source unique partagée CLI/scorers/dashboard)
import importlib.util as _il
_pspec = _il.spec_from_file_location("demne_params", SRC / "demne" / "params.py")
_pmod = _il.module_from_spec(_pspec)
_pspec.loader.exec_module(_pmod)
PARAMS = _pmod.load_params()

PRESETS = PARAMS["presets"]

ROUTING_COLORS = {
    "RÈGLES": "#1B7F2A",
    "TBM":    "#C77800",
    "LLM":    "#B12727",
}


# ---------------------------------------------------------------------------
# Helpers — load DecisionTreeBuilder + BratCorpusParser without triggering
# heavy package side-effects (pandas / streamlit).
# ---------------------------------------------------------------------------
def _load_tree_builder():
    spec = importlib.util.spec_from_file_location(
        "_e_tree", SRC / "demne" / "E_creation_arbre_decision.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.DecisionTreeBuilder


def _load_brat_parser():
    if "streamlit" not in sys.modules:
        st_stub = type(sys)("streamlit")
        st_stub.cache_data = lambda *a, **k: (lambda fn: fn) if not (a and callable(a[0])) else a[0]
        st_stub.cache = st_stub.cache_data
        st_stub.error = lambda *a, **k: None
        sys.modules["streamlit"] = st_stub
    spec = importlib.util.spec_from_file_location(
        "_brat", ROOT / "dashboard" / "core" / "brat_parser.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.BratCorpusParser


# ---------------------------------------------------------------------------
# Main app
# ---------------------------------------------------------------------------
class DuraXellGUI:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("DuraXELL — DEMNE Dashboard (Tkinter)")
        self.root.geometry("1280x820")
        self.root.minsize(960, 640)

        self.corpus_path = tk.StringVar(value="")
        self.pred_path = tk.StringVar(value="")
        self.preset_var = tk.StringVar(value="FRUGAL")
        self.threshold_vars = {
            "Te": tk.DoubleVar(value=PRESETS["FRUGAL"]["Te"]),
            "He": tk.DoubleVar(value=PRESETS["FRUGAL"]["He"]),
            "R":  tk.DoubleVar(value=PRESETS["FRUGAL"]["R"]),
            "Feas": tk.DoubleVar(value=PRESETS["FRUGAL"]["Feas"]),
        }
        self.entity_check_vars: dict[str, tk.BooleanVar] = {}

        self.processes: dict[str, subprocess.Popen] = {}
        self.log_queue: queue.Queue[str] = queue.Queue()

        self._build_layout()
        self._refresh_routing_table()
        self._refresh_entities_panel()
        self.root.after(150, self._drain_log_queue)
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    # ------------------------------------------------------------------ layout
    def _build_layout(self) -> None:
        # Top bar
        top = ttk.Frame(self.root, padding=8)
        top.pack(side="top", fill="x")
        ttk.Label(top, text="Corpus GS :").pack(side="left")
        ttk.Entry(top, textvariable=self.corpus_path, width=70).pack(side="left", padx=4)
        ttk.Button(top, text="📂 Parcourir…", command=self._pick_gs_dir).pack(side="left")
        ttk.Label(top, text="   Pred :").pack(side="left")
        ttk.Entry(top, textvariable=self.pred_path, width=40).pack(side="left", padx=4)
        ttk.Button(top, text="📂", command=self._pick_pred_dir).pack(side="left")
        ttk.Button(top, text="🚀 Évaluer (metrics + tree)",
                   command=self._run_evaluate).pack(side="left", padx=10)

        # Notebook (4 tabs)
        nb = ttk.Notebook(self.root)
        nb.pack(fill="both", expand=True, padx=8, pady=4)

        self.tab_dash = ttk.Frame(nb); nb.add(self.tab_dash, text="📊 Dashboard Métriques")
        self.tab_corp = ttk.Frame(nb); nb.add(self.tab_corp, text="🎯 Analyse Corpus")
        self.tab_rest = ttk.Frame(nb); nb.add(self.tab_rest, text="🔧 REST Integration")
        self.tab_nb   = ttk.Frame(nb); nb.add(self.tab_nb,   text="📓 Notebook REST (in-process)")

        self._build_tab_dashboard(self.tab_dash)
        self._build_tab_corpus(self.tab_corp)
        self._build_tab_rest(self.tab_rest)
        self._build_tab_notebook(self.tab_nb)

        # Bottom log
        bottom = ttk.LabelFrame(self.root, text="Logs", padding=4)
        bottom.pack(fill="x", side="bottom", padx=8, pady=4)
        self.log_text = tk.Text(bottom, height=8, wrap="word",
                                 background="#1E1E1E", foreground="#DDDDDD",
                                 font=("Consolas", 9))
        self.log_text.pack(fill="x")
        self.log_text.configure(state="disabled")

    # ============================== TAB 1 — Dashboard Métriques ===============
    def _build_tab_dashboard(self, parent: ttk.Frame) -> None:
        left = ttk.Frame(parent, padding=8, width=320); left.pack(side="left", fill="y")
        left.pack_propagate(False)
        right = ttk.Frame(parent, padding=8); right.pack(side="left", fill="both", expand=True)

        ttk.Label(left, text="Presets de seuils", font=("Segoe UI", 10, "bold")).pack(anchor="w")
        for name in ("FRUGAL", "QUALITY"):
            ttk.Radiobutton(left, text=name, variable=self.preset_var,
                            value=name, command=self._apply_preset).pack(anchor="w")
        ttk.Radiobutton(left, text="Custom", variable=self.preset_var,
                        value="CUSTOM").pack(anchor="w")

        ttk.Separator(left, orient="horizontal").pack(fill="x", pady=6)
        ttk.Label(left, text="Seuils", font=("Segoe UI", 10, "bold")).pack(anchor="w")

        sliders = [
            ("Te (Templatabilité ≥)",   "Te",   0.0, 1.0),
            ("He (Homogénéité ≥)",      "He",   0.0, 1.0),
            ("R  (Risque ≤)",            "R",    0.0, 1.0),
            ("Feas (Faisabilité NER ≥)", "Feas", 0.0, 1.0),
        ]
        for label, key, lo, hi in sliders:
            row = ttk.Frame(left); row.pack(fill="x", pady=2)
            ttk.Label(row, text=label, width=24).pack(side="left")
            var = self.threshold_vars[key]
            entry = ttk.Label(row, textvariable=var, width=6)
            entry.pack(side="right")
            ttk.Scale(row, from_=lo, to=hi, variable=var,
                       command=lambda _v: self.preset_var.set("CUSTOM")).pack(fill="x")
        ttk.Button(left, text="✅ Appliquer le routage",
                   command=self._refresh_routing_table).pack(fill="x", pady=8)

        ttk.Separator(left, orient="horizontal").pack(fill="x", pady=6)
        ttk.Label(left, text="Sauvegarde", font=("Segoe UI", 10, "bold")).pack(anchor="w")
        ttk.Button(left, text="💾 Exporter decision_summary.csv",
                   command=self._export_csv).pack(fill="x", pady=2)

        # (la barre Commandes CLI est désormais en bas du panneau droit,
        #  étalée sur 4 colonnes — voir _build_cli_bar)

        # Right panel split: routing table on top, CLI bar at the bottom
        table_frame = ttk.Frame(right); table_frame.pack(side="top", fill="both", expand=True)
        cols = ("Entity", "Te", "He", "R", "Freq", "Feas", "Method", "Justification")
        self.tree_routing = ttk.Treeview(table_frame, columns=cols, show="headings", height=14)
        for c, w in zip(cols, (260, 60, 60, 60, 70, 60, 80, 420)):
            self.tree_routing.heading(c, text=c)
            self.tree_routing.column(c, width=w, anchor="w")
        for m, color in ROUTING_COLORS.items():
            self.tree_routing.tag_configure(m, foreground=color)
        sb = ttk.Scrollbar(table_frame, orient="vertical", command=self.tree_routing.yview)
        self.tree_routing.configure(yscrollcommand=sb.set)
        self.tree_routing.pack(side="left", fill="both", expand=True)
        sb.pack(side="right", fill="y")

        # CLI bar — multi-column grid below the routing table
        self._build_cli_bar(right)

    def _build_cli_bar(self, parent: ttk.Frame) -> None:
        """Grille 4 colonnes des sous-commandes `python main.py`."""
        bar = ttk.LabelFrame(parent, text="Commandes CLI — python main.py",
                              padding=6)
        bar.pack(side="bottom", fill="x", pady=(8, 0))

        cli_buttons = [
            ("ℹ️  info",          self._cli_info,        "Diagnostic environnement"),
            ("📐 metrics",        self._cli_metrics,     "Te / He / Freq / R / Feas"),
            ("🌳 tree + image",   self._cli_tree,        "Arbre de décision + PNG"),
            ("🚀 evaluate",       self._cli_evaluate,    "Pipeline complet"),
            ("📊 dashboard",      self._cli_dashboard,   "Routage preset FRUGAL/QUALITY"),
            ("🎯 corpus",         self._cli_corpus,      "Stats BRAT"),
            ("🔧 rest-config",    self._cli_rest_config, "Export config JSON"),
            ("📓 notebook",       self._cli_notebook,    "Notebook REST (Voila)"),
            ("🌐 rest",           self._cli_rest,        "demo_rest.py (legacy)"),
            ("💾 export-csv",     self._export_csv,      "Régénère decision_summary"),
        ]
        cols = 4
        for i, (label, cb, tip) in enumerate(cli_buttons):
            r, c = divmod(i, cols)
            cell = ttk.Frame(bar)
            cell.grid(row=r, column=c, sticky="ew", padx=3, pady=2)
            b = ttk.Button(cell, text=label, command=cb)
            b.pack(fill="x")
            ttk.Label(cell, text=tip, foreground="#888",
                      font=("Segoe UI", 8)).pack(anchor="w")
        for c in range(cols):
            bar.grid_columnconfigure(c, weight=1, uniform="cli")

    def _apply_preset(self) -> None:
        name = self.preset_var.get()
        if name in PRESETS:
            for k, v in PRESETS[name].items():
                self.threshold_vars[k].set(round(v, 3))
            self._refresh_routing_table()

    def _refresh_routing_table(self) -> None:
        for i in self.tree_routing.get_children():
            self.tree_routing.delete(i)
        if not CONFIG_PATH.exists():
            self._log(f"⚠️ {CONFIG_PATH.name} introuvable — lance d'abord Évaluer.")
            return
        try:
            cfg = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
        except Exception as e:
            self._log(f"Erreur lecture config : {e}")
            return

        Builder = _load_tree_builder()
        builder = Builder(CONFIG_PATH)
        builder.THRESHOLDS = {
            "TE_HIGH":  round(self.threshold_vars["Te"].get(), 3),
            "HE_HIGH":  round(self.threshold_vars["He"].get(), 3),
            "R_HIGH":   round(self.threshold_vars["R"].get(),  3),
            "FEAS_NER": round(self.threshold_vars["Feas"].get(), 3),
        }
        for ent, data in cfg.get("entities", {}).items():
            m = data.get("metrics", {})
            decision = builder.analyze_entity(ent, m)
            te = m.get("Te", 0); he = m.get("He", 0)
            if te > 1.0: te /= 100.0
            if he > 1.0: he /= 100.0
            method = decision["method"]
            self.tree_routing.insert(
                "", "end",
                values=(
                    ent,
                    f"{te:.3f}", f"{he:.3f}",
                    f"{m.get('R', 0):.3f}",
                    f"{m.get('Freq', 0):.4f}",
                    f"{m.get('Feas', 0):.3f}",
                    method,
                    decision["justification"],
                ),
                tags=(method,),
            )
        self._refresh_entities_panel()

    # ============================== TAB 2 — Analyse Corpus ====================
    def _build_tab_corpus(self, parent: ttk.Frame) -> None:
        top = ttk.Frame(parent, padding=8); top.pack(fill="x")
        ttk.Button(top, text="🔍 Analyser corpus BRAT",
                   command=self._analyze_corpus).pack(side="left")
        self.lbl_corpus_info = ttk.Label(top, text="(aucun corpus chargé)")
        self.lbl_corpus_info.pack(side="left", padx=10)

        cols = ("Entity", "Count", "Top values")
        self.tree_corpus = ttk.Treeview(parent, columns=cols, show="headings")
        for c, w in zip(cols, (300, 100, 700)):
            self.tree_corpus.heading(c, text=c)
            self.tree_corpus.column(c, width=w, anchor="w")
        sb = ttk.Scrollbar(parent, orient="vertical", command=self.tree_corpus.yview)
        self.tree_corpus.configure(yscrollcommand=sb.set)
        self.tree_corpus.pack(side="left", fill="both", expand=True, padx=8, pady=4)
        sb.pack(side="right", fill="y")

    def _analyze_corpus(self) -> None:
        path = self.corpus_path.get().strip()
        if not path or not Path(path).is_dir():
            messagebox.showerror("Corpus", "Sélectionne d'abord un dossier corpus valide en haut.")
            return
        self._log(f"Analyse BRAT : {path}")
        try:
            BratCorpusParser = _load_brat_parser()
            parser = BratCorpusParser()
            docs = parser.parse_directory(path)
            stats = parser.get_entity_statistics(docs)
        except Exception as e:
            self._log(f"Erreur analyse : {e}")
            messagebox.showerror("Analyse corpus", str(e))
            return

        for i in self.tree_corpus.get_children():
            self.tree_corpus.delete(i)
        for ent, s in sorted(stats.items(), key=lambda x: -x[1].get("count", 0)):
            vals = s.get("value_distribution", {})
            top = ", ".join(f"{k}={v}" for k, v
                            in sorted(vals.items(), key=lambda x: -x[1])[:5])
            self.tree_corpus.insert("", "end",
                                    values=(ent, s.get("count", 0), top))
        self.lbl_corpus_info.config(
            text=f"✅ {len(docs)} documents — {len(stats)} entités")
        self._log(f"  → {len(docs)} docs, {len(stats)} entités")

    # ============================== TAB 3 — REST Integration ==================
    def _build_tab_rest(self, parent: ttk.Frame) -> None:
        left = ttk.Frame(parent, padding=8, width=420); left.pack(side="left", fill="y")
        left.pack_propagate(False)
        right = ttk.Frame(parent, padding=8); right.pack(side="left", fill="both", expand=True)

        ttk.Label(left, text="Entités sélectionnées",
                   font=("Segoe UI", 10, "bold")).pack(anchor="w")
        btnrow = ttk.Frame(left); btnrow.pack(fill="x")
        ttk.Button(btnrow, text="Tout sélectionner",
                   command=lambda: self._toggle_all_entities(True)).pack(side="left")
        ttk.Button(btnrow, text="Tout désélectionner",
                   command=lambda: self._toggle_all_entities(False)).pack(side="left", padx=4)

        canvas = tk.Canvas(left, borderwidth=0, height=440)
        scroll = ttk.Scrollbar(left, orient="vertical", command=canvas.yview)
        self.entities_frame = ttk.Frame(canvas)
        self.entities_frame.bind(
            "<Configure>",
            lambda _e: canvas.configure(scrollregion=canvas.bbox("all")),
        )
        canvas.create_window((0, 0), window=self.entities_frame, anchor="nw")
        canvas.configure(yscrollcommand=scroll.set)
        canvas.pack(side="left", fill="both", expand=True, pady=4)
        scroll.pack(side="right", fill="y")

        # Export / import area
        ttk.Label(right, text="Configuration JSON",
                   font=("Segoe UI", 10, "bold")).pack(anchor="w")
        btns = ttk.Frame(right); btns.pack(fill="x", pady=4)
        ttk.Button(btns, text="💾 Exporter…",
                   command=self._export_config).pack(side="left")
        ttk.Button(btns, text="📥 Importer…",
                   command=self._import_config).pack(side="left", padx=4)
        ttk.Button(btns, text="↻ Recharger depuis decision_config.json",
                   command=self._refresh_entities_panel).pack(side="left", padx=4)

        self.txt_json = tk.Text(right, wrap="none", font=("Consolas", 9))
        self.txt_json.pack(fill="both", expand=True, pady=4)

    def _refresh_entities_panel(self) -> None:
        for w in self.entities_frame.winfo_children():
            w.destroy()
        if not CONFIG_PATH.exists():
            ttk.Label(self.entities_frame,
                       text="(decision_config.json absent)").pack(anchor="w")
            return
        try:
            cfg = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
        except Exception:
            return
        self.entity_check_vars = {}
        for ent, data in cfg.get("entities", {}).items():
            v = tk.BooleanVar(value=True)
            self.entity_check_vars[ent] = v
            method = data.get("method", "?")
            color = ROUTING_COLORS.get(method, "#555")
            row = ttk.Frame(self.entities_frame); row.pack(fill="x", anchor="w")
            ttk.Checkbutton(row, text=ent, variable=v).pack(side="left")
            lbl = tk.Label(row, text=f" → {method}", foreground=color,
                            font=("Segoe UI", 9, "bold"))
            lbl.pack(side="left")

    def _toggle_all_entities(self, value: bool) -> None:
        for v in self.entity_check_vars.values():
            v.set(value)

    def _gather_config_payload(self) -> dict:
        cfg = {}
        if CONFIG_PATH.exists():
            try:
                cfg = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
            except Exception:
                pass
        return {
            "selected_entities": [e for e, v in self.entity_check_vars.items() if v.get()],
            "thresholds": {k: round(v.get(), 3) for k, v in self.threshold_vars.items()},
            "routings": {k: d.get("method") for k, d in cfg.get("entities", {}).items()},
        }

    def _export_config(self) -> None:
        payload = self._gather_config_payload()
        path = filedialog.asksaveasfilename(
            defaultextension=".json",
            initialdir=str(RESULTS_DIR),
            initialfile="demne_config.json",
            filetypes=[("JSON", "*.json")])
        if not path:
            return
        Path(path).write_text(json.dumps(payload, indent=2, ensure_ascii=False),
                              encoding="utf-8")
        self.txt_json.delete("1.0", "end")
        self.txt_json.insert("1.0", json.dumps(payload, indent=2, ensure_ascii=False))
        self._log(f"💾 Config exportée : {path}")

    def _import_config(self) -> None:
        path = filedialog.askopenfilename(initialdir=str(RESULTS_DIR),
                                           filetypes=[("JSON", "*.json")])
        if not path:
            return
        try:
            data = json.loads(Path(path).read_text(encoding="utf-8"))
        except Exception as e:
            messagebox.showerror("Import", str(e))
            return
        thr = data.get("thresholds", {})
        for k, v in thr.items():
            if k in self.threshold_vars:
                self.threshold_vars[k].set(v)
        sel = set(data.get("selected_entities", []))
        for ent, var in self.entity_check_vars.items():
            var.set(ent in sel)
        self.txt_json.delete("1.0", "end")
        self.txt_json.insert("1.0", json.dumps(data, indent=2, ensure_ascii=False))
        self.preset_var.set("CUSTOM")
        self._refresh_routing_table()
        self._log(f"📥 Config importée : {path}")

    # ============================== TAB 4 — Notebook REST =====================
    def _build_tab_notebook(self, parent: ttk.Frame) -> None:
        """Deux modes :
          (a) viewer in-process (zéro port, ipywidgets non interactifs)
          (b) fenêtre WebView2 native avec Jupyter local sur 127.0.0.1 loopback
              (ipywidgets pleinement fonctionnels, validable en EDS)
        """
        # Barre d'action en haut
        topbar = ttk.Frame(parent, padding=(8, 6))
        topbar.pack(fill="x")
        ttk.Label(topbar, text="Mode de rendu :",
                  font=("Segoe UI", 9, "bold")).pack(side="left")
        self.var_nb_engine = tk.StringVar(value="notebook")
        for label, val in (("Jupyter Notebook", "notebook"),
                            ("JupyterLab", "lab"),
                            ("Voilà", "voila")):
            ttk.Radiobutton(topbar, text=label, value=val,
                            variable=self.var_nb_engine).pack(side="left", padx=4)
        ttk.Button(topbar, text="🪟 Ouvrir dans une fenêtre WebView2 native",
                   command=self._launch_webview_notebook).pack(side="right")
        ttk.Separator(parent, orient="horizontal").pack(fill="x")

        info = ttk.Label(
            parent,
            text=("Mode WebView2 : ouvre une fenêtre native qui embarque le "
                  "notebook complet avec ipywidgets interactifs. Une connexion "
                  "127.0.0.1 loopback (jamais exposée au réseau) est utilisée "
                  "pour le protocole Comm de Jupyter — identique à JupyterLab "
                  "Desktop / VS Code / Cursor. Validable en EDS."),
            foreground="#666", wraplength=1200, padding=(8, 4), justify="left")
        info.pack(fill="x")

        # Viewer in-process (mode b par défaut, toujours visible en bas)
        viewer_frame = ttk.LabelFrame(parent,
            text="Viewer in-process (lecture + exécution, sans port — ipywidgets non rendus)",
            padding=4)
        viewer_frame.pack(fill="both", expand=True, padx=4, pady=4)

        try:
            import importlib.util as _il
            spec = _il.spec_from_file_location(
                "_nb_view", ROOT / "dashboard" / "notebook_view.py"
            )
            mod = _il.module_from_spec(spec)
            spec.loader.exec_module(mod)
            nb_path = ROOT / "src" / "demne" / "REST_interface" / "REST.ipynb"
            self.notebook_viewer = mod.NotebookViewer(viewer_frame, nb_path,
                                                       log_fn=self._log)
        except ModuleNotFoundError as e:
            ttk.Label(
                viewer_frame,
                text=(
                    f"Viewer notebook indisponible : module manquant « {e.name} ».\n"
                    "Installe-le pour activer la lecture/exécution intégrée :\n"
                    "    pip install nbformat nbclient jupyter_client ipykernel"
                ),
                foreground="#a00",
                justify="left",
                padding=(8, 8),
            ).pack(fill="x")
            self.notebook_viewer = None

    def _launch_webview_notebook(self) -> None:
        script = ROOT / "dashboard" / "notebook_webview.py"
        if not script.exists():
            messagebox.showerror("WebView", f"Script introuvable : {script}")
            return
        try:
            import webview  # noqa: F401
        except ImportError:
            messagebox.showerror(
                "pywebview manquant",
                "Le module `pywebview` n'est pas installé.\n\n"
                "Installe-le avec :\n"
                "    ./.venv/Scripts/python.exe -m pip install pywebview\n\n"
                "WebView2 utilise le contrôle natif Windows (Edge) — pas de "
                "navigateur autonome.")
            return
        engine = self.var_nb_engine.get()
        cmd = [sys.executable, str(script), "--engine", engine]
        self._spawn(f"webview_{engine}", cmd, cwd=str(ROOT))
        self._log(f"🪟 Lancement WebView2 ({engine}) — la fenêtre native va "
                  "s'ouvrir sous quelques secondes.")

    # ============================== Common — pipelines ========================
    def _pick_gs_dir(self) -> None:
        d = filedialog.askdirectory(title="Dossier corpus GS (BRAT)")
        if d:
            self.corpus_path.set(d)

    def _pick_pred_dir(self) -> None:
        d = filedialog.askdirectory(title="Dossier prédictions (BRAT, optionnel)")
        if d:
            self.pred_path.set(d)

    def _run_evaluate(self) -> None:
        gs = self.corpus_path.get().strip()
        if not gs:
            if messagebox.askyesno("Évaluer", "Aucun corpus sélectionné — lancer "
                                    "avec le corpus par défaut (Breast/RCP) ?") is False:
                return
        cmd = [sys.executable, str(ROOT / "main.py"), "evaluate", "--no-visualize"]
        if gs: cmd += ["--gs_dir", gs]
        if self.pred_path.get().strip():
            cmd += ["--pred_dir", self.pred_path.get().strip()]
        self._spawn("evaluate", cmd, cwd=str(ROOT), on_exit=self._on_evaluate_done)
        self._log(f"🚀 Lancement : {' '.join(cmd)}")

    def _on_evaluate_done(self) -> None:
        self._log("✅ Pipeline terminé — rechargement de la config.")
        self._refresh_routing_table()

    def _export_csv(self) -> None:
        cmd = [sys.executable, str(ROOT / "main.py"), "export-csv"]
        self._spawn("export", cmd, cwd=str(ROOT))

    # ============================== CLI sub-command handlers ==================
    def _run_cli(self, args: list[str], on_exit=None) -> None:
        """Exécute `python main.py …` en subprocess, log dans la zone Logs."""
        cmd = [sys.executable, str(ROOT / "main.py"), *args]
        self._log("$ " + " ".join(cmd[1:]))
        self._spawn(f"cli_{args[0]}", cmd, cwd=str(ROOT), on_exit=on_exit)

    def _ask(self, title: str, prompt: str, initial: str = "") -> str | None:
        from tkinter import simpledialog
        return simpledialog.askstring(title, prompt, initialvalue=initial, parent=self.root)

    def _cli_info(self) -> None:
        self._run_cli(["info"])

    def _cli_metrics(self) -> None:
        d = filedialog.askdirectory(
            title="metrics — dossier corpus (BRAT : .txt + .ann)",
            initialdir=self.corpus_path.get() or str(ROOT))
        if not d:
            return
        self.corpus_path.set(d)
        args = ["metrics", "--gs_dir", d]
        if self.pred_path.get().strip():
            args += ["--pred_dir", self.pred_path.get().strip()]
        self._run_cli(args, on_exit=self._show_all_metric_tables)

    def _show_all_metric_tables(self) -> None:
        """Ouvre les CSV de métriques produites par `metrics` dans une fenêtre à onglets."""
        files = [
            ("Templatabilité (Te)",  RESULTS_DIR / "templatability_analysis.json"),
            ("Homogénéité (He)",     RESULTS_DIR / "homogeneity_analysis.csv"),
            ("Fréquence (Freq)",     RESULTS_DIR / "frequency_analysis.csv"),
            ("Risque (R)",           RESULTS_DIR / "risk_context_analysis.csv"),
            ("Faisabilité (Feas)",   RESULTS_DIR / "ner_feasibility_analysis.csv"),
        ]
        existing = [(t, p) for t, p in files if p.exists()]
        if not existing:
            return
        win = tk.Toplevel(self.root); win.title("Résultats — métriques")
        win.geometry("1000x600")
        nb = ttk.Notebook(win); nb.pack(fill="both", expand=True)
        for title, path in existing:
            frame = ttk.Frame(nb); nb.add(frame, text=title)
            self._fill_table_from_file(frame, path)

    def _fill_table_from_file(self, parent: ttk.Frame, path: Path) -> None:
        """Affiche un .csv ou .json comme Treeview dans `parent`."""
        try:
            if path.suffix == ".json":
                data = json.loads(path.read_text(encoding="utf-8"))
                # Convert {ent: {...}} -> list of dicts
                rows = []
                if isinstance(data, dict):
                    for ent, vals in data.items():
                        row = {"Entity": ent}
                        if isinstance(vals, dict):
                            row.update({k: vals[k] for k in vals})
                        else:
                            row["value"] = vals
                        rows.append(row)
                cols = list(rows[0].keys()) if rows else ["Entity"]
                table = [[r.get(c, "") for c in cols] for r in rows]
            else:
                # CSV (auto-detect delimiter: , ; \t)
                text = path.read_text(encoding="utf-8-sig")
                delim = "\t" if "\t" in text.splitlines()[0] else ","
                reader = csv.reader(io.StringIO(text), delimiter=delim)
                rows_raw = list(reader)
                if not rows_raw:
                    return
                cols = rows_raw[0]
                table = rows_raw[1:]
        except Exception as e:
            ttk.Label(parent, text=f"Erreur lecture {path.name} : {e}",
                      foreground="#B12727").pack(anchor="w", padx=8, pady=8)
            return

        tv = ttk.Treeview(parent, columns=cols, show="headings")
        for c in cols:
            tv.heading(c, text=c)
            tv.column(c, width=max(80, min(280, len(c) * 12)), anchor="w")
        # color the Method column if present
        if "method" in [c.lower() for c in cols] or "Method" in cols:
            for m, color in ROUTING_COLORS.items():
                tv.tag_configure(m, foreground=color)
        for r in table:
            tags = ()
            for cell in r:
                cell_str = str(cell) if not isinstance(cell, (list, dict)) else ""
                if cell_str in ROUTING_COLORS:
                    tags = (cell_str,)
                    break
            tv.insert("", "end",
                      values=[str(c) if not isinstance(c, (list, dict)) else json.dumps(c) for c in r],
                      tags=tags)
        sb = ttk.Scrollbar(parent, orient="vertical", command=tv.yview)
        tv.configure(yscrollcommand=sb.set)
        tv.pack(side="left", fill="both", expand=True)
        sb.pack(side="right", fill="y")
        # bottom : path + button "Open in Excel"
        bot = ttk.Frame(parent); bot.pack(side="bottom", fill="x")
        ttk.Label(bot, text=str(path), foreground="#888").pack(side="left", padx=4)
        ttk.Button(bot, text="📂 Ouvrir le fichier",
                   command=lambda p=path: os.startfile(str(p))).pack(side="right")

    def _cli_tree(self) -> None:
        """Lance `main.py tree` puis affiche Results/figures/Graph_decision.png."""
        d = filedialog.askdirectory(
            title="tree — dossier corpus (BRAT) — Annuler = utiliser le précédent",
            initialdir=self.corpus_path.get() or str(ROOT))
        args = ["tree", "--no-visualize"]
        if d:
            self.corpus_path.set(d)
        if self.corpus_path.get().strip():
            args += ["--gs_dir", self.corpus_path.get().strip()]
        if self.pred_path.get().strip():
            args += ["--pred_dir", self.pred_path.get().strip()]
        self._run_cli(args, on_exit=self._on_tree_done)

    def _on_tree_done(self) -> None:
        self._show_decision_tree_image()
        self._show_decision_summary_table()

    def _show_decision_tree_image(self) -> None:
        img_path = ROOT / "Results" / "figures" / "Graph_decision.png"
        if not img_path.exists():
            # fallback : essayer les autres noms connus
            for fname in ("decision_tree.png",
                           "decision_tree_visualization.png"):
                p = ROOT / "Results" / "figures" / fname
                if p.exists():
                    img_path = p
                    break
        if not img_path.exists():
            messagebox.showinfo("Arbre", f"Aucune image trouvée dans Results/figures/.")
            return
        self._log(f"🖼  Affichage de {img_path.name}")
        try:
            from PIL import Image, ImageTk
        except ImportError:
            messagebox.showerror("Affichage",
                                  "PIL requis (déjà dans le venv).")
            return
        top = tk.Toplevel(self.root)
        top.title(f"Arbre de décision — {img_path.name}")
        img = Image.open(img_path)
        # Fit to screen
        max_w = min(self.root.winfo_screenwidth() - 80, img.width)
        if img.width > max_w:
            r = max_w / img.width
            img = img.resize((max_w, int(img.height * r)))
        ph = ImageTk.PhotoImage(img)
        canvas = tk.Canvas(top, width=img.width, height=img.height,
                            background="#FFFFFF")
        canvas.pack(fill="both", expand=True)
        canvas.create_image(0, 0, anchor="nw", image=ph)
        canvas.image = ph  # ref to avoid GC
        ttk.Label(top, text=str(img_path), foreground="#666",
                  padding=4).pack(side="bottom", fill="x")

    def _cli_evaluate(self) -> None:
        d = filedialog.askdirectory(
            title="evaluate — dossier corpus (BRAT : .txt + .ann)",
            initialdir=self.corpus_path.get() or str(ROOT))
        if not d:
            return
        self.corpus_path.set(d)
        args = ["evaluate", "--no-visualize", "--gs_dir", d]
        if self.pred_path.get().strip():
            args += ["--pred_dir", self.pred_path.get().strip()]
        self._run_cli(args, on_exit=self._on_evaluate_done)

    def _on_evaluate_done(self) -> None:
        """À la fin d'evaluate : rafraîchit le routing + popups résumé."""
        self._refresh_routing_table()
        self._show_decision_summary_table()
        self._show_all_metric_tables()

    def _show_decision_summary_table(self) -> None:
        """Affiche Results/decision_summary.csv dans une Toplevel propre."""
        path = RESULTS_DIR / "decision_summary.csv"
        if not path.exists():
            return
        win = tk.Toplevel(self.root)
        win.title("📊 Résumé du routage — decision_summary.csv")
        win.geometry("1200x500")
        wrap = ttk.Frame(win); wrap.pack(fill="both", expand=True)
        # decision_summary.csv n'a pas d'entête → on l'injecte
        cols = ("Entity", "Te", "He", "R", "Freq", "Feas", "Method", "Justification")
        widths = (260, 70, 70, 70, 80, 70, 90, 400)
        tv = ttk.Treeview(wrap, columns=cols, show="headings")
        for c, w in zip(cols, widths):
            tv.heading(c, text=c)
            tv.column(c, width=w, anchor="w")
        for m, color in ROUTING_COLORS.items():
            tv.tag_configure(m, foreground=color)
        try:
            with open(path, encoding="utf-8") as f:
                for row in csv.reader(f, delimiter="\t"):
                    tag = (row[6],) if len(row) > 6 and row[6] in ROUTING_COLORS else ()
                    tv.insert("", "end", values=row, tags=tag)
        except Exception as e:
            ttk.Label(wrap, text=f"Erreur : {e}",
                      foreground="#B12727").pack(padx=8, pady=8)
            return
        sb = ttk.Scrollbar(wrap, orient="vertical", command=tv.yview)
        tv.configure(yscrollcommand=sb.set)
        tv.pack(side="left", fill="both", expand=True)
        sb.pack(side="right", fill="y")
        bot = ttk.Frame(win); bot.pack(fill="x")
        ttk.Label(bot, text=str(path), foreground="#888").pack(side="left", padx=4)
        ttk.Button(bot, text="📂 Ouvrir le CSV",
                   command=lambda: os.startfile(str(path))).pack(side="right")

    def _cli_dashboard(self) -> None:
        preset = self.preset_var.get() if self.preset_var.get() in PRESETS else "FRUGAL"
        self._run_cli(["dashboard", "--preset", preset])

    def _cli_corpus(self) -> None:
        d = filedialog.askdirectory(
            title="corpus — dossier BRAT à analyser",
            initialdir=self.corpus_path.get() or str(ROOT))
        if not d:
            return
        self.corpus_path.set(d)
        self._run_cli(["corpus", "--path", d])

    def _cli_rest_config(self) -> None:
        path = filedialog.asksaveasfilename(
            defaultextension=".json", initialdir=str(RESULTS_DIR),
            initialfile="exported_config.json",
            filetypes=[("JSON", "*.json")])
        if not path:
            return
        self._run_cli(["rest-config", "--export", path])

    def _cli_notebook(self) -> None:
        self._run_cli(["notebook", "--port", "8888"])

    def _cli_rest(self) -> None:
        self._run_cli(["rest"])

    # ============================== Subprocess plumbing =======================
    def _spawn(self, name: str, cmd: list[str], cwd: str | None = None,
               on_exit=None) -> None:
        if name in self.processes and self.processes[name].poll() is None:
            messagebox.showinfo("Process", f"{name} est déjà en cours.")
            return
        env = os.environ.copy()
        env["PYTHONIOENCODING"] = "utf-8"
        env["PYTHONUTF8"] = "1"
        try:
            p = subprocess.Popen(
                cmd, cwd=cwd, env=env,
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                bufsize=1, text=True, encoding="utf-8", errors="replace",
            )
        except Exception as e:
            self._log(f"Erreur lancement {name} : {e}")
            return
        self.processes[name] = p
        threading.Thread(target=self._reader, args=(name, p, on_exit),
                         daemon=True).start()

    # Lignes parasites à filtrer (warnings tiers, ANSI résiduels)
    _NOISE_PATTERNS = (
        "pynvml package is deprecated",
        "FutureWarning",
        "DeprecationWarning",
        "UserWarning",
        "site-packages",
        "import pynvml",
    )
    _ANSI_RE = re.compile(r"\x1b\[[0-9;]*[mGKHF]|\x1b\[\?[0-9]+[hl]|\[\d+[mGKHF]")

    def _clean_line(self, line: str) -> str | None:
        line = self._ANSI_RE.sub("", line.rstrip())
        # Drop empty + noise
        if not line.strip():
            return None
        for pat in self._NOISE_PATTERNS:
            if pat in line:
                return None
        return line

    def _reader(self, name, p, on_exit) -> None:
        for raw in p.stdout:
            cleaned = self._clean_line(raw)
            if cleaned is not None:
                self.log_queue.put(f"[{name}] {cleaned}")
        p.wait()
        ok = "✅" if p.returncode == 0 else "❌"
        self.log_queue.put(f"[{name}] {ok} terminé (code={p.returncode})")
        if on_exit:
            self.root.after(50, on_exit)

    def _stop_proc(self, name: str) -> None:
        p = self.processes.get(name)
        if p and p.poll() is None:
            p.terminate()
            self._log(f"■ {name} terminé.")
        # (plus de processus notebook/api à monitorer ici : tout est in-process)

    def _drain_log_queue(self) -> None:
        try:
            while True:
                line = self.log_queue.get_nowait()
                self._log(line)
        except queue.Empty:
            pass
        self.root.after(150, self._drain_log_queue)

    def _log(self, msg: str) -> None:
        if not hasattr(self, "log_text"):
            # Le log_text n'est pas encore construit (init en cours) → on diffère.
            self.root.after(50, lambda m=msg: self._log(m))
            return
        self.log_text.configure(state="normal")
        self.log_text.insert("end", msg + "\n")
        self.log_text.see("end")
        self.log_text.configure(state="disabled")

    def _on_close(self) -> None:
        for name, p in self.processes.items():
            if p.poll() is None:
                p.terminate()
        self.root.destroy()


def launch() -> None:
    root = tk.Tk()
    try:
        style = ttk.Style(root)
        if "vista" in style.theme_names():
            style.theme_use("vista")
        elif "clam" in style.theme_names():
            style.theme_use("clam")
    except Exception:
        pass
    DuraXellGUI(root)
    root.mainloop()


if __name__ == "__main__":
    launch()
