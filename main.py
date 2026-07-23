"""DuraXELL CLI — Central hub for DEMNE.

Provides:
  - Metric pipeline:     metrics, tree, evaluate
  - Dashboard parity:    dashboard, corpus, rest-config, notebook
  - Misc:                rest, info, export-csv

Mirrors the Streamlit pages so the full workflow can be driven from the terminal.
"""

# pylint: disable=broad-exception-caught,unused-argument
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import subprocess
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Bootstrap: make `demne.*` importable + force UTF-8 stdout on Windows
# ---------------------------------------------------------------------------
ROOT = Path(__file__).parent.resolve()
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

try:
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
    sys.stderr.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
except Exception:
    pass

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_ESMO_DIR = Path(os.environ.get("DEMNE_ESMO_DIR", "data/ESMO2025"))
DEFAULT_GS = _ESMO_DIR / "Breast/RCP/evaluation_set_breast_cancer_GS"
DEFAULT_PRED = _ESMO_DIR / "Breast/RCP/evaluation_set_breast_cancer_pred_rules"


def discover_entities(corpus_path: str | Path | None) -> list[str]:
    """Discover the list of entity types annotated in a BRAT corpus.

    Strategy:
      1. Read `annotation.conf` [entities] section if present.
      2. Otherwise, scan all T* lines in .ann files and collect types.
      3. Fall back to [].
    """
    if not corpus_path:
        return []
    p = Path(corpus_path)
    if not p.exists():
        return []

    # 1) annotation.conf
    conf = p / "annotation.conf"
    if conf.exists():
        try:
            in_entities = False
            ents: list[str] = []
            for raw in conf.read_text(encoding="utf-8").splitlines():
                line = raw.strip()
                if not line or line.startswith("#"):
                    continue
                if line.startswith("["):
                    in_entities = line.lower().startswith("[entities")
                    continue
                if in_entities:
                    name = line.split()[0].strip()
                    if name and name not in ents:
                        ents.append(name)
            if ents:
                return ents
        except Exception:
            pass

    # 2) scan .ann files (T-lines : "T1\tType start end\ttext")
    found: list[str] = []
    seen: set[str] = set()
    for ann in p.glob("*.ann"):
        try:
            for line in ann.read_text(encoding="utf-8").splitlines():
                if not line.startswith("T"):
                    continue
                parts = line.split("\t")
                if len(parts) < 2:
                    continue
                meta = parts[1].split()
                if not meta:
                    continue
                etype = meta[0]
                if etype not in seen:
                    seen.add(etype)
                    found.append(etype)
        except Exception:
            continue
    return found


def discover_entities_from_config() -> list[str]:
    """Return the entities listed in data/decision_config.json (if available)."""
    if not CONFIG_PATH.exists():
        return []
    try:
        cfg = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
        return list(cfg.get("entities", {}).keys())
    except Exception:
        return []


# Tunable values (presets + thresholds) from data/demne_params.json — single source
# of truth shared with the scorers, the decision tree and the dashboard.
import importlib.util as _il

_pspec = _il.spec_from_file_location("demne_params", SRC / "demne" / "params.py")
_pmod = _il.module_from_spec(_pspec)
_pspec.loader.exec_module(_pmod)
PARAMS = _pmod.load_params()

PRESETS = PARAMS["presets"]

CONFIG_PATH = ROOT / "data" / "decision_config.json"
RESULTS_DIR = ROOT / "Results"
DECISION_CSV = RESULTS_DIR / "decision_summary.csv"


# ===========================================================================
# Metric pipeline commands
# ===========================================================================
# Noise lines to filter out in sub-processes (eco2ai, pandas, apscheduler)
_NOISE_KEYS = (
    "pynvml",
    "FutureWarning",
    "DeprecationWarning",
    "UserWarning",
    "NoNeededLibrary",
    "site-packages",
    "warnings.warn",
    "incompatible dtype",
    "apscheduler",
    "Tracker._func_for_sched",
    "_write_to_csv",
    "get_loc",
    "PyObjectHashTable",
    "hashtable_class_helper",
    "index.pyx",
    "raise KeyError",
    "KeyError: 'id'",
    "Traceback (most recent call last)",
    "Generalized error in Eco2AI",
    "carbon emission tracking data",
    "Carbon emission tracking",
    'Job "',
    '  File "',
    "  return ",
    "       ^^^^",
    "    indexer = self.columns.get_loc",
    "        retval = job.func",
    "    return self._engine.get_loc",
    "    return self._write_to_csv",
    "    if attributes_dataframe",
    "                            ~~~",
    "                            ^^",
    "The above exception was the direct cause",
    "If you use a VPN",
    "It is recommended to disable VPN",
    "manually set up the ISO-Alpha-2 code",
    "You can find the ISO-Alpha-2 code",
    "It's impossible to determine cpu number",
    "For now, number of cpu devices",
    "attributes_dataframe",
    "attributes_array",
    "loc[row_index]",
    "retval = job.func",
    "(*job.args, **job.kwargs)",
)


_DROP_PATTERNS = (
    "Comparing GS",
    "[DrBERT MMD] Simulation",
    "Yield computed for",
    "Annotation Yield computed",
    "Computing Annotation Yield",
    "Starting cross-validation",
    "Average decision stability",
    "Score C",
    " C =",
    " C: ",
)
_STRIP_RE = re.compile(
    r",\s*(DS|LLM_N|Yield|Domain[_ ]?Shift|Concordance)\s*=\s*[\d.]+", re.IGNORECASE
)


def _filter_noise(line: str) -> str | None:
    """Retourne la ligne nettoyée (ou None pour la supprimer)."""
    line = line.rstrip()
    if not line.strip():
        return None
    for k in _NOISE_KEYS:
        if k in line:
            return None
    for k in _DROP_PATTERNS:
        if k in line:
            return None
    cleaned = _STRIP_RE.sub("", line)
    return cleaned


def _run_script(script: str, gs_dir: str | None, pred_dir: str | None) -> int:
    cmd = [sys.executable, str(ROOT / script)]
    if gs_dir:
        cmd += ["--gs_dir", gs_dir]
    if pred_dir:
        cmd += ["--pred_dir", pred_dir]
    short = Path(script).name
    print(f"\n>>> {short}  --gs_dir {gs_dir}")
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUTF8"] = "1"
    env["PYTHONUNBUFFERED"] = "1"
    env["DISABLE_ECO2AI"] = "1"
    # Les scorers importent `demne.*` : sans src/ sur le PYTHONPATH, le
    # sous-processus échoue en ModuleNotFoundError si le paquet n'est pas installé.
    env["PYTHONPATH"] = os.pathsep.join(filter(None, [str(SRC), env.get("PYTHONPATH", "")]))
    proc = subprocess.Popen(
        cmd,
        cwd=str(ROOT),
        env=env,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        bufsize=1,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    assert proc.stdout is not None
    for line in proc.stdout:
        cleaned = _filter_noise(line)
        if cleaned is not None:
            print(cleaned)
    proc.wait()
    return proc.returncode


def cmd_metrics(args: argparse.Namespace) -> None:
    """Run all E_*.py metric scripts on the corpus."""
    scripts_gs_pred = [
        "src/demne/E_templatability.py",
        "src/demne/E_homogeneity.py",
        "src/demne/E_frequency.py",
        "src/demne/E_risk_context.py",
        "src/demne/E_feasibility_NER.py",
    ]
    scripts_gs_only = [
        "src/demne/E_tfidf_coverage.py",
    ]
    gs = args.gs_dir or str(DEFAULT_GS)
    pred = args.pred_dir or str(DEFAULT_PRED)
    print(f"Metrics pipeline on:\n  GS   = {gs}")
    for s in scripts_gs_pred:
        _run_script(s, gs, pred)
    for s in scripts_gs_only:
        _run_script(s, gs, None)


def cmd_tree(args: argparse.Namespace) -> None:
    """Display the static DEMNE decision tree diagram.

    Does not run the pipeline or CSV export — use `evaluate` for that.
    """
    del args  # required by argparse signature, content unused
    _open_decision_tree_image()


def _run_tree_pipeline(args: argparse.Namespace) -> None:
    """Decision tree build pipeline (called by `evaluate`)."""
    gs = args.gs_dir or str(DEFAULT_GS)
    pred = args.pred_dir or str(DEFAULT_PRED)
    _run_script("src/demne/E_creation_arbre_decision.py", gs, pred)
    if not args.no_visualize:
        print("\nVisualizing the tree...")
        subprocess.run(
            [sys.executable, str(ROOT / "src/demne/visualize_decision_tree.py")],
            cwd=str(ROOT),
            check=False,
        )
    _export_decision_csv()


def _open_decision_tree_image() -> None:
    """Open Results/figures/Graph_decision.png in the default viewer.

    Fallback: decision_tree.png, decision_tree_visualization.png.
    """
    candidates = [
        ROOT / "Results" / "figures" / "Graph_decision.png",
        ROOT / "Results" / "figures" / "decision_tree.png",
        ROOT / "Results" / "figures" / "decision_tree_visualization.png",
    ]
    img = next((p for p in candidates if p.exists()), None)
    if not img:
        print("⚠️  No tree image found in Results/figures/.")
        return
    print(f"\n🖼  Opening tree: {img}")
    try:
        if sys.platform == "win32":
            os.startfile(str(img))  # type: ignore[attr-defined]
        elif sys.platform == "darwin":
            subprocess.Popen(["open", str(img)])
        else:
            subprocess.Popen(["xdg-open", str(img)])
    except Exception as e:
        print(f"   Could not open viewer: {e}")


def cmd_evaluate(args: argparse.Namespace) -> None:
    """Full pipeline: metrics → tree (pipeline) → CSV export."""
    cmd_metrics(args)
    _run_tree_pipeline(args)


def _export_decision_csv() -> None:
    """Emit Results/decision_summary.csv with tab-separated rows:
    Entity \t Te \t He \t R \t Freq \t Feas \t Method \t Justification
    """
    if not CONFIG_PATH.exists():
        print(f"No decision_config.json found at {CONFIG_PATH}")
        return
    with open(CONFIG_PATH, encoding="utf-8") as f:
        cfg = json.load(f)
    RESULTS_DIR.mkdir(exist_ok=True)

    rows = []
    entities_in_cfg = cfg.get("entities", {})
    for ent, d in entities_in_cfg.items():
        m = d.get("metrics", {})
        te = m.get("Te", 0.0)
        he = m.get("He", 0.0)
        # Convert percent → fraction if needed
        if te > 1.0:
            te /= 100.0
        if he > 1.0:
            he /= 100.0
        tfidf = m.get("tfidf_score")
        rows.append(
            [
                ent,
                f"{te:.4f}",
                f"{he:.4f}",
                f"{m.get('R', 0.0):.4f}",
                f"{m.get('Freq', 0.0):.4f}",
                f"{m.get('Feas', 0.0):.4f}",
                f"{tfidf:.4f}" if tfidf is not None else "",
                d.get("method", ""),
                d.get("justification", ""),
            ]
        )

    header = [
        "Entity",
        "Te",
        "He",
        "R",
        "Freq",
        "Feas",
        "TFIDF_recall",
        "Method",
        "Justification",
    ]
    with open(DECISION_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(header)
        for r in rows:
            w.writerow(r)
    try:
        rel_csv = os.path.relpath(str(DECISION_CSV))
    except ValueError:
        rel_csv = str(DECISION_CSV)
    print(f"\n=== CSV synthèse écrit : {rel_csv} ===")

    # Pretty-print as a fixed-width table with wrapping for long cells
    import textwrap as _tw

    col_w = {
        "Entity": 30,
        "Te": 6,
        "He": 6,
        "R": 6,
        "Freq": 8,
        "Feas": 6,
        "TFIDF_recall": 8,
        "Method": 8,
        "Justification": 55,
    }  # noqa: N806
    sep = "+" + "+".join("-" * (w + 2) for w in col_w.values()) + "+"

    def _fmt_row(cells: list[str]) -> list[str]:
        """Return lines for a table row, wrapping cells that exceed column width."""
        wrapped = [_tw.wrap(str(c), col_w[h]) or [""] for c, h in zip(cells, col_w, strict=True)]
        height = max(len(w) for w in wrapped)
        lines = []
        for i in range(height):
            parts = []
            for w, h in zip(wrapped, col_w, strict=True):
                cell = w[i] if i < len(w) else ""
                parts.append(f" {cell:<{col_w[h]}} ")
            lines.append("|" + "|".join(parts) + "|")
        return lines

    print(sep)
    for line in _fmt_row(header):
        print(line)
    print(sep)
    for r in rows:
        for line in _fmt_row(r):
            print(line)
        print(sep)


# ===========================================================================
# Dashboard parity commands (mirror Streamlit pages)
# ===========================================================================
def cmd_dashboard(args: argparse.Namespace) -> None:
    """Mirror Page 1 — Metrics Dashboard.

    Apply a threshold preset (FRUGAL / QUALITY / custom) and re-run the
    decision tree on the current metrics, printing the routing table.
    """
    if not CONFIG_PATH.exists():
        print("⚠️  No decision_config.json — run 'metrics' then 'tree' first.")
        return

    preset_name = (args.preset or "FRUGAL").upper()
    thresholds = PRESETS.get(preset_name, PRESETS["FRUGAL"]).copy()
    for k in ("Te", "He", "R", "Feas"):
        v = getattr(args, k.lower(), None)
        if v is not None:
            thresholds[k] = v

    print(f"Preset: {preset_name}   Thresholds: {thresholds}")

    with open(CONFIG_PATH, encoding="utf-8") as f:
        cfg = json.load(f)

    spec = _il.spec_from_file_location("_e_tree", SRC / "demne" / "E_creation_arbre_decision.py")
    mod = _il.module_from_spec(spec)
    spec.loader.exec_module(mod)
    builder = mod.DecisionTreeBuilder(CONFIG_PATH)
    builder.THRESHOLDS = {
        "TE_HIGH": thresholds["Te"],
        "HE_HIGH": thresholds["He"],
        "R_HIGH": thresholds["R"],
        "FEAS_NER": thresholds["Feas"],
        "Y": PARAMS["decision_thresholds"].get("TFIDF_Y", 0.70),
    }

    print(
        f"\n{'Entity':<25} | {'Te':>6} | {'He':>6} | {'R':>6} | {'Freq':>8} | {'Feas':>6} | {'Recall':>7} | {'Method':<8} | Justification"
    )
    print("-" * 148)
    for ent, data in cfg.get("entities", {}).items():
        m = data.get("metrics", {})
        decision = builder.analyze_entity(ent, m)
        te = m.get("Te", 0)
        he = m.get("He", 0)
        if te > 1.0:
            te /= 100.0
        if he > 1.0:
            he /= 100.0
        tfidf = m.get("tfidf_score")
        tfidf_str = f"{tfidf:7.3f}" if tfidf is not None else "    n/a"
        print(
            f"{ent:<25} | {te:6.3f} | {he:6.3f} | {m.get('R',0):6.3f} | "
            f"{m.get('Freq',0):8.4f} | {m.get('Feas',0):6.3f} | {tfidf_str} | {decision['method']:<8} | {decision['justification']}"
        )


def cmd_corpus(args: argparse.Namespace) -> None:
    """Mirror Page 2 — BRAT Corpus Analysis."""
    # Stub out streamlit before importing brat_parser (which decorates with @st.cache_data)
    if "streamlit" not in sys.modules:
        st_stub = type(sys)("streamlit")
        st_stub.cache_data = lambda *a, **k: (lambda fn: fn) if not (a and callable(a[0])) else a[0]
        st_stub.cache = st_stub.cache_data  # pylint: disable=no-member
        sys.modules["streamlit"] = st_stub
    try:
        spec = _il.spec_from_file_location("_brat", ROOT / "dashboard" / "core" / "brat_parser.py")
        mod = _il.module_from_spec(spec)
        spec.loader.exec_module(mod)
        brat_corpus_parser_cls = mod.BratCorpusParser
    except Exception as e:
        print(f"Could not load BratCorpusParser: {e}")
        return

    path = args.path or str(DEFAULT_GS)
    print(f"BRAT corpus analysis: {path}")
    parser = brat_corpus_parser_cls()
    docs = parser.parse_directory(path)
    stats = parser.get_entity_statistics(docs)
    print(f"\n✅ {len(docs)} documents loaded.\n")
    print(f"{'Entity':<30} | {'Count':>7} | Top values")
    print("-" * 100)
    for ent, s in sorted(stats.items(), key=lambda x: -x[1].get("count", 0)):
        vals = s.get("value_distribution", {})
        top = ", ".join(f"{k}={v}" for k, v in sorted(vals.items(), key=lambda x: -x[1])[:3])
        print(f"{ent:<30} | {s.get('count',0):>7} | {top}")


def cmd_rest_config(args: argparse.Namespace) -> None:
    """Mirror Page 3 — REST Integration (export/import config JSON)."""
    if args.export:
        if not CONFIG_PATH.exists():
            print("No decision_config.json to export.")
            return
        with open(CONFIG_PATH, encoding="utf-8") as f:
            cfg = json.load(f)
        payload = {
            "selected_entities": list(cfg.get("entities", {}).keys()),
            "thresholds": cfg.get("global_thresholds", {}),
            "routings": {k: v.get("method") for k, v in cfg.get("entities", {}).items()},
        }
        out = Path(args.export)
        out.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"Config exportée : {out}")
    elif args.import_:
        data = json.loads(Path(args.import_).read_text(encoding="utf-8"))
        print("Config importée :")
        print(json.dumps(data, indent=2, ensure_ascii=False))
    else:
        print("Utilise --export <fichier.json> ou --import <fichier.json>")


def cmd_notebook(args: argparse.Namespace) -> None:
    """Mirror Page 4 — launch the REST notebook (Voila) and/or the REST API."""
    procs = []
    if not args.api_only:
        notebook = ROOT / "src" / "demne" / "REST_interface" / "REST.ipynb"
        if not notebook.exists():
            print(f"Notebook introuvable : {notebook}")
        else:
            print(f"Démarrage Voila sur {notebook} (port {args.port})...")
            voila_cmd = [
                sys.executable,
                "-m",
                "voila",
                str(notebook),
                "--no-browser",
                f"--port={args.port}",
                "--Voila.ip=127.0.0.1",
            ]
            procs.append(subprocess.Popen(voila_cmd, cwd=str(ROOT)))
            print(f"  → http://127.0.0.1:{args.port}")

    if not args.notebook_only:
        api = ROOT / "src" / "demne" / "REST_interface" / "demo_rest.py"
        print(f"Démarrage API REST : {api}")
        procs.append(subprocess.Popen([sys.executable, str(api)], cwd=api.parent))

    if not procs:
        return
    print("\nCtrl-C pour arrêter.")
    try:
        for p in procs:
            p.wait()
    except KeyboardInterrupt:
        for p in procs:
            p.terminate()
        print("Arrêt demandé.")


def cmd_rest(args: argparse.Namespace) -> None:
    """Legacy: launches demo_rest.py only."""
    subprocess.run(
        [sys.executable, str(ROOT / "src/demne/REST_interface/demo_rest.py")],
        cwd=str(ROOT / "src/demne/REST_interface"),
        check=False,
    )


def cmd_info(args: argparse.Namespace) -> None:
    def _rel(p: Path) -> str:
        try:
            return str(Path(p).relative_to(ROOT))
        except ValueError:
            return str(p)

    print("DuraXELL version: 2.0")
    print(f"GS path   : {_rel(DEFAULT_GS)}")
    print(f"Pred path : {_rel(DEFAULT_PRED)}")
    discovered = discover_entities(DEFAULT_GS) or discover_entities_from_config()
    if discovered:
        src = "annotation.conf/.ann" if discover_entities(DEFAULT_GS) else "decision_config.json"
        print(f"Entities  : {len(discovered)} découvertes ({src})")
        for e in discovered:
            print(f"            - {e}")
    else:
        print("Entities  : aucune (lance d'abord `evaluate --gs_dir <corpus>`)")
    print(f"Config    : {_rel(CONFIG_PATH)} ({'OK' if CONFIG_PATH.exists() else 'absent'})")


def cmd_export_csv(args: argparse.Namespace) -> None:
    _export_decision_csv()


def cmd_gui(args: argparse.Namespace) -> None:
    """Lance la GUI native Tkinter (parité avec le dashboard Streamlit).

    Utile pour les environnements sans navigateur (EDS / Citrix / RDP).
    Aucune dépendance externe : uniquement stdlib.
    """
    spec = _il.spec_from_file_location("_gui_app", ROOT / "dashboard" / "gui_app.py")
    mod = _il.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.launch()


# ===========================================================================
# Argparse wiring
# ===========================================================================
def _add_corpus_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--gs_dir", type=str, default=None, help="Gold-standard BRAT dir")
    p.add_argument("--pred_dir", type=str, default=None, help="Predictions BRAT dir")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="DuraXELL CLI — DEMNE pipeline & dashboard parity",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Examples:
  python main.py info
  python main.py metrics --gs_dir <path>
  python main.py tree --gs_dir <path>
  python main.py evaluate                  # full pipeline + CSV export
  python main.py dashboard --preset FRUGAL
  python main.py corpus --path <BRAT-dir>
  python main.py rest-config --export config.json
  python main.py notebook --port 8888
""",
    )
    sub = parser.add_subparsers(dest="command")

    p = sub.add_parser("metrics", help="Run E_*.py metric scripts on the corpus")
    _add_corpus_args(p)
    p.set_defaults(func=cmd_metrics)

    p = sub.add_parser("tree", help="Build decision tree + export CSV summary")
    _add_corpus_args(p)
    p.add_argument("--no-visualize", action="store_true")
    p.set_defaults(func=cmd_tree)

    p = sub.add_parser("evaluate", help="Full pipeline: metrics → tree → CSV")
    _add_corpus_args(p)
    p.add_argument("--no-visualize", action="store_true")
    p.set_defaults(func=cmd_evaluate)

    p = sub.add_parser("dashboard", help="Mirror Streamlit Page 1 — apply preset & show routing")
    p.add_argument("--preset", choices=["FRUGAL", "QUALITY"], default="FRUGAL")
    p.add_argument("--te", type=float)
    p.add_argument("--he", type=float)
    p.add_argument("--r", type=float)
    p.add_argument("--feas", type=float)
    p.set_defaults(func=cmd_dashboard)

    p = sub.add_parser("corpus", help="Mirror Streamlit Page 2 — BRAT corpus stats")
    p.add_argument("--path", type=str, default=None)
    p.set_defaults(func=cmd_corpus)

    p = sub.add_parser("rest-config", help="Mirror Streamlit Page 3 — export/import JSON config")
    p.add_argument("--export", type=str, default=None)
    p.add_argument("--import", dest="import_", type=str, default=None)
    p.set_defaults(func=cmd_rest_config)

    p = sub.add_parser(
        "notebook", help="Mirror Streamlit Page 4 — launch Voila on REST.ipynb + REST API"
    )
    p.add_argument("--port", type=int, default=8888)
    p.add_argument("--notebook-only", action="store_true")
    p.add_argument("--api-only", action="store_true")
    p.set_defaults(func=cmd_notebook)

    p = sub.add_parser("rest", help="Launch demo_rest.py only (legacy)")
    p.set_defaults(func=cmd_rest)

    p = sub.add_parser(
        "export-csv", help="Re-export the decision_summary.csv from decision_config.json"
    )
    p.set_defaults(func=cmd_export_csv)

    p = sub.add_parser("gui", help="GUI Tkinter native (parité Streamlit, sans navigateur)")
    p.set_defaults(func=cmd_gui)

    p = sub.add_parser("info", help="Print environment info")
    p.set_defaults(func=cmd_info)

    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        return
    args.func(args)


if __name__ == "__main__":
    main()
