"""DuraXELL CLI — Central hub for DEMNE.

Provides:
  - Direct extraction:   extract, extract-all, batch
  - Metric pipeline:     metrics, tree, evaluate
  - Dashboard parity:    dashboard, corpus, rest-config, notebook
  - Misc:                rest, serve, info, export-csv

Mirrors the Streamlit pages so the full workflow can be driven from the terminal.
"""

from __future__ import annotations

import argparse
import csv
import glob
import io
import json
import logging
import os
import subprocess
import sys
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# Bootstrap: make `duraxell.*` importable + force UTF-8 stdout on Windows
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
DEFAULT_GS = Path(
    r"D:\1_CLEM\ESIEE SCHOOL\PARCOURS RECHERCHE\Le juste usage des LLM et"
    r" méthode NLP en cancélorlogie\ESMO2025\Breast\RCP\evaluation_set_breast_cancer_GS"
)
DEFAULT_PRED = Path(
    r"D:\1_CLEM\ESIEE SCHOOL\PARCOURS RECHERCHE\Le juste usage des LLM et"
    r" méthode NLP en cancélorlogie\ESMO2025\Breast\RCP\evaluation_set_breast_cancer_pred_rules"
)


def discover_entities(corpus_path: str | Path | None) -> list[str]:
    """Découvre la liste des entités annotées dans un corpus BRAT.

    Stratégie :
      1. Lire `annotation.conf` section [entities] s'il existe.
      2. Sinon, scanner toutes les lignes T* des .ann et collecter les types.
      3. À défaut, retourner [].
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
    """Retourne les entités présentes dans data/decision_config.json (si dispo)."""
    if not CONFIG_PATH.exists():
        return []
    try:
        cfg = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
        return list(cfg.get("entities", {}).keys())
    except Exception:
        return []

PRESETS = {
    "FRUGAL": {"Te": 0.10, "He": 0.85, "R": 0.25, "Feas": 0.50},
    "QUALITY": {"Te": 0.25, "He": 0.55, "R": 0.15, "Feas": 0.70},
}

CONFIG_PATH = ROOT / "data" / "decision_config.json"
RESULTS_DIR = ROOT / "Results"
DECISION_CSV = RESULTS_DIR / "decision_summary.csv"


# ===========================================================================
# Direct extraction commands
# ===========================================================================
def cmd_extract(args: argparse.Namespace) -> None:
    from duraxell.cascade_orchestrator import CascadeOrchestrator

    orch = CascadeOrchestrator()
    print(f"Extraction for entity '{args.entity}':")
    res = orch.extract(args.doc, args.entity)
    print(
        f"  Result: {res.value} | Method: {res.method_used} | "
        f"Confidence: {res.confidence} | Energy: {res.energy_kwh:.6f} kWh"
    )


def cmd_extract_all(args: argparse.Namespace) -> None:
    from duraxell.cascade_orchestrator import CascadeOrchestrator

    orch = CascadeOrchestrator()
    if args.entities:
        entities = [e.strip() for e in args.entities.split(",")]
    else:
        entities = (
            discover_entities(args.corpus_dir)
            or discover_entities_from_config()
        )
    if not entities:
        print("Aucune entité fournie ni découvrable. Utilisez --entities ou --corpus_dir.")
        return
    print(f"Extraction de {len(entities)} entités depuis le document...")
    print("-" * 80)
    total_energy = 0.0
    for ent in entities:
        res = orch.extract(args.doc, ent)
        status = res.value if res.value else "NON TROUVÉ"
        print(
            f"  {ent:<25s} => {status:<20s} | {res.method_used:<12s} | "
            f"conf={res.confidence:.2f} | E={res.energy_kwh:.6f} kWh"
        )
        total_energy += res.energy_kwh
    print("-" * 80)
    print(f"  Énergie totale : {total_energy:.6f} kWh")


def cmd_batch(args: argparse.Namespace) -> None:
    logging.basicConfig(level=logging.WARNING)
    from duraxell.cascade_orchestrator import CascadeOrchestrator

    orch = CascadeOrchestrator()
    if args.entities:
        entities = [e.strip() for e in args.entities.split(",")]
    else:
        # Discover from the input_dir itself (BRAT) → fallback to decision_config.json
        entities = (
            discover_entities(args.input_dir)
            or discover_entities_from_config()
        )
    if not entities:
        print("Aucune entité découverte dans --input_dir (pas d'annotation.conf "
              "ni de .ann) et decision_config.json vide. Utilisez --entities.")
        return
    files = sorted(glob.glob(os.path.join(args.input_dir, "*.txt")))
    if not files:
        print(f"Aucun fichier .txt trouvé dans {args.input_dir}")
        return

    decision_cfg = {}
    if CONFIG_PATH.exists():
        with open(CONFIG_PATH, encoding="utf-8") as f:
            decision_cfg = json.load(f).get("entities", {})

    RESULTS_DIR.mkdir(exist_ok=True)
    out_path = RESULTS_DIR / "batch_extraction_results.csv"

    rows = []
    total = len(files) * len(entities)
    done = 0
    t0 = time.time()
    print(f"Batch : {len(files)} fichiers × {len(entities)} entités = {total} extractions")
    print(f"Sortie CSV : {out_path}")
    print("-" * 70)

    for fpath in files:
        text = open(fpath, encoding="utf-8").read()
        fname = os.path.basename(fpath)
        for ent in entities:
            res = orch.extract(text, ent)
            cfg = decision_cfg.get(ent, {})
            mets = cfg.get("metrics", {})
            rows.append({
                "fichier": fname,
                "entite": ent,
                "valeur": res.value or "",
                "methode_utilisee": res.method_used,
                "confiance": round(res.confidence, 4),
                "energie_kwh": round(res.energy_kwh, 8),
                "Te": mets.get("Te", ""),
                "He": mets.get("He", ""),
                "Freq": round(mets["Freq"], 6) if "Freq" in mets else "",
                "Feas": mets.get("Feas", ""),
                "methode_recommandee": cfg.get("method", ""),
                "justification": cfg.get("justification", ""),
            })
            done += 1
            if done % 50 == 0 or done == total:
                print(f"  [{done}/{total}] {time.time()-t0:.1f}s — {fname} / {ent} => {res.value or '-'}")

    with open(out_path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()), delimiter=",")
        w.writeheader()
        w.writerows(rows)
    found = sum(1 for r in rows if r["valeur"])
    print("-" * 70)
    print(f"Terminé en {time.time()-t0:.1f}s | {found}/{total} trouvés ({100*found/total:.0f}%)")
    print(f"CSV écrit : {out_path}")


# ===========================================================================
# Metric pipeline commands
# ===========================================================================
# Lignes de bruit à filtrer dans les sous-process (eco2ai, pandas, apscheduler)
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
    "Job \"",
    "  File \"",
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


def _filter_noise(line: str) -> bool:
    line = line.rstrip()
    if not line.strip():
        return False
    for k in _NOISE_KEYS:
        if k in line:
            return False
    return True


def _run_script(script: str, gs_dir: str | None, pred_dir: str | None) -> int:
    cmd = [sys.executable, str(ROOT / script)]
    if gs_dir:
        cmd += ["--gs_dir", gs_dir]
    if pred_dir:
        cmd += ["--pred_dir", pred_dir]
    short = Path(script).name
    print(f"\n>>> {short}  --gs_dir {gs_dir}" + (f"  --pred_dir {pred_dir}" if pred_dir else ""))
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUTF8"] = "1"
    env["PYTHONUNBUFFERED"] = "1"
    proc = subprocess.Popen(
        cmd, cwd=str(ROOT), env=env,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        bufsize=1, text=True, encoding="utf-8", errors="replace",
    )
    assert proc.stdout is not None
    for line in proc.stdout:
        if _filter_noise(line):
            print(line.rstrip())
    proc.wait()
    return proc.returncode


def cmd_metrics(args: argparse.Namespace) -> None:
    """Run all E_*.py metric scripts on the corpus."""
    scripts = [
        "src/duraxell/E_templatability.py",
        "src/duraxell/E_homogeneity.py",
        "src/duraxell/E_frequency.py",
        "src/duraxell/E_risk_context.py",
        "src/duraxell/E_feasibility_NER.py",
    ]
    gs = args.gs_dir or str(DEFAULT_GS)
    pred = args.pred_dir or str(DEFAULT_PRED)
    print(f"Pipeline métriques sur :\n  GS   = {gs}\n  Pred = {pred}")
    for s in scripts:
        _run_script(s, gs, pred)


def cmd_tree(args: argparse.Namespace) -> None:
    """Affiche le schéma statique de l'arbre de décision DEMNE.

    Ne lance plus le pipeline ni l'export CSV — utilise `evaluate` pour ça.
    """
    del args  # signature requise par argparse, contenu non utilisé
    _open_decision_tree_image()


def _run_tree_pipeline(args: argparse.Namespace) -> None:
    """Pipeline de construction de l'arbre (utilisé par `evaluate`)."""
    gs = args.gs_dir or str(DEFAULT_GS)
    pred = args.pred_dir or str(DEFAULT_PRED)
    _run_script("src/duraxell/E_creation_arbre_decision.py", gs, pred)
    if not args.no_visualize:
        print("\nVisualisation de l'arbre...")
        subprocess.run(
            [sys.executable, str(ROOT / "src/duraxell/visualize_decision_tree.py")],
            cwd=str(ROOT),
        )
    _export_decision_csv()


def _open_decision_tree_image() -> None:
    """Ouvre Results/figures/Graph_decision_bisss.png dans le visionneuse par défaut.

    Fallback : Graph_decision_biss.png, decision_tree.png, decision_tree_visualization.png.
    """
    candidates = [
        ROOT / "Results" / "figures" / "Graph_decision_bisss.png",
        ROOT / "Results" / "figures" / "Graph_decision_biss.png",
        ROOT / "Results" / "figures" / "decision_tree.png",
        ROOT / "Results" / "figures" / "decision_tree_visualization.png",
    ]
    img = next((p for p in candidates if p.exists()), None)
    if not img:
        print("⚠️  Aucune image d'arbre trouvée dans Results/figures/.")
        return
    print(f"\n🖼  Ouverture de l'arbre : {img}")
    try:
        if sys.platform == "win32":
            os.startfile(str(img))  # type: ignore[attr-defined]
        elif sys.platform == "darwin":
            subprocess.Popen(["open", str(img)])
        else:
            subprocess.Popen(["xdg-open", str(img)])
    except Exception as e:
        print(f"   Impossible d'ouvrir le visualiseur : {e}")


def cmd_evaluate(args: argparse.Namespace) -> None:
    """Full pipeline: metrics → tree (pipeline) → CSV export."""
    cmd_metrics(args)
    _run_tree_pipeline(args)


def _export_decision_csv() -> None:
    """Emit Results/decision_summary.csv with tab-separated rows:
    Entity \t Te \t He \t R \t Freq \t Feas \t Method \t Justification
    """
    if not CONFIG_PATH.exists():
        print(f"Pas de decision_config.json trouvé à {CONFIG_PATH}")
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
        rows.append([
            ent,
            f"{te:.4f}",
            f"{he:.4f}",
            f"{m.get('R', 0.0):.4f}",
            f"{m.get('Freq', 0.0):.4f}",
            f"{m.get('Feas', 0.0):.4f}",
            d.get("method", ""),
            d.get("justification", ""),
        ])

    with open(DECISION_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, delimiter="\t")
        for r in rows:
            w.writerow(r)
    print(f"\n=== CSV synthèse écrit : {DECISION_CSV} ===")
    for r in rows:
        print("\t".join(r))


# ===========================================================================
# Dashboard parity commands (mirror Streamlit pages)
# ===========================================================================
def cmd_dashboard(args: argparse.Namespace) -> None:
    """Mirror Page 1 — Dashboard Métriques.

    Apply a threshold preset (FRUGAL / QUALITY / custom) and re-run the
    decision tree on the current metrics, printing the routing table.
    """
    if not CONFIG_PATH.exists():
        print("⚠️  Pas de decision_config.json — lance d'abord 'metrics' puis 'tree'.")
        return

    preset_name = (args.preset or "FRUGAL").upper()
    thresholds = PRESETS.get(preset_name, PRESETS["FRUGAL"]).copy()
    for k in ("Te", "He", "R", "Feas"):
        v = getattr(args, k.lower(), None)
        if v is not None:
            thresholds[k] = v

    print(f"Preset : {preset_name}   Seuils : {thresholds}")

    with open(CONFIG_PATH, encoding="utf-8") as f:
        cfg = json.load(f)

    # Bypass duraxell/__init__.py (which pulls pandas via cascade_orchestrator)
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "_e_tree", SRC / "duraxell" / "E_creation_arbre_decision.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    builder = mod.DecisionTreeBuilder(CONFIG_PATH)
    builder.THRESHOLDS = {
        "TE_HIGH": thresholds["Te"],
        "HE_HIGH": thresholds["He"],
        "R_HIGH": thresholds["R"],
        "FEAS_NER": thresholds["Feas"],
    }

    print(f"\n{'Entity':<25} | {'Te':>6} | {'He':>6} | {'R':>6} | {'Feas':>6} | {'Method':<8} | Justification")
    print("-" * 120)
    for ent, data in cfg.get("entities", {}).items():
        m = data.get("metrics", {})
        decision = builder.analyze_entity(ent, m)
        te = m.get("Te", 0)
        he = m.get("He", 0)
        if te > 1.0:
            te /= 100.0
        if he > 1.0:
            he /= 100.0
        print(
            f"{ent:<25} | {te:6.3f} | {he:6.3f} | {m.get('R',0):6.3f} | "
            f"{m.get('Feas',0):6.3f} | {decision['method']:<8} | {decision['justification']}"
        )


def cmd_corpus(args: argparse.Namespace) -> None:
    """Mirror Page 2 — Analyse Corpus BRAT."""
    # Stub out streamlit before importing brat_parser (which decorates with @st.cache_data)
    if "streamlit" not in sys.modules:
        st_stub = type(sys)("streamlit")
        st_stub.cache_data = lambda *a, **k: (lambda fn: fn) if not (a and callable(a[0])) else a[0]
        st_stub.cache = st_stub.cache_data
        sys.modules["streamlit"] = st_stub
    try:
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "_brat", ROOT / "dashboard" / "core" / "brat_parser.py"
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        BratCorpusParser = mod.BratCorpusParser
    except Exception as e:
        print(f"Impossible de charger BratCorpusParser : {e}")
        return

    path = args.path or str(DEFAULT_GS)
    print(f"Analyse du corpus BRAT : {path}")
    parser = BratCorpusParser()
    docs = parser.parse_directory(path)
    stats = parser.get_entity_statistics(docs)
    print(f"\n✅ {len(docs)} documents chargés.\n")
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
            print("Pas de decision_config.json à exporter.")
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
        notebook = ROOT / "src" / "duraxell" / "REST_interface" / "REST.ipynb"
        if not notebook.exists():
            print(f"Notebook introuvable : {notebook}")
        else:
            print(f"Démarrage Voila sur {notebook} (port {args.port})...")
            voila_cmd = [
                sys.executable, "-m", "voila", str(notebook),
                "--no-browser", f"--port={args.port}", "--Voila.ip=127.0.0.1",
            ]
            procs.append(subprocess.Popen(voila_cmd, cwd=str(ROOT)))
            print(f"  → http://127.0.0.1:{args.port}")

    if not args.notebook_only:
        api = ROOT / "src" / "duraxell" / "REST_interface" / "demo_rest.py"
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
        [sys.executable, str(ROOT / "src/duraxell/REST_interface/demo_rest.py")],
        cwd=str(ROOT / "src/duraxell/REST_interface"),
    )


def cmd_info(args: argparse.Namespace) -> None:
    print("DuraXELL version: 2.0")
    print(f"Root      : {ROOT}")
    print(f"GS path   : {DEFAULT_GS}")
    print(f"Pred path : {DEFAULT_PRED}")
    discovered = discover_entities(DEFAULT_GS) or discover_entities_from_config()
    if discovered:
        src = "annotation.conf/.ann" if discover_entities(DEFAULT_GS) else "decision_config.json"
        print(f"Entities  : {len(discovered)} découvertes ({src})")
        for e in discovered:
            print(f"            - {e}")
    else:
        print("Entities  : aucune (lance d'abord `evaluate --gs_dir <corpus>`)")
    print(f"Config    : {CONFIG_PATH} ({'OK' if CONFIG_PATH.exists() else 'absent'})")


def cmd_export_csv(args: argparse.Namespace) -> None:
    _export_decision_csv()


def cmd_gui(args: argparse.Namespace) -> None:
    """Lance la GUI native Tkinter (parité avec le dashboard Streamlit).

    Utile pour les environnements sans navigateur (EDS / Citrix / RDP).
    Aucune dépendance externe : uniquement stdlib.
    """
    import importlib.util as _il
    spec = _il.spec_from_file_location(
        "_gui_app", ROOT / "dashboard" / "gui_app.py"
    )
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
  python main.py tree --gs_dir <path> --pred_dir <path>
  python main.py evaluate                  # full pipeline + CSV export
  python main.py dashboard --preset FRUGAL
  python main.py corpus --path <BRAT-dir>
  python main.py rest-config --export config.json
  python main.py notebook --port 8888
""",
    )
    sub = parser.add_subparsers(dest="command")

    p = sub.add_parser("extract", help="Extract single entity from text")
    p.add_argument("--doc", required=True)
    p.add_argument("--entity", required=True)
    p.set_defaults(func=cmd_extract)

    p = sub.add_parser("extract-all", help="Extract all entities from one document")
    p.add_argument("--doc", required=True)
    p.add_argument("--entities", default=None,
                   help="Liste séparée par virgule (sinon découverte automatique)")
    p.add_argument("--corpus_dir", default=None,
                   help="Dossier BRAT pour découvrir les entités à extraire")
    p.set_defaults(func=cmd_extract_all)

    p = sub.add_parser("batch", help="Extract entities from all .txt in a directory")
    p.add_argument("--input_dir", required=True)
    p.add_argument("--entities", default=None)
    p.set_defaults(func=cmd_batch)

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

    p = sub.add_parser("notebook", help="Mirror Streamlit Page 4 — launch Voila on REST.ipynb + REST API")
    p.add_argument("--port", type=int, default=8888)
    p.add_argument("--notebook-only", action="store_true")
    p.add_argument("--api-only", action="store_true")
    p.set_defaults(func=cmd_notebook)

    p = sub.add_parser("rest", help="Launch demo_rest.py only (legacy)")
    p.set_defaults(func=cmd_rest)

    p = sub.add_parser("export-csv", help="Re-export the decision_summary.csv from decision_config.json")
    p.set_defaults(func=cmd_export_csv)

    p = sub.add_parser("gui", help="GUI Tkinter native (parité Streamlit, sans navigateur)")
    p.set_defaults(func=cmd_gui)

    p = sub.add_parser("serve", help="(TODO) Serve DuraXELL via HTTP")
    p.set_defaults(func=lambda a: print("serve : not yet implemented"))

    p = sub.add_parser("info", help="Print environment info")
    p.set_defaults(func=cmd_info)

    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        return
    args.func(args)


if __name__ == "__main__":
    main()
