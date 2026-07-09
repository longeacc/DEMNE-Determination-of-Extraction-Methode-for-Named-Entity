"""E_tfidf — TFIDF_Extractability: score computation and DEMNE routing.

Reads a BRAT corpus (.ann/.txt), aggregates mentions and contexts per entity type,
computes the contextual synonymy TF-IDF score, and saves Results/tfidf_analysis.csv.

CLI:
    python src/demne/E_tfidf.py --gs_dir <path>
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

ROOT = Path(__file__).resolve().parents[3]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from demne._table import print_table
from demne.params import load_params

PARAMS = load_params()
_dt = PARAMS["decision_thresholds"]
RESULTS_DIR = ROOT / "Results"
DEFAULT_GS = ROOT / "data" / "ESMO2025" / "Breast" / "RCP" / "evaluation_set_breast_cancer_GS"


# ---------------------------------------------------------------------------
# Internal utility
# ---------------------------------------------------------------------------
def _bisect_left(arr: list, x: int) -> int:
    lo, hi = 0, len(arr)
    while lo < hi:
        mid = (lo + hi) // 2
        if arr[mid] < x:
            lo = mid + 1
        else:
            hi = mid
    return lo


# ---------------------------------------------------------------------------
# Lecture BRAT
# ---------------------------------------------------------------------------
def build_corpus_contexts(brat_ann_path, brat_txt_path, window_tokens: int = 30):
    """Lit BRAT .ann + .txt -> {surface_form_lower: [context_window_string]}."""
    txt = Path(brat_txt_path).read_text(encoding="utf-8")
    tokens = [(m.start(), m.group()) for m in re.finditer(r"\S+", txt)]
    starts = [t[0] for t in tokens]

    contexts: dict[str, list[str]] = {}
    for line in Path(brat_ann_path).read_text(encoding="utf-8").splitlines():
        if not line.startswith("T"):
            continue
        parts = line.split("\t")
        if len(parts) < 3:
            continue
        seg = parts[1].split()
        if len(seg) < 3:
            continue
        try:
            span_start = int(seg[1])
            span_end = int(seg[-1])
        except ValueError:
            continue
        surface = parts[2].strip().lower()
        if not surface:
            continue
        lo = _bisect_left(starts, span_start)
        hi = _bisect_left(starts, span_end)
        a = max(0, lo - window_tokens)
        b = min(len(tokens), hi + window_tokens)
        window = " ".join(tok for _, tok in tokens[a:b])
        contexts.setdefault(surface, []).append(window)
    return contexts


def collect_entity_data(corpus_dir: Path | str, window_tokens: int = 30) -> dict[str, dict]:
    """Read BRAT corpus -> {entity_type: {mentions, corpus_contexts}}.

    Returns {etype: {"mentions": [...], "corpus_contexts": {form: [ctx...]}}}
    """
    corpus_dir = Path(corpus_dir)
    entity_data: dict[str, dict] = {}

    def _new_entry():
        return {"mentions": [], "corpus_contexts": defaultdict(list)}

    for ann_path in sorted(corpus_dir.glob("**/*.ann")):
        txt_path = ann_path.with_suffix(".txt")
        if not txt_path.exists():
            continue
        try:
            txt = txt_path.read_text(encoding="utf-8")
            ann_text = ann_path.read_text(encoding="utf-8")
        except Exception:
            continue

        token_matches = [(m.start(), m.group()) for m in re.finditer(r"\S+", txt)]
        tok_starts = [t[0] for t in token_matches]

        for line in ann_text.splitlines():
            if not line.startswith("T"):
                continue
            parts = line.split("\t")
            if len(parts) < 3:
                continue
            seg = parts[1].split()
            if len(seg) < 3:
                continue
            etype = seg[0]
            try:
                span_start = int(seg[1])
                span_end = int(seg[-1])
            except ValueError:
                continue
            surface = parts[2].strip().lower()
            if not surface:
                continue

            lo_tok = _bisect_left(tok_starts, span_start)
            hi_tok = _bisect_left(tok_starts, span_end)
            a = max(0, lo_tok - window_tokens)
            b = min(len(token_matches), hi_tok + window_tokens)
            context = " ".join(tok for _, tok in token_matches[a:b])

            if etype not in entity_data:
                entity_data[etype] = _new_entry()
            entity_data[etype]["mentions"].append(surface)
            entity_data[etype]["corpus_contexts"][surface].append(context)

    return entity_data


# ---------------------------------------------------------------------------
# Score TFIDF
# ---------------------------------------------------------------------------
def compute_tfidf_extractability(
    entity_name, mentions, corpus_contexts, X=5, sim_threshold=0.50, Y=None  # noqa: N803
):
    """Contextual TF-IDF extractability score for an entity."""
    if Y is None:
        Y = _dt.get("TFIDF_Y", 0.70)  # noqa: N806
    mentions_lower = [m.lower().strip() for m in mentions]
    unique_forms = set(mentions_lower)

    if len(mentions) < 3:
        return {"tfidf_score": 0.0, "routes_to_rules": False, "clusters": [], "top_X_recalls": []}
    if len(unique_forms) == 1:
        only = sorted(unique_forms)[0]
        return {
            "tfidf_score": 1.0,
            "routes_to_rules": 1.0 >= Y,
            "clusters": [{"forms": [only], "total_count": len(mentions)}],
            "top_X_recalls": [1.0],
        }

    forms_list = sorted(unique_forms)

    docs = []
    for f in forms_list:
        raw = " ".join(corpus_contexts.get(f, [""]))
        strip = re.compile(r"(?i)\b" + re.escape(f) + r"\b")
        docs.append(strip.sub(" ", raw))

    vectorizer = TfidfVectorizer(
        analyzer="word", ngram_range=(1, 2), max_features=500, min_df=1, sublinear_tf=True
    )
    try:
        m_tfidf = vectorizer.fit_transform(docs)
    except ValueError:
        return {"tfidf_score": 0.0, "routes_to_rules": False, "clusters": [], "top_X_recalls": []}

    sim_matrix = cosine_similarity(m_tfidf)
    form_counts = Counter(mentions_lower)
    idx = {f: i for i, f in enumerate(forms_list)}
    clusters = []
    assigned: set[str] = set()
    for form_i in forms_list:
        if form_i in assigned:
            continue
        cluster = [form_i]
        assigned.add(form_i)
        for form_j in forms_list:
            if form_j in assigned:
                continue
            avg_sim = np.mean([sim_matrix[idx[form_j], idx[f]] for f in cluster])
            if avg_sim >= sim_threshold:
                cluster.append(form_j)
                assigned.add(form_j)
        clusters.append({"forms": cluster, "total_count": sum(form_counts[f] for f in cluster)})
    clusters.sort(key=lambda c: c["total_count"], reverse=True)

    total_mentions = len(mentions)
    top_x_recalls = []
    for cluster in clusters[: min(X, len(clusters))]:
        pattern = re.compile(r"(?i)\b(" + "|".join(re.escape(f) for f in cluster["forms"]) + r")\b")
        matched = sum(1 for m in mentions_lower if pattern.search(m))
        top_x_recalls.append(matched / total_mentions)
    tfidf_score = float(np.mean(top_x_recalls)) if top_x_recalls else 0.0

    return {
        "tfidf_score": round(tfidf_score, 4),
        "routes_to_rules": tfidf_score >= Y,
        "clusters": clusters,
        "top_X_recalls": [round(r, 4) for r in top_x_recalls],
    }


# ---------------------------------------------------------------------------
# Routing DEMNE
# ---------------------------------------------------------------------------
def updated_demne_routing(metrics, thresholds):
    """DEMNE decision graph with TFIDF_Extractability.

    Te >= Te_HIGH AND He >= He_HIGH  -> R- node
    otherwise                        -> TF-IDF node
      TF-IDF present AND >= Y        -> R- node
      otherwise                      -> Feas node

    R- node: R <= R_HIGH  -> RULES  /  otherwise -> Feas node
    Feas node: Feas >= Feas_NER -> TBM / otherwise -> LLM
    """
    r = metrics["R"]
    feas = metrics["Feas"]

    def _noeud_feas():
        return "TBM" if feas >= thresholds["Feas_NER"] else "LLM"

    def _noeud_r():
        return "RULES" if r <= thresholds["R_HIGH"] else _noeud_feas()

    if metrics["Te"] >= thresholds["Te_HIGH"] and metrics["He"] >= thresholds["He_HIGH"]:
        return _noeud_r()

    tfidf = metrics.get("tfidf_score")
    if tfidf is not None and tfidf >= thresholds["Y"]:
        return _noeud_r()

    return _noeud_feas()


# ---------------------------------------------------------------------------
# API corpus
# ---------------------------------------------------------------------------
def compute_tfidf_for_all_entities(
    corpus_dir: Path | str,
    top_x: int | None = None,
    sim_threshold: float | None = None,
    window_tokens: int | None = None,
) -> dict[str, dict]:
    """Compute tfidf_extractability for all entities in a BRAT corpus."""
    x = top_x if top_x is not None else _dt.get("TFIDF_X", 10)
    sim_threshold = sim_threshold if sim_threshold is not None else _dt.get("TFIDF_SIM", 0.50)
    y = _dt.get("TFIDF_Y", 0.70)
    window_tokens = (
        window_tokens if window_tokens is not None else _dt.get("TFIDF_WINDOW_TOKENS", 30)
    )
    entity_data = collect_entity_data(corpus_dir, window_tokens=window_tokens)
    return {
        etype: compute_tfidf_extractability(
            etype,
            data["mentions"],
            dict(data["corpus_contexts"]),
            X=x,
            sim_threshold=sim_threshold,
            Y=y,
        )
        for etype, data in entity_data.items()
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def run(
    gs_dir: Path, results_dir: Path, top_x: int, sim_threshold: float, y: float | None = None
) -> None:
    """Compute TFIDF per entity and save Results/tfidf_analysis.csv."""
    if y is None:
        y = _dt.get("TFIDF_Y", 0.70)
    print(f"TFIDF_Extractability — corpus : {gs_dir}")

    entity_data = collect_entity_data(gs_dir)
    if not entity_data:
        print("  No .ann/.txt annotations found.")
        return

    results_dir.mkdir(parents=True, exist_ok=True)
    out_csv = results_dir / "tfidf_analysis.csv"

    rows = []
    table_rows = []
    for etype in sorted(entity_data.keys()):
        data = entity_data[etype]
        res = compute_tfidf_extractability(
            etype,
            data["mentions"],
            dict(data["corpus_contexts"]),
            X=top_x,
            sim_threshold=sim_threshold,
            Y=y,
        )
        n_mentions = len(data["mentions"])
        rows.append(
            {
                "Entity": etype,
                "TFIDF_Score": res["tfidf_score"],
                "Clusters_Count": len(res["clusters"]),
                "Top_X_Recalls": str(res["top_X_recalls"]),
                "Routes_To_Rules": res["routes_to_rules"],
                "Mentions_Count": n_mentions,
            }
        )
        table_rows.append(
            [
                etype,
                f"{res['tfidf_score']:.4f}",
                str(len(res["clusters"])),
                str(n_mentions),
                str(res["routes_to_rules"]),
            ]
        )
    print_table(
        ["Entity", "TFIDF", "Clusters", "Mentions", "->Rules"],
        table_rows,
        [45, 7, 9, 9, 8],
    )

    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "Entity",
                "TFIDF_Score",
                "Clusters_Count",
                "Top_X_Recalls",
                "Routes_To_Rules",
                "Mentions_Count",
            ],
        )
        w.writeheader()
        w.writerows(rows)

    try:
        rel = str(out_csv.relative_to(ROOT))
    except ValueError:
        rel = str(out_csv)
    print(f"\n=== tfidf_analysis.csv written: {rel} ===")


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute TFIDF_Extractability per entity")
    parser.add_argument("--gs_dir", type=str, default=None, help="BRAT gold-standard corpus")
    parser.add_argument("--results_dir", type=str, default=None, help="CSV output directory")
    parser.add_argument(
        "--pred_dir", type=str, default=None, help="(ignored, pipeline compatibility)"
    )
    args = parser.parse_args()

    gs_dir = Path(args.gs_dir) if args.gs_dir else DEFAULT_GS
    results_dir = Path(args.results_dir) if args.results_dir else RESULTS_DIR
    top_x = _dt.get("TFIDF_X", 10)
    sim_threshold = _dt.get("TFIDF_SIM", 0.50)
    y = _dt.get("TFIDF_Y", 0.70)
    run(gs_dir, results_dir, top_x, sim_threshold, y=y)


if __name__ == "__main__":
    main()
