"""tfidf_coverage — extractability by keyword COVERAGE (TF-IDF).

Intuition: an entity can be extracted by rules even when its observations are not
homogeneous, AS LONG AS a small set of keywords (the top-X TF-IDF words of its
mentions) covers a good share of all its mentions.

Differs from the former E_tfidf (contextual synonymy):
  - unit       : WORDS (tokens) of the mentions, not whole surface forms
  - basis      : the mentions themselves, not the contexts around them
  - clustering : NONE → no sim_threshold, no window
  - score      : F1 = recall (coverage) × precision (anti false positives)

For entity E:
  ranking   = words of E's mentions ranked by TF-IDF weight (E vs the other entities)
  top-X     = the first X words
  recall    = |mentions of E containing ≥1 top word| / |mentions of E|
  precision = |mentions of E hit| / |mentions (all entities) hit|
              → a too generic word (present in other entities) lowers the
                precision = false positives are penalised
  f1        = 2·recall·precision / (recall + precision)
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from demne._brat import load_brat_corpus

_ROOT = Path(__file__).resolve().parents[2]
_RESULTS_DIR = _ROOT / "Results"

_TOKEN_RE = re.compile(r"[^a-zA-Z0-9%]+")


def _tokenize(text: str) -> list[str]:
    return [w for w in _TOKEN_RE.split(text.lower()) if len(w) > 1]


def coverage_f1_by_x(
    entity_mentions: dict[str, list[str]], max_x: int = 15
) -> dict[str, dict[int, float]]:
    """Coverage-F1 per entity for every X in 1..max_x.

    Args:
        entity_mentions: {entity_type: [mention strings]} for ONE corpus.
        max_x: largest top-X to precompute.

    Returns:
        {entity_type: {x: f1}} — empty if fewer than 2 entities (TF-IDF impossible).
    """
    ents = [e for e, m in entity_mentions.items() if m]
    if len(ents) < 2:
        return {}

    pooled = [" ".join(entity_mentions[e]) for e in ents]
    vec = TfidfVectorizer(analyzer="word", ngram_range=(1, 1), min_df=1)
    try:
        matrix = vec.fit_transform(pooled)
    except ValueError:  # vocabulaire vide
        return {}
    feats = np.array(vec.get_feature_names_out())

    out: dict[str, dict[int, float]] = {}
    for i, e in enumerate(ents):
        row = matrix[i].toarray().ravel()
        order = np.argsort(row)[::-1]
        ranked = [feats[j] for j in order if row[j] > 0]  # words present, by TF-IDF weight
        ment_e = [m.lower() for m in entity_mentions[e]]
        per_x: dict[int, float] = {}
        for x in range(1, max_x + 1):
            top = ranked[:x]
            if not top:
                per_x[x] = 0.0
                continue
            pat = re.compile(r"\b(" + "|".join(re.escape(w) for w in top) + r")\b")
            hit_e = sum(1 for m in ment_e if pat.search(m))
            recall = hit_e / len(ment_e)
            hit_all = sum(1 for ee in ents for m in entity_mentions[ee] if pat.search(m.lower()))
            precision = hit_e / hit_all if hit_all else 0.0
            per_x[x] = (
                round(2 * recall * precision / (recall + precision), 4)
                if recall + precision > 0
                else 0.0
            )
        out[e] = per_x
    return out


def compute_coverage_for_all_entities(
    corpus_dir: str | Path, top_x: int | None = None
) -> dict[str, dict]:
    """Corpus API (dashboard/CLI): {etype: {"tfidf_score"}}.

    Reads the mentions of the BRAT corpus and computes the coverage-F1 at the current
    top-X (TFIDF_X of demne_params.json if not given). `tfidf_score` holds that same
    value (key expected by the dashboard/routing).
    """
    from demne.params import load_params

    x = int(top_x if top_x is not None else load_params()["decision_thresholds"].get("TFIDF_X", 5))
    mentions = collect_mentions(corpus_dir)
    by_x = coverage_f1_by_x(mentions, max_x=max(x, 1))
    return {et: {"tfidf_score": by_x.get(et, {}).get(x, 0.0)} for et in mentions}


def collect_mentions(corpus_dir: str | Path) -> dict[str, list[str]]:
    """Read the BRAT .ann files → {entity_type: [mention texts]}.

    BRAT entity line: ``T<id>\\t<TYPE> <start> <end>\\t<text>``.
    """
    ment: dict[str, list[str]] = {}
    for doc in load_brat_corpus([corpus_dir], recursive=True, quiet=True):
        for ann in doc["annotations"]:
            text = ann.text.strip()
            if text:
                ment.setdefault(ann.entity_type, []).append(text)
    return ment


def run(
    gs_dir: str | Path,
    results_dir: str | Path,
    top_x: int | None = None,
    y: float | None = None,
    **_kw: object,
) -> None:
    """CLI/pipeline: write ``Results/tfidf_analysis.csv`` with the coverage-F1.

    Schema compatible with decision_tree_builder (`_read_csv` reads TFIDF_Score).
    `**_kw` swallows former arguments (e.g. sim_threshold) that are now useless.
    """
    import csv

    from demne.params import load_params

    dt = load_params()["decision_thresholds"]
    x = int(top_x if top_x is not None else dt.get("TFIDF_X", 5))
    y = float(y if y is not None else dt.get("TFIDF_Y", 0.70))

    mentions = collect_mentions(gs_dir)
    if not mentions:
        print(f"  Aucune annotation .ann trouvée dans {gs_dir}.")
        return
    by_x = coverage_f1_by_x(mentions, max_x=max(x, 1))

    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    out_csv = results_dir / "tfidf_analysis.csv"
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f, fieldnames=["Entity", "TFIDF_Score", "Routes_To_Rules", "Mentions_Count"]
        )
        w.writeheader()
        for et in sorted(mentions):
            f1 = by_x.get(et, {}).get(x, 0.0)
            w.writerow(
                {
                    "Entity": et,
                    "TFIDF_Score": round(f1, 4),
                    "Routes_To_Rules": f1 >= y,
                    "Mentions_Count": len(mentions[et]),
                }
            )
    print(f"coverage-F1 → {out_csv} (X={x}, Y={y})")


def main() -> None:
    """CLI (main.py pipeline compatibility): write Results/tfidf_analysis.csv."""
    import argparse

    ap = argparse.ArgumentParser(description="Coverage-F1 extractability per entity")
    ap.add_argument("--gs_dir", default=None, help="corpus BRAT gold-standard")
    ap.add_argument("--results_dir", default=None, help="dossier de sortie CSV")
    ap.add_argument("--pred_dir", default=None, help="(ignoré, compat pipeline)")
    a = ap.parse_args()
    gs = Path(a.gs_dir) if a.gs_dir else _ROOT
    results = Path(a.results_dir) if a.results_dir else _RESULTS_DIR
    run(gs, results)


if __name__ == "__main__":
    main()
