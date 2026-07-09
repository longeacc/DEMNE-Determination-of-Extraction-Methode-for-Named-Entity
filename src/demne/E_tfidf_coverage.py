"""E_tfidf_coverage — Extractabilité par COUVERTURE de mots-clés (TF-IDF).

Intuition : une entité peut être extraite par règles même si ses observations ne
sont pas homogènes, DÈS LORS qu'un petit ensemble de mots-clés (les top-X mots
TF-IDF de ses mentions) couvre une bonne partie de toutes ses mentions.

Diffère de E_tfidf (synonymie contextuelle) :
  - unité   : MOTS (tokens) des mentions, pas les formes de surface entières
  - base    : les mentions elles-mêmes, pas les contextes autour
  - clustering : AUCUN → pas de sim_threshold, pas de fenêtre
  - score   : F1 = recall (couverture) × précision (anti faux-positifs)

Pour l'entité E :
  ranking   = mots des mentions de E classés par poids TF-IDF (E vs autres entités)
  top-X     = les X premiers mots
  recall    = |mentions de E contenant ≥1 top-mot| / |mentions de E|
  precision = |mentions de E touchées| / |mentions (toutes entités) touchées|
              → un mot trop générique (présent chez d'autres entités) fait chuter
                la précision = faux positifs pénalisés
  f1        = 2·recall·precision / (recall + precision)
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

# Chargé par file-spec (dashboard) → garantir que `src/` est sur le path pour
# `from demne.E_tfidf import …` / `from demne.params import …`.
_SRC = Path(__file__).resolve().parents[1]
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))
_ROOT = _SRC.parent
_RESULTS_DIR = _ROOT / "Results"

_TOKEN_RE = re.compile(r"[^a-zA-Z0-9%]+")


def _tokenize(text: str) -> list[str]:
    return [w for w in _TOKEN_RE.split(text.lower()) if len(w) > 1]


def coverage_f1_by_x(
    entity_mentions: dict[str, list[str]], max_x: int = 15
) -> dict[str, dict[int, float]]:
    """Coverage-F1 par entité pour chaque X de 1..max_x.

    Args:
        entity_mentions: {type_entité: [chaînes de mention]} pour UN corpus.
        max_x: plus grand top-X à précalculer.

    Returns:
        {type_entité: {x: f1}} — vide si < 2 entités (TF-IDF impossible).
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
        ranked = [feats[j] for j in order if row[j] > 0]  # mots présents, par poids TF-IDF
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


def compute_coverage_for_all_entities(corpus_dir, top_x: int | None = None) -> dict[str, dict]:
    """API corpus (dashboard/CLI) : {etype: {"tfidf_score"}}.

    Lit les mentions du corpus BRAT, calcule la coverage-F1 au top-X courant
    (TFIDF_X de demne_params.json si non fourni). `tfidf_score` = même valeur
    (compat clé attendue par le dashboard/routing).
    """
    from demne.params import load_params

    x = int(top_x if top_x is not None else load_params()["decision_thresholds"].get("TFIDF_X", 5))
    mentions = collect_mentions(corpus_dir)
    by_x = coverage_f1_by_x(mentions, max_x=max(x, 1))
    return {et: {"tfidf_score": by_x.get(et, {}).get(x, 0.0)} for et in mentions}


def collect_mentions(corpus_dir) -> dict[str, list[str]]:
    """Lit les fichiers BRAT .ann → {type_entité: [textes de mention]}.

    Ligne d'entité BRAT : ``T<id>\\t<TYPE> <start> <end>\\t<texte>``.
    """
    ment: dict[str, list[str]] = {}
    for ann in Path(corpus_dir).rglob("*.ann"):
        for line in ann.read_text(encoding="utf-8").splitlines():
            if not line.startswith("T"):
                continue
            parts = line.split("\t")
            if len(parts) < 3:
                continue
            etype = parts[1].split(" ", 1)[0]
            text = parts[2].strip()
            if text:
                ment.setdefault(etype, []).append(text)
    return ment


def run(gs_dir, results_dir, top_x=None, y=None, **_kw) -> None:
    """CLI/pipeline : écrit ``Results/tfidf_analysis.csv`` avec la coverage-F1.

    Schéma compatible avec E_creation_arbre_decision (`_read_csv` lit TFIDF_Score).
    `**_kw` absorbe d'anciens arguments (ex. sim_threshold) désormais inutiles.
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
    """CLI (compat pipeline main.py) : écrit Results/tfidf_analysis.csv."""
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
