"""Routage DuraXell — graphe exact de la figure de référence DEMNE.

Nœud R− PARTAGÉ entre la branche Te/He et la branche TF-IDF.
TF-IDF 'Oui' ne route PAS directement RULES : il passe par R− d'abord.
"""

import importlib.util as _il
from pathlib import Path

# Seuils par défaut = data/demne_params.json (source unique partagée CLI/scorers/dashboard)
_pspec = _il.spec_from_file_location(
    "demne_params", Path(__file__).resolve().parents[2] / "src" / "demne" / "params.py"
)
_pmod = _il.module_from_spec(_pspec)
_pspec.loader.exec_module(_pmod)
_DT = _pmod.load_params()["decision_thresholds"]


def compute_routing(metrics: dict[str, float], thresholds: dict[str, float]) -> tuple[str, str]:
    """Arbre de décision DEMNE — graphe à nœud R− partagé.

    Graphe (figure de référence) :
      Te++ ET He++ → R− → R≤R_HIGH : RULES / R>R_HIGH : Feas++
      sinon        → TF-IDF ?
                       score≥Y → R− (même nœud !) → RULES / Feas++
                       score<Y ou absent → Feas++
      Feas++ → Feas≥FEAS_NER : TBM / sinon : LLM

    Args:
        metrics: {Te, He, R, Feas, tfidf_score (optionnel)} sur échelle [0-1].
        thresholds: {Te, He, R, Feas, Y (optionnel)} sur échelle [0-1].

    Returns:
        Tuple (méthode, justification).
    """
    te: float = metrics.get("Te", metrics.get("te", 0.0))
    he: float = metrics.get("He", metrics.get("he", 0.0))
    r: float = metrics.get("R", metrics.get("r", 0.0))
    feas: float = metrics.get("Feas", metrics.get("feas", 0.0))
    tfidf = metrics.get("tfidf_score")

    t_te: float = thresholds.get("Te", _DT["TE_HIGH"])
    t_he: float = thresholds.get("He", _DT["HE_HIGH"])
    t_r: float = thresholds.get("R", _DT["R_HIGH"])
    t_feas: float = thresholds.get("Feas", _DT["FEAS_NER"])
    t_y: float = thresholds.get("Y", _DT.get("TFIDF_Y", 0.70))

    def _noeud_feas() -> tuple[str, str]:
        if feas >= t_feas:
            return "TBM", f"Feas={feas:.3f}≥{t_feas}"
        return "LLM", "Conditions RÈGLES et TBM non satisfaites"

    def _noeud_r(label: str) -> tuple[str, str]:
        if r <= t_r:
            return "RÈGLES", f"{label} — R={r:.3f}≤{t_r}"
        return _noeud_feas()

    # Branche Te++ AND He++
    if te >= t_te and he >= t_he:
        return _noeud_r(f"Te={te:.2f}≥{t_te}, He={he:.2f}≥{t_he}")

    # Branche TF-IDF (synonymie conceptuelle)
    if tfidf is not None and tfidf >= t_y:
        return _noeud_r(f"TFIDF={tfidf:.3f}≥Y={t_y}")

    # Convergence Feas++
    return _noeud_feas()
