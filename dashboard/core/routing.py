"""DuraXell routing — exact graph matching the DEMNE reference figure.

The R− node is SHARED between the Te/He branch and the TF-IDF branch.
TF-IDF 'Yes' does NOT route directly to RULES: it first passes through R−.
"""

import importlib.util as _il
from pathlib import Path

# Default thresholds = data/demne_params.json (single source shared by CLI/scorers/dashboard)
_pspec = _il.spec_from_file_location(
    "demne_params", Path(__file__).resolve().parents[2] / "src" / "demne" / "params.py"
)
_pmod = _il.module_from_spec(_pspec)
_pspec.loader.exec_module(_pmod)
_DT = _pmod.load_params()["decision_thresholds"]


def compute_routing(metrics: dict[str, float], thresholds: dict[str, float]) -> tuple[str, str]:
    """DEMNE decision graph — shared R− node.

    Graph (reference figure):
      Te++ AND He++ → R− → R≤R_HIGH: RULES / R>R_HIGH: Feas++
      otherwise     → TF-IDF ?
                         score≥Y → R− (same node!) → RULES / Feas++
                         score<Y or absent → Feas++
      Feas++ → Feas≥FEAS_NER: TBM / otherwise: LLM

    Args:
        metrics: {Te, He, R, Feas, tfidf_score (optional)} on scale [0-1].
        thresholds: {Te, He, R, Feas, Y (optional)} on scale [0-1].

    Returns:
        Tuple (method, justification).
    """
    te: float = metrics.get("Te", metrics.get("te", 0.0))
    he: float = metrics.get("He", metrics.get("he", 0.0))
    r: float = metrics.get("R", metrics.get("r", 0.0))
    feas: float = metrics.get("Feas", metrics.get("feas", 0.0))

    t_te: float = thresholds.get("Te", _DT["TE_HIGH"])
    t_he: float = thresholds.get("He", _DT["HE_HIGH"])
    t_r: float = thresholds.get("R", _DT["R_HIGH"])
    t_feas: float = thresholds.get("Feas", _DT["FEAS_NER"])
    t_y: float = thresholds.get("Y", _DT.get("TFIDF_Y", 0.70))
    use_f1: bool = bool(_DT.get("TFIDF_USE_F1", True))

    # Routing metric: f1_score preferred when TFIDF_USE_F1=True, else recall
    if use_f1 and metrics.get("f1_score") is not None:
        tfidf_routing = metrics.get("f1_score")
        tfidf_label = "F1"
    else:
        tfidf_routing = metrics.get("tfidf_score")
        tfidf_label = "recall"

    def _noeud_feas() -> tuple[str, str]:
        if feas >= t_feas:
            return "TBM", f"Feas={feas:.3f}≥{t_feas}"
        return "LLM", "RULES and TBM conditions not met"

    def _noeud_r(label: str) -> tuple[str, str]:
        if r <= t_r:
            return "RULES", f"{label} — R={r:.3f}≤{t_r}"
        return _noeud_feas()

    # Te++ AND He++ branch
    if te >= t_te and he >= t_he:
        return _noeud_r(f"Te={te:.2f}≥{t_te}, He={he:.2f}≥{t_he}")

    # TF-IDF branch (conceptual synonymy) — He → TFIDF → R → RULES/Feas
    if tfidf_routing is not None and tfidf_routing >= t_y:
        return _noeud_r(f"TFIDF {tfidf_label}={tfidf_routing:.3f}≥Y={t_y}")

    # Feas++ convergence point
    return _noeud_feas()
