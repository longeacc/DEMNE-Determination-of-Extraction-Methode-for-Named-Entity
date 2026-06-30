"""Routage simplifié DuraXell — Arbre à 4 nœuds, 3 sorties."""

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
    """Arbre de décision simplifié : Te++ → He++ → R− → RÈGLES | Feas++ → TBM | LLM.

    Args:
        metrics: Métriques de l'entité (Te, He, R, Feas sur échelle [0-1]).
        thresholds: Seuils de routage (Te, He, R, Feas sur échelle [0-1]).

    Returns:
        Tuple (méthode, justification).
    """
    te: float = metrics.get("Te", metrics.get("te", 0.0))
    he: float = metrics.get("He", metrics.get("he", 0.0))
    r: float = metrics.get("R", metrics.get("r", 0.0))
    feas: float = metrics.get("Feas", metrics.get("feas", 0.0))

    # Seuils par défaut = decision_thresholds de data/demne_params.json (échelle 0-1)
    t_te: float = thresholds.get("Te", _DT["TE_HIGH"])
    t_he: float = thresholds.get("He", _DT["HE_HIGH"])
    t_r: float = thresholds.get("R", _DT["R_HIGH"])
    t_feas: float = thresholds.get("Feas", _DT["FEAS_NER"])

    # Branche RÈGLES : Te élevée + He élevée + R faible
    if te >= t_te and he >= t_he and r <= t_r:
        return "RÈGLES", f"Te={te:.2f}≥{t_te}, He={he:.2f}≥{t_he}, R={r:.3f}≤{t_r}"

    # Branche TBM : Faisabilité suffisante
    if feas >= t_feas:
        return "TBM", f"Feas={feas:.3f}≥{t_feas}"

    # Branche LLM : fallback
    return "LLM", "Conditions RÈGLES et TBM non satisfaites"
