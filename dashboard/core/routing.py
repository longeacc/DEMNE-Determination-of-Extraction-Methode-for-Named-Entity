"""Routage simplifié DuraXell — Arbre à 4 nœuds, 3 sorties."""


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

    # Seuils par défaut = optimum du grid search DEMNE (échelle 0-1)
    # {Te_HIGH: 0.1, He_HIGH: 0.85, R_HIGH: 0.25, Feas_NER: 0.2}
    t_te: float = thresholds.get("Te", 0.10)
    t_he: float = thresholds.get("He", 0.85)
    t_r: float = thresholds.get("R", 0.25)
    t_feas: float = thresholds.get("Feas", 0.20)

    # Branche RÈGLES : Te élevée + He élevée + R faible
    if te >= t_te and he >= t_he and r <= t_r:
        return "RÈGLES", f"Te={te:.2f}≥{t_te}, He={he:.2f}≥{t_he}, R={r:.3f}≤{t_r}"

    # Branche TBM : Faisabilité suffisante
    if feas >= t_feas:
        return "TBM", f"Feas={feas:.3f}≥{t_feas}"

    # Branche LLM : fallback
    return "LLM", "Conditions RÈGLES et TBM non satisfaites"
