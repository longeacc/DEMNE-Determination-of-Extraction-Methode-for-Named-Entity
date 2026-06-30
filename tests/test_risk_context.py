from demne.E_risk_context import RiskContextScorer


def test_negation_detection():
    scorer = RiskContextScorer()

    # Patterns effectivement détectés (voir NEGATION_PATTERNS dans E_risk_context.py)
    # "non" seul est volontairement exclu pour éviter les faux positifs cliniques
    assert scorer.has_negation("Absence de recepteurs estrogeniques", "recepteurs")
    assert scorer.has_negation("Sans expression HER2 detectee", "HER2")
    assert scorer.has_negation("Il n'y a aucun marquage observe", "marquage")
    assert scorer.has_negation("HER2 ne pas surexprimer", "HER2")

    # Cas positif — aucune negation
    assert not scorer.has_negation("HER2 surexprime (3+)", "HER2")


def test_uncertainty_detection():
    scorer = RiskContextScorer()

    assert scorer.has_uncertainty("Possible amplification", "amplification")
    assert scorer.has_uncertainty("Statut à confirmer", "Statut")
    assert not scorer.has_uncertainty("Biopsie franche", "Biopsie")


def test_risk_score():
    scorer = RiskContextScorer()

    # Contexte simple -> Risque faible (0.00)
    score_low = scorer.compute_score(["ER positif 100%"], "ER")
    assert score_low < 0.2

    # Contexte avec négation et incertiture -> Risque élevé
    score_high = scorer.compute_score(
        ["Pas clair si ER positif ou négatif", "statut discordant ER"], "ER"
    )
    assert score_high >= 0.35
