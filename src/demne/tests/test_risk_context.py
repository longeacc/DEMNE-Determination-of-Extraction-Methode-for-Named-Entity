import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "../demne"))

from demne.E_risk_context import RiskContextScorer


def test_negation_detection():
    scorer = RiskContextScorer()

    # Cas clairs de négation
    assert scorer.has_negation("HER2 non surexprimé", "HER2")
    assert scorer.has_negation("Absence de récepteurs œstrogènes", "récepteurs")
    assert scorer.has_negation("Pas de mutation détectée", "mutation")

    # Cas positif
    assert not scorer.has_negation("HER2 surexprimé (3+)", "HER2")


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
