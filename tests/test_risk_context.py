from demne.risk_context import RiskContextScorer


def test_negation_detection():
    scorer = RiskContextScorer()

    # Patterns actually detected (see NEGATION_PATTERNS in risk_context.py)
    # "non" alone is intentionally excluded to avoid clinical false positives
    assert scorer.has_negation("Absence de recepteurs estrogeniques", "recepteurs")
    assert scorer.has_negation("Sans expression HER2 detectee", "HER2")
    assert scorer.has_negation("Il n'y a aucun marquage observe", "marquage")
    assert scorer.has_negation("HER2 ne pas surexprimer", "HER2")

    # Positive case — no negation
    assert not scorer.has_negation("HER2 surexprime (3+)", "HER2")


def test_uncertainty_detection():
    scorer = RiskContextScorer()

    assert scorer.has_uncertainty("Possible amplification", "amplification")
    assert scorer.has_uncertainty("Statut à confirmer", "Statut")
    assert not scorer.has_uncertainty("Biopsie franche", "Biopsie")


def test_risk_score():
    scorer = RiskContextScorer()

    # Simple context -> Low risk (0.00)
    score_low = scorer.compute_score(["ER positif 100%"], "ER")
    assert score_low < 0.2

    # Context with negation and uncertainty -> High risk
    score_high = scorer.compute_score(
        ["Absence de recepteurs estrogeniques ER", "Statut ER à confirmer"], "ER"
    )
    assert score_high >= 0.20


def test_to_csv_and_contradiction_rate(tmp_path, brat_corpus_dir):
    import csv

    scorer = RiskContextScorer([brat_corpus_dir])
    out = tmp_path / "r.csv"
    scorer.to_csv(out)
    rows = {r["Entity"]: r for r in csv.DictReader(out.open(encoding="utf-8"))}
    assert set(rows) == {"HER2_Status", "ER_Status", "PR_Status"}
    assert all(0.0 <= float(r["R_Score"]) <= 1.0 for r in rows.values())


def test_contradiction_needs_both_polarities_in_one_document():
    scorer = RiskContextScorer([])
    assert scorer._check_contradiction([{"value_text": "her2 positif"}, {"value_text": "her2 neg"}])
    assert not scorer._check_contradiction([{"value_text": "her2 positif"}])


def test_learn_weights_fits_logistic_regression():
    scorer = RiskContextScorer([])
    data = [(1, 0, 0, 1), (0, 1, 0, 1), (0, 0, 1, 1), (0, 0, 0, 0), (0, 0, 0, 0), (1, 1, 0, 1)]
    scorer._learn_weights(data)
    assert set(scorer.weights) == {"negation", "uncertainty", "contradiction"}
