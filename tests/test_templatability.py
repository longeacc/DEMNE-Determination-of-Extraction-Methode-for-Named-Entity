from demne.E_templatability import TemplatabilityScorer


def test_templatability_scores():
    scorer = TemplatabilityScorer([])

    # Cas 1 : Structure très rigide -> Te élevé
    # Simulation: pattern unique répété
    data_rigid = ["ER 100%"] * 100
    score_rigid = scorer.compute_from_list(data_rigid)
    assert score_rigid > 80.0, f"Expected > 80.0 for rigid data, got {score_rigid}"

    # Cas 2 : Structure très variée -> Te faible
    # On utilise des textes structurellement différents
    data_chaotic = [
        "ER positif",
        "Pas de marquage significatif",
        "Absence totale de récepteurs",
        "Marquage faible à modéré",
        "Score Allred de 5/8",
        "RO: + (10%)",
        "Statut inconnu",
        "échantillon non contributif",
        "voir compte rendu anatomopathologique",
        "RO neg",
    ] * 10
    score_chaotic = scorer.compute_from_list(data_chaotic)
    assert score_chaotic < 60.0, f"Expected < 60.0 for chaotic data, got {score_chaotic}"

    # Cas 3 : Normalisation Regex
    # "HER2 3+" et "HER2 2+" devraient être vus comme similaires après normalisation "D+"
    # Si le scorer normalise bien, il devrait trouver un pattern dominant
    data_semi = ["HER2 3+", "HER2 2+", "HER2 1+", "HER2 0"] * 25
    score_semi = scorer.compute_from_list(data_semi)
    assert score_semi > 25.0, "Normalization should capture digit variations"
