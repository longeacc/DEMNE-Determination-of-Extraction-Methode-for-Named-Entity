from demne.templatability import TemplatabilityScorer


def test_templatability_scores():
    scorer = TemplatabilityScorer([])

    # Case 1: Very rigid structure -> high Te
    # Simulation: single pattern repeated
    data_rigid = ["ER 100%"] * 100
    score_rigid = scorer.compute_from_list(data_rigid)
    assert score_rigid > 80.0, f"Expected > 80.0 for rigid data, got {score_rigid}"

    # Case 2: Very varied structure -> low Te
    # Using structurally different texts
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

    # Case 3: Regex normalization
    # "HER2 3+" and "HER2 2+" should be seen as similar after "D+" normalization.
    # If the scorer normalizes correctly, it should find a dominant pattern.
    data_semi = ["HER2 3+", "HER2 2+", "HER2 1+", "HER2 0"] * 25
    score_semi = scorer.compute_from_list(data_semi)
    assert score_semi > 25.0, "Normalization should capture digit variations"


def test_to_json_round_trip(tmp_path, brat_corpus):
    import json

    scorer = TemplatabilityScorer(brat_corpus)
    scorer.compute_all()  # to_json writes the cached per-entity statistics
    out = tmp_path / "te.json"
    scorer.to_json(out)
    data = json.loads(out.read_text(encoding="utf-8"))
    assert set(data) == {"HER2_Status", "ER_Status", "PR_Status"}
    assert all("templatability_score" in v and "count" in v for v in data.values())
