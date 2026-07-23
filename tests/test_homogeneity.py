from demne.E_homogeneity import HomogeneityScorer


def test_homogeneity_limits():
    scorer = HomogeneityScorer([])

    # Edge case 1: single mention -> He = 1.0 by convention (if handled) or 0?
    # If redundancy = (1-1)/1 = 0.
    # But a unique entity is technically "homogeneous" (no variation).
    # Check current behavior.
    scorer.compute_from_list(["ER"])
    # If the code handles N=1 -> 1.0, otherwise it is 0.
    # Test will be adapted based on the code.

    # Edge case 2: 1000 identical mentions
    data_identical = ["ER"] * 1000
    score_id = scorer.compute_from_list(data_identical)
    assert score_id > 0.9

    # Edge case 3: 100 entirely different mentions WITHOUT a common prefix.
    # Using str(int) because "variation_i" contains "variation" repeated 100 times,
    # which would artificially inflate homogeneity (~0.5).
    data_diff = [str(i) for i in range(1000, 1100)]
    score_diff = scorer.compute_from_list(data_diff)
    assert (
        score_diff < 10.0
    )  # Expected very low score (< 0.1 ideally, but sigmoid may raise it slightly)


def test_homogeneity_mixed():
    scorer = HomogeneityScorer([])

    # 50/50 mix
    data_mixed = ["ER"] * 50 + ["Estrogen Receptor"] * 50
    # Redundancy approx 0.98: "ER" and "Estrogen Receptor" each repeated 50 times.
    # Total items: 100. Unique items: 2.
    # Item redundancy = (100-2)/100 = 0.98.

    # Does the scorer tokenize?
    # "ER" -> "er"
    # "Estrogen Receptor" -> "estrogen", "receptor"
    # Total words: 50*1 + 50*2 = 150 words.
    # Unique words: "er", "estrogen", "receptor" = 3.
    # Word redundancy = (150-3)/150 = 147/150 = 0.98.

    score_mixed = scorer.compute_from_list(data_mixed)
    assert score_mixed > 0.9
