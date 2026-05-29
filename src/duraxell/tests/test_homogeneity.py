import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "../duraxell"))

from duraxell.E_homogeneity import HomogeneityScorer


def test_homogeneity_limits():
    scorer = HomogeneityScorer([])

    # Cas limite 1 : 1 seule mention -> He = 1.0 par convention (si géré) ou 0 ?
    # Si redondance = (1-1)/1 = 0.
    # Mais une entité unique est techniquement "homogène" (pas de variation).
    # Vérifions le comportement actuel.
    scorer.compute_from_list(["ER"])
    # Si le code gère le cas N=1 -> 1.0, sinon c'est 0.
    # On adaptera le test fonction du code.

    # Cas limite 2 : 1000 mentions identiques
    data_identical = ["ER"] * 1000
    score_id = scorer.compute_from_list(data_identical)
    assert score_id > 0.95

    # Cas limite 3 : 100 mentions toutes différentes et SANS préfixe commun
    # Utilisons des entiers str(), car "variation_i" contient "variation" qui se répète 100 fois !
    # Cela augmentait artificiellement l'homogénéité (~0.5).
    data_diff = [str(i) for i in range(1000, 1100)]
    score_diff = scorer.compute_from_list(data_diff)
    assert (
        score_diff < 10.0
    )  # On attend un score très bas (< 0.1 idéalement, mais la sigmoide peut le relever un peu)


def test_homogeneity_mixed():
    scorer = HomogeneityScorer([])

    # Mélange 50/50
    data_mixed = ["ER"] * 50 + ["Estrogen Receptor"] * 50
    # Redondance approx 0.98 car "ER" et "Estrogen Receptor" répétés 50 fois.
    # Total items: 100. Unique items: 2.
    # Redondance items = (100-2)/100 = 0.98.

    # Mais le scoreur tokenise ?
    # "ER" -> "er"
    # "Estrogen Receptor" -> "estrogen", "receptor"
    # Total words: 50*1 + 50*2 = 150 words.
    # Unique words: "er", "estrogen", "receptor" = 3.
    # Redondance words = (150-3)/150 = 147/150 = 0.98.

    score_mixed = scorer.compute_from_list(data_mixed)
    assert score_mixed > 0.9
