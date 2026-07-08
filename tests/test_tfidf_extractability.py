"""Tests for the TFIDF_Extractability metric and its integration into the
DEMNE decision tree (E_creation_arbre_decision).

Two blocks:
  A. Unit tests of demne.tfidf_extractability (compute, routing, grid search,
     BRAT context extraction).
  B. Integration tests of DecisionTreeBuilder.analyze_entity:
     - strict backward-compatibility when no TFIDF data is supplied;
     - conceptual-synonymy rescue (low He but high tfidf_score -> RÈGLES);
     - R modulation / escalation behaviour.
"""

import sys
from unittest.mock import MagicMock

import pytest

# eco2ai is an optional heavy dep pulled in by E_creation_arbre_decision.
sys.modules.setdefault("eco2ai", MagicMock())

from demne.E_creation_arbre_decision import DecisionTreeBuilder
from demne.E_tfidf import (
    build_corpus_contexts,
    compute_tfidf_extractability,
    grid_search_tfidf,
    updated_demne_routing,
)


# --------------------------------------------------------------------------- #
# Fixtures : jeux de données synthétiques réutilisés                          #
# --------------------------------------------------------------------------- #
@pytest.fixture
def synonyms_case():
    """évolution_tumorale : synonymes conceptuels, contextes identiques."""
    mentions = ["majoration", "progression", "augmentation", "croissance"] * 10
    contexts = {
        "majoration": ["lésion tumorale majoration diamètre mm mesure"] * 10,
        "progression": ["lésion tumorale progression diamètre mm mesure"] * 10,
        "augmentation": ["lésion tumorale augmentation diamètre mm mesure"] * 10,
        "croissance": ["lésion tumorale croissance diamètre mm mesure"] * 10,
    }
    return mentions, contexts


@pytest.fixture
def heterogeneous_case():
    """pays_origine : formes variées, contextes cliniques non partagés."""
    mentions = ["France", "Maroc", "Algérie", "Tunisie", "Mali",
                "Sénégal", "Cameroun", "Côte d'Ivoire"] * 5
    contexts = {
        "france": ["né en France domicilié paris"] * 5,
        "maroc": ["originaire du Maroc rabat famille"] * 5,
        "algérie": ["venu d'Algérie oran antécédents"] * 5,
        "tunisie": ["séjour tunisie retour récent"] * 5,
        "mali": ["origine mali bamako voyage"] * 5,
        "sénégal": ["sénégal dakar migration récente"] * 5,
        "cameroun": ["cameroun yaoundé antécédents tropicaux"] * 5,
        "côte d'ivoire": ["abidjan côte ivoire expatrié"] * 5,
    }
    return mentions, contexts


# =========================================================================== #
# BLOC A — compute_tfidf_extractability                                       #
# =========================================================================== #
def test_edge_case_too_few_mentions():
    out = compute_tfidf_extractability("HER2_FISH", ["positif", "négatif"], {})
    assert out["tfidf_score"] == 0.0
    assert out["routes_to_rules"] is False
    assert out["clusters"] == []
    assert "R_modulation_factor" not in out


def test_edge_case_single_unique_form():
    out = compute_tfidf_extractability("mono", ["stable", "stable", "stable"], {})
    assert out["tfidf_score"] == 1.0
    assert out["routes_to_rules"] is True
    assert "R_modulation_factor" not in out


def test_conceptual_synonyms_high_score(synonyms_case):
    mentions, contexts = synonyms_case
    out = compute_tfidf_extractability("évolution_tumorale", mentions, contexts,
                                       X=5, sim_threshold=0.50)
    assert out["tfidf_score"] >= 0.70
    assert out["routes_to_rules"] is True
    # les 4 formes doivent fusionner en un seul cluster couvrant tout
    assert len(out["clusters"]) == 1
    assert out["top_X_recalls"] == [1.0]


def test_heterogeneous_low_score(heterogeneous_case):
    mentions, contexts = heterogeneous_case
    out = compute_tfidf_extractability("pays_origine", mentions, contexts,
                                       X=5, sim_threshold=0.50)
    assert out["tfidf_score"] <= 0.35
    assert out["routes_to_rules"] is False


def test_no_r_modulation_factor_in_output(synonyms_case, heterogeneous_case):
    """R_modulation_factor supprimé : plus de modulation du risque dans TFIDF."""
    for mentions, contexts in (synonyms_case, heterogeneous_case):
        out = compute_tfidf_extractability("x", mentions, contexts)
        assert "R_modulation_factor" not in out


def test_score_is_rounded_and_in_unit_interval(synonyms_case):
    mentions, contexts = synonyms_case
    out = compute_tfidf_extractability("x", mentions, contexts)
    assert 0.0 <= out["tfidf_score"] <= 1.0
    assert out["tfidf_score"] == round(out["tfidf_score"], 4)


def test_missing_contexts_do_not_crash():
    # mentions présentes mais aucun contexte -> docs vides -> score défini
    mentions = ["alpha", "beta", "gamma"]
    out = compute_tfidf_extractability("x", mentions, {})
    assert 0.0 <= out["tfidf_score"] <= 1.0


# =========================================================================== #
# BLOC A — updated_demne_routing (graphe figure de référence)                 #
# Nœud R- PARTAGÉ entre branche Te/He et branche TF-IDF.                     #
# TF-IDF Oui → R- → RULES ou Feas (pas directement RULES).                   #
# =========================================================================== #
_TH = {"Te_HIGH": 0.10, "He_HIGH": 0.85, "R_HIGH": 0.25, "Feas_NER": 0.50, "Y": 0.70}


def _m(**kw):
    base = {"Te": 0.0, "He": 0.0, "R": 0.0, "Feas": 0.0}
    base.update(kw)
    return base


def test_routing_te_he_r_low_returns_rules():
    # Te+He élevés + R faible → RULES (branche classique)
    assert updated_demne_routing(_m(Te=0.5, He=0.9, R=0.1), _TH) == "RULES"


def test_routing_te_he_r_high_falls_to_feas():
    # Te+He élevés + R élevé (risk of conflict) → Feas décide
    assert updated_demne_routing(_m(Te=0.5, He=0.9, R=0.9, Feas=0.9), _TH) == "TBM"
    assert updated_demne_routing(_m(Te=0.5, He=0.9, R=0.9, Feas=0.1), _TH) == "LLM"


def test_routing_te_only_no_rules_without_he():
    # Te élevé mais He faible → branche TF-IDF, pas RULES direct
    assert updated_demne_routing(_m(Te=0.5, He=0.1, R=0.1, Feas=0.9), _TH) == "TBM"


def test_routing_tfidf_high_r_low_returns_rules():
    # TF-IDF élevé + R faible → nœud R- → RULES
    assert updated_demne_routing(_m(tfidf_score=0.8, R=0.1, Feas=0.9), _TH) == "RULES"


def test_routing_tfidf_high_r_high_falls_to_feas():
    # TF-IDF élevé mais R élevé (risk of conflict) → Feas, pas RULES
    assert updated_demne_routing(_m(tfidf_score=0.8, R=0.9, Feas=0.9), _TH) == "TBM"
    assert updated_demne_routing(_m(tfidf_score=0.8, R=0.9, Feas=0.1), _TH) == "LLM"


def test_routing_tfidf_below_y_falls_to_feas():
    # TF-IDF < Y → Feas directement (pas de nœud R-)
    assert updated_demne_routing(_m(tfidf_score=0.50, R=0.1, Feas=0.9), _TH) == "TBM"
    assert updated_demne_routing(_m(tfidf_score=0.50, R=0.1, Feas=0.1), _TH) == "LLM"


def test_routing_tfidf_absent_falls_to_feas():
    # Sans tfidf_score : nœud TF-IDF ignoré → Feas directement
    assert updated_demne_routing(_m(R=0.30, Feas=0.9), _TH) == "TBM"
    assert updated_demne_routing(_m(R=0.30, Feas=0.1), _TH) == "LLM"


def test_routing_feas_gate_tbm_vs_llm():
    assert updated_demne_routing(_m(Feas=0.9), _TH) == "TBM"
    assert updated_demne_routing(_m(Feas=0.1), _TH) == "LLM"


# =========================================================================== #
# BLOC A — build_corpus_contexts (parsing BRAT)                               #
# =========================================================================== #
def test_build_corpus_contexts_from_brat(tmp_path):
    txt = "Le patient presente une progression tumorale nette au scanner de controle."
    ann = "T1\tEVOL 22 33\tprogression\n"  # span de "progression"
    txt_p = tmp_path / "doc.txt"
    ann_p = tmp_path / "doc.ann"
    txt_p.write_text(txt, encoding="utf-8")
    ann_p.write_text(ann, encoding="utf-8")

    ctx = build_corpus_contexts(ann_p, txt_p, window_tokens=3)
    assert "progression" in ctx
    window = ctx["progression"][0].lower()
    # la fenêtre contient des mots voisins du span
    assert "tumorale" in window
    assert "progression" in window


def test_build_corpus_contexts_ignores_non_term_lines(tmp_path):
    txt = "alpha beta gamma delta"
    ann = "#1\tAnnotatorNotes T1\tnote\nR1\trel Arg1:T1 Arg2:T2\n"
    txt_p = tmp_path / "d.txt"
    ann_p = tmp_path / "d.ann"
    txt_p.write_text(txt, encoding="utf-8")
    ann_p.write_text(ann, encoding="utf-8")
    ctx = build_corpus_contexts(ann_p, txt_p)
    assert ctx == {}


# =========================================================================== #
# BLOC A — grid_search_tfidf                                                  #
# =========================================================================== #
def test_grid_search_space_and_best(synonyms_case, heterogeneous_case):
    syn_m, syn_c = synonyms_case
    het_m, het_c = heterogeneous_case
    entities = [
        {"entity_name": "evo", "Te": 0.02, "He": 0.30,
         "mentions": syn_m, "corpus_contexts": syn_c},
        {"entity_name": "pays", "Te": 0.01, "He": 0.10,
         "mentions": het_m, "corpus_contexts": het_c},
        # entité NON éligible (Te >= Te_HIGH) : doit être exclue du F1
        {"entity_name": "skip", "Te": 0.50, "He": 0.10,
         "mentions": ["a", "b", "c"], "corpus_contexts": {}},
    ]
    refs = ["RULES", "LLM", "RULES"]
    res = grid_search_tfidf(entities, refs, te_high=0.10, he_high=0.85)

    assert len(res["all_results"]) == 650  # 10 X × 13 Y × 5 sim
    assert res["best_X"] in range(1, 11)
    assert res["best_sim_threshold"] in {0.30, 0.40, 0.50, 0.60, 0.70}
    assert 0.40 <= res["best_Y"] <= 1.00
    # synonyme=RULES / hétérogène=LLM sont parfaitement séparables -> F1=1.0
    assert res["best_f1"] == 1.0


def test_grid_search_eligibility_filter():
    # toutes les entités inéligibles -> aucun TP/FP/FN -> F1 = 0
    entities = [{"entity_name": "x", "Te": 0.9, "He": 0.9,
                 "mentions": ["a", "b", "c"], "corpus_contexts": {}}]
    res = grid_search_tfidf(entities, ["RULES"], te_high=0.10, he_high=0.85)
    assert res["best_f1"] == 0.0


# =========================================================================== #
# BLOC B — Intégration dans DecisionTreeBuilder                               #
# =========================================================================== #
@pytest.fixture
def builder():
    return DecisionTreeBuilder(config_path="dummy.json")


def test_backward_compat_baseline_unchanged(builder):
    """Sans TFIDF, l'arbre doit être identique au baseline (test_decision_tree.py).
    Le nœud R- est maintenant PARTAGÉ : Te+He élevés + R élevé → Feas (TBM), pas LLM.
    """
    # Te+He élevés, R faible → RULES
    assert builder.analyze_entity(
        "StructureOnly", {"Te": 90.0, "Te_count": 20, "He": 80.0, "R": 0.1}
    )["method"] == "RÈGLES"
    # Te élevé, He faible, R faible, Feas élevé → TBM (branche TF-IDF, absent → Feas)
    assert builder.analyze_entity(
        "GoodRules", {"Te": 50.0, "Te_count": 20, "He": 20.0, "R": 0.1, "Feas": 0.8}
    )["method"] == "TBM"
    # Te faible, pas de TF-IDF, Feas faible → LLM
    assert builder.analyze_entity(
        "CommonEntity", {"Te": 10.0, "Te_count": 20, "He": 10.0, "R": 0.2, "Feas": 0.1}
    )["method"] == "LLM"
    # Te+He élevés, R élevé (risk of conflict) → Feas élevé → TBM (comportement préservé)
    assert builder.analyze_entity(
        "RiskyEntity", {"Te": 90.0, "Te_count": 20, "He": 80.0, "R": 0.8, "Feas": 0.8}
    )["method"] == "TBM"


def test_tfidf_node_bypassed_when_te_and_he_high(builder):
    """Te+He élevés + R faible → RULES via branche classique, TFIDF jamais consulté."""
    res = builder.analyze_entity(
        "x", {"Te": 90.0, "Te_count": 20, "He": 80.0, "R": 0.1, "tfidf_score": 0.95}
    )
    assert res["method"] == "RÈGLES"
    assert "TFIDF" not in res["justification"]


def test_tfidf_rescue_low_he_r_low_routes_rules(builder):
    """He faible → TF-IDF élevé + R faible → nœud R- → RÈGLES."""
    metrics = {"Te": 5.0, "Te_count": 20, "He": 10.0, "R": 0.1, "Feas": 0.8,
               "tfidf_score": 0.95}
    res = builder.analyze_entity("évolution_tumorale", metrics)
    assert res["method"] == "RÈGLES"
    assert "TFIDF" in res["justification"]


def test_tfidf_rescue_low_he_r_high_falls_to_feas(builder):
    """He faible → TF-IDF élevé MAIS R élevé (risk of conflict) → Feas, pas RULES."""
    metrics = {"Te": 5.0, "Te_count": 20, "He": 10.0, "R": 0.9, "Feas": 0.8,
               "tfidf_score": 0.95}
    assert builder.analyze_entity("x", metrics)["method"] == "TBM"


def test_tfidf_below_y_falls_to_feas(builder):
    """TF-IDF < Y → Feas directement (pas de nœud R-)."""
    metrics = {"Te": 5.0, "Te_count": 20, "He": 10.0, "R": 0.9, "Feas": 0.8,
               "tfidf_score": 0.30}
    # R élevé mais ignoré (TF-IDF < Y → Feas direct)
    assert builder.analyze_entity("x", metrics)["method"] == "TBM"


def test_tfidf_absent_falls_to_feas_directly(builder):
    """Sans tfidf_score : nœud TF-IDF ignoré → Feas directement."""
    assert builder.analyze_entity(
        "x", {"Te": 5.0, "Te_count": 20, "He": 10.0, "R": 0.9, "Feas": 0.8}
    )["method"] == "TBM"
    assert builder.analyze_entity(
        "x", {"Te": 5.0, "Te_count": 20, "He": 10.0, "R": 0.9, "Feas": 0.1}
    )["method"] == "LLM"
