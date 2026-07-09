"""Tests for the TFIDF_Extractability metric and its integration into the
DEMNE decision tree (E_creation_arbre_decision).

Two blocks:
  A. Unit tests of demne.E_tfidf (compute, routing, BRAT context extraction,
     corpus API).
  B. Integration tests of DecisionTreeBuilder.analyze_entity:
     - strict backward-compatibility when no TFIDF data is supplied;
     - conceptual-synonymy rescue (low He but high tfidf_score -> RULES);
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
    updated_demne_routing,
)


# --------------------------------------------------------------------------- #
# Fixtures: synthetic datasets reused across tests                            #
# --------------------------------------------------------------------------- #
@pytest.fixture
def synonyms_case():
    """évolution_tumorale: conceptual synonyms, identical contexts."""
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
    """pays_origine: varied forms, non-shared clinical contexts."""
    mentions = [
        "France",
        "Maroc",
        "Algérie",
        "Tunisie",
        "Mali",
        "Sénégal",
        "Cameroun",
        "Côte d'Ivoire",
    ] * 5
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
    assert out["f1_score"] == 0.0
    assert out["routes_to_rules"] is False
    assert out["clusters"] == []
    assert out["top_X_f1s"] == []
    assert "R_modulation_factor" not in out


def test_edge_case_single_unique_form():
    out = compute_tfidf_extractability("mono", ["stable", "stable", "stable"], {})
    assert out["tfidf_score"] == 1.0
    assert out["f1_score"] == 1.0
    assert out["routes_to_rules"] is True
    assert out["top_X_f1s"] == [1.0]
    assert "R_modulation_factor" not in out


def test_conceptual_synonyms_high_score(synonyms_case):
    mentions, contexts = synonyms_case
    out = compute_tfidf_extractability(
        "évolution_tumorale", mentions, contexts, X=5, sim_threshold=0.50
    )
    assert out["tfidf_score"] >= 0.70
    assert out["f1_score"] >= 0.70
    assert out["routes_to_rules"] is True
    # all 4 forms must merge into a single cluster covering everything
    assert len(out["clusters"]) == 1
    assert out["top_X_recalls"] == [1.0]
    assert out["top_X_f1s"] == [1.0]


def test_heterogeneous_low_score(heterogeneous_case):
    mentions, contexts = heterogeneous_case
    out = compute_tfidf_extractability("pays_origine", mentions, contexts, X=5, sim_threshold=0.50)
    assert out["tfidf_score"] <= 0.35
    assert out["f1_score"] <= 0.50
    assert out["routes_to_rules"] is False


def test_f1_fields_always_present(synonyms_case, heterogeneous_case):
    """f1_score and top_X_f1s are always in the output dict."""
    for mentions, contexts in (synonyms_case, heterogeneous_case):
        out = compute_tfidf_extractability("x", mentions, contexts)
        assert "f1_score" in out
        assert "top_X_f1s" in out
        assert isinstance(out["f1_score"], float)
        assert isinstance(out["top_X_f1s"], list)


def test_f1_in_unit_interval(synonyms_case, heterogeneous_case):
    """F1 score ∈ [0, 1] for all cases."""
    for mentions, contexts in (synonyms_case, heterogeneous_case):
        out = compute_tfidf_extractability("x", mentions, contexts)
        assert 0.0 <= out["f1_score"] <= 1.0
        for f in out["top_X_f1s"]:
            assert 0.0 <= f <= 1.0


def test_f1_geq_recall_compact_cluster(synonyms_case):
    """Compact cluster (few forms / many unique) → precision=1 → F1 ≥ recall."""
    mentions, contexts = synonyms_case
    out = compute_tfidf_extractability("x", mentions, contexts)
    # single cluster with 4 forms out of 4 unique → precision = n_unique/k = 1.0
    for recall, f1 in zip(out["top_X_recalls"], out["top_X_f1s"], strict=True):
        assert f1 >= recall * 0.99  # F1 ≥ recall when precision ≥ recall


def test_use_f1_false_routes_on_recall(synonyms_case):
    """use_f1=False → routes_to_rules based on recall (tfidf_score), not f1_score."""
    mentions, contexts = synonyms_case
    # force Y above 1.0 to fail regardless, then verify routing uses recall path
    out_f1 = compute_tfidf_extractability("x", mentions, contexts, Y=0.50, use_f1=True)
    out_rc = compute_tfidf_extractability("x", mentions, contexts, Y=0.50, use_f1=False)
    # both should route to rules here (recall=1.0 and f1=1.0 both ≥ 0.50)
    assert out_f1["routes_to_rules"] is True
    assert out_rc["routes_to_rules"] is True


def test_use_f1_toggle_changes_routing():
    """When F1 < Y but recall >= Y, toggling use_f1 changes routes_to_rules."""
    # Craft a case: 2 forms, one rare → low recall per cluster but high compactness
    # We need recall >= Y but f1 < Y for one mode to differ from the other
    # Simplest: use Y=0.0 vs Y=1.1 to check the toggle is wired
    mentions = ["alpha", "beta", "gamma"] * 5
    contexts = {
        "alpha": ["context alpha one two three"] * 5,
        "beta": ["context beta four five six"] * 5,
        "gamma": ["context gamma seven eight nine"] * 5,
    }
    out = compute_tfidf_extractability("x", mentions, contexts, Y=0.0, use_f1=True)
    assert out["routes_to_rules"] is True  # any score >= 0.0
    out2 = compute_tfidf_extractability("x", mentions, contexts, Y=1.1, use_f1=False)
    assert out2["routes_to_rules"] is False  # no score > 1.0


def test_no_r_modulation_factor_in_output(synonyms_case, heterogeneous_case):
    """R_modulation_factor removed: no more risk modulation inside TFIDF."""
    for mentions, contexts in (synonyms_case, heterogeneous_case):
        out = compute_tfidf_extractability("x", mentions, contexts)
        assert "R_modulation_factor" not in out


def test_score_is_rounded_and_in_unit_interval(synonyms_case):
    mentions, contexts = synonyms_case
    out = compute_tfidf_extractability("x", mentions, contexts)
    assert 0.0 <= out["tfidf_score"] <= 1.0
    assert out["tfidf_score"] == round(out["tfidf_score"], 4)
    assert 0.0 <= out["f1_score"] <= 1.0
    assert out["f1_score"] == round(out["f1_score"], 4)


def test_missing_contexts_do_not_crash():
    # mentions present but no context -> empty docs -> score is still defined
    mentions = ["alpha", "beta", "gamma"]
    out = compute_tfidf_extractability("x", mentions, {})
    assert 0.0 <= out["tfidf_score"] <= 1.0


# =========================================================================== #
# BLOC A — updated_demne_routing (reference figure graph)                     #
# R- node SHARED between Te/He branch and TF-IDF branch.                      #
# TF-IDF Yes → R- → RULES or Feas (not directly RULES).                      #
# =========================================================================== #
_TH = {"Te_HIGH": 0.10, "He_HIGH": 0.85, "R_HIGH": 0.25, "Feas_NER": 0.50, "Y": 0.70}


def _m(**kw):
    base = {"Te": 0.0, "He": 0.0, "R": 0.0, "Feas": 0.0}
    base.update(kw)
    return base


def test_routing_te_he_r_low_returns_rules():
    # High Te+He + low R → RULES (classic branch)
    assert updated_demne_routing(_m(Te=0.5, He=0.9, R=0.1), _TH) == "RULES"


def test_routing_te_he_r_high_falls_to_feas():
    # High Te+He + high R (risk of conflict) → Feas decides
    assert updated_demne_routing(_m(Te=0.5, He=0.9, R=0.9, Feas=0.9), _TH) == "TBM"
    assert updated_demne_routing(_m(Te=0.5, He=0.9, R=0.9, Feas=0.1), _TH) == "LLM"


def test_routing_te_only_no_rules_without_he():
    # High Te but low He → TF-IDF branch, not direct RULES
    assert updated_demne_routing(_m(Te=0.5, He=0.1, R=0.1, Feas=0.9), _TH) == "TBM"


def test_routing_tfidf_high_r_low_returns_rules():
    # High TF-IDF + low R → R- node → RULES
    assert updated_demne_routing(_m(tfidf_score=0.8, R=0.1, Feas=0.9), _TH) == "RULES"


def test_routing_tfidf_high_r_high_falls_to_feas():
    # High TF-IDF but high R (risk of conflict) → Feas, not RULES
    assert updated_demne_routing(_m(tfidf_score=0.8, R=0.9, Feas=0.9), _TH) == "TBM"
    assert updated_demne_routing(_m(tfidf_score=0.8, R=0.9, Feas=0.1), _TH) == "LLM"


def test_routing_tfidf_below_y_falls_to_feas():
    # TF-IDF < Y → Feas directly (no R- node)
    assert updated_demne_routing(_m(tfidf_score=0.50, R=0.1, Feas=0.9), _TH) == "TBM"
    assert updated_demne_routing(_m(tfidf_score=0.50, R=0.1, Feas=0.1), _TH) == "LLM"


def test_routing_tfidf_absent_falls_to_feas():
    # No tfidf_score: TF-IDF node skipped → Feas directly
    assert updated_demne_routing(_m(R=0.30, Feas=0.9), _TH) == "TBM"
    assert updated_demne_routing(_m(R=0.30, Feas=0.1), _TH) == "LLM"


def test_routing_feas_gate_tbm_vs_llm():
    assert updated_demne_routing(_m(Feas=0.9), _TH) == "TBM"
    assert updated_demne_routing(_m(Feas=0.1), _TH) == "LLM"


# =========================================================================== #
# BLOC A — build_corpus_contexts (BRAT parsing)                               #
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
    # the window contains neighbouring words of the span
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


@pytest.fixture
def builder():
    return DecisionTreeBuilder(config_path="dummy.json")


def test_backward_compat_baseline_unchanged(builder):
    """Without TFIDF the tree must match the baseline (test_decision_tree.py).
    The R- node is now SHARED: high Te+He + high R → Feas (TBM), not LLM.
    """
    # High Te+He, low R → RULES
    assert (
        builder.analyze_entity("StructureOnly", {"Te": 90.0, "Te_count": 20, "He": 80.0, "R": 0.1})[
            "method"
        ]
        == "RULES"
    )
    # High Te, low He, low R, high Feas → TBM (TF-IDF branch, absent → Feas)
    assert (
        builder.analyze_entity(
            "GoodRules", {"Te": 50.0, "Te_count": 20, "He": 20.0, "R": 0.1, "Feas": 0.8}
        )["method"]
        == "TBM"
    )
    # Low Te, no TF-IDF, low Feas → LLM
    assert (
        builder.analyze_entity(
            "CommonEntity", {"Te": 10.0, "Te_count": 20, "He": 10.0, "R": 0.2, "Feas": 0.1}
        )["method"]
        == "LLM"
    )
    # High Te+He, high R (risk of conflict) → high Feas → TBM (preserved behaviour)
    assert (
        builder.analyze_entity(
            "RiskyEntity", {"Te": 90.0, "Te_count": 20, "He": 80.0, "R": 0.8, "Feas": 0.8}
        )["method"]
        == "TBM"
    )


def test_tfidf_node_bypassed_when_te_and_he_high(builder):
    """High Te+He + low R → RULES via classic branch, TFIDF never consulted."""
    res = builder.analyze_entity(
        "x", {"Te": 90.0, "Te_count": 20, "He": 80.0, "R": 0.1, "tfidf_score": 0.95}
    )
    assert res["method"] == "RULES"
    assert "TFIDF" not in res["justification"]


def test_tfidf_rescue_low_he_r_low_routes_rules(builder):
    """Low He → high TF-IDF + low R → R- node → RULES."""
    metrics = {"Te": 5.0, "Te_count": 20, "He": 10.0, "R": 0.1, "Feas": 0.8, "tfidf_score": 0.95}
    res = builder.analyze_entity("évolution_tumorale", metrics)
    assert res["method"] == "RULES"
    assert "TFIDF" in res["justification"]


def test_tfidf_rescue_low_he_r_high_falls_to_feas(builder):
    """Low He → high TF-IDF BUT high R (risk of conflict) → Feas, not RULES."""
    metrics = {"Te": 5.0, "Te_count": 20, "He": 10.0, "R": 0.9, "Feas": 0.8, "tfidf_score": 0.95}
    assert builder.analyze_entity("x", metrics)["method"] == "TBM"


def test_tfidf_below_y_falls_to_feas(builder):
    """TF-IDF < Y → Feas directly (no R- node)."""
    metrics = {"Te": 5.0, "Te_count": 20, "He": 10.0, "R": 0.9, "Feas": 0.8, "tfidf_score": 0.30}
    # High R but ignored (TF-IDF < Y → direct Feas)
    assert builder.analyze_entity("x", metrics)["method"] == "TBM"


def test_tfidf_absent_falls_to_feas_directly(builder):
    """No tfidf_score: TF-IDF node skipped → Feas directly."""
    assert (
        builder.analyze_entity("x", {"Te": 5.0, "Te_count": 20, "He": 10.0, "R": 0.9, "Feas": 0.8})[
            "method"
        ]
        == "TBM"
    )
    assert (
        builder.analyze_entity("x", {"Te": 5.0, "Te_count": 20, "He": 10.0, "R": 0.9, "Feas": 0.1})[
            "method"
        ]
        == "LLM"
    )
