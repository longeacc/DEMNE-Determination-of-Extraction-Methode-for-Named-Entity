"""Tests de l'extractabilité par COUVERTURE de mots-clés + intégration à l'arbre.

Deux blocs :
  A. Unitaires de demne.E_tfidf_coverage (coverage-F1, lecture BRAT, API corpus).
     La coverage-F1 remplace la synonymie contextuelle (ancien E_tfidf) : elle
     sépare mieux les classes de routage (cf. optimisation).
  B. Intégration DecisionTreeBuilder.analyze_entity :
     - rétro-compatibilité stricte sans TFIDF ;
     - rescue par synonymie (He bas mais tfidf_score haut -> RULES) ;
     - modulation par R.
"""

from __future__ import annotations

import csv
import sys
from unittest.mock import MagicMock

import pytest

# eco2ai is an optional heavy dep pulled in by E_creation_arbre_decision.
sys.modules.setdefault("eco2ai", MagicMock())

from demne.E_creation_arbre_decision import DecisionTreeBuilder
from demne.E_tfidf_coverage import (
    collect_mentions,
    compute_coverage_for_all_entities,
    coverage_f1_by_x,
    run,
)


# =========================================================================== #
# BLOC A — coverage_f1_by_x                                                    #
# =========================================================================== #
def test_less_than_two_entities_returns_empty():
    assert coverage_f1_by_x({"A": ["foo", "bar"]}) == {}


def test_keyword_covers_all_mentions_high_f1():
    # "her2" couvre toutes les mentions de E et n'apparaît pas ailleurs → f1 ~1
    ment = {
        "E": ["her2 positif", "her2 score 3", "her2 amplifie"],
        "Other": ["ki67 10%", "ki67 fort"],
    }
    out = coverage_f1_by_x(ment, max_x=3)
    assert out["E"][1] > 0.9


def test_generic_word_lowers_precision():
    # "positif" présent chez E ET Other → recall ok mais précision basse → f1 < 1
    ment = {"E": ["positif", "positif", "positif"], "Other": ["positif", "positif"]}
    out = coverage_f1_by_x(ment, max_x=1)
    assert 0.0 < out["E"][1] < 1.0


def test_keys_and_range(tmp_path=None):
    ment = {"E": ["a b", "a c", "a d"], "F": ["x y", "x z"]}
    out = coverage_f1_by_x(ment, max_x=5)
    assert set(out["E"].keys()) == set(range(1, 6))
    assert all(0.0 <= v <= 1.0 for v in out["E"].values())


# =========================================================================== #
# BLOC A — collect_mentions + API corpus                                      #
# =========================================================================== #
def _write_brat(d, name, lines):
    (d / f"{name}.ann").write_text("\n".join(lines), encoding="utf-8")
    (d / f"{name}.txt").write_text("dummy", encoding="utf-8")


def test_collect_mentions_reads_ann(tmp_path):
    _write_brat(
        tmp_path, "doc1",
        ["T1\tHER2 10 14\tHER2 positif", "T2\tKi67 20 24\tKi 67 10%", "A1\tattr T1 x"],
    )
    ment = collect_mentions(tmp_path)
    assert ment["HER2"] == ["HER2 positif"]
    assert ment["Ki67"] == ["Ki 67 10%"]
    assert "A1" not in ment  # lignes d'attribut ignorées


def test_compute_and_run_write_csv(tmp_path):
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    _write_brat(
        corpus, "d",
        ["T1\tHER2 0 4\ther2 a", "T2\tHER2 5 9\ther2 b", "T3\tKi 10 12\tki 1", "T4\tKi 13 15\tki 2"],
    )
    res = compute_coverage_for_all_entities(corpus, top_x=2)
    assert "HER2" in res and "f1_score" in res["HER2"]
    assert res["HER2"]["f1_score"] == res["HER2"]["tfidf_score"]

    out = tmp_path / "out"
    run(corpus, out, top_x=2, y=0.5)
    csv_path = out / "tfidf_analysis.csv"
    assert csv_path.exists()
    rows = list(csv.DictReader(open(csv_path, encoding="utf-8")))
    assert {r["Entity"] for r in rows} == {"HER2", "Ki"}
    assert all("TFIDF_Score" in r for r in rows)


def test_run_empty_corpus_no_crash(tmp_path):
    run(tmp_path, tmp_path / "out")  # aucun .ann → pas de crash, pas de CSV
    assert not (tmp_path / "out" / "tfidf_analysis.csv").exists()


# =========================================================================== #
# BLOC B — intégration DecisionTreeBuilder.analyze_entity                      #
# =========================================================================== #
@pytest.fixture
def builder():
    return DecisionTreeBuilder(config_path="dummy.json")


def test_backward_compat_baseline_unchanged(builder):
    """Sans TFIDF l'arbre doit matcher le baseline. Nœud R- partagé :
    Te+He haut + R haut → Feas (TBM), pas LLM.
    """
    assert (
        builder.analyze_entity("StructureOnly", {"Te": 90.0, "Te_count": 20, "He": 80.0, "R": 0.1})[
            "method"
        ]
        == "RULES"
    )
    assert (
        builder.analyze_entity(
            "GoodRules", {"Te": 50.0, "Te_count": 20, "He": 20.0, "R": 0.1, "Feas": 0.8}
        )["method"]
        == "TBM"
    )
    assert (
        builder.analyze_entity(
            "CommonEntity", {"Te": 10.0, "Te_count": 20, "He": 10.0, "R": 0.2, "Feas": 0.1}
        )["method"]
        == "LLM"
    )
    assert (
        builder.analyze_entity(
            "RiskyEntity", {"Te": 90.0, "Te_count": 20, "He": 80.0, "R": 0.8, "Feas": 0.8}
        )["method"]
        == "TBM"
    )


def test_tfidf_node_bypassed_when_te_and_he_high(builder):
    res = builder.analyze_entity(
        "x", {"Te": 90.0, "Te_count": 20, "He": 80.0, "R": 0.1, "tfidf_score": 0.95}
    )
    assert res["method"] == "RULES"
    assert "TFIDF" not in res["justification"]


def test_tfidf_rescue_low_he_r_low_routes_rules(builder):
    metrics = {"Te": 5.0, "Te_count": 20, "He": 10.0, "R": 0.1, "Feas": 0.8, "tfidf_score": 0.95}
    res = builder.analyze_entity("évolution_tumorale", metrics)
    assert res["method"] == "RULES"
    assert "TFIDF" in res["justification"]


def test_tfidf_rescue_low_he_r_high_falls_to_feas(builder):
    metrics = {"Te": 5.0, "Te_count": 20, "He": 10.0, "R": 0.9, "Feas": 0.8, "tfidf_score": 0.95}
    assert builder.analyze_entity("x", metrics)["method"] == "TBM"


def test_tfidf_below_y_falls_to_feas(builder):
    metrics = {"Te": 5.0, "Te_count": 20, "He": 10.0, "R": 0.9, "Feas": 0.8, "tfidf_score": 0.30}
    assert builder.analyze_entity("x", metrics)["method"] == "TBM"


def test_tfidf_absent_falls_to_feas_directly(builder):
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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
