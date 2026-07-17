import sys
from unittest.mock import MagicMock

sys.modules["eco2ai"] = MagicMock()

from demne.E_creation_arbre_decision import DecisionTreeBuilder


def test_decision_tree_logic():
    builder = DecisionTreeBuilder(config_path="dummy.json")

    metrics_struct = {"Te": 90.0, "Te_count": 20, "He": 80.0, "R": 0.1}
    res1 = builder.analyze_entity("StructureOnly", metrics_struct)
    assert res1["method"] == "LLM"

    metrics_tbm = {"Te": 50.0, "Te_count": 20, "He": 20.0, "R": 0.1, "Feas": 0.8}
    res2 = builder.analyze_entity("GoodRules", metrics_tbm)
    assert res2["method"] == "TBM"

    metrics_llm = {"Te": 10.0, "Te_count": 20, "He": 10.0, "R": 0.2, "Feas": 0.1}
    res3 = builder.analyze_entity("CommonEntity", metrics_llm)
    assert res3["method"] == "LLM"

    metrics_risky = {"Te": 90.0, "Te_count": 20, "He": 80.0, "R": 0.8, "Feas": 0.8}
    res4 = builder.analyze_entity("RiskyEntity", metrics_risky)
    assert res4["method"] == "TBM"

    metrics_low_count = {"Te": 90.0, "Te_count": 1, "He": 80.0, "R": 0.1, "Feas": 0.8}
    res5 = builder.analyze_entity("RareRisky", metrics_low_count)
    assert res5["method"] == "TBM"
