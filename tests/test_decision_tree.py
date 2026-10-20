import pytest

from demne.decision_tree_builder import DecisionTreeBuilder


def test_decision_tree_logic():
    builder = DecisionTreeBuilder(config_path="dummy.json")

    metrics_struct = {"Te": 90.0, "Te_count": 20, "He": 80.0, "R": 0.1}
    res1 = builder.analyze_entity("StructureOnly", metrics_struct)
    assert res1["method"] == "RULES"

    metrics_tbm = {"Te": 50.0, "Te_count": 20, "He": 20.0, "R": 0.1, "Feas": 0.8}
    res2 = builder.analyze_entity("GoodRules", metrics_tbm)
    assert res2["method"] == "TBM"

    metrics_llm = {"Te": 10.0, "Te_count": 20, "He": 10.0, "R": 0.2, "Feas": 0.1}
    res3 = builder.analyze_entity("CommonEntity", metrics_llm)
    assert res3["method"] == "LLM"

    metrics_risky = {"Te": 90.0, "Te_count": 20, "He": 80.0, "R": 0.9, "Feas": 0.8}
    res4 = builder.analyze_entity("RiskyEntity", metrics_risky)
    assert res4["method"] == "TBM"

    metrics_low_count = {"Te": 90.0, "Te_count": 1, "He": 80.0, "R": 0.1, "Feas": 0.8}
    res5 = builder.analyze_entity("RareRisky", metrics_low_count)
    assert res5["method"] == "TBM"


def test_visualize_decision_tree_writes_layered_figure(tmp_path):
    import json

    import matplotlib

    matplotlib.use("Agg")
    from demne.visualize_decision_tree import visualize_decision_tree

    cfg = tmp_path / "decision_config.json"
    cfg.write_text(
        json.dumps(
            {
                "entities": {
                    "Ki67": {"method": "RULES", "metrics": {"Te": 0.9}},
                    "Histologie": {"method": "LLM", "metrics": {"Te": 0.1}},
                }
            }
        ),
        encoding="utf-8",
    )
    out = tmp_path / "figs" / "tree.png"
    visualize_decision_tree(str(cfg), str(out))
    assert out.exists() and out.stat().st_size > 0


def test_visualize_decision_tree_missing_config_is_a_noop(tmp_path, capsys):
    from demne.visualize_decision_tree import visualize_decision_tree

    visualize_decision_tree(str(tmp_path / "nope.json"), str(tmp_path / "x.png"))
    assert "not found" in capsys.readouterr().out
    assert not (tmp_path / "x.png").exists()


# ---------------------------------------------------------------------------
# Config build / save / report, CSV loading, stability check
# ---------------------------------------------------------------------------
def _metrics_for_five_entities():
    return {
        f"E{i}": {"Te": te, "Te_count": 20, "He": he, "R": 0.1, "Feas": feas}
        for i, (te, he, feas) in enumerate(
            [(0.9, 0.9, 0.8), (0.5, 0.2, 0.8), (0.1, 0.1, 0.1), (0.8, 0.7, 0.6), (0.3, 0.3, 0.9)]
        )
    }


def test_build_save_and_report_round_trip(tmp_path):
    import json

    builder = DecisionTreeBuilder(config_path=tmp_path / "decision_config.json")
    cfg = builder.build_full_config(_metrics_for_five_entities())
    assert set(cfg["entities"]) == {f"E{i}" for i in range(5)}
    assert cfg["global_thresholds"] == builder.THRESHOLDS

    builder.save_config()
    saved = json.loads((tmp_path / "decision_config.json").read_text(encoding="utf-8"))
    assert saved["entities"]["E0"]["method"] == cfg["entities"]["E0"]["method"]

    builder.export_text_report(tmp_path / "report.txt")
    report = (tmp_path / "report.txt").read_text(encoding="utf-8")
    assert "## E0" in report and "Global Thresholds" in report


def test_load_metrics_from_csv_reads_every_metric_file(tmp_path):
    from demne.decision_tree_builder import load_metrics_from_csv

    (tmp_path / "templatability_analysis_c.json").write_text(
        '{"Ki67": {"templatability_score": 0.8, "count": 12}}', encoding="utf-8"
    )
    (tmp_path / "homogeneity_analysis_c.csv").write_text("Entity,He\nKi67,0.7\n", encoding="utf-8")
    (tmp_path / "risk_context_analysis.csv").write_text(
        "Entity,R_Score\nKi67,0.2\n", encoding="utf-8"
    )  # no corpus suffix -> falls back to the generic name
    (tmp_path / "frequency_analysis_c.csv").write_text(
        "Entity,Frequency\nKi67,0.01\n", encoding="utf-8"
    )
    (tmp_path / "ner_feasibility_analysis_c.csv").write_text(
        "Entity,Feas_Score\nKi67,0.5\n", encoding="utf-8"
    )
    (tmp_path / "tfidf_analysis.csv").write_text("Entity,TFIDF_Score\nKi67,0.9\n", encoding="utf-8")

    m = load_metrics_from_csv(tmp_path, "c")["Ki67"]
    assert m["Te"] == 0.8 and m["Te_count"] == 12
    assert (m["He"], m["R"], m["Freq"], m["Feas"], m["tfidf_score"]) == (0.7, 0.2, 0.01, 0.5, 0.9)


def test_load_metrics_legacy_percent_homogeneity(tmp_path):
    from demne.decision_tree_builder import load_metrics_from_csv

    (tmp_path / "homogeneity_analysis.csv").write_text(
        "Entity,He_Score_Percent\nKi67,70\n", encoding="utf-8"
    )
    assert load_metrics_from_csv(tmp_path)["Ki67"]["He"] == pytest.approx(0.7)


def test_threshold_stability_check_restores_thresholds(capsys):
    builder = DecisionTreeBuilder(config_path="dummy.json")
    before = dict(builder.THRESHOLDS)
    builder.threshold_stability_check(_metrics_for_five_entities(), k=2, seed=1)
    assert "stability" in capsys.readouterr().out
    assert builder.THRESHOLDS == before


def test_threshold_stability_check_needs_enough_entities(capsys):
    builder = DecisionTreeBuilder(config_path="dummy.json")
    builder.threshold_stability_check({"A": {"Te": 0.5}}, k=3)
    assert "Not enough entities" in capsys.readouterr().out
