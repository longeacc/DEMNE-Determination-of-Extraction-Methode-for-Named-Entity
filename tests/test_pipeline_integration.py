"""
End-to-end integration tests for the full DEMNE pipeline.

What is tested here?
--------------------
Each metric scorer (Frequency, Templatability, Homogeneity, Risk) runs on the
same synthetic BRAT corpus, its output feeds DecisionTreeBuilder, and the final
routing decision is verified.  This catches regressions that unit tests miss:
interface breakage between scorers, threshold drift, and wrong metric wiring.

Structure
---------
  BLOC 1 — Individual scorers on the shared BRAT corpus (smoke + sanity)
  BLOC 2 — Full pipeline: scorers → metrics dict → DecisionTreeBuilder
  BLOC 3 — Threshold sensitivity (kfold stability helper)

All fixtures come from conftest.py (no imports needed).
"""

# eco2ai is mocked by the session-scoped `eco2ai_mock` fixture in conftest.py
# before any demne module is imported.

import csv

import pytest

from demne.decision_tree_builder import DecisionTreeBuilder
from demne.frequency import FrequencyScorer
from demne.homogeneity import HomogeneityScorer
from demne.risk_context import RiskContextScorer
from demne.templatability import TemplatabilityScorer

# ===========================================================================
# BLOC 1 — Individual scorers on the shared corpus
# ===========================================================================


class TestFrequencyScorerOnCorpus:
    """FrequencyScorer ingests BRAT files from disk via compute_all()."""

    def test_discovers_all_entity_types(self, brat_corpus_dir):
        scorer = FrequencyScorer([brat_corpus_dir])
        scorer.compute_all()
        assert "HER2_Status" in scorer.entity_counts
        assert "ER_Status" in scorer.entity_counts
        assert "PR_Status" in scorer.entity_counts

    def test_counts_match_annotations(self, brat_corpus_dir):
        # doc_her2: HER2×3 / doc_er: ER×2 PR×1 / doc_risk: HER2×1 PR×1 ER×1
        # → HER2_Status=4, ER_Status=3, PR_Status=2
        scorer = FrequencyScorer([brat_corpus_dir])
        scorer.compute_all()
        assert scorer.entity_counts["HER2_Status"] == 4
        assert scorer.entity_counts["ER_Status"] == 3
        assert scorer.entity_counts["PR_Status"] == 2

    def test_total_tokens_positive(self, brat_corpus_dir):
        scorer = FrequencyScorer([brat_corpus_dir])
        scorer.compute_all()
        assert scorer.total_tokens > 0

    def test_frequencies_are_in_unit_interval(self, brat_corpus_dir):
        scorer = FrequencyScorer([brat_corpus_dir])
        scorer.compute_all()
        for row in scorer.get_stats():
            assert 0.0 <= row["Frequency"] <= 1.0

    def test_to_csv_writes_readable_file(self, brat_corpus_dir, tmp_path):
        scorer = FrequencyScorer([brat_corpus_dir])
        scorer.compute_all()
        out = tmp_path / "freq.csv"
        scorer.to_csv(out)
        with open(out, encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        entities = {r["Entity"] for r in rows}
        assert "HER2_Status" in entities


class TestTemplatabilityOnCorpus:
    """
    TemplatabilityScorer uses the in-memory corpus (list-of-dicts).

    HER2_Status mentions are "3+", "positif", "positif" — two unique patterns
    after normalisation → moderate Te.  ER_Status has "positif" × 2 → high Te.
    """

    def test_computes_all_entities(self, brat_corpus):
        scorer = TemplatabilityScorer(brat_corpus)
        scores = scorer.compute_all()
        assert "HER2_Status" in scores
        assert "ER_Status" in scores

    def test_scores_in_valid_range(self, brat_corpus):
        scorer = TemplatabilityScorer(brat_corpus)
        scores = scorer.compute_all()
        for etype, score in scores.items():
            assert 0.0 <= score <= 100.0, f"{etype}: Te={score} out of [0,100]"

    def test_er_status_higher_than_her2(self, brat_corpus):
        # ER_Status: "positif"×2 → single pattern → higher Te
        # HER2_Status: "3+", "positif"×2 → two patterns → lower Te
        scorer = TemplatabilityScorer(brat_corpus)
        scores = scorer.compute_all()
        assert scores["ER_Status"] >= scores["HER2_Status"]


class TestHomogeneityOnCorpus:
    def test_computes_all_entities(self, brat_corpus):
        scorer = HomogeneityScorer(brat_corpus)
        scores = scorer.compute_all()
        assert "ER_Status" in scores
        assert "HER2_Status" in scores

    def test_er_status_high_he(self, brat_corpus):
        # ER_Status mentions are all "positif" → near-perfect redundancy
        scorer = HomogeneityScorer(brat_corpus)
        scores = scorer.compute_all()
        assert scores["ER_Status"] > 0.5

    def test_scores_in_unit_interval(self, brat_corpus):
        scorer = HomogeneityScorer(brat_corpus)
        for etype, score in scorer.compute_all().items():
            assert 0.0 <= score <= 1.0, f"{etype}: He={score} out of [0,1]"


class TestRiskContextOnCorpus:
    """
    RiskContextScorer loads files from disk and extracts context windows.

    doc_risk.txt contains "possible", "à confirmer", "probable" → HER2_Status
    and ER_Status should get elevated R scores.  doc_her2 and doc_er have no
    uncertainty language → their R scores should be lower.
    """

    def test_computes_without_crash(self, brat_corpus_dir):
        scorer = RiskContextScorer([brat_corpus_dir])
        results = scorer.compute_all()
        assert isinstance(results, list)

    def test_her2_has_elevated_risk(self, brat_corpus_dir):
        # doc_risk.txt annotates HER2_Status with "amplifié" in an uncertain context
        scorer = RiskContextScorer([brat_corpus_dir])
        results = scorer.compute_all()
        her2 = next((r for r in results if r["Entity"] == "HER2_Status"), None)
        assert her2 is not None, "HER2_Status missing from risk results"
        # At least one mention sits in an uncertain context → R > 0
        assert her2["R_Score"] > 0.0

    def test_result_fields_present(self, brat_corpus_dir):
        scorer = RiskContextScorer([brat_corpus_dir])
        for row in scorer.compute_all():
            assert "Entity" in row
            assert "R_Score" in row
            assert 0.0 <= row["R_Score"] <= 1.0


# ===========================================================================
# BLOC 2 — Full pipeline: metrics dict → DecisionTreeBuilder
# ===========================================================================


class TestFullPipelineRouting:
    """
    Verify that the four routing outcomes all fire correctly when the metric
    dict is built from synthetic values (fastest path — no scorer I/O).

    The `synthetic_metrics` fixture (conftest.py) provides a dict that covers
    every branch of the DEMNE decision graph.
    """

    @pytest.fixture
    def builder(self):
        # DecisionTreeBuilder loads thresholds from params.json at __init__
        return DecisionTreeBuilder(config_path="dummy.json")

    def test_her2_routes_to_rules(self, builder, synthetic_metrics):
        res = builder.analyze_entity("HER2_Status", synthetic_metrics["HER2_Status"])
        assert res["method"] == "RULES"

    def test_pays_routes_to_tbm(self, builder, synthetic_metrics):
        res = builder.analyze_entity("Pays_Origine", synthetic_metrics["Pays_Origine"])
        assert res["method"] == "TBM"

    def test_social_routes_to_llm(self, builder, synthetic_metrics):
        res = builder.analyze_entity("Contexte_Social", synthetic_metrics["Contexte_Social"])
        assert res["method"] == "LLM"

    def test_tfidf_rescue_routes_to_rules(self, builder, synthetic_metrics):
        res = builder.analyze_entity("Evolution_Tumorale", synthetic_metrics["Evolution_Tumorale"])
        assert res["method"] == "RULES"
        assert "TFIDF" in res["justification"]

    def test_build_full_config_covers_all_entities(self, builder, synthetic_metrics):
        config = builder.build_full_config(synthetic_metrics)
        assert set(config["entities"].keys()) == set(synthetic_metrics.keys())

    def test_all_decisions_have_required_keys(self, builder, synthetic_metrics):
        config = builder.build_full_config(synthetic_metrics)
        for ename, data in config["entities"].items():
            assert "method" in data, f"{ename}: missing 'method'"
            assert "justification" in data, f"{ename}: missing 'justification'"
            assert "trace" in data, f"{ename}: missing 'trace'"
            assert data["method"] in {
                "RULES",
                "TBM",
                "LLM",
            }, f"{ename}: unexpected method '{data['method']}'"

    def test_save_config_writes_json(self, builder, synthetic_metrics, tmp_path):
        out = tmp_path / "decision_config.json"
        builder.config_path = out
        builder.build_full_config(synthetic_metrics)
        builder.save_config()
        assert out.exists()
        import json

        with open(out, encoding="utf-8") as f:
            data = json.load(f)
        assert "entities" in data
        assert "global_thresholds" in data


# ===========================================================================
# BLOC 3 — Pipeline with real scorers feeding the decision tree
# ===========================================================================


class TestPipelineEndToEnd:
    """
    Closest thing to a real run: scorers ingest BRAT files, produce metric
    dicts, those feed DecisionTreeBuilder, and final decisions are verified.

    This is the test that would have caught the 'pycrfsuite' removal regression
    and the TF-IDF routing change simultaneously.
    """

    def test_full_run_produces_decisions_for_known_entities(self, brat_corpus_dir, brat_corpus):
        # --- Step 1: Frequency ---
        freq_scorer = FrequencyScorer([brat_corpus_dir])
        freq_scorer.compute_all()
        freq_stats = {r["Entity"]: r["Frequency"] for r in freq_scorer.get_stats()}

        # --- Step 2: Templatability ---
        te_scorer = TemplatabilityScorer(brat_corpus)
        te_scores = te_scorer.compute_all()

        # --- Step 3: Homogeneity ---
        he_scorer = HomogeneityScorer(brat_corpus)
        he_scores = he_scorer.compute_all()

        # --- Step 4: Risk ---
        risk_scorer = RiskContextScorer([brat_corpus_dir])
        risk_results = {r["Entity"]: r["R_Score"] for r in risk_scorer.compute_all()}

        # --- Step 5: Assemble metrics ---
        # Union of all entity names seen by any scorer
        all_entities = set(freq_stats) | set(te_scores) | set(he_scores) | set(risk_results)
        metrics_db = {}
        for ent in all_entities:
            metrics_db[ent] = {
                "Te": te_scores.get(ent, 0.0),
                "Te_count": freq_scorer.entity_counts.get(ent, 0),
                "He": he_scores.get(ent, 0.0),
                "R": risk_results.get(ent, 0.0),
                "Feas": 0.5,  # neutral placeholder (no feasibility scorer in unit tests)
            }

        # --- Step 6: Decision tree ---
        builder = DecisionTreeBuilder(config_path="dummy.json")
        config = builder.build_full_config(metrics_db)

        # Every entity that entered must produce a decision
        assert set(config["entities"].keys()) == all_entities

        # Every decision must be a valid method
        for ent, data in config["entities"].items():
            assert data["method"] in {
                "RULES",
                "TBM",
                "LLM",
            }, f"{ent}: unexpected method '{data['method']}'"

    def test_export_text_report(self, brat_corpus_dir, brat_corpus, tmp_path):
        freq_scorer = FrequencyScorer([brat_corpus_dir])
        freq_scorer.compute_all()
        te_scorer = TemplatabilityScorer(brat_corpus)
        he_scorer = HomogeneityScorer(brat_corpus)

        metrics_db = {
            ent: {
                "Te": te_scorer.compute_all().get(ent, 0.0),
                "Te_count": freq_scorer.entity_counts.get(ent, 0),
                "He": he_scorer.compute_all().get(ent, 0.0),
                "R": 0.1,
                "Feas": 0.5,
            }
            for ent in te_scorer.compute_all()
        }

        builder = DecisionTreeBuilder(config_path=tmp_path / "cfg.json")
        builder.build_full_config(metrics_db)
        builder.save_config()

        report = tmp_path / "report.txt"
        builder.export_text_report(report)

        content = report.read_text(encoding="utf-8")
        assert "DEMNE Decision Tree Report" in content
        assert "Global Thresholds" in content
        # At least one entity section must appear
        assert any(ent in content for ent in metrics_db)


# ===========================================================================
# BLOC 4 — main.py command line (sub-commands, removed commands, error paths)
# ===========================================================================
def _load_main():
    import importlib.util
    from pathlib import Path

    path = Path(__file__).resolve().parents[1] / "main.py"
    spec = importlib.util.spec_from_file_location("demne_cli_main", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def cli(monkeypatch, tmp_path):
    """main.py with its config/results paths redirected to a temp folder."""
    main_mod = _load_main()
    monkeypatch.setattr(main_mod, "CONFIG_PATH", tmp_path / "decision_config.json")
    monkeypatch.setattr(main_mod, "RESULTS_DIR", tmp_path / "Results")

    def run(*argv):
        monkeypatch.setattr("sys.argv", ["main.py", *argv])
        main_mod.main()

    return run


@pytest.fixture
def decision_config(tmp_path):
    builder = DecisionTreeBuilder(config_path=tmp_path / "decision_config.json")
    builder.build_full_config(
        {
            "Ki67": {"Te": 0.9, "He": 0.9, "R": 0.1, "Feas": 0.8, "Te_count": 20},
            "Histologie": {"Te": 0.1, "He": 0.1, "R": 0.1, "Feas": 0.1, "Te_count": 20},
        }
    )
    builder.save_config()
    return tmp_path / "decision_config.json"


def test_cli_info_prints_the_package_version(cli, capsys):
    import demne

    cli("info")
    assert f"DEMNE version: {demne.__version__}" in capsys.readouterr().out


def test_cli_corpus_lists_entities(cli, capsys, brat_corpus_dir):
    cli("corpus", "--path", str(brat_corpus_dir))
    out = capsys.readouterr().out
    assert "3 documents loaded" in out and "HER2_Status" in out


def test_cli_removed_commands_are_rejected(cli):
    for cmd in ("tree", "export-csv", "rest"):
        with pytest.raises(SystemExit):
            cli(cmd)
    with pytest.raises(SystemExit):
        cli("rest-config", "--import", "x.json")


def test_cli_rest_config_export(cli, decision_config, tmp_path):
    import json

    out = tmp_path / "exported.json"
    cli("rest-config", "--export", str(out))
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["routings"] == {"Ki67": "RULES", "Histologie": "LLM"}


def test_cli_dashboard_presets_are_ordered(cli, decision_config, capsys):
    """FRUGAL never routes to a costlier tier than CALIBRATED, QUALITY never to a cheaper one."""
    rank = {"RULES": 0, "TBM": 1, "LLM": 2}
    methods = {}
    for preset in ("CALIBRATED", "FRUGAL", "QUALITY"):
        cli("dashboard", "--preset", preset)
        lines = [ln for ln in capsys.readouterr().out.splitlines() if "|" in ln][1:]
        methods[preset] = [rank[ln.split("|")[7].strip()] for ln in lines]
    assert all(f <= c for f, c in zip(methods["FRUGAL"], methods["CALIBRATED"], strict=True))
    assert all(q >= c for q, c in zip(methods["QUALITY"], methods["CALIBRATED"], strict=True))


def test_cli_dashboard_without_config_explains_what_to_do(cli, capsys):
    cli("dashboard")
    assert "evaluate" in capsys.readouterr().out


def test_cli_optimize_params_requires_a_mode(cli):
    with pytest.raises(SystemExit):
        cli("optimize_params")
    with pytest.raises(SystemExit):
        cli("optimize_params", "--in_domain", "--all_corpora")


def test_cli_optimize_params_explains_missing_corpora(cli, monkeypatch, tmp_path, capsys):
    monkeypatch.chdir(tmp_path)  # no data/*_CORPORA here
    with pytest.raises(SystemExit) as exc:
        cli("optimize_params", "--in_domain")
    assert exc.value.code == 2
    assert "OPT_CORPORA" in capsys.readouterr().err
