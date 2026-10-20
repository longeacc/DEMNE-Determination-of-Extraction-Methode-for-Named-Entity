"""Guards the single-source routing graph (demne.routing.route).

The CLI (demne.DecisionTree.analyze_entity) and the dashboard
(dashboard.core.routing.compute_routing) must now BOTH delegate to the same
graph, so they can never silently diverge. This replaces the old manual
"drift-guard" comment with an executable check.
"""

import importlib.util as _il
import itertools
from pathlib import Path

import pytest

from demne import DecisionTree  # noqa: E402
from demne.params import load_params  # noqa: E402
from demne.routing import route  # noqa: E402

# Load the dashboard routing module in isolation (no Streamlit): it depends only
# on `demne`, so we exercise the real dashboard adapter without the UI stack.
_rp = Path(__file__).resolve().parents[1] / "dashboard" / "core" / "routing.py"
_spec = _il.spec_from_file_location("dash_routing", _rp)
_dash = _il.module_from_spec(_spec)
_spec.loader.exec_module(_dash)
compute_routing = _dash.compute_routing

_DT = load_params()["decision_thresholds"]
_THRESH = {
    "Te": _DT["TE_HIGH"],
    "He": _DT["HE_HIGH"],
    "R": _DT["R_HIGH"],
    "Feas": _DT["FEAS_NER"],
    "Y": _DT.get("TFIDF_Y", 0.70),
}


def test_cli_and_dashboard_route_identically():
    builder = DecisionTree(config_path="dummy.json")
    grid = [0.0, 0.05, 0.1, 0.2, 0.5, 0.85, 0.9, 1.0]
    rs = [0.0, 0.25, 0.9]
    fs = [0.0, 0.2, 0.8]
    tfs = [None, 0.0, 0.5, 0.7, 0.95]
    n = 0
    for te, he, r, feas, tf in itertools.product(grid, grid, rs, fs, tfs):
        metrics = {"Te": te, "He": he, "R": r, "Feas": feas, "Te_count": 99}
        if tf is not None:
            metrics["tfidf_score"] = tf
        cli = builder.analyze_entity("X", metrics)["method"]
        dash = compute_routing(metrics, _THRESH)[0]
        assert cli == dash, (te, he, r, feas, tf, cli, dash)
        n += 1
    assert n > 1000


def test_route_is_pure_and_deterministic():
    kw = dict(
        te=0.9,
        he=0.9,
        r=0.1,
        feas=0.8,
        tfidf_score=0.95,
        te_high=0.1,
        he_high=0.85,
        r_high=0.25,
        feas_ner=0.2,
        tfidf_y=0.7,
    )
    a = route(**kw)
    b = route(**kw)
    assert a["method"] == b["method"] == "RULES"
    assert a["trace"] == b["trace"]


def test_dashboard_tuple_is_the_detail_dict():
    """The tuple API is exactly (method, justification) of the CLI-shaped dict."""
    metrics = {"Te": 0.9, "He": 0.9, "R": 0.1, "Feas": 0.8}
    detail = _dash.compute_routing_detail(metrics, _THRESH)
    assert set(detail) == {"method", "justification", "trace"}
    assert compute_routing(metrics, _THRESH) == (detail["method"], detail["justification"])


# ---------------------------------------------------------------------------
# Dashboard metrics: parity with the CLI scorers, contradiction term, entities
# ---------------------------------------------------------------------------
def _load_dashboard_core():
    import sys

    name = "_test_dash_core"
    if name not in sys.modules:
        core = Path(__file__).resolve().parents[1] / "dashboard" / "core"
        spec = _il.spec_from_file_location(
            name, core / "__init__.py", submodule_search_locations=[str(core)]
        )
        mod = _il.module_from_spec(spec)
        sys.modules[name] = mod
        spec.loader.exec_module(mod)
    return sys.modules[name]


def test_dashboard_risk_equals_cli_risk_including_contradiction(tmp_path):
    from demne.risk_context import RiskContextScorer

    (tmp_path / "d1.txt").write_text("her2 positif puis her2 negatif", encoding="utf-8")
    (tmp_path / "d1.ann").write_text(
        "T1\tHER2 0 12\ther2 positif\nT2\tHER2 18 30\ther2 negatif\n", encoding="utf-8"
    )
    (tmp_path / "d2.txt").write_text("er positif", encoding="utf-8")
    (tmp_path / "d2.ann").write_text("T1\tHER2 0 10\ther2 positif\n", encoding="utf-8")

    cli_r = {r["Entity"]: r["R_Score"] for r in RiskContextScorer([tmp_path]).compute_all()}
    core = _load_dashboard_core()
    docs = core.BratCorpusParser().parse_directory(str(tmp_path))
    dash_r = core.MetricsCalculator().compute_all_metrics(docs, "HER2")["R"]
    assert dash_r == pytest.approx(cli_r["HER2"], abs=1e-4)
    assert dash_r > 0  # one of the two documents contradicts itself


def test_default_entities_follow_the_last_decision_config(tmp_path, monkeypatch):
    import json

    metrics = _load_dashboard_core().metrics
    cfg = tmp_path / "decision_config.json"
    cfg.write_text(json.dumps({"entities": {"A": {}, "B": {}}}), encoding="utf-8")
    monkeypatch.setattr(metrics, "_DECISION_CONFIG", cfg)
    assert metrics.default_entities() == ["A", "B"]
    monkeypatch.setattr(metrics, "_DECISION_CONFIG", tmp_path / "missing.json")
    assert metrics.default_entities() == list(metrics.DEMO_METRICS)
