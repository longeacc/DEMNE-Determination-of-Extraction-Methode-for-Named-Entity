"""Unit tests for demne.params (single source of truth for thresholds)."""

import json

import pytest

from demne.params import DEFAULTS, load_params


def test_defaults_when_no_file(tmp_path):
    p = load_params(start=tmp_path / "params.py")
    assert p["decision_thresholds"]["TE_HIGH"] == DEFAULTS["decision_thresholds"]["TE_HIGH"]


def test_user_values_merge_over_defaults(tmp_path):
    d = tmp_path / "data"
    d.mkdir()
    (d / "demne_params.json").write_text(
        json.dumps({"decision_thresholds": {"TE_HIGH": 0.99}}), encoding="utf-8"
    )
    p = load_params(start=tmp_path / "params.py")
    assert p["decision_thresholds"]["TE_HIGH"] == 0.99
    # untouched keys keep their default
    assert p["decision_thresholds"]["HE_HIGH"] == DEFAULTS["decision_thresholds"]["HE_HIGH"]


def test_corrupt_json_warns_and_falls_back(tmp_path):
    d = tmp_path / "data"
    d.mkdir()
    (d / "demne_params.json").write_text("{ not valid json", encoding="utf-8")
    with pytest.warns(RuntimeWarning):
        p = load_params(start=tmp_path / "params.py")
    assert p["decision_thresholds"]["TE_HIGH"] == DEFAULTS["decision_thresholds"]["TE_HIGH"]
