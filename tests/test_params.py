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


def test_presets_follow_calibrated_thresholds(tmp_path):
    d = tmp_path / "data"
    d.mkdir()
    (d / "demne_params.json").write_text(
        json.dumps(
            {
                "decision_thresholds": {
                    "TE_HIGH": 0.3,
                    "HE_HIGH": 0.6,
                    "R_HIGH": 0.5,
                    "FEAS_NER": 0.4,
                },
                "preset_margin": 0.1,
                # stale stored presets must be ignored
                "presets": {"FRUGAL": {"Te": 0.9, "He": 0.9, "R": 0.0, "Feas": 0.9}},
            }
        ),
        encoding="utf-8",
    )
    pr = load_params(start=tmp_path / "params.py")["presets"]
    assert pr["CALIBRATED"] == {"Te": 0.3, "He": 0.6, "R": 0.5, "Feas": 0.4}
    # FRUGAL relaxes every gate toward the cheaper tier, QUALITY tightens it
    assert pr["FRUGAL"] == {"Te": 0.2, "He": 0.5, "R": 0.6, "Feas": 0.3}
    assert pr["QUALITY"] == {"Te": 0.4, "He": 0.7, "R": 0.4, "Feas": 0.5}


def test_corrupt_json_warns_and_falls_back(tmp_path):
    d = tmp_path / "data"
    d.mkdir()
    (d / "demne_params.json").write_text("{ not valid json", encoding="utf-8")
    with pytest.warns(RuntimeWarning):
        p = load_params(start=tmp_path / "params.py")
    assert p["decision_thresholds"]["TE_HIGH"] == DEFAULTS["decision_thresholds"]["TE_HIGH"]


def test_has_contradiction_needs_both_polarities():
    from demne.params import has_contradiction

    assert has_contradiction(["her2 positif", "her2 négatif"])
    assert not has_contradiction(["her2 positif", "er positif"])
    assert not has_contradiction([])
