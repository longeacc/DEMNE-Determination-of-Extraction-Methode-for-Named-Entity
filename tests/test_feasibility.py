"""Unit test for the NER feasibility scorer (demne.E_feasibility_NER).

Exercises the real file-based pipeline against a temporary Results directory,
so the Feas = alpha*min(1, count/sat) + beta*He formula (and the He-default
branch) are actually covered.
"""

import csv

from demne.E_feasibility_NER import compute_feasibility
from demne.params import load_params


def test_feasibility_formula(tmp_path):
    (tmp_path / "frequency_analysis.csv").write_text(
        "Entity,Frequency,Count\nAAA,0.5,300\nBBB,0.01,0\n", encoding="utf-8"
    )
    # BBB is deliberately absent here -> must fall back to he_default
    (tmp_path / "homogeneity_analysis.csv").write_text("Entity,He\nAAA,0.8\n", encoding="utf-8")

    compute_feasibility(gs_dir_str=None, pred_dir_str=None, results_dir=tmp_path)

    out = tmp_path / "ner_feasibility_analysis_Breast.csv"
    assert out.exists()
    with open(out, encoding="utf-8") as f:
        got = {r["Entity"]: float(r["Feas_Score"]) for r in csv.DictReader(f)}

    fw = load_params()["feasibility_weights"]
    a, b, he_default = fw["alpha_freq"], fw["beta_he"], fw["he_default"]
    sat = fw.get("sat_count", 300)

    assert got["AAA"] == round(a * min(1.0, 300 / sat) + b * 0.8, 3)
    assert got["BBB"] == round(a * min(1.0, 0 / sat) + b * he_default, 3)
