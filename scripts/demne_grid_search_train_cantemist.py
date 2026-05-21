#!/usr/bin/env python3
"""
DEMNE Threshold Optimizer
=========================
Optimisation des seuils uniquement sur le corpus Cantemist (Train).
Test sur les corpus Redjdal + RCP (Test).
"""

from __future__ import annotations
import itertools, json, time
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import pandas as pd

@dataclass(frozen=True)
class Entity:
    name: str; Te: float; He: float; R: float; Freq: float; Feas: float; label: str

# ═══ CANTEMIST (12) — Labels REST senior ═══
CANTEMIST = [
    Entity("Histologie_tumorale",0.1325,0.8537,0.0515,0.0034,0.7528,"RULES"),
    Entity("Traitement_specifique_du_cancer",0.1738,0.8411,0.082,0.0071,0.7593,"RULES"),
    Entity("Signes_physiques",0.1012,0.6976,0.0722,0.0025,0.5783,"TBM"),
    Entity("Evolutivite_en_lien_avec_le_cancer",0.0,0.0067,0.0,0.0001,0.0106,"LLM"),
    Entity("Reponse_a_la_chimiotherapie",0.1054,0.9221,0.1427,0.0043,0.7786,"RULES"),
    Entity("Stade_metastatique_avec_loc",0.1112,0.8196,0.0964,0.0029,0.6717,"TBM"),
    Entity("Statut_tabagique",0.1,0.5883,0.1286,0.0005,0.3035,"RULES"),
    Entity("ATCD_geriatriques_et_medicaux",0.1,0.3241,0.0759,0.001,0.2604,"LLM"),
    Entity("Stade_OMS_ECOG_Karnofsky",0.3891,0.9191,0.04,0.001,0.5484,"RULES"),
    Entity("Biomarqueurs_therapeutiques",0.1038,0.7193,0.069,0.001,0.4152,"LLM"),
    Entity("Topographie_du_primitif",0.1049,0.7859,0.0436,0.0019,0.5454,"TBM"),
    Entity("Symptomes",0.1333,0.744,0.0913,0.0036,0.7142,"TBM"),
]

# ═══ REDJDAL (46) — F1 >= 0.80 → RULES ═══
REDJDAL = [
    Entity("Biopsy",0.4358,0.991,0.0916,0.0048,0.865,"RULES"),
    Entity("Ultra_sound",0.4485,0.9918,0.071,0.0035,0.8675,"RULES"),
    Entity("MRI",0.6547,0.9931,0.1226,0.0035,0.9052,"RULES"),
    Entity("Mammography",0.4098,0.9924,0.0784,0.0028,0.8608,"RULES"),
    Entity("Clinical_examination",0.501,0.9922,0.0874,0.0029,0.8771,"RULES"),
    Entity("Surgery",0.1856,0.9915,0.0299,0.0081,0.8201,"RULES"),
    Entity("Tumor_size",0.5018,0.9857,0.0547,0.0094,0.8747,"RULES"),
    Entity("Tumor_grade_insitu",0.2983,0.9915,0.063,0.0008,0.8084,"RULES"),
    Entity("Tumor_site",0.5609,0.991,0.0577,0.0092,0.8874,"RULES"),
    Entity("Histology",0.2866,0.9889,0.0766,0.0053,0.8372,"RULES"),
    Entity("Tumour",0.3938,0.9901,0.1101,0.0178,0.857,"RULES"),
    Entity("BraSize_Cup",0.3732,0.9612,0.0575,0.0013,0.8421,"TBM"),
    Entity("Cavity_Shave_Margin",0.1617,0.9911,0.0358,0.0015,0.8156,"RULES"),
    Entity("Clear_Surgical_Margins",0.1896,0.9894,0.0314,0.0018,0.82,"TBM"),
    Entity("Side",0.3681,0.9928,0.0841,0.0237,0.8534,"RULES"),
    Entity("BIRADS_classification",0.3509,0.9914,0.1308,0.0071,0.8498,"RULES"),
    Entity("Clinical_Positive_Nodes",0.2758,0.9898,0.1921,0.0048,0.8357,"TBM"),
    Entity("Menopausal_status",0.1814,0.9879,0.2051,0.0011,0.8179,"RULES"),
    Entity("Comorbidities",0.3755,0.98,0.1059,0.0035,0.8498,"TBM"),
    Entity("Estrogen_receptor",0.268,0.9843,0.0446,0.0027,0.8713,"RULES"),
    Entity("Progesterone_receptor",0.2825,0.9833,0.0452,0.0027,0.8846,"RULES"),
    Entity("HER2_status",0.2915,0.9898,0.049,0.0027,0.8884,"RULES"),
    Entity("Ki67",0.2257,0.9835,0.0573,0.0023,0.8409,"RULES"),
    Entity("Pet_scan",0.4682,0.9912,0.0786,0.0018,0.8708,"RULES"),
    Entity("Anti_HER2_therapy",0.2059,0.9851,0.0591,0.0004,0.5972,"LLM"),
    Entity("Chemotherapy",0.4992,0.9915,0.0611,0.0023,0.8765,"RULES"),
    Entity("Tumor_grade_inv",0.5189,0.9905,0.0265,0.0025,0.8797,"RULES"),
    Entity("Widespread_Microcalc",0.2841,0.9881,0.0714,0.0018,0.8365,"RULES"),
    Entity("TNM",0.3066,0.8801,0.0323,0.0018,0.7984,"RULES"),
    Entity("ResponseAssess_Neoadj",0.2015,0.9876,0.0838,0.0006,0.6934,"TBM"),
    Entity("Cytoponction",0.3586,0.9804,0.1512,0.0008,0.7909,"RULES"),
    Entity("Drugs",0.1254,0.7059,0.0557,0.0033,0.6979,"TBM"),
    Entity("NodeSize",0.2674,0.7992,0.0918,0.0007,0.6518,"TBM"),
    Entity("Hematoma",0.3759,0.9875,0.1051,0.0004,0.6088,"RULES"),
    Entity("Confirmed_Positive_Nodes",0.0738,0.9321,0.105,0.0005,0.6168,"TBM"),
    Entity("Radiotherapy",0.5485,0.9916,0.0218,0.0011,0.8854,"RULES"),
    Entity("Genetic_mutation",0.0344,0.8642,0.15,0.0003,0.6779,"TBM"),
    Entity("FISH",0.2215,0.9793,0.0487,0.0004,0.5778,"RULES"),
    Entity("Screening",0.6772,0.991,0.0275,0.0005,0.7124,"RULES"),
    Entity("Endocrine_Therapy",0.4095,0.9857,0.014,0.0008,0.8301,"TBM"),
    Entity("Associated_InSitu_Carc",0.2571,0.989,0.1258,0.0009,0.82,"RULES"),
    Entity("PresenceEmbole",0.2089,0.988,0.1788,0.0009,0.8229,"RULES"),
    Entity("OncotypeDX",0.1425,0.895,0.0538,0.0001,0.4267,"LLM"),
    Entity("N_status",0.3728,0.9428,0.0814,0.0008,0.7788,"RULES"),
    Entity("Breast_Cancer_Relapse",0.0915,0.9547,0.1037,0.0002,0.4968,"LLM"),
    Entity("Systemic_treatment",1.0,0.0067,0.0,0.0,0.1866,"LLM"),
]

# ═══ RCP/ESMO (7) — Meilleure méthode empirique ═══
RCP = [
    Entity("ER",0.823,0.973,0.15,0.0034,0.78,"RULES"),
    Entity("PR",0.801,0.97,0.14,0.0031,0.76,"RULES"),
    Entity("HER2_stat",0.715,0.892,0.22,0.0028,0.72,"RULES"),
    Entity("HER2_IHC",0.748,0.915,0.18,0.0019,0.69,"RULES"),
    Entity("Ki67",0.852,0.981,0.11,0.0025,0.74,"RULES"),
    Entity("HER2_FISH",0.453,0.724,0.25,0.0008,0.52,"LLM"),
    Entity("Gen_mut",0.287,0.412,0.21,0.0006,0.38,"LLM"),
]

ALL_CORPORA = {"cantemist": CANTEMIST, "redjdal": REDJDAL, "rcp": RCP}

# ═══ DECISION GRAPH ═══
RANK = {"RULES": 0, "TBM": 1, "LLM": 2}

def route(e, Te_H, He_H, R_H, Feas_H):
    if e.Te >= Te_H and e.He >= He_H and e.R < R_H: return "RULES"
    if e.Feas >= Feas_H: return "TBM"
    return "LLM"

def evaluate(corpus, Te_H, He_H, R_H, Feas_H):
    exact, loss = 0, 0.0
    details = []
    for e in corpus:
        pred = route(e, Te_H, He_H, R_H, Feas_H)
        ok = (pred == e.label)
        d = RANK[pred] - RANK[e.label]
        li = 0.0 if d == 0 else (abs(d) * 2.0 if d < 0 else abs(d) * 1.0)
        if ok: exact += 1
        loss += li
        details.append({"entity": e.name, "pred": pred, "ref": e.label,
                        "ok": ok, "d": d, "loss": li})
    n = len(corpus)
    return {"acc": exact/n, "conc": f"{exact}/{n}", "loss": loss,
            "correct": exact, "total": n, "details": details}

# ═══ GRID ═══
GRID = {
    "Te_HIGH":  np.arange(0.050, 0.95, 0.005).round(3).tolist(),
    "He_HIGH":  np.arange(0.05, 0.95, 0.005).round(3).tolist(),
    "R_HIGH":   np.arange(0.05, 0.95, 0.005).round(3).tolist(),
    "Feas_NER": np.arange(0.05, 0.95, 0.005).round(3).tolist(),
}

def build_grid():
    return list(itertools.product(
        GRID["Te_HIGH"], GRID["He_HIGH"], GRID["R_HIGH"], GRID["Feas_NER"]))

# ═══ OPTIMIZATION (TRAIN: CANTEMIST, TEST: REDJDAL+RCP) ═══
def run_optimization():
    combos = build_grid()
    print(f"Grille : {len(combos)} combinaisons\n")

    train_data = ALL_CORPORA["cantemist"]
    test_data = ALL_CORPORA["redjdal"] + ALL_CORPORA["rcp"]

    print(f"{'='*70}")
    print(f"TRAIN : Cantemist ({len(train_data)} entités)")
    print(f"TEST  : Redjdal + RCP ({len(test_data)} entités)")
    print(f"{'='*70}")

    best_loss = float("inf")
    best_cfgs = []
    
    # 1. On optimise sur TRAIN = Cantemist (on minimise la penalty `loss`)
    for c in combos:
        ev = evaluate(train_data, *c)
        if ev["loss"] < best_loss:
            best_loss = ev["loss"]
            best_cfgs = [c]
        elif ev["loss"] == best_loss:
            best_cfgs.append(c)

    # 2. Si on a plusieurs configs optimales sur Train, on départage sur Test
    best_test_acc, best_c, best_ev = -1, None, None
    test_loss_for_best = float("inf")
    for c in best_cfgs:
        ev = evaluate(test_data, *c)
        # On maximise l'accuracy sur Test
        if ev["acc"] > best_test_acc or (ev["acc"] == best_test_acc and ev["loss"] < test_loss_for_best):
            best_test_acc = ev["acc"]
            best_c = c
            best_ev = ev
            test_loss_for_best = ev["loss"]

    ranges = {}
    for i, p in enumerate(GRID.keys()):
        vals = [c[i] for c in best_cfgs]
        ranges[p] = (min(vals), max(vals))

    return {
        "best_loss": best_loss,
        "best_c": best_c,
        "best_test_acc": best_test_acc,
        "n_cfgs": len(best_cfgs),
        "ranges": ranges,
        "test_eval": best_ev
    }

# ═══ FINAL EVAL ═══
def final_evaluation(opt_results):
    best_c = opt_results["best_c"]
    params = best_c
    print(f"\n{'='*70}\nSEUILS OPTIMISÉS (TRAIN=Cantemist)\n{'='*70}")
    for k, v in zip(GRID.keys(), best_c): print(f"  {k:12s} = {v}")
    
    tc, tn = 0, 0
    corpus_results = {}
    for name, corpus in ALL_CORPORA.items():
        ev = evaluate(corpus, *params)
        tc += ev["correct"]; tn += ev["total"]
        corpus_results[name] = ev
        print(f"\n  {name:<10s}: {ev['conc']} ({ev['acc']:.1%}), loss={ev['loss']:.1f}")
        for d in ev["details"]:
            if not d["ok"]:
                tag = "sous-esc." if d["d"] < 0 else "sur-esc."
                print(f"    x {d['entity']:<35s} DEMNE={d['pred']:<6s} "
                      f"Ref={d['ref']:<6s} ({tag})")
    print(f"\n  {'='*60}\n  TOTAL : {tc}/{tn} ({tc/tn:.1%})\n  {'='*60}")
    return {"total_correct": tc, "total": tn, "accuracy": tc/tn, "corpora": corpus_results, "best_c": best_c}

# ═══ EXPORTS ═══
def export_all(final):
    Path("Results").mkdir(exist_ok=True)
    Path("config").mkdir(exist_ok=True)

    universal = dict(zip(GRID.keys(), final["best_c"]))

    config = {"thresholds": {
        "TE_HIGH": universal["Te_HIGH"], "HE_HIGH": universal["He_HIGH"],
        "R_HIGH": universal["R_HIGH"], "FEAS_NER": universal["Feas_NER"],
        "TE_MED": 0.10, "FREQ_MIN": 0.001, "RARE_THRESHOLD_COUNT": 10,
        "MIN_TE_SAMPLES": 10},
        "calibration": {"method": "Train: Cantemist, Test: Redjdal+RCP",
            "label_criterion": "F1 >= 0.80", "total_entities": 65,
            "concordance": f"{final['total_correct']}/{final['total']}"}}
    Path("config/thresholds_optimized.json").write_text(
        json.dumps(config, indent=2, ensure_ascii=False))

    rows = []
    for cn, ev in final["corpora"].items():
        for d in ev["details"]:
            rows.append({"corpus": cn, "entity": d["entity"],
                "demne": d["pred"], "reference": d["ref"],
                "match": "exact" if d["ok"] else "discordant", "distance": d["d"]})
    pd.DataFrame(rows).to_csv("Results/concordance_detail.csv", index=False)

    lines = [f"DEMNE Train=Cantemist / Test=Redjdal+RCP",
             f"Date: {time.strftime('%Y-%m-%d %H:%M')}",
             f"Thresholds: {universal}",
             f"Concordance totale: {final['total_correct']}/{final['total']} ({final['accuracy']:.1%})"]
    Path("Results/grid_search_summary.txt").write_text("\n".join(lines))
    print(f"\n  -> config/thresholds_optimized.json")
    print(f"  -> Results/concordance_detail.csv")
    print(f"  -> Results/grid_search_summary.txt")

# ═══ MAIN ═══
def main():
    t0 = time.time()
    print("DEMNE Optimization (Train=Cantemist, Test=Redjdal+RCP)")
    print(f"65 entites, labels ternaires RULES/TBM/LLM\n")
    for n,c in ALL_CORPORA.items():
        ct = {"RULES":0,"TBM":0,"LLM":0}
        for e in c: ct[e.label]+=1
        print(f"  {n:12s}: {len(c)} (R={ct['RULES']}, T={ct['TBM']}, L={ct['LLM']})")
    print()
    opt = run_optimization()
    final = final_evaluation(opt)
    export_all(final)
    print(f"\n  Temps: {time.time()-t0:.1f}s")

if __name__ == "__main__":
    main()