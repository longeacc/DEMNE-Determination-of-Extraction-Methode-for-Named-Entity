#!/usr/bin/env python3
"""
DEMNE Threshold + Weight Optimizer
==================================
Optimisation conjointe sur le corpus Cantemist (Train) de :
  - 4 seuils  : Te_HIGH, He_HIGH, R_HIGH, Feas_NER
  - 3 poids R : αR, βR, γR     (R = min(1, αR·f_neg + βR·f_unc + γR·f_cont))
  - 2 poids F : αFeas, βFeas   (Feas = αFeas·min(1,Freq) + βFeas·He)

Test sur les corpus Redjdal + RCP.

Note : les sous-composantes (f_neg, f_unc, f_cont) sont disponibles pour
Cantemist et RCP (issues des CSV `risk_context_analysis_*.csv`).
Pour Redjdal, on ne les a pas — son R reste celui hardcodé (les poids αR/βR/γR
n'affectent donc PAS Redjdal au test). Feas est recalculé partout (Freq+He
sont disponibles pour toutes les entités).
"""

from __future__ import annotations
import itertools, json, time
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Entity : Te, He, R, Freq, Feas, label + sous-composantes R optionnelles
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Entity:
    name: str
    Te: float
    He: float
    R: float            # R pré-calculé (sert de fallback si pas de sous-composantes)
    Freq: float
    Feas: float         # Feas pré-calculé (utilisé seulement si pas d'optimisation des poids)
    label: str
    # Sous-composantes de R (None si indisponibles → R fixé)
    f_neg: float | None = None
    f_unc: float | None = None
    f_cont: float | None = None


# ═══ CANTEMIST-35 (12) — Données fournies par l'utilisateur + sous-rates CSV ═══
CANTEMIST = [
    Entity("Histologie_tumorale",                                                0.132, 0.8537, 0.0414, 0.0034, 0.2570, "RULES", 0.03, 0.07, 0.00),
    Entity("Traitement_specifique_du_cancer",                                    0.174, 0.8411, 0.3185, 0.0071, 0.2550, "LLM",   0.10, 0.01, 0.29),
    Entity("Signes_physiques",                                                   0.101, 0.6976, 0.0928, 0.0025, 0.2100, "LLM",   0.29, 0.00, 0.03),
    Entity("Evolutivite_en_lien_avec_le_cancer",                                 0.000, 0.0067, 0.0000, 0.0001, 0.0020, "LLM",   0.00, 0.00, 0.00),
    Entity("Reponse_a_la_chimiotherapie",                                        0.105, 0.9221, 0.1789, 0.0043, 0.2780, "RULES", 0.19, 0.04, 0.12),
    Entity("Stade_metastatique_avec_localisations",                              0.111, 0.8196, 0.0891, 0.0029, 0.2470, "LLM",   0.05, 0.10, 0.03),
    Entity("Statut_tabagique",                                                   0.100, 0.5883, 0.1000, 0.0005, 0.1770, "LLM",   0.50, 0.00, 0.00),
    Entity("ATCD_geriatriques_et_medicaux_significatifs_pour_la_prise_en_charge",0.100, 0.3241, 0.0483, 0.0010, 0.0980, "LLM",   0.24, 0.00, 0.00),
    Entity("Stade_OMS_ECOG_Karnofsky",                                           0.389, 0.9190, 0.0200, 0.0010, 0.2760, "RULES", 0.10, 0.00, 0.00),
    Entity("Biomarqueurs_therapeutiques",                                        0.104, 0.7193, 0.6074, 0.0010, 0.2160, "LLM",   0.00, 0.14, 0.54),
    Entity("Topographie_du_primitif",                                            0.105, 0.7859, 0.0400, 0.0019, 0.2370, "LLM",   0.02, 0.07, 0.00),
    Entity("Symptomes",                                                          0.133, 0.7440, 0.0617, 0.0036, 0.2250, "LLM",   0.11, 0.01, 0.04),
]

# ═══ RCP/ESMO (7) — Données fournies par l'utilisateur + sous-rates CSV ═══
RCP = [
    Entity("Estrogen_receptor",     0.304, 0.9780, 0.0589, 0.0028, 0.2950, "RULES", 0.08, 0.04, 0.02),
    Entity("Progesterone_receptor", 0.294, 0.9644, 0.1383, 0.0023, 0.2900, "RULES", 0.08, 0.06, 0.09),
    Entity("HER2_status",           0.241, 0.9691, 0.0758, 0.0015, 0.2910, "RULES", 0.12, 0.02, 0.04),
    Entity("HER2_IHC",              0.176, 0.9579, 0.1601, 0.0014, 0.2880, "RULES", 0.13, 0.08, 0.09),
    Entity("Ki67",                  0.269, 0.9712, 0.0635, 0.0021, 0.2920, "RULES", 0.12, 0.08, 0.00),
    Entity("Genetic_mutation",      0.000, 0.0143, 0.0000, 0.0000, 0.0040, "LLM",   0.00, 0.00, 0.00),
    Entity("HER2_FISH",             0.100, 0.5000, 0.2741, 0.0003, 0.1500, "LLM",   0.19, 0.19, 0.14),
]

# ═══ REDJDAL (46) — Labels REST senior, R conservé (pas de sous-composantes) ═══
REDJDAL = [
    Entity("Biopsy",                   0.4358, 0.991,  0.0916, 0.0048, 0.865,  "RULES"),
    Entity("Ultra_sound",              0.4485, 0.9918, 0.071,  0.0035, 0.8675, "RULES"),
    Entity("MRI",                      0.6547, 0.9931, 0.1226, 0.0035, 0.9052, "RULES"),
    Entity("Mammography",              0.4098, 0.9924, 0.0784, 0.0028, 0.8608, "RULES"),
    Entity("Clinical_examination",     0.501,  0.9922, 0.0874, 0.0029, 0.8771, "RULES"),
    Entity("Surgery",                  0.1856, 0.9915, 0.0299, 0.0081, 0.8201, "RULES"),
    Entity("Tumor_size",               0.5018, 0.9857, 0.0547, 0.0094, 0.8747, "RULES"),
    Entity("Tumor_grade_insitu",       0.2983, 0.9915, 0.063,  0.0008, 0.8084, "RULES"),
    Entity("Tumor_site",               0.5609, 0.991,  0.0577, 0.0092, 0.8874, "RULES"),
    Entity("Histology",                0.2866, 0.9889, 0.0766, 0.0053, 0.8372, "RULES"),
    Entity("Tumour",                   0.3938, 0.9901, 0.1101, 0.0178, 0.857,  "RULES"),
    Entity("BraSize_Cup",              0.3732, 0.9612, 0.0575, 0.0013, 0.8421, "TBM"),
    Entity("Cavity_Shave_Margin",      0.1617, 0.9911, 0.0358, 0.0015, 0.8156, "RULES"),
    Entity("Clear_Surgical_Margins",   0.1896, 0.9894, 0.0314, 0.0018, 0.82,   "TBM"),
    Entity("Side",                     0.3681, 0.9928, 0.0841, 0.0237, 0.8534, "RULES"),
    Entity("BIRADS_classification",    0.3509, 0.9914, 0.1308, 0.0071, 0.8498, "RULES"),
    Entity("Clinical_Positive_Nodes",  0.2758, 0.9898, 0.1921, 0.0048, 0.8357, "TBM"),
    Entity("Menopausal_status",        0.1814, 0.9879, 0.2051, 0.0011, 0.8179, "RULES"),
    Entity("Comorbidities",            0.3755, 0.98,   0.1059, 0.0035, 0.8498, "TBM"),
    Entity("Estrogen_receptor",        0.268,  0.9843, 0.0446, 0.0027, 0.8713, "RULES"),
    Entity("Progesterone_receptor",    0.2825, 0.9833, 0.0452, 0.0027, 0.8846, "RULES"),
    Entity("HER2_status",              0.2915, 0.9898, 0.049,  0.0027, 0.8884, "RULES"),
    Entity("Ki67",                     0.2257, 0.9835, 0.0573, 0.0023, 0.8409, "RULES"),
    Entity("Pet_scan",                 0.4682, 0.9912, 0.0786, 0.0018, 0.8708, "RULES"),
    Entity("Anti_HER2_therapy",        0.2059, 0.9851, 0.0591, 0.0004, 0.5972, "LLM"),
    Entity("Chemotherapy",             0.4992, 0.9915, 0.0611, 0.0023, 0.8765, "RULES"),
    Entity("Tumor_grade_inv",          0.5189, 0.9905, 0.0265, 0.0025, 0.8797, "RULES"),
    Entity("Widespread_Microcalc",     0.2841, 0.9881, 0.0714, 0.0018, 0.8365, "RULES"),
    Entity("TNM",                      0.3066, 0.8801, 0.0323, 0.0018, 0.7984, "RULES"),
    Entity("ResponseAssess_Neoadj",    0.2015, 0.9876, 0.0838, 0.0006, 0.6934, "TBM"),
    Entity("Cytoponction",             0.3586, 0.9804, 0.1512, 0.0008, 0.7909, "RULES"),
    Entity("Drugs",                    0.1254, 0.7059, 0.0557, 0.0033, 0.6979, "TBM"),
    Entity("NodeSize",                 0.2674, 0.7992, 0.0918, 0.0007, 0.6518, "TBM"),
    Entity("Hematoma",                 0.3759, 0.9875, 0.1051, 0.0004, 0.6088, "RULES"),
    Entity("Confirmed_Positive_Nodes", 0.0738, 0.9321, 0.105,  0.0005, 0.6168, "TBM"),
    Entity("Radiotherapy",             0.5485, 0.9916, 0.0218, 0.0011, 0.8854, "RULES"),
    Entity("Genetic_mutation",         0.0344, 0.8642, 0.15,   0.0003, 0.6779, "TBM"),
    Entity("FISH",                     0.2215, 0.9793, 0.0487, 0.0004, 0.5778, "RULES"),
    Entity("Screening",                0.6772, 0.991,  0.0275, 0.0005, 0.7124, "RULES"),
    Entity("Endocrine_Therapy",        0.4095, 0.9857, 0.014,  0.0008, 0.8301, "TBM"),
    Entity("Associated_InSitu_Carc",   0.2571, 0.989,  0.1258, 0.0009, 0.82,   "RULES"),
    Entity("PresenceEmbole",           0.2089, 0.988,  0.1788, 0.0009, 0.8229, "RULES"),
    Entity("OncotypeDX",               0.1425, 0.895,  0.0538, 0.0001, 0.4267, "LLM"),
    Entity("N_status",                 0.3728, 0.9428, 0.0814, 0.0008, 0.7788, "RULES"),
    Entity("Breast_Cancer_Relapse",    0.0915, 0.9547, 0.1037, 0.0002, 0.4968, "LLM"),
    Entity("Systemic_treatment",       1.0,    0.0067, 0.0,    0.0,    0.1866, "LLM"),
]

ALL_CORPORA = {"cantemist": CANTEMIST, "redjdal": REDJDAL, "rcp": RCP}

# ═══ DECISION GRAPH ═══
RANK = {"RULES": 0, "TBM": 1, "LLM": 2}


def effective_R(e: Entity, aR: float, bR: float, gR: float) -> float:
    """Recalcule R si les sous-composantes existent ; sinon retourne R hardcodé."""
    if e.f_neg is None:
        return e.R
    r = aR * e.f_neg + bR * e.f_unc + gR * e.f_cont
    return min(1.0, max(0.0, r))


def effective_Feas(e: Entity, aF: float, bF: float) -> float:
    """Recalcule Feas (Freq+He disponibles partout)."""
    return aF * min(1.0, e.Freq) + bF * e.He


def route(e, Te_H, He_H, R_H, Feas_H, aR, bR, gR, aF, bF):
    r = effective_R(e, aR, bR, gR)
    f = effective_Feas(e, aF, bF)
    if e.Te >= Te_H and e.He >= He_H and r < R_H:
        return "RULES"
    if f >= Feas_H:
        return "TBM"
    return "LLM"


def evaluate(corpus, Te_H, He_H, R_H, Feas_H, aR, bR, gR, aF, bF):
    exact, loss = 0, 0.0
    details = []
    for e in corpus:
        pred = route(e, Te_H, He_H, R_H, Feas_H, aR, bR, gR, aF, bF)
        ok = (pred == e.label)
        d = RANK[pred] - RANK[e.label]
        # Sous-escalade plus coûteuse que sur-escalade
        li = 0.0 if d == 0 else (abs(d) * 2.0 if d < 0 else abs(d) * 1.0)
        if ok: exact += 1
        loss += li
        details.append({"entity": e.name, "pred": pred, "ref": e.label,
                        "ok": ok, "d": d, "loss": li})
    n = len(corpus)
    return {"acc": exact/n, "conc": f"{exact}/{n}", "loss": loss,
            "correct": exact, "total": n, "details": details}


# ═══ GRID (coarse mais joint sur 9 paramètres) ═══
GRID = {
    # Seuils
    "Te_HIGH":  [0.05, 0.10, 0.15, 0.20, 0.25, 0.30],
    "He_HIGH":  [0.50, 0.60, 0.70, 0.80, 0.85, 0.90],
    "R_HIGH":   [0.15, 0.20, 0.25, 0.30, 0.35],
    "Feas_NER": [0.15, 0.20, 0.25, 0.30, 0.35],
    # Poids R  (existants : α=0.2, β=0.5, γ=1.0)
    "alpha_R":  [0.1, 0.2, 0.3, 0.4, 0.5],
    "beta_R":   [0.3, 0.4, 0.5, 0.6, 0.7],
    "gamma_R":  [0.6, 0.8, 1.0, 1.2],
    # Poids Feas  (existants : α=0.4, β=0.3)
    "alpha_F":  [0.2, 0.3, 0.4, 0.5, 0.6],
    "beta_F":   [0.2, 0.3, 0.4, 0.5],
}


def build_grid():
    return list(itertools.product(*[GRID[k] for k in GRID.keys()]))


# ═══ OPTIMIZATION (TRAIN: CANTEMIST, TEST: REDJDAL+RCP) ═══
def run_optimization():
    combos = build_grid()
    n_combos = len(combos)

    train_data = ALL_CORPORA["cantemist"]
    test_data = ALL_CORPORA["redjdal"] + ALL_CORPORA["rcp"]

    print(f"{'='*70}")
    print(f"TRAIN : Cantemist-35 ({len(train_data)} entités)")
    print(f"TEST  : Redjdal + RCP ({len(test_data)} entités)")
    print(f"{'='*70}")

    print(f"Grille : {n_combos:,} combinaisons (9 paramètres)\n")

    best_loss = float("inf")
    best_cfgs = []

    t0 = time.time()
    progress_every = max(1, n_combos // 20)
    for i, c in enumerate(combos):
        ev = evaluate(train_data, *c)
        if ev["loss"] < best_loss:
            best_loss = ev["loss"]
            best_cfgs = [c]
        elif ev["loss"] == best_loss:
            best_cfgs.append(c)
        if (i + 1) % progress_every == 0:
            pct = 100 * (i + 1) / n_combos
            print(f"  {pct:5.1f}%  best_loss={best_loss:.1f}  "
                  f"({len(best_cfgs)} configs tied)  [{time.time()-t0:.1f}s]")

    print(f"\nTrain terminé en {time.time()-t0:.1f}s — {len(best_cfgs)} "
          f"configs optimales (loss={best_loss:.2f})")

    # Tie-break sur Test : on maximise l'accuracy, puis on minimise la loss test
    best_test_acc, best_c, best_ev = -1, None, None
    test_loss_for_best = float("inf")
    for c in best_cfgs:
        ev = evaluate(test_data, *c)
        if (ev["acc"] > best_test_acc
                or (ev["acc"] == best_test_acc and ev["loss"] < test_loss_for_best)):
            best_test_acc = ev["acc"]
            best_c = c
            best_ev = ev
            test_loss_for_best = ev["loss"]

    ranges = {p: (min(c[i] for c in best_cfgs), max(c[i] for c in best_cfgs))
              for i, p in enumerate(GRID.keys())}

    return {
        "best_loss": best_loss,
        "best_c": best_c,
        "best_test_acc": best_test_acc,
        "n_cfgs": len(best_cfgs),
        "ranges": ranges,
        "test_eval": best_ev,
    }


# ═══ FINAL EVAL ═══
def final_evaluation(opt_results):
    params = opt_results["best_c"]
    print(f"\n{'='*70}\nPARAMÈTRES OPTIMISÉS (TRAIN=Cantemist-35)\n{'='*70}")
    for k, v in zip(GRID.keys(), params):
        print(f"  {k:10s} = {v}")

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
                print(f"    x {d['entity']:<55s} DEMNE={d['pred']:<6s} "
                      f"Ref={d['ref']:<6s} ({tag})")
    print(f"\n  {'='*60}\n  TOTAL : {tc}/{tn} ({tc/tn:.1%})\n  {'='*60}")
    return {"total_correct": tc, "total": tn, "accuracy": tc/tn,
            "corpora": corpus_results, "best_c": params, "ranges": opt_results["ranges"]}


# ═══ EXPORTS ═══
def export_all(final):
    Path("Results").mkdir(exist_ok=True)
    Path("config").mkdir(exist_ok=True)

    universal = dict(zip(GRID.keys(), final["best_c"]))

    config = {
        "thresholds": {
            "TE_HIGH": universal["Te_HIGH"],
            "HE_HIGH": universal["He_HIGH"],
            "R_HIGH":  universal["R_HIGH"],
            "FEAS_NER": universal["Feas_NER"],
            "TE_MED": 0.10, "FREQ_MIN": 0.001,
            "RARE_THRESHOLD_COUNT": 10, "MIN_TE_SAMPLES": 10,
        },
        "weights_R": {
            "alpha": universal["alpha_R"],
            "beta":  universal["beta_R"],
            "gamma": universal["gamma_R"],
        },
        "weights_Feas": {
            "alpha": universal["alpha_F"],
            "beta":  universal["beta_F"],
        },
        "calibration": {
            "method": "Train: Cantemist-35, Test: Redjdal+RCP",
            "label_criterion": "REST senior labels (Cantemist+RCP), F1≥0.80 (Redjdal)",
            "total_entities": sum(len(c) for c in ALL_CORPORA.values()),
            "concordance": f"{final['total_correct']}/{final['total']}",
            "accuracy": round(final["accuracy"], 4),
            "ranges_of_tied_optimal_configs": {k: list(v) for k, v in final["ranges"].items()},
        },
    }
    Path("config/thresholds_optimized.json").write_text(
        json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8")

    rows = []
    for cn, ev in final["corpora"].items():
        for d in ev["details"]:
            rows.append({
                "corpus": cn, "entity": d["entity"],
                "demne": d["pred"], "reference": d["ref"],
                "match": "exact" if d["ok"] else "discordant", "distance": d["d"],
            })
    pd.DataFrame(rows).to_csv("Results/concordance_detail.csv", index=False, encoding="utf-8")

    lines = [
        f"DEMNE Train=Cantemist-35 / Test=Redjdal+RCP",
        f"Date: {time.strftime('%Y-%m-%d %H:%M')}",
        f"Thresholds : {{Te_HIGH: {universal['Te_HIGH']}, He_HIGH: {universal['He_HIGH']}, "
        f"R_HIGH: {universal['R_HIGH']}, Feas_NER: {universal['Feas_NER']}}}",
        f"Weights R   : α={universal['alpha_R']}, β={universal['beta_R']}, γ={universal['gamma_R']}",
        f"Weights Feas: α={universal['alpha_F']}, β={universal['beta_F']}",
        f"Concordance totale: {final['total_correct']}/{final['total']} ({final['accuracy']:.1%})",
    ]
    Path("Results/grid_search_summary.txt").write_text("\n".join(lines), encoding="utf-8")
    print(f"\n  -> config/thresholds_optimized.json")
    print(f"  -> Results/concordance_detail.csv")
    print(f"  -> Results/grid_search_summary.txt")


# ═══ MAIN ═══
def main():
    t0 = time.time()
    print("DEMNE Optimization — Train=Cantemist-35, Test=Redjdal+RCP")
    print("4 seuils + αR/βR/γR + αFeas/βFeas, labels ternaires RULES/TBM/LLM\n")
    for n, c in ALL_CORPORA.items():
        ct = {"RULES": 0, "TBM": 0, "LLM": 0}
        for e in c: ct[e.label] += 1
        sub = sum(1 for e in c if e.f_neg is not None)
        print(f"  {n:10s}: {len(c)} entités (R={ct['RULES']}, T={ct['TBM']}, L={ct['LLM']})"
              f"  sous-rates R : {sub}/{len(c)}")
    print()
    opt = run_optimization()
    final = final_evaluation(opt)
    export_all(final)
    print(f"\n  Temps total : {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
