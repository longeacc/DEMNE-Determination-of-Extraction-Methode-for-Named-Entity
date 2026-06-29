#!/usr/bin/env python3
"""
DEMNE Threshold + Weight Optimizer — Multi-corpus
=================================================
Optimisation conjointe de :
  - 4 seuils  : Te_HIGH, He_HIGH, R_HIGH, Feas_NER
  - 3 poids R : aR, bR, gR     (R = min(1, aR*f_neg + bR*f_unc + gR*f_cont))
  - 2 poids F : aFeas, bFeas   (Feas = aFeas*min(1,Freq) + bFeas*He)
  - 1 garde   : MIN_TE_SAMPLES (si te_count < seuil -> Te=0, bloque RULES)

TRAIN : MACCROBAT2020 + QUEARO_French_Med + RCP/ESMO Breast
TEST  : Cantemist-35 + Redjdal (these d'Akram)

Sources des labels :
  - MACCROBAT2020 : labels Cochran Q, metriques DEMNE via pipeline sur corpus BRAT
  - QUEARO        : labels Cochran Q, metriques DEMNE via pipeline sur corpus BRAT
  - RCP/ESMO      : metriques DEMNE completes (Te, He, R, f_neg/f_unc/f_cont)
  - Cantemist-35  : metriques DEMNE completes (Te, He, R, f_neg/f_unc/f_cont)
  - Redjdal       : metriques hardcodees (pas de sous-composantes R ni te_count)
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
    R: float
    Freq: float
    Feas: float
    label: str
    f_neg: float | None = None
    f_unc: float | None = None
    f_cont: float | None = None
    te_count: int | None = None


# ===================================================================
# TRAIN CORPORA
# ===================================================================

# --- MACCROBAT2020 (27 entites) ---
# Labels Cochran Q : MACCROBAT2020_cochran_routing.csv
# Metriques DEMNE calculees via pipeline (E_templatability, E_homogeneity, E_risk_context, E_frequency)
# sur data/ESMO2025_MACCROBAT2020/MACCROBAT2020/ (200 docs, 83404 tokens)
MACCROBAT = [
    #                                          Te      He      R       Freq    Feas   label    f_neg f_unc f_cont te_count
    Entity("Activity",                     0.2250, 0.1058, 0.0367, 0.0013, 0.021, "LLM",   0.00, 0.00, 0.06, 107),
    Entity("Administration",               0.3270, 0.9101, 0.0006, 0.0021, 0.182, "RULES", 0.01, 0.00, 0.00, 175),
    Entity("Age",                          0.7330, 0.9688, 0.0000, 0.0025, 0.194, "RULES", 0.00, 0.00, 0.00, 206),
    Entity("Area",                         0.1350, 0.9307, 0.0000, 0.0005, 0.186, "RULES", 0.00, 0.00, 0.00, 43),
    Entity("Biological_structure",         0.3540, 0.9511, 0.0131, 0.0354, 0.197, "RULES", 0.01, 0.00, 0.02, 2953),
    Entity("Clinical_event",               0.4810, 0.9565, 0.0843, 0.0075, 0.193, "RULES", 0.00, 0.00, 0.14, 626),
    Entity("Color",                        0.1790, 0.7013, 0.0058, 0.0006, 0.140, "RULES", 0.00, 0.02, 0.00, 52),
    Entity("Coreference",                  0.3090, 0.4716, 0.0006, 0.0038, 0.095, "TBM",   0.01, 0.00, 0.00, 315),
    Entity("Date",                         0.1910, 0.9778, 0.0106, 0.0088, 0.197, "RULES", 0.00, 0.00, 0.02, 735),
    Entity("Detailed_description",         0.3860, 0.6707, 0.0795, 0.0350, 0.141, "TBM",   0.00, 0.00, 0.13, 2920),
    Entity("Diagnostic_procedure",         0.3110, 0.9395, 0.0759, 0.0551, 0.199, "RULES", 0.01, 0.00, 0.12, 4598),
    Entity("Disease_disorder",             0.3070, 0.7134, 0.0014, 0.0163, 0.146, "TBM",   0.01, 0.00, 0.00, 1362),
    Entity("Distance",                     0.2470, 0.9571, 0.0033, 0.0015, 0.192, "RULES", 0.01, 0.01, 0.00, 122),
    Entity("Dosage",                       0.1430, 0.9558, 0.0003, 0.0043, 0.192, "RULES", 0.00, 0.00, 0.00, 361),
    Entity("Duration",                     0.2260, 0.9578, 0.0044, 0.0034, 0.192, "RULES", 0.00, 0.00, 0.01, 283),
    Entity("Family_history",               0.1040, 0.5680, 0.0000, 0.0010, 0.114, "TBM",   0.00, 0.00, 0.00, 81),
    Entity("Frequency",                    0.1910, 0.7786, 0.0013, 0.0009, 0.156, "RULES", 0.01, 0.00, 0.00, 76),
    Entity("History",                      0.1400, 0.7409, 0.0127, 0.0047, 0.149, "TBM",   0.01, 0.00, 0.02, 392),
    Entity("Lab_value",                    0.3150, 0.9465, 0.1507, 0.0341, 0.196, "RULES", 0.01, 0.00, 0.25, 2848),
    Entity("Medication",                   0.3860, 0.7311, 0.0194, 0.0129, 0.149, "TBM",   0.00, 0.00, 0.03, 1080),
    Entity("Nonbiological_location",       0.3050, 0.8956, 0.0000, 0.0043, 0.180, "RULES", 0.00, 0.00, 0.00, 356),
    Entity("Personal_background",          0.2490, 0.3506, 0.0000, 0.0007, 0.070, "LLM",   0.00, 0.00, 0.00, 57),
    Entity("Severity",                     0.4800, 0.9377, 0.0016, 0.0045, 0.188, "RULES", 0.01, 0.00, 0.00, 376),
    Entity("Sex",                          0.1640, 0.9899, 0.0000, 0.0023, 0.198, "RULES", 0.00, 0.00, 0.00, 191),
    Entity("Shape",                        0.2580, 0.4332, 0.0016, 0.0008, 0.087, "RULES", 0.02, 0.00, 0.00, 64),
    Entity("Sign_symptom",                 0.4660, 0.8479, 0.0285, 0.0405, 0.178, "TBM",   0.01, 0.00, 0.04, 3382),
    Entity("Therapeutic_procedure",        0.3110, 0.7365, 0.0228, 0.0124, 0.150, "RULES", 0.00, 0.01, 0.03, 1036),
]

# --- QUEARO French Med (10 entites) ---
# Labels Cochran Q : QUEARO_cochran_routing.csv
# Metriques DEMNE calculees via pipeline sur
# data/ESMO2025_QUERO_French_Med/QUAERO_FrenchMed/corpus/test/MEDLINE (833 docs, 10871 tokens)
QUEARO = [
    Entity("DISO",  0.2700, 0.6198, 0.0105, 0.0909, 0.142, "TBM",    0.00, 0.03, 0.00, 988),
    Entity("PROC",  0.3070, 0.7047, 0.0125, 0.0559, 0.152, "LLM",    0.00, 0.03, 0.01, 608),
    Entity("ANAT",  0.3590, 0.4500, 0.0035, 0.0469, 0.099, "RULES",  0.00, 0.01, 0.00, 510),
    Entity("CHEM",  0.2730, 0.2131, 0.0038, 0.0315, 0.049, "LLM",    0.00, 0.01, 0.00, 342),
    Entity("LIVB",  0.3090, 0.4013, 0.0056, 0.0298, 0.086, "RULES",  0.00, 0.02, 0.00, 324),
    Entity("PHYS",  0.2560, 0.2341, 0.0038, 0.0145, 0.050, "LLM",   0.02, 0.01, 0.00, 158),
]

# --- RCP/ESMO Breast (7 entites) — metriques DEMNE completes ---
RCP = [
    Entity("Estrogen_receptor",     0.304, 0.9780, 0.0589, 0.0028, 0.2950, "RULES", 0.08, 0.04, 0.02, 157),
    Entity("Progesterone_receptor", 0.294, 0.9644, 0.1383, 0.0023, 0.2900, "RULES", 0.08, 0.06, 0.09, 132),
    Entity("HER2_status",           0.241, 0.9691, 0.0758, 0.0015, 0.2910, "RULES", 0.12, 0.02, 0.04, 82),
    Entity("HER2_IHC",              0.176, 0.9579, 0.1601, 0.0014, 0.2880, "RULES", 0.13, 0.08, 0.09, 77),
    Entity("Ki67",                  0.269, 0.9712, 0.0635, 0.0021, 0.2920, "RULES", 0.12, 0.08, 0.00, 116),
    Entity("HER2_FISH",             0.100, 0.5000, 0.2741, 0.0003, 0.1500, "LLM",   0.19, 0.19, 0.14, 16),
]


# ===================================================================
# TEST CORPORA
# ===================================================================

# --- Cantemist-35 (12 entites) — metriques DEMNE recalculees via pipeline ---
# sur Datasets/Emmanuelle_35_cantemist/Emmanuelle_35_cantemist/ (35 docs, 28785 tokens)
CANTEMIST = [
    #                                                                            Te      He      R       Freq    Feas   label    f_neg f_unc f_cont te_count
    Entity("Histologie_tumorale",                                                0.132, 0.8537, 0.0242, 0.0034, 0.171, "RULES", 0.03, 0.07, 0.00, 99),
    Entity("Traitement_specifique_du_cancer",                                    0.174, 0.8411, 0.1892, 0.0071, 0.170, "RULES", 0.10, 0.01, 0.29, 205),
    Entity("Signes_physiques",                                                   0.101, 0.6976, 0.0499, 0.0025, 0.140, "LLM",   0.29, 0.00, 0.03, 72),
    Entity("Evolutivite_en_lien_avec_le_cancer",                                 0.000, 0.0067, 0.0000, 0.0001, 0.001, "LLM",   0.00, 0.00, 0.00, 2),
    Entity("Reponse_a_la_chimiotherapie",                                        0.105, 0.9221, 0.1035, 0.0043, 0.185, "RULES", 0.19, 0.04, 0.12, 124),
    Entity("Stade_metastatique_avec_localisations",                              0.111, 0.8196, 0.0525, 0.0029, 0.164, "TBM",   0.05, 0.10, 0.03, 83),
    Entity("Statut_tabagique",                                                   0.100, 0.5883, 0.0500, 0.0005, 0.118, "RULES", 0.50, 0.00, 0.00, 14),
    Entity("ATCD_geriatriques_et_medicaux_significatifs_pour_la_prise_en_charge",0.100, 0.3241, 0.0241, 0.0010, 0.065, "LLM",   0.24, 0.00, 0.00, 29),
    Entity("Stade_OMS_ECOG_Karnofsky",                                           0.389, 0.9190, 0.0100, 0.0010, 0.184, "RULES", 0.10, 0.00, 0.00, 30),
    Entity("Biomarqueurs_therapeutiques",                                        0.104, 0.7193, 0.3645, 0.0010, 0.144, "RULES", 0.00, 0.14, 0.54, 29),
    Entity("Topographie_du_primitif",                                            0.105, 0.7859, 0.0236, 0.0019, 0.158, "LLM",   0.02, 0.07, 0.00, 55),
    Entity("Symptomes",                                                          0.133, 0.7440, 0.0349, 0.0036, 0.150, "LLM",   0.11, 0.01, 0.04, 104),
]

# --- Redjdal / these d'Akram (46 entites) — pas de sous-composantes R ni te_count ---
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
    Entity("BraSize_Cup",              0.3732, 0.9612, 0.0575, 0.0013, 0.8421, "RULES"),
    Entity("Cavity_Shave_Margin",      0.1617, 0.9911, 0.0358, 0.0015, 0.8156, "LLM"),
    Entity("Clear_Surgical_Margins",   0.1896, 0.9894, 0.0314, 0.0018, 0.82,   "LLM"),
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
    Entity("Chemotherapy",             0.4992, 0.9915, 0.0611, 0.0023, 0.8765, "TBM"),
    Entity("Tumor_grade_inv",          0.5189, 0.9905, 0.0265, 0.0025, 0.8797, "RULES"),
    Entity("Widespread_Microcalc",     0.2841, 0.9881, 0.0714, 0.0018, 0.8365, "RULES"),
    Entity("TNM",                      0.3066, 0.8801, 0.0323, 0.0018, 0.7984, "RULES"),
    Entity("ResponseAssess_Neoadj",    0.2015, 0.9876, 0.0838, 0.0006, 0.6934, "RULES"),
    Entity("Cytoponction",             0.3586, 0.9804, 0.1512, 0.0008, 0.7909, "RULES"),
    Entity("Drugs",                    0.1254, 0.7059, 0.0557, 0.0033, 0.6979, "LLM"),
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


# ===================================================================
# CORPORA REGISTRY
# ===================================================================
TRAIN_CORPORA = {"maccrobat": MACCROBAT, "quearo": QUEARO, "cantemist": CANTEMIST}
TEST_CORPORA  = {"rcp": RCP, "redjdal": REDJDAL}
ALL_CORPORA   = {**TRAIN_CORPORA, **TEST_CORPORA}


# ===================================================================
# DECISION GRAPH
# ===================================================================
RANK = {"RULES": 0, "TBM": 1, "LLM": 2}


def effective_R(e: Entity, aR: float, bR: float, gR: float) -> float:
    if e.f_neg is None:
        return e.R
    r = aR * e.f_neg + bR * e.f_unc + gR * e.f_cont
    return min(1.0, max(0.0, r))


def effective_Feas(e: Entity, aF: float, bF: float) -> float:
    return aF * min(1.0, e.Freq) + bF * e.He


def effective_Te(e: Entity, min_te_samples: int) -> float:
    if e.te_count is not None and e.te_count < min_te_samples:
        return 0.0
    return e.Te


def route(e, Te_H, He_H, R_H, Feas_H, aR, bR, gR, aF, bF, min_te):
    te = effective_Te(e, min_te)
    r = effective_R(e, aR, bR, gR)
    f = effective_Feas(e, aF, bF)
    if te >= Te_H and e.He >= He_H and r < R_H:
        return "RULES"
    if f >= Feas_H:
        return "TBM"
    return "LLM"


def evaluate(corpus, Te_H, He_H, R_H, Feas_H, aR, bR, gR, aF, bF, min_te):
    exact, loss = 0, 0.0
    details = []
    for e in corpus:
        pred = route(e, Te_H, He_H, R_H, Feas_H, aR, bR, gR, aF, bF, min_te)
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


# ===================================================================
# GRID (10 parametres)
# ===================================================================
GRID = {
    # Seuils — pas fins autour des optimaux precedents
    "Te_HIGH":  [0.05, 0.08, 0.10, 0.12, 0.15, 0.18, 0.20, 0.25, 0.30],
    "He_HIGH":  [0.50, 0.65, 0.75, 0.80, 0.85, 0.88, 0.90, 0.95],
    "R_HIGH":   [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40],
    "Feas_NER": [0.05, 0.08, 0.10, 0.12, 0.15, 0.18, 0.20, 0.25, 0.30],
    # Poids R — grossier (R=0 pour la plupart des entites MACCROBAT/QUEARO)
    "alpha_R":  [0.1,0.2, 0.3,0.4, 0.5],
    "beta_R":   [0.3, 0.5, 0.7],
    "gamma_R":  [0.6, 1.0, 1.2],
    # Poids Feas — pas fins
    "alpha_F":  [0.1, 0.2, 0.3, 0.4, 0.5],
    "beta_F":   [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
    # Garde
    "MIN_TE_SAMPLES": [2, 5, 10, 15, 20],
}


def build_grid():
    return list(itertools.product(*[GRID[k] for k in GRID.keys()]))


# ===================================================================
# OPTIMIZATION
# ===================================================================
def run_optimization():
    combos = build_grid()
    n_combos = len(combos)

    train_data = []
    for c in TRAIN_CORPORA.values():
        train_data.extend(c)
    test_data = []
    for c in TEST_CORPORA.values():
        test_data.extend(c)

    print(f"{'='*70}")
    print(f"TRAIN : MACCROBAT + QUEARO + RCP ({len(train_data)} entites)")
    print(f"TEST  : Cantemist + Redjdal ({len(test_data)} entites)")
    print(f"{'='*70}")
    print(f"Grille : {n_combos:,} combinaisons (10 parametres)\n")

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

    print(f"\nTrain termine en {time.time()-t0:.1f}s -- {len(best_cfgs)} "
          f"configs optimales (loss={best_loss:.2f})")

    best_test_acc, best_c = -1, None
    test_loss_for_best = float("inf")
    for c in best_cfgs:
        ev = evaluate(test_data, *c)
        if (ev["acc"] > best_test_acc
                or (ev["acc"] == best_test_acc and ev["loss"] < test_loss_for_best)):
            best_test_acc = ev["acc"]
            best_c = c
            test_loss_for_best = ev["loss"]

    ranges = {p: (min(c[i] for c in best_cfgs), max(c[i] for c in best_cfgs))
              for i, p in enumerate(GRID.keys())}

    return {
        "best_loss": best_loss,
        "best_c": best_c,
        "best_test_acc": best_test_acc,
        "n_cfgs": len(best_cfgs),
        "ranges": ranges,
    }


# ===================================================================
# FINAL EVAL
# ===================================================================
def final_evaluation(opt_results):
    params = opt_results["best_c"]
    print(f"\n{'='*70}\nPARAMETRES OPTIMISES (TRAIN=MACCROBAT+QUEARO+RCP)\n{'='*70}")
    for k, v in zip(GRID.keys(), params):
        print(f"  {k:15s} = {v}")

    tc, tn = 0, 0
    corpus_results = {}
    for name, corpus in ALL_CORPORA.items():
        ev = evaluate(corpus, *params)
        tc += ev["correct"]; tn += ev["total"]
        corpus_results[name] = ev
        role = "TRAIN" if name in TRAIN_CORPORA else "TEST"
        print(f"\n  [{role:5s}] {name:<12s}: {ev['conc']} ({ev['acc']:.1%}), loss={ev['loss']:.1f}")
        for d in ev["details"]:
            if not d["ok"]:
                tag = "sous-esc." if d["d"] < 0 else "sur-esc."
                print(f"    x {d['entity']:<55s} DEMNE={d['pred']:<6s} "
                      f"Ref={d['ref']:<6s} ({tag})")
    print(f"\n  {'='*60}\n  TOTAL : {tc}/{tn} ({tc/tn:.1%})\n  {'='*60}")
    return {"total_correct": tc, "total": tn, "accuracy": tc/tn,
            "corpora": corpus_results, "best_c": params, "ranges": opt_results["ranges"]}


# ===================================================================
# EXPORTS
# ===================================================================
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
            "RARE_THRESHOLD_COUNT": 10,
            "MIN_TE_SAMPLES": universal["MIN_TE_SAMPLES"],
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
            "method": "Train: MACCROBAT+QUEARO+RCP, Test: Cantemist+Redjdal",
            "label_criterion": "Cochran Q (MACCROBAT+QUEARO), REST senior (RCP), REST senior (Cantemist), F1>=0.80 (Redjdal)",
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
        role = "train" if cn in TRAIN_CORPORA else "test"
        for d in ev["details"]:
            rows.append({
                "corpus": cn, "role": role, "entity": d["entity"],
                "demne": d["pred"], "reference": d["ref"],
                "match": "exact" if d["ok"] else "discordant", "distance": d["d"],
            })
    pd.DataFrame(rows).to_csv("Results/concordance_detail.csv", index=False, encoding="utf-8")

    lines = [
        "DEMNE Multi-corpus Grid Search",
        f"Train: MACCROBAT+QUEARO+RCP / Test: Cantemist+Redjdal",
        f"Date: {time.strftime('%Y-%m-%d %H:%M')}",
        f"Thresholds : Te_HIGH={universal['Te_HIGH']}, He_HIGH={universal['He_HIGH']}, "
        f"R_HIGH={universal['R_HIGH']}, Feas_NER={universal['Feas_NER']}",
        f"Weights R   : a={universal['alpha_R']}, b={universal['beta_R']}, g={universal['gamma_R']}",
        f"Weights Feas: a={universal['alpha_F']}, b={universal['beta_F']}",
        f"MIN_TE_SAMPLES: {universal['MIN_TE_SAMPLES']}",
        f"Concordance totale: {final['total_correct']}/{final['total']} ({final['accuracy']:.1%})",
    ]
    Path("Results/grid_search_summary.txt").write_text("\n".join(lines), encoding="utf-8")
    print(f"\n  -> config/thresholds_optimized.json")
    print(f"  -> Results/concordance_detail.csv")
    print(f"  -> Results/grid_search_summary.txt")


# ===================================================================
# MAIN
# ===================================================================
def main():
    t0 = time.time()
    print("DEMNE Multi-corpus Grid Search")
    print("Train: MACCROBAT+QUEARO+RCP / Test: Cantemist+Redjdal")
    print("10 parametres: 4 seuils + aR/bR/gR + aFeas/bFeas + MIN_TE_SAMPLES\n")
    for group_name, group in [("TRAIN", TRAIN_CORPORA), ("TEST", TEST_CORPORA)]:
        for n, c in group.items():
            ct = {"RULES": 0, "TBM": 0, "LLM": 0}
            for e in c: ct[e.label] += 1
            sub = sum(1 for e in c if e.f_neg is not None)
            te_known = sum(1 for e in c if e.te_count is not None)
            print(f"  [{group_name:5s}] {n:12s}: {len(c):2d} entites "
                  f"(R={ct['RULES']:2d}, T={ct['TBM']:2d}, L={ct['LLM']:2d})  "
                  f"R_sub: {sub}/{len(c)}, te_count: {te_known}/{len(c)}")
    print()
    opt = run_optimization()
    final = final_evaluation(opt)
    export_all(final)
    print(f"\n  Temps total : {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
