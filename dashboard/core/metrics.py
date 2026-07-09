import importlib.util as _il
import math
import re
from collections import Counter
from pathlib import Path
from typing import Any

# --- Tunable weights loaded from data/demne_params.json (single source of truth).
# Same file the CLI scorers read → dashboard and command line can never drift. ---
_pspec = _il.spec_from_file_location(
    "demne_params", Path(__file__).resolve().parents[2] / "src" / "demne" / "params.py"
)
_pmod = _il.module_from_spec(_pspec)
_pspec.loader.exec_module(_pmod)
PARAMS = _pmod.load_params()

ENTITIES = [
    "Histologie_tumorale",
    "Traitement_specifique_du_cancer",
    "Signes_physiques",
    "Evolutivite_en_lien_avec_le_cancer",
    "Reponse_a_la_chimiotherapie",
    "Stade_metastatique_avec_localisations",
    "Statut_tabagique",
    "ATCD_geriatriques_et_medicaux_significatifs_pour_la_prise_en_charge",
    "Stade_OMS_ECOG_Karnofsky",
    "Biomarqueurs_therapeutiques",
    "Topographie_du_primitif",
    "Symptomes",
]


class MetricsCalculator:
    def __init__(self, params: dict | None = None):
        # params defaults to the shared config; the optimizer passes trial-specific
        # weights/thresholds here so the SAME formulas are reused (no drift).
        self.params = params if params is not None else PARAMS
        self.NEGATION_PATTERNS = [
            r"\baucun\b",
            r"\bsans\b",
            r"\bni\b",
            r"\bpas\b",
            r"\babsence\b",
        ]
        self.UNCERTAINTY_PATTERNS = [
            r"\bprobable\b",
            r"\bpossible\b",
            r"\bà confirmer\b",
            r"\ba confirmer\b",
            r"\bsuspecté\b",
            r"\bsuspecte\b",
            r"\béquivoque\b",
            r"\bequivoque\b",
            r"\bdiscuté\b",
            r"\bincertain\b",
            r"\bhypothèse\b",
            r"\?",
            r"\bdiscordant\b",
            r"\bdiscordance\b",
        ]

    def has_negation(self, text: str) -> bool:
        return any(re.search(pat, text) for pat in self.NEGATION_PATTERNS)

    def has_uncertainty(self, text: str) -> bool:
        return any(re.search(pat, text) for pat in self.UNCERTAINTY_PATTERNS)

    def compute_all_metrics(self, documents: list[Any], entity_type: str) -> dict[str, float]:
        annotations = []
        doc_count = 0
        for doc in documents:
            doc_anns = [a for a in doc.annotations if a.entity_type == entity_type]
            if doc_anns:
                doc_count += 1
                annotations.extend(doc_anns)

        if not annotations:
            return {
                "Te": 0.0,
                "He": 0.0,
                "R": 0.0,
                "Freq": 0.0,
                "Feas": 0.0,
            }

        values = [a.value.strip() for a in annotations if a.value]
        contexts = [
            (
                a.context.lower().strip()
                if hasattr(a, "context") and a.context
                else a.value.lower().strip()
            )
            for a in annotations
        ]

        if not values:
            return {
                "Te": 0.0,
                "He": 0.0,
                "R": 0.0,
                "Freq": 0.0,
                "Feas": 0.0,
            }

        # 1. Te [0.0 - 1.0] - Abstraction and Entropy
        # Templatabilité = forme CONSTANTE, indépendante de la casse et de la longueur.
        # Lettre→L (casse-insensible) puis collapse des répétitions (LLD == LLLD,
        # "SEIN   DROIT" == "SEIN DROIT") : sinon la casse/longueur gonflent le nombre
        # de patterns → entropie haute → Te artificiellement bas. Symboles (%,+,-…)
        # conservés pour le bonus_semantic.
        normalized_patterns = []
        for v in values:
            v_norm = re.sub(r"[0-9]", "D", v)
            v_norm = re.sub(r"[A-Za-zÀ-ÖØ-Þß-ÿ]", "L", v_norm)
            v_norm = re.sub(r"(.)\1+", r"\1", v_norm)
            normalized_patterns.append(v_norm)

        num_unique = len(set(normalized_patterns))
        if num_unique <= 1:
            h_norm = 0.0
        else:
            counter = Counter(normalized_patterns)
            entropy = -sum(
                (p / len(normalized_patterns)) * math.log(p / len(normalized_patterns))
                for p in counter.values()
            )
            h_norm = entropy / math.log(num_unique)

        _tb = self.params["templatability_bonus"]
        structure_consistency = 1.0 - h_norm
        bonus_semantic = (
            _tb["symbol_bonus"]
            if any(c in p for p in set(normalized_patterns) for c in ["%", "+", "-", ">", "<"])
            else 0.0
        )
        if (
            any("D" in p for p in set(normalized_patterns))
            and structure_consistency > _tb["digit_gate"]
        ):
            bonus_semantic += _tb["digit_bonus"]

        te = min(1.0, max(0.0, structure_consistency + bonus_semantic))

        # 2. He [0.0 - 1.0] - Sigmoid mapping over Token Redundancy
        all_tokens = []
        for val in values:
            all_tokens.extend([w.lower() for w in re.split(r"[^a-zA-Z0-9%]+", val) if w.strip()])

        if not all_tokens:
            he = 0.0
        else:
            n_total = len(all_tokens)
            n_unique = len(set(all_tokens))
            redundancy = (n_total - n_unique) / n_total if n_total > 0 else 0
            _sig = self.params["homogeneity_sigmoid"]
            k, x0 = _sig["k"], _sig["x0"]
            try:
                he_raw = 1 / (1 + math.exp(-k * (redundancy - x0)))
            except OverflowError:
                he_raw = 0.0 if (-k * (redundancy - x0)) > 0 else 1.0
            he = min(1.0, max(0.0, he_raw))

        # 3. R [0.0 - 1.0] - R(E) = min(1, α_R·f_neg + β_R·f_unc + γ_R·f_cont)
        # Same weights as the CLI scorer (E_risk_context.py) via shared config.
        # f_cont (contradiction) needs document-level analysis the dashboard does not
        # run on this short-text view → its rate is 0 here, identical to the CLI's
        # per-call default (compute_score_from_stats(..., contradicted_rate=0.0)).
        _rw = self.params["risk_weights"]
        alpha_r, beta_r, gamma_r = _rw["negation"], _rw["uncertainty"], _rw["contradiction"]
        contradicted_rate = 0.0
        total_texts = len(contexts) if contexts else len(values)
        text_to_search = contexts if contexts else [v.lower() for v in values]
        negated = sum(1 for t in text_to_search if self.has_negation(t))
        uncertain = sum(1 for t in text_to_search if self.has_uncertainty(t))

        r_raw = (
            (negated / total_texts) * alpha_r
            + (uncertain / total_texts) * beta_r
            + contradicted_rate * gamma_r
            if total_texts > 0
            else 0.0
        )
        r = min(1.0, max(0.0, r_raw))

        # 4. Freq
        total_tokens = sum(len(doc.text.split()) for doc in documents) if documents else 1
        count = len(annotations)
        freq = count / max(total_tokens, 1)

        # 5. Feas — disponibilité de données pour un template :
        # Feas(E) = α_Feas · min(1, count/C) + β_Feas · He
        # On sature le COMPTE absolu (assez d'exemples pour énumérer un template),
        # pas la densité count/tokens : une entité fréquente dans un gros corpus a
        # une densité faible mais plein d'exemples — la densité la pénalisait à tort.
        _fw = self.params["feasibility_weights"]
        alpha_feas = _fw["alpha_freq"]
        beta_feas = _fw["beta_he"]
        sat_count = _fw.get("sat_count", 300)
        feas_freq = min(1.0, count / sat_count) if sat_count > 0 else 0.0
        feas = min(1.0, max(0.0, alpha_feas * feas_freq + beta_feas * he))

        return {
            "Te": round(te, 4),
            "He": round(he, 4),
            "R": round(r, 4),
            "Freq": round(freq, 4),
            "Feas": round(feas, 4),
        }


# Valeurs de référence issues du pipeline DEMNE réel (Results/decision_summary.csv,
# Train=Cantemist-35 / Test=Redjdal+RCP). Mises à jour pour rester cohérentes avec
# la ligne de commande (formules Feas α=β=0.2 et R α=0.1/β=0.3/γ=0.6).
DEMO_METRICS = {
    "Histologie_tumorale": {"Te": 0.132, "He": 0.8537, "R": 0.0242, "Freq": 0.0034, "Feas": 0.171},
    "Traitement_specifique_du_cancer": {
        "Te": 0.174,
        "He": 0.8411,
        "R": 0.1892,
        "Freq": 0.0071,
        "Feas": 0.170,
    },
    "Signes_physiques": {"Te": 0.101, "He": 0.6976, "R": 0.0499, "Freq": 0.0025, "Feas": 0.140},
    "Evolutivite_en_lien_avec_le_cancer": {
        "Te": 0.000,
        "He": 0.0067,
        "R": 0.0000,
        "Freq": 0.0001,
        "Feas": 0.001,
    },
    "Reponse_a_la_chimiotherapie": {
        "Te": 0.105,
        "He": 0.9221,
        "R": 0.1035,
        "Freq": 0.0043,
        "Feas": 0.185,
    },
    "Stade_metastatique_avec_localisations": {
        "Te": 0.111,
        "He": 0.8196,
        "R": 0.0525,
        "Freq": 0.0029,
        "Feas": 0.164,
    },
    "Statut_tabagique": {"Te": 0.100, "He": 0.5883, "R": 0.0500, "Freq": 0.0005, "Feas": 0.118},
    "ATCD_geriatriques_et_medicaux_significatifs_pour_la_prise_en_charge": {
        "Te": 0.100,
        "He": 0.3241,
        "R": 0.0241,
        "Freq": 0.0010,
        "Feas": 0.065,
    },
    "Stade_OMS_ECOG_Karnofsky": {
        "Te": 0.389,
        "He": 0.9190,
        "R": 0.0100,
        "Freq": 0.0010,
        "Feas": 0.184,
    },
    "Biomarqueurs_therapeutiques": {
        "Te": 0.104,
        "He": 0.7193,
        "R": 0.3645,
        "Freq": 0.0010,
        "Feas": 0.144,
    },
    "Topographie_du_primitif": {
        "Te": 0.105,
        "He": 0.7859,
        "R": 0.0236,
        "Freq": 0.0019,
        "Feas": 0.158,
    },
    "Symptomes": {"Te": 0.133, "He": 0.7440, "R": 0.0349, "Freq": 0.0036, "Feas": 0.150},
}

ROUTING_COLORS = {
    "RULES": "#2E7D32",
    "TBM": "#F57C00",
    "LLM": "#C62828",
}
