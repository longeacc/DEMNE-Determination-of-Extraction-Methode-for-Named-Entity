import math
import re
from collections import Counter
from typing import Any

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
    def __init__(self):
        self.rules = {
            "Estrogen_receptor": re.compile(
                r"(?:RE|RO|ER|Estrogen)[\s:]*(\d+\s*%|positif|négatif|positive|negative|\+|\-)",
                re.IGNORECASE,
            ),
            "Progesterone_receptor": re.compile(
                r"(?:RP|PR|Progesterone)[\s:]*(\d+\s*%|positif|négatif|positive|negative|\+|\-)",
                re.IGNORECASE,
            ),
            "Ki67": re.compile(r"Ki[\-\s]?67[\s:]*(\d+\s*%)", re.IGNORECASE),
            "HER2_status": re.compile(
                r"HER[\-\s]?2[\s:]*(\d\+?|positif|négatif|équivoque|score)",
                re.IGNORECASE,
            ),
            "HER2_IHC": re.compile(
                r"HER[\-\s]?2[\s:]*(\d\+?|positif|négatif|équivoque|score)",
                re.IGNORECASE,
            ),
            "HER2_FISH": re.compile(r"FISH|amplifié", re.IGNORECASE),
            "Genetic_mutation": re.compile(r"mutation|variant|BRCA", re.IGNORECASE),
        }
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
        self.CONTRADICTION_PATTERNS = [
            r"\bcependant\b",
            r"\bmais\b",
            r"\bnéanmoins\b",
            r"\bau contraire\b",
            r"\bmalgré\b",
            r"\btoutefois\b",
            r"\bà l'inverse\b",
        ]

    def has_negation(self, text: str) -> bool:
        return any(re.search(pat, text) for pat in self.NEGATION_PATTERNS)

    def has_uncertainty(self, text: str) -> bool:
        return any(re.search(pat, text) for pat in self.UNCERTAINTY_PATTERNS)

    def has_contradiction(self, text: str) -> bool:
        return any(re.search(pat, text) for pat in self.CONTRADICTION_PATTERNS)

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
                "Yield": 0.0,
                "Feas": 0.0,
                "DomainShift": 0.0,
                "LLM_Necessity": 0.0,
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
                "Yield": 0.0,
                "Feas": 0.0,
                "DomainShift": 0.0,
                "LLM_Necessity": 0.0,
            }

        # 1. Te [0.0 - 1.0] - Abstraction and Entropy
        normalized_patterns = []
        for v in values:
            v_norm = re.sub(r"[0-9]", "D", v)
            v_norm = re.sub(r"[A-ZÀ-ÖØ-Þ]", "X", v_norm)
            v_norm = re.sub(r"[a-zß-ÿ]", "x", v_norm)
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

        structure_consistency = 1.0 - h_norm
        bonus_semantic = (
            0.1
            if any(c in p for p in set(normalized_patterns) for c in ["%", "+", "-", ">", "<"])
            else 0.0
        )
        if any("D" in p for p in set(normalized_patterns)) and structure_consistency > 0.6:
            bonus_semantic += 0.1

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
            k, x0 = 10, 0.5
            try:
                he_raw = 1 / (1 + math.exp(-k * (redundancy - x0)))
            except OverflowError:
                he_raw = 0.0 if (-k * (redundancy - x0)) > 0 else 1.0
            he = min(1.0, max(0.0, he_raw))

        # 3. R [0.0 - 1.0] - Risk Context calculation
        total_texts = len(contexts) if contexts else len(values)
        text_to_search = contexts if contexts else [v.lower() for v in values]
        negated = sum(1 for t in text_to_search if self.has_negation(t))
        uncertain = sum(1 for t in text_to_search if self.has_uncertainty(t))
        contradictory = sum(1 for t in text_to_search if self.has_contradiction(t))

        # Poids ALIGNÉS sur la ligne de commande (E_risk_context.py) :
        # R(E) = min(1, α_R·f_neg + β_R·f_unc + γ_R·f_contradiction)
        ALPHA_R, BETA_R, GAMMA_R = 0.1, 0.3, 0.6
        r_raw = (
            (negated / total_texts) * ALPHA_R
            + (uncertain / total_texts) * BETA_R
            + (contradictory / total_texts) * GAMMA_R
            if total_texts > 0
            else 0.0
        )
        r = min(1.0, max(0.0, r_raw))

        # 4. Freq
        total_tokens = sum(len(doc.text.split()) for doc in documents) if documents else 1
        count = len(annotations)
        freq = count / max(total_tokens, 1)

        # 5. Yield
        if entity_type in self.rules:
            matches = sum(1 for c in contexts if self.rules[entity_type].search(c))
            y = matches / len(contexts) if len(contexts) > 0 else 0.0
        else:
            y = min(1.0, max(0.0, te * 0.6 + he * 0.3))

        # 6. Feas — formule ALIGNÉE sur la ligne de commande (E_feasibility_NER.py)
        # Feas(E) = α_Feas · min(1, Freq) + β_Feas · He, avec α_Feas = β_Feas = 0.2
        # He ∈ [0, 1] (sortie sigmoïde) et Freq = occurrences / total_tokens du corpus.
        ALPHA_FEAS = 0.2
        BETA_FEAS = 0.2
        feas = min(1.0, max(0.0, ALPHA_FEAS * min(1.0, freq) + BETA_FEAS * he))

        # 7. Domain Shift
        base_shift = 0.15
        he_penalty = max(0.0, (1.0 - he) / 2.0)
        te_penalty = max(0.0, (1.0 - te) / 3.0)
        min(1.0, max(0.0, base_shift + he_penalty + te_penalty))

        # 8. LLM Necessity
        necessity = 0.30 * (1.0 - y) + 0.25 * (r * 4.0) + 0.25 * (1.0 - feas) + 0.20 * (1.0 - he)
        min(1.0, max(0.0, necessity))

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
    "RÈGLES": "#2E7D32",
    "TBM": "#F57C00",
    "LLM": "#C62828",
}
