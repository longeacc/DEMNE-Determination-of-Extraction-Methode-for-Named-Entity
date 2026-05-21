import math
import re
from collections import Counter
from typing import Any

ENTITIES = [
    "ER",
    "PR",
    "HER2_status",
    "HER2_IHC",
    "Ki67",
    "HER2_FISH",
    "Genetic_mutation",
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

        r_raw = (
            (negated / total_texts) * 0.2
            + (uncertain / total_texts) * 0.5
            + (contradictory / total_texts) * 1.0
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

        # 6. Feas
        count = len(annotations)
        freq_factor = min(1.0, count / 100.0)
        feas = min(1.0, max(0.0, 0.6 * freq_factor + 0.4 * he))

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


DEMO_METRICS = {
    "ER": {"Te": 0.92, "He": 0.88, "R": 0.06, "Freq": 0.85, "Feas": 0.95, "C": 0.02},
    "PR": {"Te": 0.90, "He": 0.86, "R": 0.08, "Freq": 0.83, "Feas": 0.93, "C": 0.02},
    "HER2_status": {
        "Te": 0.72,
        "He": 0.68,
        "R": 0.12,
        "Freq": 0.75,
        "Feas": 0.85,
        "C": 0.12,
    },
    "HER2_IHC": {
        "Te": 0.68,
        "He": 0.65,
        "R": 0.15,
        "Freq": 0.70,
        "Feas": 0.82,
        "C": 0.15,
    },
    "Ki67": {"Te": 0.55, "He": 0.52, "R": 0.10, "Freq": 0.72, "Feas": 0.78, "C": 0.25},
    "HER2_FISH": {
        "Te": 0.42,
        "He": 0.45,
        "R": 0.25,
        "Freq": 0.25,
        "Feas": 0.40,
        "C": 0.45,
    },
    "Genetic_mutation": {
        "Te": 0.35,
        "He": 0.38,
        "R": 0.00,
        "Freq": 0.15,
        "Feas": 0.08,
        "C": 0.65,
    },
}

ROUTING_COLORS = {
    "RÈGLES": "#2E7D32",
    "TBM": "#F57C00",
    "LLM": "#C62828",
}
