"""
Calculate 'Risk Context' (R) Score.
Measures the complexity of the linguistic context surrounding an entity.

R Score ranges from 0.0 (Simple) to 1.0 (Complex/Dangerous).
High R indicates the entity is surrounded by:
- Negations (simple adjustment needed)
- Uncertainty (probabilistic language)
- Contradictions (conflicting values in same doc)
"""

# pylint: disable=unused-argument,broad-exception-caught

import csv
import os
import re
from collections import defaultdict
from pathlib import Path

from demne._brat import load_brat_corpus
from demne._table import print_table
from demne._tracking import start_tracker, stop_tracker
from demne.params import (
    NEGATION_PATTERNS,
    UNCERTAINTY_PATTERNS,
    VAL_NEG,
    VAL_POS,
    has_contradiction,
    load_params,
)

# --- Tunable weights loaded from data/demne_params.json (single source of truth) ---
PARAMS = load_params()


class RiskContextScorer:
    """
    Compute the Contextual Risk score (R).
    Analyses the text surrounding an entity (token window) to detect:
    1. Negation (slight increase of R)
    2. Uncertainty (strong increase of R)
    3. Contradiction (maximum R)
    """

    def __init__(self, data_dirs: list[Path] | None = None):
        self.data_dirs = data_dirs or []
        self.document_data: defaultdict[str, list] = defaultdict(list)
        self.entities_stats: dict = {}

        # --- DETECTOR CONFIGURATION ---

        # Analysis window: number of characters to read around the entity
        self.WINDOW_SIZE = PARAMS["risk_window_size"]

        # Lexicons shared with the dashboard and the optimizer (demne.params): the
        # thresholds are calibrated on them, so there is exactly one copy.
        self.NEGATION_PATTERNS = list(NEGATION_PATTERNS)
        self.UNCERTAINTY_PATTERNS = list(UNCERTAINTY_PATTERNS)
        self.VAL_POS = set(VAL_POS)
        self.VAL_NEG = set(VAL_NEG)

        # R(E) = min(1, α_R · f_neg + β_R · f_unc + γ_R · f_fam)
        _rw = PARAMS["risk_weights"]
        self.ALPHA_R = _rw["negation"]
        self.BETA_R = _rw["uncertainty"]
        self.GAMMA_R = _rw["contradiction"]
        self.weights = {
            "negation": self.ALPHA_R,
            "uncertainty": self.BETA_R,
            "contradiction": self.GAMMA_R,
        }

    def _learn_weights(self, annotated_data: list[tuple[int, int, int, int]]) -> None:
        """
        EXPERIMENTAL, not wired to any command: needs REAL ground-truth labels.
        (A former `--learn_weights` flag fed it random labels, so it was removed;
        use `main.py optimize_params` to calibrate the R weights.)

        Learn R weights via Logistic Regression on a validation set
        (Chapman et al., 2001 - weighted NegEx-style approach).
        annotated_data: list of tuples (has_neg, has_uncert, has_contradiction, is_risky_ground_truth)
        """
        try:
            import numpy as np
            from sklearn.linear_model import LogisticRegression

            x = np.array([[d[0], d[1], d[2]] for d in annotated_data])
            y = np.array([d[3] for d in annotated_data])

            # Positive-weight constraint
            clf = LogisticRegression(fit_intercept=False)
            clf.fit(x, y)

            self.weights["negation"] = float(clf.coef_[0][0])
            self.weights["uncertainty"] = float(clf.coef_[0][1])
            self.weights["contradiction"] = float(clf.coef_[0][2])
            print(f"R weights recalibrated via LR: {self.weights}")
        except ImportError:
            print("scikit-learn unavailable for LR — falling back to heuristics.")
        except Exception as e:
            print(f"Error during weight learning: {e}")

    def is_negation_risky(self, value: str, context: str) -> bool:
        """Is the negation a RISK for rule-based extraction?

        It is only if the marker bears on the entity WITHOUT being part of the
        mention:
        - marker INSIDE the mention ("Absence d'emboles") → the rule captures it too:
          the negation is part of the pattern, not a risk;
        - marker OUT OF SCOPE (further than scope_tokens before the mention, or after
          it) → it does not negate the entity: "ménopausée depuis 48 ans, pas de
          THM" — the « pas » bears on THM;
        - marker IN SCOPE and outside the mention ("pas de récidive") → the rule
          extracts the entity and misses the negation: risk.

        MUST stay identical to MetricsCalculator (drift-guard).
        """
        val = (value or "").lower().strip()
        ctx = (context or val).lower()
        if not val:
            return False
        if any(re.search(pat, val) for pat in self.NEGATION_PATTERNS):
            return False  # negation inside the mention → captured by the rule
        idx = ctx.find(val)
        before = ctx[:idx] if idx >= 0 else ctx
        scope = int(PARAMS["risk_weights"].get("scope_tokens", 6))
        window = " ".join(before.split()[-scope:])
        return any(re.search(pat, window) for pat in self.NEGATION_PATTERNS)

    def has_negation(self, text: str, entity_type: str = "") -> bool:
        """Negation marker present in `text` (no scope analysis).

        Do not use it to compute R: see `is_negation_risky`.
        """
        text = text.lower()
        return any(re.search(pat, text) for pat in self.NEGATION_PATTERNS)

    def has_uncertainty(self, text: str, entity_type: str = "") -> bool:
        """Return True if the text contains an uncertainty pattern."""
        text = text.lower()
        return any(re.search(pat, text) for pat in self.UNCERTAINTY_PATTERNS)

    def compute_score_from_stats(
        self,
        negated_count: int,
        uncertain_count: int,
        total_count: int,
        contradicted_rate: float = 0.0,
    ) -> float:
        """
        Single method for computing the R score from base statistics.
        Ensures formula consistency across all call sites.
        """
        if total_count == 0:
            return 0.0

        f_neg = negated_count / total_count
        f_unc = uncertain_count / total_count

        # R(E) = min(1, α_R · f_neg + β_R · f_unc + γ_R · f_fam)
        raw_risk = (
            (self.ALPHA_R * f_neg) + (self.BETA_R * f_unc) + (self.GAMMA_R * contradicted_rate)
        )
        return min(1.0, raw_risk)

    def compute_score(self, texts: list[str], entity_type: str) -> float:
        """
        Compute the R score over a list of short sentences (no document-level analysis).
        Delegates to compute_score_from_stats.
        """
        if not texts:
            return 0.0

        total = len(texts)
        negated = sum(1 for t in texts if self.has_negation(t))
        uncertain = sum(1 for t in texts if self.has_uncertainty(t))

        return self.compute_score_from_stats(negated, uncertain, total, contradicted_rate=0.0)

    def _load_data(self) -> None:
        """Load .ann and .txt files to build annotation context via unified parser."""
        print("Loading data (Annotations + Text)...")
        corpus = load_brat_corpus(
            self.data_dirs, include_text=False, context_window=self.WINDOW_SIZE
        )
        for doc in corpus:
            file_id = doc["file_id"]
            for ann in doc["annotations"]:
                self.document_data[file_id].append(
                    {
                        "type": ann.entity_type,
                        "value_text": ann.text.lower(),
                        "context": ann.context.lower(),
                    }
                )

    def _check_contradiction(self, entries: list[dict]) -> bool:
        """Detect whether an entity has contradictory values within the SAME document."""
        return has_contradiction([e["value_text"] for e in entries])

    def compute_all(self) -> list[dict]:
        """Compute the aggregated R score per entity type."""
        self._load_data()

        # Group everything by entity type for global stats
        entity_stats: defaultdict[str, dict[str, int]] = defaultdict(
            lambda: {"total": 0, "negated": 0, "uncertain": 0, "contradicted_docs": 0}
        )
        entity_docs: defaultdict[str, defaultdict[str, list]] = defaultdict(
            lambda: defaultdict(list)
        )

        # 1. Local analysis (Negation / Uncertainty) for each occurrence
        for filename, entries in self.document_data.items():
            for entry in entries:
                etype = entry["type"]
                ctx = entry["context"]

                entity_stats[etype]["total"] += 1
                entity_docs[etype][filename].append(entry)

                # Negation WITH scope analysis: a plain keyword over the whole window
                # gives up to 100 % false positives on clinical text.
                if self.is_negation_risky(entry.get("value_text", ""), ctx):
                    entity_stats[etype]["negated"] += 1

                # Check uncertainty
                if any(re.search(pat, ctx) for pat in self.UNCERTAINTY_PATTERNS):
                    entity_stats[etype]["uncertain"] += 1

        # 2. Global analysis (Contradiction) per document
        for etype, docs in entity_docs.items():
            for _filename, entries in docs.items():
                if self._check_contradiction(entries):
                    entity_stats[etype]["contradicted_docs"] += 1

        # 3. Final R score computation
        results = []
        for etype, stats in entity_stats.items():
            n_total = stats["total"]
            if n_total == 0:
                continue

            # Relative frequencies
            f_neg = stats["negated"] / n_total
            f_unc = stats["uncertain"] / n_total

            # For contradiction: ratio of contradicted documents
            # Total doc count per entity approximated as len(entity_docs[etype])
            n_docs = len(entity_docs[etype])
            f_cont = stats["contradicted_docs"] / n_docs if n_docs > 0 else 0

            # Delegate to unified scoring method
            risk_score = self.compute_score_from_stats(
                negated_count=stats["negated"],
                uncertain_count=stats["uncertain"],
                total_count=n_total,
                contradicted_rate=f_cont,
            )

            self.entities_stats[etype] = {"f_neg": f_neg, "f_unc": f_unc, "f_cont": f_cont}

            results.append(
                {
                    "Entity": etype,
                    "R_Score": round(risk_score, 4),
                    "Negation_Rate": round(f_neg, 2),
                    "Uncertainty_Rate": round(f_unc, 2),
                    "Contradiction_Rate": round(f_cont, 2),
                    "Count": n_total,
                }
            )

        return sorted(results, key=lambda x: x["R_Score"], reverse=True)

    def to_csv(self, output_path: Path) -> None:
        """Write the per-entity R table to `output_path` (CSV)."""
        data = self.compute_all()
        with open(output_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "Entity",
                    "R_Score",
                    "Negation_Rate",
                    "Uncertainty_Rate",
                    "Contradiction_Rate",
                    "Count",
                ],
            )
            writer.writeheader()
            writer.writerows(data)
        try:
            rel = os.path.relpath(str(output_path))
        except ValueError:
            rel = str(output_path)
        print(f"Saved to {rel}")


# ==================================================================================
# MAIN EXECUTION
# ==================================================================================
def main() -> None:
    """CLI entry point: writes Results/risk_context_analysis_<corpus>.csv."""
    import argparse

    _tracker = start_tracker("DEMNE-risk_context", "Calculating Contextual Risk")

    parser = argparse.ArgumentParser()
    parser.add_argument("--gs_dir", type=str, default=None)
    parser.add_argument("--pred_dir", type=str, default=None)
    args, _ = parser.parse_known_args()

    # RELATIVE PATHS
    script_dir = Path(__file__).parent
    root_dir = script_dir.parent.parent

    if args.gs_dir:
        data_dirs = [Path(args.gs_dir)]
    else:
        data_dirs = [
            root_dir / "src/demne/NER/data/Breast/train",
            root_dir / "src/demne/NER/data/Breast/val",
            root_dir / "src/demne/NER/data/Breast/test",
        ]

    corpus_name = Path(args.gs_dir).parent.name if args.gs_dir else "Breast"
    output_file = root_dir / "Results" / f"risk_context_analysis_{corpus_name}.csv"
    output_file.parent.mkdir(parents=True, exist_ok=True)

    print("=== Starting Risk Context (R) analysis ===")
    scorer = RiskContextScorer(data_dirs)

    scorer.to_csv(output_file)

    # === CRITICAL TESTS ===
    print("\n=== CRITICAL TESTS VERIFICATION ===")

    # Simulate artificial cases to validate the logic
    test_scorer = RiskContextScorer([])

    # Test 1: "HER2 non surexprimé" (simple negation)
    test_scorer.document_data["test1.txt"] = [
        {
            "type": "TEST_NEG",
            "value_text": "her2",
            "context": "le statut est her2 non surexprimé sur la lame",
        }
    ]

    # Test 2: discordant HER2 status (explicit contradiction — one POS and one NEG in same doc)
    test_scorer.document_data["test2.txt"] = [
        {
            "type": "TEST_CONTRA",
            "value_text": "her2 positif",
            "context": "biopsie montre her2 positif",
        },
        {
            "type": "TEST_CONTRA",
            "value_text": "her2 negatif",
            "context": "piece operatoire montre her2 negatif",
        },
    ]

    # Test 3: uncertainty
    test_scorer.document_data["test3.txt"] = [
        {
            "type": "TEST_UNCERT",
            "value_text": "tumeur",
            "context": "origine probable de la tumeur a confirmer",
        }
    ]

    res = test_scorer.compute_all()
    print_table(
        ["Entity", "Score R", "Neg", "Unc", "Contra"],
        [
            [
                r["Entity"],
                f"{r['R_Score']:.2f}",
                str(r["Negation_Rate"]),
                str(r["Uncertainty_Rate"]),
                str(r["Contradiction_Rate"]),
            ]
            for r in res
        ],
        [25, 8, 6, 6, 8],
    )

    stop_tracker(_tracker)


if __name__ == "__main__":
    main()
