"""
Calculate Linguistic Homogeneity Scores (He).

Homogeneity measures linguistic redundancy:
- High He (near 100%): The entity is always expressed with the exact same words.
- Low He (near 0%): Every time the entity appears, it uses different words.

Formula: He = (Total_Words - Unique_Words) / Total_Words
Normalization: Sigmoid transform to spread values between 0 and 1 (returned as %).
"""

# pylint: disable=broad-exception-caught

import csv
import importlib.util as _il
import math
import os
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

from demne._table import print_table

# --- Tunable weights loaded from data/demne_params.json (single source of truth) ---
_pspec = _il.spec_from_file_location("demne_params", Path(__file__).resolve().parent / "params.py")
assert _pspec is not None and _pspec.loader is not None
_pmod = _il.module_from_spec(_pspec)
_pspec.loader.exec_module(_pmod)
PARAMS = _pmod.load_params()

# Eco2AI for energy tracking
try:
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        from eco2ai import Tracker, set_params

    HAS_ECO2AI = True
except ImportError:
    HAS_ECO2AI = False

if __name__ == "__main__" and HAS_ECO2AI and not os.environ.get("DISABLE_ECO2AI"):
    set_params(
        project_name="Consumtion_of_E_homogeneity.py",
        experiment_description="Calculating Linguistic Homogeneity",
        file_name="data/Consumtion_of_Duraxell.csv",
    )
    tracker = Tracker()
    tracker.start()


class HomogeneityScorer:
    """
    Compute the Homogeneity score (He) for each entity.
    He measures vocabulary repetition.
    """

    def __init__(self, corpus: list[dict[str, Any]]):
        """
        Args:
            corpus: List of documents containing text and annotations.
        """
        self.entities_values = defaultdict(list)
        for doc in corpus:
            annotations = doc.get("annotations", [])
            for ann in annotations:
                # Handle both object and dict access
                if hasattr(ann, "text") and hasattr(ann, "entity_type"):
                    text = ann.text
                    etype = ann.entity_type
                elif isinstance(ann, dict):
                    text = ann.get("text", "")
                    etype = ann.get("entity_type", "Unknown")
                else:
                    continue
                self.entities_values[etype].append(text)

        # Calibration for Sigmoid
        # k=10, x0=0.5 means the transition from low to high happens around 0.5 redundancy
        # "ER positif" repeated 100 times -> Raw=0.99 -> Sigmoid(0.99) ~ 1.0
        # "ER positif" / "Recepteur Estro" (50/50 mix) -> Raw ~ 0.5 -> Sigmoid(0.5) = 0.5
        _sig = PARAMS["homogeneity_sigmoid"]
        self.k = _sig["k"]
        self.x0 = _sig["x0"]

    def compute_from_list(self, values: list[str]) -> float:
        """
        Compute the He score directly from a list of text values.
        """
        # Reset internal storage for this computation
        self.entities_values["TEMP_LIST"] = values
        res = self.compute("TEMP_LIST")
        return res

    def _tokenize(self, text: str) -> list[str]:
        """Découpe en tokens minuscules.

        Séparateurs identiques à MetricsCalculator (drift-guard) : '+' et '-' sont
        traités comme des séparateurs. Toute divergence ici fait diverger He entre
        main.py (CSV) et le dashboard.
        """
        return [w.lower() for w in re.split(r"[^a-zA-Z0-9%]+", text) if w.strip()]

    def _sigmoid(self, x: float) -> float:
        """
        Transforms raw redundancy (0-1) into a specialized score.
        Spread values to make decision easier (Polarization).
        """
        try:
            val = 1 / (1 + math.exp(-self.k * (x - self.x0)))
            return val
        except OverflowError:
            return 0.0 if (-self.k * (x - self.x0)) > 0 else 1.0

    def compute(self, entity_type: str) -> float:
        """
        Calculate He score in % [0, 100].
        He = (Te_words - Ue_words) / Te_words
        """
        values = self.entities_values.get(entity_type, [])
        if not values:
            return 0.0

        all_tokens = []
        for val in values:
            all_tokens.extend(self._tokenize(val))

        n_total = len(all_tokens)  # Te_words
        n_unique = len(set(all_tokens))  # Ue_words

        # Edge case: Single occurrence or No words
        if n_total <= 1:
            # Convention: If seen only once, predictability is unknown/null for rules.
            # We assume He=0 because we have no evidence of redundancy.
            return 0.0

        # Raw Redundancy (Compactness)
        # If I say "ER+" 100 times: Total=100, Unique=1. Raw = 99/100 = 0.99
        # If I say 100 different things: Total=100, Unique=100. Raw = 0/100 = 0.0
        he_raw = (n_total - n_unique) / n_total

        # Apply Sigmoid polarization
        he_final = self._sigmoid(he_raw)

        return he_final  # ∈ [0, 1]

    def compute_all(self) -> dict[str, float]:
        """Compute He for all entities."""
        scores = {}
        for entity in self.entities_values:
            scores[entity] = self.compute(entity)
        return scores

    def to_csv(self, output_path: str):
        """Save analysis to CSV."""
        scores = self.compute_all()
        rows: list[dict[str, float | str | int]] = []
        for entity, score in scores.items():
            values = self.entities_values[entity]
            all_tokens = []
            for v in values:
                all_tokens.extend(self._tokenize(v))

            rows.append(
                {
                    "Entity": entity,
                    "He": round(score, 4),
                    "Total_Occurrences": len(values),
                    "Total_Words": len(all_tokens),
                    "Unique_Words": len(set(all_tokens)),
                }
            )

        rows.sort(key=lambda x: x["He"], reverse=True)

        with open(output_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "Entity",
                    "He",
                    "Total_Occurrences",
                    "Total_Words",
                    "Unique_Words",
                ],
            )
            writer.writeheader()
            writer.writerows(rows)

        try:
            rel = os.path.relpath(str(output_path))
        except ValueError:
            rel = str(output_path)
        print(f"Results saved to {rel}")


# ==================================================================================
# MAIN EXECUTION
# ==================================================================================
def load_brat_corpus_simple(data_dirs):
    """Simple loader reusing logic from templatability."""
    corpus = []
    processed_files = set()
    for d in data_dirs:
        path = Path(d)
        if not path.exists():
            continue
        for ann in path.glob("*.ann"):
            if ann.name in processed_files:
                continue
            processed_files.add(ann.name)

            # Simplified parsing
            anns = []
            try:
                with open(ann, encoding="utf-8") as f:
                    for line in f:
                        if line.startswith("T"):
                            parts = line.strip().split("\t")
                            if len(parts) >= 3:
                                etype = parts[1].split()[0]
                                text = parts[2]
                                anns.append({"entity_type": etype, "text": text})
            except Exception:
                pass

            corpus.append({"file_id": ann.name, "annotations": anns})
    return corpus


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--gs_dir", type=str, default=None)
    parser.add_argument("--pred_dir", type=str, default=None)
    args = parser.parse_args()

    # RELATIVE PATHS
    script_dir = Path(__file__).parent
    root_dir = script_dir.parent.parent

    if args.gs_dir:
        abs_data_dirs = [Path(args.gs_dir)]
    else:
        data_dirs = [
            "src/demne/NER/data/Breast/train",
            "src/demne/NER/data/Breast/val",
            "src/demne/NER/data/Breast/test",
        ]
        abs_data_dirs = [root_dir / d for d in data_dirs]

    corpus_name = Path(args.gs_dir).parent.name if args.gs_dir else "Breast"
    output_file = root_dir / "Results" / f"homogeneity_analysis_{corpus_name}.csv"
    output_file.parent.mkdir(parents=True, exist_ok=True)

    # 1. Load context
    print("Loading corpus...")
    corpus = load_brat_corpus_simple(abs_data_dirs)

    # 2. Compute
    scorer = HomogeneityScorer(corpus)
    scores = scorer.compute_all()

    # 3. Print Top 5
    print("\nTop 5 Homogeneity Scores (He):")
    top5 = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:5]
    print_table(["Entity", "He"], [[e, f"{s:.3f}"] for e, s in top5], [45, 8])

    # 4. Save
    scorer.to_csv(output_file)

    # 5. Canonical Test Verification
    print("\n=== VERIFICATION TEST CANONIQUE ===")
    test_corpus = [
        {
            "annotations": [
                {"entity_type": "TEST_CANONIQUE", "text": "ER positif 80%"} for _ in range(100)
            ]
        }
    ]
    test_scorer = HomogeneityScorer(test_corpus)
    score_canon = test_scorer.compute("TEST_CANONIQUE")
    print(f"Test 'ER positif 80%' x 100 -> Score: {score_canon:.4f} (Attendu: proche de 1.0)")

    if HAS_ECO2AI and not os.environ.get("DISABLE_ECO2AI"):
        tracker.stop()


if __name__ == "__main__":
    main()
