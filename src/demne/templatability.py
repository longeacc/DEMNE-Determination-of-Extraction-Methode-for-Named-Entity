"""
Calculate templatability of biomarkers and named entities.

Templatability is the capacity of an entity to follow predictable structured patterns
(formats, constant prefixes/suffixes).

Example: TNM staging always follows the pattern T[0-4]N[0-3]M[0-1].
"""

# pylint: disable=broad-exception-caught
import json
import math
import os
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

# Re-export for backward compatibility
__all__ = ["BratAnnotation", "TemplatabilityScorer", "load_brat_corpus"]

from demne._brat import BratAnnotation, load_brat_corpus
from demne._table import print_table
from demne._tracking import start_tracker, stop_tracker
from demne.params import load_params

# --- Tunable weights loaded from data/demne_params.json (single source of truth) ---
PARAMS = load_params()


class TemplatabilityScorer:
    """
    Compute the Templatability score (Te) for each biomedical entity.
    Te measures the degree of structural predictability of expression patterns.
    """

    def __init__(self, corpus: list[dict[str, Any]]):
        """
        Initialize the scorer with a corpus of annotated documents.

        Args:
            corpus: list of annotated documents {
                'text': str,
                'annotations': list[BratAnnotation] or list[dict],
                'file_id': str (optional)
            }
        """
        self.corpus = corpus
        # Pre-process: group entity values by type
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

        # Cache for compute results
        self.results_cache: dict[str, Any] = {}

    def compute_from_list(self, values: list[str]) -> float:
        """Compute Te score directly from a list of strings."""
        self.entities_values["TEMP_LIST"] = values
        return self.compute("TEMP_LIST")

    def normalize_pattern(self, text: str) -> str:
        """Normalise an entity string into an abstract pattern.

        Templatability measures a CONSTANT shape: independent of case and length.
        e.g. "HER2 3+" / "her2 3+" -> "LD+"; "SEIN   DROIT" / "sein droit" -> "L L".
        Without it, case and exact length inflate the number of distinct patterns
        -> higher entropy -> artificially low Te. Must stay consistent with the
        dashboard MetricsCalculator (see the routing-equivalence intent).
        """
        pattern = text.strip()
        # 1. Letters -> 'L' (case-insensitive). BEFORE the digits: otherwise the letter
        #    rule would overwrite the marker 'D' (which is a letter), making digits
        #    and letters indistinguishable and the digit_bonus inert.
        pattern = re.sub(r"[A-Za-zÀ-ÖØ-Þß-ÿ]", "L", pattern)
        # 2. Chiffres -> 'D'
        pattern = re.sub(r"[0-9]", "D", pattern)
        # 3. Collapse repeats (LLD == LLLD, "L   L" == "L L")
        pattern = re.sub(r"(.)\1+", r"\1", pattern)
        return pattern

    def _calculate_entropy(self, patterns: list[str]) -> float:
        """Calculate Shannon entropy of pattern distribution."""
        if not patterns:
            return 0.0

        counter = Counter(patterns)
        total = len(patterns)
        entropy = 0.0

        for count in counter.values():
            p = count / total
            entropy -= p * math.log(p)

        return entropy

    def compute(self, entity_type: str) -> float:
        """
        Return a Te score ∈ [0, 100].
        Method:
        1. Extract all mentions of entity_type from the corpus
        2. Normalize patterns: "HER2 3+" → "XXXX D+" (regex abstraction)
        3. Compute entropy of the normalized pattern distribution (H)
        4. Normalize entropy against maximum possible (H_norm = h / ln(n_unique))
        5. Compute structural consistency: 1.0 - H_norm
        6. Add semantic bonus if standard markers are present (+ / - / % / > / <)
        7. Te = (structural_consistency + semantic_bonus) * 100
        """
        values = self.entities_values.get(entity_type, [])
        if not values:
            return 0.0

        total_count = len(values)
        normalized_patterns = [self.normalize_pattern(v) for v in values]

        # Entropy calculation
        h = self._calculate_entropy(normalized_patterns)

        # Normalize entropy: H_max = log(N) where N is number of unique patterns observed
        # Or better: N is count of items? No, entropy is maximized when uniform distribution over unique patterns
        # Standard relative entropy usually divides by log(len(unique_patterns))
        # But if unique_patterns is 1, log(1)=0 -> division by zero.
        # Here we want a measure of predictability.
        # If entropy is 0 -> perfectly predictable -> Te should be 1.
        # If entropy is high -> unpredictable -> Te should be 0.

        unique_patterns = set(normalized_patterns)
        num_unique = len(unique_patterns)

        if num_unique <= 1:
            h_norm = 0.0
        else:
            h_norm = h / math.log(num_unique)

        # Structure Score based on entropy (as per documentation: 1 - h_norm)
        structure_consistency = 1.0 - h_norm
        pattern_counts = Counter(normalized_patterns)

        # Semantic Bonus for standard markers
        _tb = PARAMS["templatability_bonus"]
        bonus_semantic = 0.0
        # Check for numeric patterns, symbols
        has_digit = any("D" in p for p in unique_patterns)
        has_symbol = any(c in p for p in unique_patterns for c in ["%", "+", "-", ">", "<"])
        if has_symbol:
            bonus_semantic += _tb["symbol_bonus"]
        if has_digit and structure_consistency > _tb["digit_gate"]:
            bonus_semantic += _tb["digit_bonus"]

        # Te calculation
        # Baseline is structure_consistency
        raw_score = structure_consistency + bonus_semantic
        raw_score = min(1.0, max(0.0, raw_score))

        # Convert to percentage [0-100]
        te_val = raw_score * 100.0

        # Store detailed stats for report
        self.results_cache[entity_type] = {
            "count": total_count,
            "unique_patterns": num_unique,
            "entropy_consistency": structure_consistency,
            "entropy": h,
            "templatability_score": te_val,
            "top_patterns": pattern_counts.most_common(5),
        }

        return te_val

    def compute_all(self) -> dict[str, float]:
        """Compute Te for all entities in the corpus (as %)."""
        scores = {}
        for entity_type in self.entities_values.keys():
            scores[entity_type] = self.compute(entity_type)
        return scores

    def to_json(self, output_path: str | Path) -> None:
        """Save results to templatability_analysis.json."""
        output = {}
        for entity_type, stats in self.results_cache.items():
            # Convert stats to JSON serializable format
            output[entity_type] = {
                "count": stats["count"],
                "unique_patterns": stats["unique_patterns"],
                "templatability_score": round(
                    stats["templatability_score"], 1
                ),  # Round to 1 decimal place
                "top_patterns": [f"{p} ({c})" for p, c in stats["top_patterns"]],
            }

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(output, f, indent=4, ensure_ascii=False)
        try:
            rel = os.path.relpath(str(output_path))
        except ValueError:
            rel = str(output_path)
        print(f"Results saved to {rel}")


# ==================================================================================
# SCRIPT UTILS (Load Data & Run)
# ==================================================================================


def main() -> None:
    """CLI entry point: writes Results/templatability_analysis_<corpus>.json."""
    import argparse

    _tracker = start_tracker("DEMNE-templatability", "We Calculate...")

    parser = argparse.ArgumentParser()
    parser.add_argument("--gs_dir", type=str, default=None)
    parser.add_argument("--pred_dir", type=str, default=None)
    args = parser.parse_args()

    script_dir = Path(__file__).parent
    root_dir = script_dir.parent.parent

    # If the user passed gs_dir, use that instead of the hardcoded paths
    if args.gs_dir:
        data_dirs = [Path(args.gs_dir)]
    else:
        # Configuration
        data_dirs_rel = [
            "src/demne/NER/data/Breast/train",
            "src/demne/NER/data/Breast/val",
            "src/demne/NER/data/Breast/test",
        ]
        data_dirs = [root_dir / d for d in data_dirs_rel]

    corpus_name = Path(args.gs_dir).parent.name if args.gs_dir else "Breast"
    output_file = root_dir / "Results" / f"templatability_analysis_{corpus_name}.json"

    # Ensure output directory exists
    output_file.parent.mkdir(parents=True, exist_ok=True)

    # 1. Load Data
    corpus = load_brat_corpus([str(p) for p in data_dirs])

    # 2. Initialize Scorer
    scorer = TemplatabilityScorer(corpus)

    # 3. Compute All
    scores = scorer.compute_all()

    # 4. Print & Save
    scorer.to_json(output_file)

    # Optional: Print Top 5
    print("\nTop 5 Templatability Scores (Te normalized [0-1]):")
    top5 = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:5]
    print_table(["Entity", "Te [0-1]"], [[e, f"{s / 100:.4f}"] for e, s in top5], [45, 10])

    stop_tracker(_tracker)


if __name__ == "__main__":
    main()
