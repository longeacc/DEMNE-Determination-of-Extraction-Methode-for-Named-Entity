"""
Calculate templatability of biomarkers and named entities.

Templatability is the capacity of an entity to follow predictable structured patterns
(formats, constant prefixes/suffixes).

Example: TNM staging always follows the pattern T[0-4]N[0-3]M[0-1].
"""

# pylint: disable=broad-exception-caught
import importlib.util as _il
import json
import math
import os
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from demne._table import print_table

# --- Tunable weights loaded from data/demne_params.json (single source of truth) ---
_pspec = _il.spec_from_file_location("demne_params", Path(__file__).resolve().parent / "params.py")
_pmod = _il.module_from_spec(_pspec)
_pspec.loader.exec_module(_pmod)
PARAMS = _pmod.load_params()

# eco2ai dependencies
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
        project_name="Consumtion_of_E_templatability.py",
        experiment_description="We Calculate...",
        file_name="data/Consumtion_of_Duraxell.csv",
    )
    tracker = Tracker()
    tracker.start()


@dataclass
class BratAnnotation:
    """Represents a BRAT annotation."""

    start: int
    end: int
    text: str
    entity_type: str
    file_id: str | None = None


class TemplatabilityScorer:
    """
    Calcule le score de Templateabilité (Te) pour chaque entité biomédicale.
    Te mesure le degré de prédictibilité structurelle des patterns d'expression.
    """

    def __init__(self, corpus: list[dict[str, Any]]):
        """
        Initialize the scorer with a corpus of annotated documents.

        Args:
            corpus: liste de documents annotés {
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
        self.results_cache = {}

    def compute_from_list(self, values: list[str]) -> float:
        """
        Calcule le score Te directement depuis une liste de chaînes.
        """
        self.entities_values["TEMP_LIST"] = values
        return self.compute("TEMP_LIST")

    def normalize_pattern(self, text: str) -> str:
        """
        Normalise un texte d'entité en template abstrait.
        Ex: "HER2 3+" -> "XXX D+"
        Ex: "ER >80%" -> "XX >DD%"
        Ex: "Ki67 15-20%" -> "XXDD DD-DD%"
        """
        # 1. Strip whitespace
        pattern = text.strip()

        # 2. Abstract Digits -> 'D'
        pattern = re.sub(r"[0-9]", "D", pattern)

        # 3. Abstract Uppercase -> 'X'
        pattern = re.sub(r"[A-ZÀ-ÖØ-Þ]", "X", pattern)

        # 4. Abstract Lowercase -> 'x'
        pattern = re.sub(r"[a-zà-öø-ÿ]", "x", pattern)

        # 5. Simplify repeated types (DD -> D+, XX -> X+) - OPTIONAL, let's keep exact count for now
        # pattern = re.sub(r'D+', 'D+', pattern)
        # pattern = re.sub(r'X+', 'X+', pattern)
        # pattern = re.sub(r'x+', 'x+', pattern)

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
        Retourne un score Te ∈ [0, 100].
        Méthode :
        1. Extraire toutes les mentions de entity_type dans le corpus
        2. Normaliser les patterns : "HER2 3+" → "XXXX D+" (regex abstraction)
        3. Calculer l'entropie de la distribution des patterns normalisés (H)
        4. Normaliser l'entropie par rapport au maximum possible (H_norm = h / ln(n_unique))
        5. Calculer la cohérence structurelle: 1.0 - H_norm
        6. Ajouter un bonus sémantique si présence de marqueurs standards (+ / - / % / > / <)
        7. Te = (cohérence_structurelle + bonus_sémantique) * 100
        """
        values = self.entities_values.get(entity_type, [])
        if not values:
            return 0.0, {}

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
        """Calcule Te pour toutes les entités du corpus (en %)."""
        scores = {}
        for entity_type in self.entities_values.keys():
            scores[entity_type] = self.compute(entity_type)
        return scores

    def to_json(self, output_path: str) -> None:
        """Sauvegarder les résultats dans templatability_analysis.json"""
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


def load_brat_corpus(data_dirs: list[str]) -> list[dict[str, Any]]:
    """
    Load annotations from BRAT files (.ann + .txt) into a corpus list.
    """
    corpus = []
    processed_files = set()

    for d in data_dirs:
        path = Path(d)
        if not path.exists():
            print(f"Warning: {path} does not exist.")
            continue

        for ann_file in path.glob("*.ann"):
            if ann_file.name in processed_files:
                continue
            processed_files.add(ann_file.name)

            # Read annotations
            annotations = []
            with open(ann_file, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line.startswith("T"):
                        continue
                    try:
                        # T1  Entity 10 20  text
                        parts = line.split("\t")
                        info = parts[1].split()
                        entity_type = info[0]
                        start = int(info[1])
                        # Handle discontinuous spans "10 20;30 40" -> take end of first span for simplicity or map properly
                        end_str = info[-1]
                        if ";" in parts[1]:
                            # Simplification: take the last offset as end
                            end_str = parts[1].replace(";", " ").split()[2]
                        end = int(end_str)
                        text = parts[2]

                        annotations.append(
                            BratAnnotation(
                                start=start,
                                end=end,
                                text=text,
                                entity_type=entity_type,
                                file_id=ann_file.name,
                            )
                        )
                    except Exception:
                        continue

            # Read text (optional, not strictly needed for Te but good for corpus object)
            txt_file = ann_file.with_suffix(".txt")
            text_content = ""
            if txt_file.exists():
                with open(txt_file, encoding="utf-8") as f:
                    text_content = f.read()

            corpus.append(
                {
                    "file_id": ann_file.name,
                    "text": text_content,
                    "annotations": annotations,
                }
            )

    print(f"Loaded {len(corpus)} documents.")
    return corpus


def main():
    import argparse

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
    print("\nTop 5 Templatability Scores (Te normalisé [0-1]) :")
    top5 = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:5]
    print_table(["Entity", "Te [0-1]"], [[e, f"{s / 100:.4f}"] for e, s in top5], [45, 10])

    if HAS_ECO2AI and not os.environ.get("DISABLE_ECO2AI"):
        tracker.stop()


if __name__ == "__main__":
    main()
