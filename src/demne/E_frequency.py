"""
Calculate Entity Frequency Score (Freq).

Freq measures how often an entity appears relative to the total corpus size.
Formula: Freq = (Total Occurrences of Entity) / (Total Words in Corpus)

Interpretation:
- Freq >> 0.001 (0.1%): Frequent entity. Sufficient data for ML training (NER).
- Freq << 0.001: Rare entity (<10 occurrences). Sparse data problem.
  -> Strategy: Use Rules (if structurally simple) or LLM (few-shot). ML will fail.

Outputs:
- frequency_analysis.csv: Detailed stats per entity.
- frequency_histogram.txt: Log-scale visual distribution.
"""

# pylint: disable=broad-exception-caught

import csv
import importlib.util as _il
import math
import os
from collections import defaultdict
from pathlib import Path

from demne._table import print_table

# --- Tunable thresholds loaded from data/demne_params.json (single source of truth) ---
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
        project_name="Consumtion_of_E_frequency.py",
        experiment_description="Calculating Entity Frequency",
        file_name="data/Consumtion_of_Duraxell.csv",
    )
    tracker = Tracker()
    tracker.start()


class FrequencyScorer:
    """
    Compute the relative frequency of each entity.
    Decide whether the data volume is sufficient for statistical NER training.
    """

    def __init__(self, data_dirs):
        self.data_dirs = [Path(d) for d in data_dirs]
        self.entity_counts = defaultdict(int)
        self.total_tokens = 0
        self.total_docs = 0

        # Absolute threshold for the "Rare Warning" (data/demne_params.json → frequency)
        self.RARE_THRESHOLD_COUNT = PARAMS["frequency"]["rare_threshold_count"]

    def _count_words(self, text: str) -> int:
        """Tokenise by non-whitespace sequences (consistent with E_tfidf)."""
        if not text:
            return 0
        import re

        return len(re.findall(r"\S+", text))

    def ingest_document(self, text: str, annotation_lines: list[str]):
        """
        Update counters for a given document.
        Allows testing the logic without depending on the filesystem.
        """
        # 1. Update total token count (frequency denominator)
        self.total_tokens += self._count_words(text)
        self.total_docs += 1

        # 2. Update entity count (frequency numerator)
        for line in annotation_lines:
            line = line.strip()
            if line.startswith("T"):
                parts = line.split("\t")
                # Format brat: T1  Entity_Type Start End  Text
                # Sometimes: T1  Entity_Type Start End;Start End Text (discontinuous)
                if len(parts) >= 2:
                    # parts[1] contains "Entity_Type Start End"; first token is the type
                    type_info = parts[1]
                    etype = type_info.split(" ")[0]
                    self.entity_counts[etype] += 1

    def compute_all(self):
        """Scan the corpus to count tokens and entities via ingest_document."""
        print("Scanning corpus for frequencies...")

        # Pair .txt and .ann files: iterate over .txt and look up the matching .ann

        for d in self.data_dirs:
            if not d.exists():
                continue

            for txt_file in d.glob("*.txt"):
                try:
                    text_content = txt_file.read_text(encoding="utf-8")

                    ann_file = txt_file.with_suffix(".ann")
                    ann_lines = []
                    if ann_file.exists():
                        ann_lines = ann_file.read_text(encoding="utf-8").splitlines()

                    self.ingest_document(text_content, ann_lines)

                except Exception as e:
                    print(f"Error processing {txt_file}: {e}")

        print(f"Total Corpus: {self.total_docs} docs, {self.total_tokens} words.")

    def get_stats(self):
        """Generate computed statistics."""
        results = []
        for etype, count in self.entity_counts.items():
            freq = count / self.total_tokens if self.total_tokens > 0 else 0

            # Recommendation
            if count < self.RARE_THRESHOLD_COUNT:
                reco = "RARE -> RULES/LLM"
            else:
                reco = "FREQUENT -> ML POSSIBLE"

            results.append(
                {
                    "Entity": etype,
                    "Frequency": freq,  # Raw value (e.g. 0.0004)
                    "Count": count,
                    "Per_1k_tokens": freq * 1000,
                    "Strategy_Hint": reco,
                }
            )

        return sorted(results, key=lambda x: x["Frequency"], reverse=True)

    def draw_histogram(self, results):
        """Draw a logarithmic ASCII histogram."""
        print("\n=== FREQUENCY DISTRIBUTION (Log Scale) ===")
        rows = []
        for r in results:
            count = r["Count"]
            if count == 0:
                continue
            bar_len = int(math.log10(count) * 4) + 1
            rows.append([r["Entity"], str(count), "█" * bar_len])
        print_table(["Entity", "Count", "Hist (Log Scale)"], rows, [45, 6, 16])

    def to_csv(self, output_path):
        data = self.get_stats()
        with open(output_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "Entity",
                    "Frequency",
                    "Count",
                    "Per_1k_tokens",
                    "Strategy_Hint",
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
def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--gs_dir", type=str, default=None)
    parser.add_argument("--pred_dir", type=str, default=None)
    args = parser.parse_args()

    script_dir = Path(__file__).parent
    root_dir = script_dir.parent.parent

    if args.gs_dir:
        data_dirs = [Path(args.gs_dir)]
    else:
        # Same data sources for consistency
        data_dirs = [
            root_dir / "src/demne/NER/data/Breast/train",
            root_dir / "src/demne/NER/data/Breast/val",
            root_dir / "src/demne/NER/data/Breast/test",
        ]

    corpus_name = Path(args.gs_dir).parent.name if args.gs_dir else "Breast"
    output_file = root_dir / "Results" / f"frequency_analysis_{corpus_name}.csv"
    output_file.parent.mkdir(parents=True, exist_ok=True)

    scorer = FrequencyScorer(data_dirs)
    scorer.compute_all()

    stats = scorer.get_stats()
    scorer.draw_histogram(stats)
    scorer.to_csv(output_file)

    if HAS_ECO2AI and not os.environ.get("DISABLE_ECO2AI"):
        tracker.stop()


if __name__ == "__main__":
    main()
