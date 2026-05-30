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

import csv
import math
import os
from collections import defaultdict
from pathlib import Path

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
    Calcule la fréquence relative de chaque entité.
    Décide si le volume de données est suffisant pour un entraînement statistique (NER).
    """

    def __init__(self, data_dirs):
        self.data_dirs = [Path(d) for d in data_dirs]
        self.entity_counts = defaultdict(int)
        self.total_tokens = 0
        self.total_docs = 0

        # Seuil critique (défini dans decision_config.json: "FREQ_SUFFICIENT": 0.001)
        # Mais ici on utilise un seuil absolu pour le "Rare Warning"
        self.RARE_THRESHOLD_COUNT = 10

    def _count_words(self, text: str) -> int:
        """Tokenisation simple par espace."""
        if not text:
            return 0
        return len(text.split())

    def ingest_document(self, text: str, annotation_lines: list[str]):
        """
        Met à jour les compteurs pour un document donné.
        Permet de tester la logique sans dépendre du système de fichiers.
        """
        # 1. Mise à jour du total de tokens (Dénominateur de la fréquence)
        self.total_tokens += self._count_words(text)
        self.total_docs += 1

        # 2. Mise à jour du compte des entités (Numérateur)
        for line in annotation_lines:
            line = line.strip()
            if line.startswith("T"):
                parts = line.split("\t")
                # Format brat: T1  Entity_Type Start End  Text
                # Parfois: T1  Entity_Type Start End;Start End Text (discontinu)
                if len(parts) >= 2:
                    # parts[1] contient "Entity_Type Start End"
                    # On prend le premier mot comme Entity_Type
                    type_info = parts[1]
                    etype = type_info.split(" ")[0]
                    self.entity_counts[etype] += 1

    def compute_all(self):
        """Parcourt le corpus pour compter tokens et entités via ingest_document."""
        print("Scanning corpus for frequencies...")

        # On doit apparier .txt et .ann
        # Approche simplifiée : on itère sur les .txt et on cherche le .ann correspondant

        for d in self.data_dirs:
            if not d.exists():
                continue

            for txt_file in d.glob("*.txt"):
                try:
                    # Lire le texte
                    text_content = txt_file.read_text(encoding="utf-8")

                    # Chercher l'annotation correspondante
                    ann_file = txt_file.with_suffix(".ann")
                    ann_lines = []
                    if ann_file.exists():
                        ann_lines = ann_file.read_text(encoding="utf-8").splitlines()

                    # Ingérer
                    self.ingest_document(text_content, ann_lines)

                except Exception as e:
                    print(f"Error processing {txt_file}: {e}")

        print(f"Total Corpus: {self.total_docs} docs, {self.total_tokens} words.")

    def get_stats(self):
        """Génère les stats calculées."""
        results = []
        for etype, count in self.entity_counts.items():
            freq = count / self.total_tokens if self.total_tokens > 0 else 0

            # Recommendation
            if count < self.RARE_THRESHOLD_COUNT:
                reco = "RARE -> REGLES/LLM"
            else:
                reco = "FREQUENT -> ML POSSIBLE"

            results.append(
                {
                    "Entity": etype,
                    "Frequency": freq,  # Valeur brute (ex: 0.0004)
                    "Count": count,
                    "Per_1k_tokens": freq * 1000,
                    "Strategy_Hint": reco,
                }
            )

        return sorted(results, key=lambda x: x["Frequency"], reverse=True)

    def draw_histogram(self, results):
        """Dessine un histogramme ASCII logarithmique."""
        print("\n=== DISTRIBUTION DES FREQUENCES (Echelle Log) ===")
        print(f"{'Entity':<25} | {'Count':<6} | {'Hist (Log Scale)'}")
        print("-" * 60)

        for r in results:
            count = r["Count"]
            if count == 0:
                continue
            # Log scale bar: log10(1)=0, log10(10)=1, log10(100)=2...
            bar_len = int(math.log10(count) * 4) + 1
            bar = "█" * bar_len
            print(f"{r['Entity']:<25} | {count:<6} | {bar}")

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
        # MÃªme sources de donnÃ©es pour cohÃ©rence
        data_dirs = [
            root_dir / "src/duraxell/NER/data/Breast/train",
            root_dir / "src/duraxell/NER/data/Breast/val",
            root_dir / "src/duraxell/NER/data/Breast/test",
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
