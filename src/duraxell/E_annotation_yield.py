"""
Calculate 'Annotation Yield' (Yield) Score.

Yield measures the "return on investment" of simple rule-based annotation.
High Yield means simple rules capture most of the Gold Standard (GS) correctly.
Low Yield means rules miss a lot, implying complex patterns requiring ML/LLM or human effort.

Formula: Yield = F1-Score (Harmonic mean of Precision and Recall) of Rules vs GS.
Range: 0.0 (Rules fail completely) to 1.0 (Rules represent the GS perfectly).

Matching mode: IoU-based (Intersection over Union) with configurable threshold.
Two spans match if they share the same entity type AND IoU >= threshold (default 0.5).
This is more realistic than exact offset matching for regex-based extraction.

This metric is crucial for the Decision Tree:
- High Yield -> Rules are sufficient.
- Low Yield -> We need ML or LLM.
"""

import os
from collections import defaultdict
from pathlib import Path

# Eco2AI for energy tracking
try:
    from eco2ai import Tracker, set_params

    HAS_ECO2AI = True
except ImportError:
    HAS_ECO2AI = False

if __name__ == "__main__" and HAS_ECO2AI and not os.environ.get("DISABLE_ECO2AI"):
    set_params(
        project_name="Consumtion_of_E_annotation_yield.py",
        experiment_description="Calculating Annotation Yield (Rules vs GS)",
        file_name="Consumtion_of_Duraxell.csv",
    )
    tracker = Tracker()
    tracker.start()

# ---------------------------------------------------------------------------
# Default IoU threshold – two spans match when IoU >= this value.
# 0.5 is the standard threshold in NER / information extraction tasks.
# ---------------------------------------------------------------------------
DEFAULT_IOU_THRESHOLD = 0.5


class AnnotationYieldScorer:
    """
    Compare les annotations prédites par les règles (Pred) avec les annotations manuelles (GS).
    Calcule le F1-multiclass ou par entité.

    Utilise un matching par IoU (Intersection over Union) au lieu d'un matching
    exact des offsets, ce qui est plus réaliste pour l'évaluation de regex qui
    peuvent capturer des bornes légèrement différentes du Gold Standard.
    """

    def __init__(
        self,
        gs_dir: Path = None,
        pred_dir: Path = None,
        iou_threshold: float = DEFAULT_IOU_THRESHOLD,
    ):
        self.gs_dir = gs_dir
        self.pred_dir = pred_dir
        self.iou_threshold = iou_threshold
        self.tp = defaultdict(int)
        self.fp = defaultdict(int)
        self.fn = defaultdict(int)
        self.population_stats = defaultdict(lambda: {"P": 0, "R": 0, "F1": 0})

    # ------------------------------------------------------------------
    # Parsing
    # ------------------------------------------------------------------

    def _parse_ann(self, file_path: Path) -> list[tuple[str, int, int]]:
        """
        Extract simplified annotations: [(Entity, Start, End), ...]
        """
        if not file_path or not file_path.exists():
            return []

        try:
            with open(file_path, encoding="utf-8") as f:
                content = f.read()
                return self._parse_ann_content(content)
        except Exception:
            pass
        return []

    def _parse_ann_content(self, content: str) -> list[tuple[str, int, int]]:
        """Parses annotation content string into list of (Type, Start, End)."""
        anns = []
        for line in content.splitlines():
            if line.startswith("T"):
                parts = line.strip().split("\t")
                if len(parts) >= 2:
                    meta = parts[1].split()
                    if len(meta) >= 3:
                        etype = meta[0]
                        try:
                            start = int(meta[1])
                            end = int(meta[-1])
                            anns.append((etype, start, end))
                        except ValueError:
                            pass
        return anns

    # ------------------------------------------------------------------
    # IoU helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _iou(s1: int, e1: int, s2: int, e2: int) -> float:
        """Compute Intersection-over-Union of two [start, end) spans."""
        inter = max(0, min(e1, e2) - max(s1, s2))
        union = (e1 - s1) + (e2 - s2) - inter
        return inter / union if union > 0 else 0.0

    def _match_iou(
        self, gs_spans: list[tuple[int, int]], pred_spans: list[tuple[int, int]]
    ) -> tuple[int, int, int]:
        """
        Greedy one-to-one IoU matching for spans of the SAME entity type.
        Returns (tp, fp, fn).
        """
        if not gs_spans and not pred_spans:
            return 0, 0, 0
        if not gs_spans:
            return 0, len(pred_spans), 0
        if not pred_spans:
            return 0, 0, len(gs_spans)

        # Build all candidate pairs with IoU >= threshold, sorted best-first
        pairs = []
        for gi, (gs, ge) in enumerate(gs_spans):
            for pi, (ps, pe) in enumerate(pred_spans):
                score = self._iou(gs, ge, ps, pe)
                if score >= self.iou_threshold:
                    pairs.append((score, gi, pi))
        pairs.sort(key=lambda t: -t[0])

        matched_g = set()
        matched_p = set()
        tp = 0
        for _score, gi, pi in pairs:
            if gi in matched_g or pi in matched_p:
                continue
            tp += 1
            matched_g.add(gi)
            matched_p.add(pi)

        fp = len(pred_spans) - len(matched_p)
        fn = len(gs_spans) - len(matched_g)
        return tp, fp, fn

    # ------------------------------------------------------------------
    # Counting
    # ------------------------------------------------------------------

    def update_counts(self, gs_content: str, pred_content: str):
        """Update TP/FP/FN counts using IoU-based matching."""
        gs_anns = self._parse_ann_content(gs_content)
        pred_anns = self._parse_ann_content(pred_content)

        # Group by entity type
        gs_by_type: dict[str, list[tuple[int, int]]] = defaultdict(list)
        pred_by_type: dict[str, list[tuple[int, int]]] = defaultdict(list)

        for etype, s, e in gs_anns:
            gs_by_type[etype].append((s, e))
        for etype, s, e in pred_anns:
            pred_by_type[etype].append((s, e))

        all_types = set(gs_by_type.keys()) | set(pred_by_type.keys())
        for etype in all_types:
            tp, fp, fn = self._match_iou(
                gs_by_type.get(etype, []),
                pred_by_type.get(etype, []),
            )
            self.tp[etype] += tp
            self.fp[etype] += fp
            self.fn[etype] += fn

    def compute_all(self):
        """Scan folders and match files."""
        if not self.gs_dir or not self.pred_dir:
            return {}

        print(f"Comparing GS ({self.gs_dir.name}) vs Pred ({self.pred_dir.name})...")

        gs_files = list(self.gs_dir.glob("*.ann"))

        for gs_file in gs_files:
            pred_file = self.pred_dir / gs_file.name

            try:
                gs_content = gs_file.read_text(encoding="utf-8")
                pred_content = ""
                if pred_file.exists():
                    pred_content = pred_file.read_text(encoding="utf-8")

                self.update_counts(gs_content, pred_content)
            except Exception as e:
                print(f"Error processing {gs_file}: {e}")

        return self.get_scores()

    def get_scores(self) -> dict[str, dict[str, float]]:
        """Calculate Precision, Recall, F1 per entity."""
        scores = {}
        all_types = set(self.tp.keys()) | set(self.fp.keys()) | set(self.fn.keys())

        for etype in all_types:
            tp = self.tp[etype]
            fp = self.fp[etype]
            fn = self.fn[etype]

            p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0

            scores[etype] = {"Precision": p, "Recall": r, "F1-Yield": f1}

        return scores

    def print_report(self):
        scores = self.get_scores()
        print(f"\n=== ANNOTATION YIELD (F1-score Rules vs GS, IoU≥{self.iou_threshold}) ===")
        print(
            f"{'Entity':<25} | {'Yield (F1)':<10} | {'Prec':<6} | {'Rec':<6} | {'TP':<4} | {'FP':<4} | {'FN':<4}"
        )
        print("-" * 80)

        for etype, metas in sorted(scores.items(), key=lambda x: x[1]["F1-Yield"], reverse=True):
            f1 = metas["F1-Yield"]
            p = metas["Precision"]
            r = metas["Recall"]
            print(
                f"{etype:<25} | {f1:<10.4f} | {p:<6.2f} | {r:<6.2f} | {self.tp[etype]:<4} | {self.fp[etype]:<4} | {self.fn[etype]:<4}"
            )

        return scores


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--gs_dir", type=str, default=None)
    parser.add_argument("--pred_dir", type=str, default=None)
    args = parser.parse_args()

    script_dir = Path(__file__).parent

    # Paths to comparison sets
    if args.gs_dir and args.pred_dir:
        gs_dir = Path(args.gs_dir)
        pred_dir = Path(args.pred_dir)
    else:
        base_esmo = Path(os.environ.get("DEMNE_ESMO_DIR", "data/ESMO2025"))
        gs_dir = base_esmo / "Breast/RCP/evaluation_set_breast_cancer_GS"
        pred_dir = base_esmo / "Breast/RCP/evaluation_set_breast_cancer_pred_rules"

    # If using from root
    if not gs_dir.exists():
        gs_dir = base_esmo / "Breast/RCP/evaluation_set_breast_cancer_GS"
        pred_dir = base_esmo / "Breast/RCP/evaluation_set_breast_cancer_pred_rules"

    if not gs_dir.exists():
        print("Datasets for Yield calculation not found.")
        print(f"GS Search Path: {gs_dir}")

        # Fallback simulation
        print("\n[SIMULATION MODE] Generating dummy Yield scores...")
        sim_scores = {
            "Estrogen_receptor": 0.85,
            "HER2": 0.92,
            "Ki67": 0.75,
            "Infiltrating_carcinoma": 0.40,
        }
        print(f"{'Entity':<25} | {'Yield (F1)':<10}")
        print("-" * 40)
        for k, v in sim_scores.items():
            print(f"{k:<25} | {v:.4f} (Simulated)")
        return

    scorer = AnnotationYieldScorer(gs_dir, pred_dir)
    scorer.compute_all()
    scorer.print_report()

    if HAS_ECO2AI and "tracker" in globals():
        tracker.stop()


if __name__ == "__main__":
    main()
