import json
import os
import sys

# Add repo root to path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from demne.REST_interface.convergence_analyzer import ConvergenceAnalyzer
from demne.REST_interface.rest_annotator import RESTAnnotator
from demne.REST_interface.rest_decision_bridge import RESTDecisionBridge
from demne.REST_interface.rest_evaluator import RESTEvaluator


def main():
    print("================================================================")
    print("      REST-INTERFACE INTEGRATION DEMO (DuraXELL)")
    print("================================================================")

    # 1. Load documents (simulated here for standalone demo)
    print("\n[STEP 1] Loading Pilot Corpus...")
    docs = [
        (
            "doc_001",
            "Patient presents with Invasive Ductal Carcinoma. Estrogen Receptor is positive (100%). HER2 is negative score 0.",
        ),
        (
            "doc_002",
            "Breast cancer diagnosis. ER: 90% positive. PR: 20% positive. HER2 status: negative.",
        ),
        (
            "doc_003",
            "Biopsy results: ER positive (strong intensity). HER2 negative (1+). Ki67 index is 15%.",
        ),
        (
            "doc_004",
            "Tumor phenotype: Estrogen Receptor positive. Progesterone Receptor negative. HER2 equivocal (2+).",
        ),
        (
            "doc_005",
            "Pathology report. ER neg. PR neg. HER2 positive (3+). Triple negative status excluded.",
        ),
    ]
    print(f"   > Loaded {len(docs)} simulated clinical documents.")

    # 2. Annotation Pilote (RESTAnnotator)
    print("\n[ETAPE 2] Annotation Rapide (Simulation Expert)...")
    annotator = RESTAnnotator(output_dir="Evaluation/REST_Annotations")
    # 'automated_test' mode uses regex to simulate an expert finding entities
    annotations = annotator.annotate_batch(
        docs, entity_types=["Estrogen_receptor", "HER2", "Ki67"], mode="automated_test"
    )
    print(f"   > Total annotations collected: {len(annotations)}")

    # 3. Empirical Evaluation (RESTEvaluator)
    print("\n[STEP 3] Computing Empirical Metrics (Bottom-Up)...")
    evaluator = RESTEvaluator()
    rest_reports = []

    for entity in ["Estrogen_receptor", "HER2", "Ki67"]:
        report = evaluator.evaluate_entity(entity, annotations)
        rest_reports.append(report)
        print(
            f"   > Entity '{entity}': Te_obs={report.empirical_te:.2f}, He_obs={report.empirical_he:.2f}"
        )

    # 4. Load Decision Tree (Top-Down)
    print("\n[STEP 4] Comparing with Decision Tree (Top-Down)...")
    config_path = "data/decision_config.json"
    if os.path.exists(config_path):
        with open(config_path, encoding="utf-8") as f:
            tree_config = json.load(f)
    else:
        print("   (data/decision_config.json not found — using mock)")
        tree_config = {
            "Estrogen_receptor": {
                "method": "RULES-BASED NER LEAF",
                "metrics": {"Te": 0.9},
            },
            "HER2": {"method": "RULES-BASED NER LEAF", "metrics": {"Te": 0.85}},
            "Ki67": {"method": "LIGHTWEIGHT ML NER LEAF", "metrics": {"Te": 0.4}},
        }

    # 5. Decision Bridge
    bridge = RESTDecisionBridge()
    convergence_results = bridge.compare(tree_config, rest_reports)

    # 6. Convergence Analysis
    print("\n[STEP 5] Convergence Report...")
    analyzer = ConvergenceAnalyzer()
    analyzer.analyze_convergence(convergence_results)

    print("\n================================================================")
    print("      DEMO COMPLETE - CHECK RESULTS/FIGURES")
    print("================================================================")


if __name__ == "__main__":
    main()
