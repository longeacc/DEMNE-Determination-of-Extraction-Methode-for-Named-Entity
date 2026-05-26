import json
import os
import sys

# Ajout du chemin racine pour les imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from duraxell.REST_interface.convergence_analyzer import ConvergenceAnalyzer
from duraxell.REST_interface.rest_annotator import RESTAnnotator
from duraxell.REST_interface.rest_decision_bridge import RESTDecisionBridge
from duraxell.REST_interface.rest_evaluator import RESTEvaluator


def main():
    print("================================================================")
    print("      DÉMONSTRATION INTEGRATION REST-INTERFACE (DuraXELL)")
    print("================================================================")

    # 1. Chargement de documents (Simulés ici pour la démo autonome)
    print("\n[ETAPE 1] Chargement du Corpus Pilote...")
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
    print(f"   > Chargé {len(docs)} documents cliniques simulés.")

    # 2. Annotation Pilote (RESTAnnotator)
    print("\n[ETAPE 2] Annotation Rapide (Simulation Expert)...")
    annotator = RESTAnnotator(output_dir="Evaluation/REST_Annotations")
    # On utilise le mode 'automated_test' qui utilise des regex pour simuler un expert trouvant les entités
    annotations = annotator.annotate_batch(
        docs, entity_types=["Estrogen_receptor", "HER2", "Ki67"], mode="automated_test"
    )
    print(f"   > Total annotations collectées : {len(annotations)}")

    # 3. Évaluation Empirique (RESTEvaluator)
    print("\n[ETAPE 3] Calcul des Métriques Empiriques (Bottom-Up)...")
    evaluator = RESTEvaluator()
    rest_reports = []

    for entity in ["Estrogen_receptor", "HER2", "Ki67"]:
        report = evaluator.evaluate_entity(entity, annotations)
        rest_reports.append(report)
        print(
            f"   > Entité '{entity}': Te_obs={report.empirical_te:.2f}, He_obs={report.empirical_he:.2f}"
        )

    # 4. Chargement Arbre de Décision (Top-Down)
    print("\n[ETAPE 4] Comparaison avec l'Arbre de Décision (Top-Down)...")
    config_path = "data/decision_config.json"
    if os.path.exists(config_path):
        with open(config_path, encoding="utf-8") as f:
            tree_config = json.load(f)
    else:
        print("   (Fichier data/decision_config.json absent, utilisation mock)")
        tree_config = {
            "Estrogen_receptor": {
                "method": "FEUILLE NER À BASE DE RÈGLES",
                "metrics": {"Te": 0.9},
            },
            "HER2": {"method": "FEUILLE NER À BASE DE RÈGLES", "metrics": {"Te": 0.85}},
            "Ki67": {"method": "FEUILLE ML LÉGER NER", "metrics": {"Te": 0.4}},
        }

    # 5. Pont de Décision (Bridge)
    bridge = RESTDecisionBridge()
    convergence_results = bridge.compare(tree_config, rest_reports)

    # 6. Analyse Convergence
    print("\n[ETAPE 5] Rapport de Convergence...")
    analyzer = ConvergenceAnalyzer()
    analyzer.analyze_convergence(convergence_results)

    print("\n================================================================")
    print("      DÉMONSTRATION TERMINÉE - CHECK RESULTS/FIGURES")
    print("================================================================")


if __name__ == "__main__":
    main()
