"""
Build and execute the DuraXELL Decision Tree.
Generates 'decision_config.json' and 'output_decision.txt'.

Decision Tree Logic (Priority Order):
1. Templatability (Te) & Homogeneity (He) (Structure) -> HIGH? -> RULES
2. Risk Context (R) -> HIGH? -> LLM / REVIEW
3. Frequency (Freq) & Annotation Yield -> RULES vs ML (NER) vs LLM

Outputs:
- decision_config.json: Machine-readable config for the orchestrator.
- output_decision.txt: Human-readable report.
"""

import csv
import json
from pathlib import Path
from typing import Any

# Eco2AI tracking
try:
    from eco2ai import Tracker, set_params

    HAS_ECO2AI = True
except ImportError:
    HAS_ECO2AI = False

if __name__ == "__main__" and HAS_ECO2AI:
    set_params(
        project_name="Consumtion_of_E_creation_arbre_decision.py",
        experiment_description="Building Decision Tree Config",
        file_name="Consumtion_of_Duraxell.csv",
    )
    tracker = Tracker()
    tracker.start()

# Import metrics scorers (Assuming they are in same package or path)
try:
    # We try local imports if running as script in same dir
    from E_annotation_yield import AnnotationYieldScorer
except ImportError:
    try:
        from duraxell.E_annotation_yield import AnnotationYieldScorer
    except ImportError:
        try:
            from .E_annotation_yield import AnnotationYieldScorer
        except (ImportError, SystemError):
            pass


class DecisionTreeBuilder:
    def __init__(self, config_path: Path):
        self.config_path = config_path
        self.decisions = {}

        # --- CALIBRATED THRESHOLDS (To heavily favor RULES) ---
        self.THRESHOLDS = {
            "TE_HIGH": 0.10,
            "HE_HIGH": 0.85,
            "R_HIGH": 0.25,
            "FEAS_NER": 0.50,
        }

    # Nombre minimum d'occurrences pour que Te soit fiable (Aligné avec THRESHOLDS_JUSTIFICATION.md)
    MIN_TE_SAMPLES = 2

    def validate_thresholds_kfold(self, entities_metrics: dict[str, dict[str, float]], k: int = 5):
        """
        Validation croisée k-fold sur les seuils : partitionner le corpus en k plis,
        calibrer les seuils sur k-1 plis, mesurer la stabilité des décisions sur le pli restant.
        """
        import random

        entities = list(entities_metrics.keys())
        if len(entities) < k:
            print(f"Pas assez d'entités pour une CV {k}-fold.")
            return

        random.shuffle(entities)

        # Split en k plis
        folds = [entities[i::k] for i in range(k)]
        stabilities = []

        print(f"--- Début de la validation croisée {k}-fold des seuils ---")

        for i in range(k):
            test_entities = folds[i]
            train_entities = [ent for j, f in enumerate(folds) if j != i for ent in f]

            # Simulation d'une calibration : modification mineure d'un seuil basée sur le train set
            train_te_vals = sorted([entities_metrics[ent].get("Te", 0.0) for ent in train_entities])
            if train_te_vals:
                # 75e percentile manuel
                idx = int(len(train_te_vals) * 0.75)
                calibrated_te_high = (
                    train_te_vals[idx] if idx < len(train_te_vals) else self.THRESHOLDS["TE_HIGH"]
                )
            else:
                calibrated_te_high = self.THRESHOLDS["TE_HIGH"]

            # Stocker l'ancien
            old_te_high = self.THRESHOLDS["TE_HIGH"]
            # Appliquer le seuil calibré
            self.THRESHOLDS["TE_HIGH"] = calibrated_te_high

            # Mesurer l'accord (stabilité) entre les règles par défaut et les calibrées
            matches = 0
            for ent in test_entities:
                metrics = entities_metrics[ent]
                # Modèle origine
                self.THRESHOLDS["TE_HIGH"] = old_te_high
                orig_decision = self.analyze_entity(ent, metrics).get("method", "")
                # Modèle calibré
                self.THRESHOLDS["TE_HIGH"] = calibrated_te_high
                new_decision = self.analyze_entity(ent, metrics).get("method", "")

                if orig_decision == new_decision:
                    matches += 1

            stability = matches / max(1, len(test_entities))
            stabilities.append(stability)
            self.THRESHOLDS["TE_HIGH"] = old_te_high  # Reset

        avg_stability = sum(stabilities) / len(stabilities)
        print(f"Stabilité moyenne des décisions (K-Fold, k={k}): {avg_stability:.2%}")

    def analyze_entity(self, entity: str, metrics: dict[str, float]) -> dict[str, Any]:
        """Arbre de décision simplifié : Te++ → He++ → R− → RÈGLES | Feas++ → TBM | LLM.

        Args:
            entity: Nom de l'entité cible (ex: 'Estrogen_receptor').
            metrics: Dictionnaire {Te, He, R, Feas, Freq, Te_count, ...}.

        Returns:
            Dictionnaire avec 'method', 'justification', 'trace'.
        """
        te: float = metrics.get("Te", 0.0)
        te_count: int = metrics.get("Te_count", 0)
        he: float = metrics.get("He", 0.0)
        r_score: float = metrics.get("R", 0.0)
        feas: float = metrics.get("Feas", 0.0)

        # Convert Te and He to 0-1 scale to match new thresholds correctly
        if te > 1.0:
            te /= 100.0
        if he > 1.0:
            he /= 100.0

        # Garde-fou existant : Te non fiable si trop peu d'échantillons
        if te_count < self.MIN_TE_SAMPLES:
            te = 0.0

        path_trace: list[str] = []

        # NOEUD 1 : Templatabilité élevée ?
        path_trace.append("Te++ ?")
        if te >= self.THRESHOLDS["TE_HIGH"]:
            path_trace.append("Oui → He++ ?")
            # NOEUD 2 : Homogénéité élevée ?
            if he >= self.THRESHOLDS["HE_HIGH"]:
                path_trace.append("Oui → R− ?")
                # NOEUD 3 : Risque contextuel acceptable ?
                if r_score <= self.THRESHOLDS["R_HIGH"]:
                    path_trace.append("Oui → [RÈGLES]")
                    return {
                        "method": "RÈGLES",
                        "justification": f"Te={te:.1f}≥{self.THRESHOLDS['TE_HIGH']}, He={he:.1f}≥{self.THRESHOLDS['HE_HIGH']}, R={r_score:.3f}≤{self.THRESHOLDS['R_HIGH']}",
                        "trace": path_trace,
                    }
                else:
                    path_trace.append(
                        f"Non (R={r_score:.3f} > {self.THRESHOLDS['R_MAX']}) → Feas++ ?"
                    )
            else:
                path_trace.append(f"Non (He={he:.1f} < {self.THRESHOLDS['HE_HIGH']}) → Feas++ ?")
        else:
            path_trace.append(f"Non (Te={te:.1f} < {self.THRESHOLDS['TE_HIGH']}) → Feas++ ?")

        # NOEUD 4 : Faisabilité NER ?
        if feas >= self.THRESHOLDS["FEAS_NER"]:
            path_trace.append(f"Oui (Feas={feas:.3f}) → [TBM]")
            return {
                "method": "TBM",
                "justification": f"Feas={feas:.3f}≥{self.THRESHOLDS['FEAS_NER']} — Transformer (DrBERT) faisable.",
                "trace": path_trace,
            }
        else:
            path_trace.append(f"Non (Feas={feas:.3f} < {self.THRESHOLDS['FEAS_NER']}) → [LLM]")
            return {
                "method": "LLM",
                "justification": f"Feas={feas:.3f}<{self.THRESHOLDS['FEAS_TBM']} — Escalade vers LLM nécessaire.",
                "trace": path_trace,
            }

    def build_full_config(self, metrics_data: dict[str, dict]):
        """Compile all decisions into the config dict."""
        config = {
            "version": "2.1",
            "global_thresholds": self.THRESHOLDS,
            "entities": {},
        }

        print("\n=== DECISION TREE EXECUTION ===")
        print(f"{'Entity':<25} | {'Method':<18} | {'Justification'}")
        print("-" * 100)

        for entity_raw, mets in metrics_data.items():
            # Clean entity name if needed
            entity = entity_raw.strip()
            decision = self.analyze_entity(entity, mets)

            # Print short reason
            reason_short = (
                (decision["justification"][:75] + "..")
                if len(decision["justification"]) > 75
                else decision["justification"]
            )
            print(f"{entity:<25} | {decision['method']:<18} | {reason_short}")

            config["entities"][entity] = {
                "metrics": mets,
                "method": decision["method"],
                "justification": decision["justification"],
                "trace": decision["trace"],
            }

        self.decisions = config
        return config

    def save_config(self):
        with open(self.config_path, "w", encoding="utf-8") as f:
            json.dump(self.decisions, f, indent=4)
        print(f"\nConfiguration saved to {self.config_path}")

    def export_text_report(self, output_txt: Path):
        with open(output_txt, "w", encoding="utf-8") as f:
            f.write("# DuraXELL Decision Tree Report\n")
            f.write("Generated by E_creation_arbre_decision.py\n\n")
            f.write("## Global Thresholds:\n")
            for k, v in self.THRESHOLDS.items():
                f.write(f"- {k}: {v}\n")
            f.write("\n")

            for entity, data in self.decisions["entities"].items():
                f.write(f"## {entity}\n")
                f.write(f"- **Method**: {data['method']}\n")
                f.write(f"- **Justification**: {data['justification']}\n")
                f.write(
                    f"- **Metrics**: {json.dumps(data['metrics'], default=str)}\n"
                )  # default=str to handle non serializable
                f.write(f"- **Trace**: {' -> '.join(data['trace'])}\n\n")
        print(f"Report saved to {output_txt}")


def load_metrics_from_csv(results_dir: Path):
    """Aggregate CSV results from previous steps into a single dict."""
    aggregated = {}  # {Entity: {Te: x, He: y...}}

    def _read_csv(filename, col_name, metric_key, multiplier=1.0):
        p = results_dir / filename
        if not p.exists():
            return
        try:
            with open(p, encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    # Try different column headers for Entity name
                    ent = row.get("Entity") or row.get("Entity_Type") or row.get("Entité")
                    if not ent:
                        continue

                    try:
                        val = float(row.get(col_name, 0))
                        if ent not in aggregated:
                            aggregated[ent] = {}
                        aggregated[ent][metric_key] = val * multiplier
                    except ValueError:
                        pass
        except Exception as e:
            print(f"Error reading {filename}: {e}")

    # 1. Te (Templatability)
    # Often in JSON, but let's check CSVs too
    if (results_dir / "templatability_analysis.json").exists():
        try:
            with open(results_dir / "templatability_analysis.json") as f:
                data = json.load(f)
                for ent, vals in data.items():
                    if ent not in aggregated:
                        aggregated[ent] = {}
                    aggregated[ent]["Te"] = vals.get("templatability_score", 0)
                    aggregated[ent]["Te_count"] = vals.get("count", 0)
        except Exception:
            pass

    # 2. He (Homogeneity)
    _read_csv("homogeneity_analysis.csv", "He_Score_Percent", "He")

    # 3. R (Risk)
    _read_csv("risk_context_analysis.csv", "R_Score", "R")

    # 4. Freq (Frequency)
    _read_csv("frequency_analysis.csv", "Frequency", "Freq")

    # 5. NER Feasibility metrics
    _read_csv("ner_feasibility_analysis.csv", "Feas_Score", "Feas")

    return aggregated


def main():
    script_dir = Path(__file__).parent
    # Standard DuraXELL directory structure
    results_dir = script_dir / "Rules/Results"
    root_dir = script_dir.parent
    config_file = root_dir / "data" / "decision_config.json"
    report_file = root_dir / "logs" / "output_decision.txt"

    # 1. Load existing metrics
    print("Loading metrics from Results folder...")
    metrics_db = load_metrics_from_csv(results_dir)

    # 2. Compute Yield on the fly (Hybrid approach)
    # We define paths to GS and Pred
    gs_dir = script_dir / "Breast/RCP/evaluation_set_breast_cancer_GS"
    pred_dir = script_dir / "Breast/RCP/evaluation_set_breast_cancer_pred_rules"

    # If using CHIR as well? For now stick to RCP as primary benchmark

    if gs_dir.exists() and pred_dir.exists():
        try:
            print("Computing Annotation Yield (Rules vs GS)...")
            yield_scorer = AnnotationYieldScorer(gs_dir, pred_dir)
            yield_scorer.compute_all()
            yield_scores = yield_scorer.get_scores()

            for ent, score_dict in yield_scores.items():
                f1 = (
                    score_dict.get("F1-Yield", 0.0)
                    if isinstance(score_dict, dict)
                    else float(score_dict)
                )
                if ent in metrics_db:
                    metrics_db[ent]["Yield"] = f1
                else:
                    metrics_db[ent] = {"Yield": f1}  # If missing in other stats
            print("Annotation Yield computed.")
        except NameError:
            print("AnnotationYieldScorer class not found (import failed). Skipping Yield.")
    else:
        print(f"Warning: GS/Pred folders not found for Yield calculation.\nScan path: {gs_dir}")

    # 3. Build Tree
    builder = DecisionTreeBuilder(config_file)
    builder.validate_thresholds_kfold(metrics_db, k=3)
    builder.build_full_config(metrics_db)

    # 4. Save Outputs
    builder.save_config()
    builder.export_text_report(report_file)
    if HAS_ECO2AI:
        try:
            tracker.stop()
        except Exception:
            pass


if __name__ == "__main__":
    main()
