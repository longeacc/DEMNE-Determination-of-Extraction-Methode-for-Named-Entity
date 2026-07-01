"""
Build and execute the DuraXELL Decision Tree.
Generates 'decision_config.json' and 'output_decision.txt'.

Decision Tree Logic (Priority Order):
1. Templatability (Te) & Homogeneity (He) (Structure) -> HIGH? -> RULES
2. Risk Context (R) -> HIGH? -> LLM / REVIEW
3. Feasibility (Feas) -> RULES vs ML (NER) vs LLM

Outputs:
- decision_config.json: Machine-readable config for the orchestrator.
- output_decision.txt: Human-readable report.
"""

# pylint: disable=broad-exception-caught,unused-argument

import csv
import importlib.util as _il
import json
import os
from pathlib import Path
from typing import Any

# --- Tunable thresholds loaded from data/demne_params.json (single source of truth) ---
_pspec = _il.spec_from_file_location("demne_params", Path(__file__).resolve().parent / "params.py")
_pmod = _il.module_from_spec(_pspec)
_pspec.loader.exec_module(_pmod)
PARAMS = _pmod.load_params()

# Eco2AI tracking
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
        project_name="Consumtion_of_E_creation_arbre_decision.py",
        experiment_description="Building Decision Tree Config",
        file_name="Consumtion_of_Duraxell.csv",
    )
    tracker = Tracker()
    tracker.start()


class DecisionTreeBuilder:
    def __init__(self, config_path: Path):
        self.config_path = config_path
        self.decisions = {}

        # --- CALIBRATED THRESHOLDS (data/demne_params.json → decision_thresholds) ---
        _dt = PARAMS["decision_thresholds"]
        self.THRESHOLDS = {
            "TE_HIGH": _dt["TE_HIGH"],
            "HE_HIGH": _dt["HE_HIGH"],
            "R_HIGH": _dt["R_HIGH"],
            "FEAS_NER": _dt["FEAS_NER"],
        }
        # Nombre minimum d'occurrences pour que Te soit fiable
        self.MIN_TE_SAMPLES = _dt["MIN_TE_SAMPLES"]

    def validate_thresholds_kfold(
        self, entities_metrics: dict[str, dict[str, float]], k: int | None = None
    ):
        """
        Validation croisée k-fold sur les seuils : partitionner le corpus en k plis,
        calibrer les seuils sur k-1 plis, mesurer la stabilité des décisions sur le pli restant.
        """
        import random

        if k is None:
            k = PARAMS["kfold"]["folds"]
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
                # percentile manuel (data/demne_params.json → kfold.calibration_percentile)
                idx = int(len(train_te_vals) * PARAMS["kfold"]["calibration_percentile"])
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
                        f"Non (R={r_score:.3f} > {self.THRESHOLDS['R_HIGH']}) → Feas++ ?"
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
                "justification": f"Feas={feas:.3f}<{self.THRESHOLDS['FEAS_NER']} — Escalade vers LLM nécessaire.",
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


def load_metrics_from_csv(results_dir: Path, corpus_name: str | None = None):
    """Aggregate CSV results from previous steps into a single dict."""
    aggregated = {}  # {Entity: {Te: x, He: y...}}

    def _resolve(filename: str) -> Path:
        stem, _, ext = filename.rpartition(".")
        if corpus_name:
            cand = results_dir / f"{stem}_{corpus_name}.{ext}"
            if cand.exists():
                return cand
        return results_dir / filename

    def _read_csv(filename, col_name, metric_key, multiplier=1.0):
        """Read `col_name` from CSV. col_name may be a string OR a list of
        (col, mult) tuples — first present column wins, its multiplier applies.
        """
        p = _resolve(filename)
        if not p.exists():
            return
        # Normalize to list of (col, mult)
        if isinstance(col_name, str):
            candidates = [(col_name, multiplier)]
        else:
            candidates = list(col_name)
        try:
            with open(p, encoding="utf-8") as f:
                reader = csv.DictReader(f)
                fieldnames = reader.fieldnames or []
                chosen = next(((c, m) for c, m in candidates if c in fieldnames), None)
                if chosen is None:
                    return
                col, mult = chosen
                for row in reader:
                    ent = row.get("Entity") or row.get("Entity_Type") or row.get("Entité")
                    if not ent:
                        continue
                    try:
                        val = float(row.get(col, 0))
                        if ent not in aggregated:
                            aggregated[ent] = {}
                        aggregated[ent][metric_key] = val * mult
                    except ValueError:
                        pass
        except Exception as e:
            print(f"Error reading {filename}: {e}")

    # 1. Te (Templatability)
    # Often in JSON, but let's check CSVs too
    te_json = _resolve("templatability_analysis.json")
    if te_json.exists():
        try:
            with open(te_json, encoding="utf-8") as f:
                data = json.load(f)
                for ent, vals in data.items():
                    if ent not in aggregated:
                        aggregated[ent] = {}
                    aggregated[ent]["Te"] = vals.get("templatability_score", 0)
                    aggregated[ent]["Te_count"] = vals.get("count", 0)
        except Exception:
            pass

    # 2. He (Homogeneity) — new schema 'He' ∈ [0,1]; legacy 'He_Score_Percent' ∈ [0,100]
    _read_csv("homogeneity_analysis.csv", [("He", 1.0), ("He_Score_Percent", 0.01)], "He")

    # 3. R (Risk)
    _read_csv("risk_context_analysis.csv", "R_Score", "R")

    # 4. Freq (Frequency)
    _read_csv("frequency_analysis.csv", "Frequency", "Freq")

    # 5. NER Feasibility metrics
    _read_csv("ner_feasibility_analysis.csv", "Feas_Score", "Feas")

    return aggregated


import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gs_dir", type=str, default=None)
    parser.add_argument("--pred_dir", type=str, default=None)
    args = parser.parse_args()

    script_dir = Path(__file__).parent
    # Standard DuraXELL directory structure
    results_dir = script_dir.parent.parent / "Results"
    root_dir = script_dir.parent.parent
    config_file = root_dir / "data" / "decision_config.json"
    report_file = root_dir / "logs" / "output_decision.txt"

    # Make dirs
    config_file.parent.mkdir(parents=True, exist_ok=True)
    report_file.parent.mkdir(parents=True, exist_ok=True)

    # 1. Load existing metrics
    print("Loading metrics from Results folder...")
    corpus_name = Path(args.gs_dir).parent.name if args.gs_dir else "Breast"
    metrics_db = load_metrics_from_csv(results_dir, corpus_name)

    # 2. Build Tree
    builder = DecisionTreeBuilder(config_file)
    builder.validate_thresholds_kfold(metrics_db)
    builder.build_full_config(metrics_db)

    # 3. Save Outputs
    builder.save_config()
    builder.export_text_report(report_file)
    if HAS_ECO2AI:
        try:
            tracker.stop()
        except Exception:
            pass


if __name__ == "__main__":
    main()
