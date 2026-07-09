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

from demne._table import print_table

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
            # TFIDF_Extractability threshold (default 0.70 if absent from JSON)
            "Y": _dt.get("TFIDF_Y", 0.70),
        }
        # Minimum occurrence count for Te to be reliable
        self.MIN_TE_SAMPLES = _dt["MIN_TE_SAMPLES"]
        # Hyperparameters for TFIDF_Extractability (lazy computation)
        self.TFIDF_X = _dt.get("TFIDF_X", 5)
        self.TFIDF_SIM = _dt.get("TFIDF_SIM", 0.50)

    def validate_thresholds_kfold(
        self, entities_metrics: dict[str, dict[str, float]], k: int | None = None
    ):
        """
        K-fold cross-validation on thresholds: partition the corpus into k folds,
        calibrate thresholds on k-1 folds, measure decision stability on the remaining fold.
        """
        import random

        if k is None:
            k = PARAMS["kfold"]["folds"]
        entities = list(entities_metrics.keys())
        if len(entities) < k:
            print(f"Not enough entities for {k}-fold CV.")
            return

        random.shuffle(entities)

        # Split into k folds
        folds = [entities[i::k] for i in range(k)]
        stabilities = []

        print(f"--- Starting {k}-fold threshold cross-validation ---")

        for i in range(k):
            test_entities = folds[i]
            train_entities = [ent for j, f in enumerate(folds) if j != i for ent in f]

            # Simulated calibration: minor threshold adjustment based on the train set
            train_te_vals = sorted([entities_metrics[ent].get("Te", 0.0) for ent in train_entities])
            if train_te_vals:
                # Manual percentile (data/demne_params.json → kfold.calibration_percentile)
                idx = int(len(train_te_vals) * PARAMS["kfold"]["calibration_percentile"])
                calibrated_te_high = (
                    train_te_vals[idx] if idx < len(train_te_vals) else self.THRESHOLDS["TE_HIGH"]
                )
            else:
                calibrated_te_high = self.THRESHOLDS["TE_HIGH"]

            old_te_high = self.THRESHOLDS["TE_HIGH"]
            # Apply calibrated threshold
            self.THRESHOLDS["TE_HIGH"] = calibrated_te_high

            # Measure agreement (stability) between default and calibrated rules
            matches = 0
            for ent in test_entities:
                metrics = entities_metrics[ent]
                # Default model
                self.THRESHOLDS["TE_HIGH"] = old_te_high
                orig_decision = self.analyze_entity(ent, metrics).get("method", "")
                # Calibrated model
                self.THRESHOLDS["TE_HIGH"] = calibrated_te_high
                new_decision = self.analyze_entity(ent, metrics).get("method", "")

                if orig_decision == new_decision:
                    matches += 1

            stability = matches / max(1, len(test_entities))
            stabilities.append(stability)
            self.THRESHOLDS["TE_HIGH"] = old_te_high  # Reset

        avg_stability = sum(stabilities) / len(stabilities)
        print(f"Average decision stability (K-Fold, k={k}): {avg_stability:.2%}")

    def analyze_entity(self, entity: str, metrics: dict[str, float]) -> dict[str, Any]:
        """DEMNE decision graph — exact graph matching the reference figure.

        Graph (see figure):
          Te++ AND He++ → R− ? → Yes: RULES / No (risk of conflict): Feas++
          Otherwise     → TF-IDF ? → Yes: R− ? → Yes: RULES / No: Feas++
                                    → No: Feas++
          Feas++ → Yes: TBM / No: LLM

        The R− node is SHARED between the Te/He branch and the TF-IDF branch.
        TF-IDF 'Yes' does NOT route directly to RULES: it first passes through R−.

        Args:
            entity: Entity name (used in reports, not in logic).
            metrics: {Te, He, R, Feas, Te_count, tfidf_score (optional, pre-computed)}.
        """
        _ = entity  # public API param — consumed by build_full_config
        te: float = metrics.get("Te", 0.0)
        te_count: int = metrics.get("Te_count", 0)
        he: float = metrics.get("He", 0.0)
        r_score: float = metrics.get("R", 0.0)
        feas: float = metrics.get("Feas", 0.0)

        if te > 1.0:
            te /= 100.0
        if he > 1.0:
            he /= 100.0
        if te_count < self.MIN_TE_SAMPLES:
            te = 0.0

        path_trace: list[str] = []

        # Helper: shared R− node (same logic from both the Te/He and TF-IDF branches)
        def _noeud_r(context_label: str):
            path_trace.append(
                f"R− ? (R={r_score:.3f} ≤ R_HIGH={self.THRESHOLDS['R_HIGH']}) [{context_label}]"
            )
            if r_score <= self.THRESHOLDS["R_HIGH"]:
                path_trace.append("Yes → [RULES]")
                return {
                    "method": "RULES",
                    "justification": (
                        f"{context_label}: R={r_score:.3f}≤{self.THRESHOLDS['R_HIGH']} — "
                        "acceptable contextual risk."
                    ),
                    "trace": path_trace,
                }
            path_trace.append(f"Non (risk of conflict, R={r_score:.3f}) → Feas++ ?")
            return None  # fall-through vers Feas

        # NOEUD 1 : Te++ ?
        path_trace.append(f"Te++ ? (Te={te:.3f}, TE_HIGH={self.THRESHOLDS['TE_HIGH']})")
        if te >= self.THRESHOLDS["TE_HIGH"]:
            # NOEUD 2 : He++ ?
            path_trace.append(f"Oui → He++ ? (He={he:.3f}, HE_HIGH={self.THRESHOLDS['HE_HIGH']})")
            if he >= self.THRESHOLDS["HE_HIGH"]:
                # NOEUD R− (branche Te/He)
                path_trace.append("Yes → R− ?")
                result = _noeud_r(
                    f"Te={te:.2f}≥{self.THRESHOLDS['TE_HIGH']}, He={he:.2f}≥{self.THRESHOLDS['HE_HIGH']}"
                )
                if result:
                    return result
                # High R → risk of conflict → fall through to Feas
            else:
                # Low He → TF-IDF
                path_trace.append(f"Non (He={he:.3f} < {self.THRESHOLDS['HE_HIGH']}) → TF-IDF ?")
                result = self._noeud_tfidf(metrics, r_score, path_trace)
                if result:
                    return result
        else:
            # Low Te → TF-IDF
            path_trace.append(f"Non (Te={te:.3f} < {self.THRESHOLDS['TE_HIGH']}) → TF-IDF ?")
            result = self._noeud_tfidf(metrics, r_score, path_trace)
            if result:
                return result

        # NODE Feas++ (convergence point for all fall-throughs)
        path_trace.append(f"Feas++ ? (Feas={feas:.3f}, FEAS_NER={self.THRESHOLDS['FEAS_NER']})")
        if feas >= self.THRESHOLDS["FEAS_NER"]:
            path_trace.append("Yes → [TBM]")
            return {
                "method": "TBM",
                "justification": (
                    f"Feas={feas:.3f}≥{self.THRESHOLDS['FEAS_NER']} — "
                    "Transformer (DrBERT) feasible."
                ),
                "trace": path_trace,
            }
        path_trace.append("No → [LLM]")
        return {
            "method": "LLM",
            "justification": (
                f"Feas={feas:.3f}<{self.THRESHOLDS['FEAS_NER']} — " "LLM escalation required."
            ),
            "trace": path_trace,
        }

    def _noeud_tfidf(self, metrics, r_score, path_trace):
        """TF-IDF graph node: uses pre-computed score only (no internal computation).

        Returns a result dict if RULES, None otherwise (fall-through to Feas).
        """
        tfidf_raw = metrics.get("tfidf_score")
        if tfidf_raw is None:
            path_trace.append("TF-IDF absent → Feas++ ?")
            return None
        tfidf_score = float(tfidf_raw)
        path_trace.append(f"TF-IDF ? (score={tfidf_score:.3f}, Y={self.THRESHOLDS['Y']})")
        if tfidf_score >= self.THRESHOLDS["Y"]:
            # TF-IDF Yes → same shared R− node as the Te/He branch
            path_trace.append("Yes → R− ? (shared node)")
            if r_score <= self.THRESHOLDS["R_HIGH"]:
                path_trace.append("Yes → [RULES] (conceptual synonymy + acceptable R)")
                return {
                    "method": "RULES",
                    "justification": (
                        f"TFIDF={tfidf_score:.3f}≥Y={self.THRESHOLDS['Y']} and "
                        f"R={r_score:.3f}≤{self.THRESHOLDS['R_HIGH']} — "
                        "conceptual synonyms extractable by rule."
                    ),
                    "trace": path_trace,
                }
            path_trace.append(f"No (risk of conflict, R={r_score:.3f}) → Feas++ ?")
            return None  # High R despite TF-IDF → Feas
        path_trace.append(f"Non (score={tfidf_score:.3f} < Y={self.THRESHOLDS['Y']}) → Feas++ ?")
        return None

    def build_full_config(self, metrics_data: dict[str, dict]):
        """Compile all decisions into the config dict."""
        config = {
            "version": "2.1",
            "global_thresholds": self.THRESHOLDS,
            "entities": {},
        }

        print("\n=== DECISION TREE EXECUTION ===")
        table_rows = []
        for entity_raw, mets in metrics_data.items():
            entity = entity_raw.strip()
            decision = self.analyze_entity(entity, mets)
            table_rows.append([entity, decision["method"], decision["justification"]])
            config["entities"][entity] = {
                "metrics": mets,
                "method": decision["method"],
                "justification": decision["justification"],
                "trace": decision["trace"],
            }
        print_table(["Entity", "Method", "Justification"], table_rows, [45, 8, 60])

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
                    ent = row.get("Entity") or row.get("Entity_Type")
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

    # 6. TFIDF_Extractability (contextual conceptual synonymy)
    _read_csv("tfidf_analysis.csv", "TFIDF_Score", "tfidf_score")

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

    # 0. Compute TF-IDF if tfidf_analysis.csv is missing or empty
    tfidf_csv = results_dir / "tfidf_analysis.csv"
    gs_dir = Path(args.gs_dir) if args.gs_dir else None
    if gs_dir and gs_dir.exists() and (not tfidf_csv.exists() or tfidf_csv.stat().st_size == 0):
        from demne.E_tfidf import run as _run_tfidf

        _dt_params = PARAMS["decision_thresholds"]
        print("Computing TF-IDF scores...")
        _run_tfidf(
            gs_dir,
            results_dir,
            top_x=_dt_params.get("TFIDF_X", 10),
            sim_threshold=_dt_params.get("TFIDF_SIM", 0.50),
            y=_dt_params.get("TFIDF_Y", 0.70),
        )

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
