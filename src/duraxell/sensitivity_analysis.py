import copy
import csv
import json
import os
import sys
from pathlib import Path

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from duraxell.E_creation_arbre_decision import DecisionTreeBuilder


def load_real_metrics(results_dir: Path) -> dict:
    metrics_db = {}

    freq_file = results_dir / "frequency_analysis.csv"
    if freq_file.exists():
        with open(freq_file, encoding="utf-8") as f:
            for row in csv.DictReader(f):
                ent = row.get("Entity") or row.get("Entity_Type")
                if ent:
                    metrics_db.setdefault(ent, {})["Freq"] = float(row.get("Frequency", 0))

    he_file = results_dir / "homogeneity_analysis.csv"
    if he_file.exists():
        with open(he_file, encoding="utf-8") as f:
            for row in csv.DictReader(f):
                ent = row.get("Entity")
                if ent:
                    metrics_db.setdefault(ent, {})["He"] = float(row.get("He_Score_Percent", 0))

    te_file = results_dir / "templatability_analysis.json"
    if te_file.exists():
        try:
            with open(te_file, encoding="utf-8") as f:
                data = json.load(f)
                for ent, vals in data.items():
                    metrics_db.setdefault(ent, {})["Te"] = vals.get("templatability_score", 0)
                    metrics_db[ent]["Te_count"] = vals.get("total_occurrences", 0)
        except Exception:
            pass

    risk_file = results_dir / "risk_context_analysis.csv"
    if risk_file.exists():
        with open(risk_file, encoding="utf-8") as f:
            for row in csv.DictReader(f):
                ent = row.get("Entity")
                if ent:
                    metrics_db.setdefault(ent, {})["R"] = float(row.get("R_Score", 0))

    feas_file = results_dir / "ner_feasibility_analysis.csv"
    if feas_file.exists():
        with open(feas_file, encoding="utf-8") as f:
            for row in csv.DictReader(f):
                ent = row.get("Entity")
                if ent:
                    metrics_db.setdefault(ent, {})["Feas"] = float(row.get("Feas_Score", 0))
                    metrics_db[ent]["DomainShift"] = float(row.get("Domain_Shift", 0))
                    metrics_db[ent]["LLM_Necessity"] = float(row.get("LLM_Necessity", 0))

    return metrics_db


def run_sensitivity_analysis(
    metrics_data: dict,
    thresholds_to_vary: list = None,
    perturbation_pct: list = None,
):
    if perturbation_pct is None:
        perturbation_pct = [-0.2, -0.1, 0.1, 0.2]
    if thresholds_to_vary is None:
        thresholds_to_vary = [
            "TE_MED",
            "TE_HIGH",
            "HE_HIGH",
            "RISK_HIGH",
            "YIELD_HIGH",
            "FEAS_NER",
            "DOMAIN_SHIFT_MAX",
            "LLM_NEC_HIGH",
        ]
    builder = DecisionTreeBuilder("dummy.json")
    base_thresholds = dict(builder.THRESHOLDS)

    results = []
    base_decisions = {}

    for entity, metrics in metrics_data.items():
        base_decisions[entity] = builder.analyze_entity(entity, metrics)["method"]

    for threshold_name in thresholds_to_vary:
        if threshold_name not in base_thresholds:
            continue
        original_val = base_thresholds[threshold_name]

        for pct in perturbation_pct:
            new_val = original_val * (1 + pct)
            builder.THRESHOLDS = copy.deepcopy(base_thresholds)
            builder.THRESHOLDS[threshold_name] = new_val

            changes, changed_entities = 0, []
            for entity, metrics in metrics_data.items():
                new_decision = builder.analyze_entity(entity, metrics)["method"]
                if new_decision != base_decisions[entity]:
                    changes += 1
                    changed_entities.append(f"{entity} ({base_decisions[entity]}->{new_decision})")

            results.append(
                {
                    "threshold": threshold_name,
                    "perturbation": f"{pct * 100:+.0f}%",
                    "original_value": original_val,
                    "new_value": new_val,
                    "n_changes": changes,
                    "changed_entities": ", ".join(changed_entities),
                    "robustness": 1.0 - (changes / len(metrics_data)) if metrics_data else 1.0,
                }
            )
    return results


if __name__ == "__main__":
    script_dir = Path(__file__).parent
    results_dir = script_dir / "Rules" / "Results"

    print("Chargement des metriques reelles...")
    metrics_data = load_real_metrics(results_dir)
    print(f"Metriques chargees pour {len(metrics_data)} entites.")

    if metrics_data:
        print("Execution de l analyse...")
        results = run_sensitivity_analysis(metrics_data)

        print("\n=== RESULTATS ===")
        for r in results:
            print(
                f"- {r['threshold']} ({r['perturbation']}): {r['new_value']:.2f} -> {r['n_changes']} changements ({r['robustness']:.2%})"
            )
            if r["n_changes"] > 0:
                print(f"    Detail: {r['changed_entities']}")

        out_csv = results_dir / "sensitivity_results.csv"
        os.makedirs(results_dir, exist_ok=True)
        with open(out_csv, "w", encoding="utf-8", newline="") as f:
            if results:
                writer = csv.DictWriter(f, fieldnames=results[0].keys())
                writer.writeheader()
                writer.writerows(results)
        print(f"\nSauvegarde: {out_csv}")
