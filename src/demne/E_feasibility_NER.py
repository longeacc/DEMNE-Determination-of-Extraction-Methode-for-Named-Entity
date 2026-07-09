# pylint: disable=broad-exception-caught,unused-argument
import csv
import importlib.util as _il
import os
from pathlib import Path

from demne._table import print_table

# --- Tunable weights loaded from data/demne_params.json (single source of truth) ---
_pspec = _il.spec_from_file_location("demne_params", Path(__file__).resolve().parent / "params.py")
assert _pspec is not None and _pspec.loader is not None
_pmod = _il.module_from_spec(_pspec)
_pspec.loader.exec_module(_pmod)
PARAMS = _pmod.load_params()

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
        project_name="Consumtion_of_E_feasibility_NER.py",
        experiment_description="NER Feasibility Computation",
        file_name="Consumtion_of_Duraxell.csv",
    )
    tracker = Tracker()
    tracker.start()


def compute_feasibility(gs_dir_str=None, pred_dir_str=None):
    print("Computing NER Feasibility per entity...")
    script_dir = Path(__file__).parent
    results_dir = script_dir.parent.parent / "Results"

    corpus_name = Path(gs_dir_str).parent.name if gs_dir_str else "Breast"

    def _resolve(stem: str, ext: str) -> Path:
        suffixed = results_dir / f"{stem}_{corpus_name}.{ext}"
        if suffixed.exists():
            return suffixed
        return results_dir / f"{stem}.{ext}"

    # 1. Load Frequencies
    freq_file = _resolve("frequency_analysis", "csv")
    frequencies = {}

    if freq_file.exists():
        with open(freq_file, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                ent = row.get("Entity") or row.get("Entity_Type") or row.get("Entity_Label")
                if ent:
                    frequencies[ent] = float(row.get("Frequency", 0.0))

    # 2. Load Homogeneity (He)
    he_file = _resolve("homogeneity_analysis", "csv")
    homogeneity = {}
    if he_file.exists():
        with open(he_file, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                ent = row.get("Entity")
                if ent:
                    if "He" in row:
                        homogeneity[ent] = float(row.get("He", 0.0))
                    else:
                        homogeneity[ent] = float(row.get("He_Score_Percent", 0.0)) / 100.0

    # Feas(E) = α_Feas · min(1, Freq) + β_Feas · He
    _fw = PARAMS["feasibility_weights"]
    alpha_feas = _fw["alpha_freq"]
    beta_feas = _fw["beta_he"]
    he_default = _fw["he_default"]

    results = []
    feas_rows = []
    for ent, freq in frequencies.items():
        he = homogeneity.get(ent, he_default)
        feas = round(alpha_feas * min(1.0, freq) + beta_feas * he, 3)
        results.append({"Entity": ent, "Feas_Score": feas})
        feas_rows.append([ent, f"{feas:.3f}"])
    print_table(["Entity", "Feas"], feas_rows, [60, 8])

    out_file = results_dir / f"ner_feasibility_analysis_{corpus_name}.csv"
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["Entity", "Feas_Score"])
        writer.writeheader()
        writer.writerows(results)

    try:
        rel = os.path.relpath(str(out_file))
    except ValueError:
        rel = str(out_file)
    print(f"Feasibility scores written to {rel}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--gs_dir", type=str, default=None)
    parser.add_argument("--pred_dir", type=str, default=None)
    args = parser.parse_args()

    compute_feasibility(args.gs_dir, args.pred_dir)

    try:
        if HAS_ECO2AI:
            tracker.stop()
    except Exception as e:
        print(
            f"\nWarning: Generalized error in Eco2AI tracking: {e}\n"
            "Carbon emission tracking data could not be saved, but analysis results are preserved."
        )
