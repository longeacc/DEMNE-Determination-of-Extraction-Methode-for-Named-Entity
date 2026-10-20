# pylint: disable=broad-exception-caught,unused-argument
import csv
import os
from pathlib import Path

from demne._table import print_table
from demne._tracking import start_tracker, stop_tracker
from demne.params import load_params

# --- Tunable weights loaded from data/demne_params.json (single source of truth) ---
PARAMS = load_params()


def compute_feasibility(
    gs_dir_str: str | None = None,
    pred_dir_str: str | None = None,
    results_dir: str | Path | None = None,
) -> None:
    """Compute Feas per entity from the frequency/homogeneity CSVs and write the result CSV."""
    print("Computing NER Feasibility per entity...")
    if results_dir is None:
        results_dir = Path(__file__).parent.parent.parent / "Results"
    else:
        results_dir = Path(results_dir)

    corpus_name = Path(gs_dir_str).parent.name if gs_dir_str else "Breast"

    def _resolve(stem: str, ext: str) -> Path:
        suffixed = results_dir / f"{stem}_{corpus_name}.{ext}"
        if suffixed.exists():
            return suffixed
        return results_dir / f"{stem}.{ext}"

    # 1. Load Frequencies
    freq_file = _resolve("frequency_analysis", "csv")
    frequencies = {}

    counts = {}
    if freq_file.exists():
        with open(freq_file, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                ent = row.get("Entity") or row.get("Entity_Type") or row.get("Entity_Label")
                if ent:
                    frequencies[ent] = float(row.get("Frequency", 0.0))
                    counts[ent] = float(row.get("Count", 0.0))

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

    # Feas(E) = α_Feas · min(1, count/C) + β_Feas · He
    # We saturate the absolute COUNT (enough examples to enumerate a template), not
    # the count/tokens density: a frequent entity in a large corpus has a low density
    # but plenty of examples — the density wrongly penalised it.
    # MUST stay identical to MetricsCalculator (drift-guard).
    _fw = PARAMS["feasibility_weights"]
    alpha_feas = _fw["alpha_freq"]
    beta_feas = _fw["beta_he"]
    he_default = _fw["he_default"]
    sat_count = _fw.get("sat_count", 300)

    results = []
    feas_rows = []
    for ent in frequencies:
        he = homogeneity.get(ent, he_default)
        feas_freq = min(1.0, counts.get(ent, 0.0) / sat_count) if sat_count > 0 else 0.0
        feas = round(alpha_feas * feas_freq + beta_feas * he, 3)
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

    _tracker = start_tracker("DEMNE-ner_feasibility", "NER Feasibility Computation")

    parser = argparse.ArgumentParser()
    parser.add_argument("--gs_dir", type=str, default=None)
    parser.add_argument("--pred_dir", type=str, default=None)
    args = parser.parse_args()

    compute_feasibility(args.gs_dir, args.pred_dir)

    stop_tracker(_tracker)
