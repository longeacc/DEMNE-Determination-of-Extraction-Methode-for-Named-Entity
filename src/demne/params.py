"""Single source of truth for DEMNE tunable weights & thresholds.

All values come from data/demne_params.json (found by walking up from this
file or any given start path). DEFAULTS below are used only for keys missing
from the JSON, so the JSON file is the authority and the CLI scorers, the
decision tree and the dashboard can never silently diverge.

Standalone-safe: load it without importing the demne package, e.g.

    import importlib.util as _il
    _spec = _il.spec_from_file_location("demne_params", <path>/"params.py")
    _pm = _il.module_from_spec(_spec); _spec.loader.exec_module(_pm)
    PARAMS = _pm.load_params()
"""

import json
from pathlib import Path

DEFAULTS = {
    "decision_thresholds": {
        "TE_HIGH": 0.10,
        "HE_HIGH": 0.85,
        "R_HIGH": 0.25,
        "FEAS_NER": 0.20,
        "MIN_TE_SAMPLES": 2,
        # --- TFIDF_Extractability (contextual conceptual synonymy) ---
        "TFIDF_Y": 0.70,  # threshold on tfidf_score -> RULES
        "TFIDF_X": 5,  # number of top clusters evaluated for recall
        "TFIDF_SIM": 0.50,  # cosine similarity threshold for greedy clustering
    },
    "risk_weights": {"negation": 0.1, "uncertainty": 0.3, "contradiction": 0.6},
    "risk_window_size": 50,
    "feasibility_weights": {"alpha_freq": 0.2, "beta_he": 0.2, "he_default": 0.5},
    "homogeneity_sigmoid": {"k": 10, "x0": 0.5},
    "templatability_bonus": {"symbol_bonus": 0.1, "digit_bonus": 0.1, "digit_gate": 0.6},
    "frequency": {"rare_threshold_count": 10},
    "kfold": {"folds": 3, "calibration_percentile": 0.75},
    "presets": {
        "FRUGAL": {"Te": 0.10, "He": 0.85, "R": 0.25, "Feas": 0.20},
        "QUALITY": {"Te": 0.25, "He": 0.55, "R": 0.15, "Feas": 0.70},
    },
}

PARAMS_FILENAME = "demne_params.json"


def _find_params_file(start=None) -> Path | None:
    here = Path(start or __file__).resolve()
    for parent in [here, *here.parents]:
        cand = parent / "data" / PARAMS_FILENAME
        if cand.exists():
            return cand
    return None


def _merge(base: dict, over: dict) -> dict:
    out = dict(base)
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _merge(out[k], v)
        else:
            out[k] = v
    return out


def _deepcopy_defaults() -> dict:
    return json.loads(json.dumps(DEFAULTS))


def load_params(start=None) -> dict:
    """Return the merged params (DEFAULTS ← JSON). Never raises on missing file."""
    f = _find_params_file(start)
    if f is None:
        return _deepcopy_defaults()
    try:
        user = json.loads(f.read_text(encoding="utf-8"))
    except Exception:  # pylint: disable=broad-exception-caught
        return _deepcopy_defaults()
    user = {k: v for k, v in user.items() if not k.startswith("_")}
    return _merge(_deepcopy_defaults(), user)


PARAMS = load_params()
