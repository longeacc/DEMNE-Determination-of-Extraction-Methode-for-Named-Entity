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
import re
import warnings
from os import PathLike
from pathlib import Path

DEFAULTS = {
    "decision_thresholds": {
        "TE_HIGH": 0.10,
        "HE_HIGH": 0.85,
        "R_HIGH": 0.25,
        "FEAS_NER": 0.20,
        "MIN_TE_SAMPLES": 2,
        # --- TFIDF_Extractability (contextual conceptual synonymy) ---
        "TFIDF_Y": 0.70,  # threshold on routing metric (recall or F1) -> RULES
        "TFIDF_X": 5,  # top-X clusters evaluated (1–10)
        "TFIDF_SIM": 0.50,  # cosine similarity threshold for greedy clustering
    },
    "risk_weights": {"negation": 0.1, "uncertainty": 0.3, "contradiction": 0.6},
    "risk_window_size": 50,
    "feasibility_weights": {"alpha_freq": 0.2, "beta_he": 0.2, "he_default": 0.5},
    "homogeneity_sigmoid": {"k": 10, "x0": 0.5},
    "templatability_bonus": {"symbol_bonus": 0.1, "digit_bonus": 0.1, "digit_gate": 0.6},
    "frequency": {"rare_threshold_count": 10},
    "kfold": {"folds": 3, "calibration_percentile": 0.75},
    # Width of the FRUGAL / QUALITY presets around the calibrated thresholds
    # (see derive_presets). The presets themselves are not stored.
    "preset_margin": 0.10,
}

PARAMS_FILENAME = "demne_params.json"

# --- Contextual-risk lexicons: one list for the CLI scorer, the dashboard and the
# optimizer (they used to carry three copies kept equal by a drift-guard). ---
# Negation markers are deliberately restrictive: "pas", "ni", "non" alone over-detect
# (e.g. "non surexprimé" is an expected clinical state); "négatif" is excluded on
# purpose (PR-, HER2- are expected states, not a contextual-risk signal).
NEGATION_PATTERNS: tuple[str, ...] = (
    r"\baucun\b",
    r"\bsans\b",
    r"\bni\b",
    r"\bpas\b",
    r"\babsence\b",
)
UNCERTAINTY_PATTERNS: tuple[str, ...] = (
    r"\bprobable\b",
    r"\bpossible\b",
    r"\bà confirmer\b",
    r"\ba confirmer\b",
    r"\bsuspecté\b",
    r"\bsuspecte\b",
    r"\béquivoque\b",
    r"\bequivoque\b",
    r"\bdiscuté\b",
    r"\bincertain\b",
    r"\bhypothèse\b",
    r"\?",
    r"\bdiscordant\b",
    r"\bdiscordance\b",
)
# Opposite polarities, to detect contradictory values inside ONE document.
VAL_POS: tuple[str, ...] = (r"positif", r"positive", r"\+", r"pos", r"exprimé", r"present")
VAL_NEG: tuple[str, ...] = (r"négatif", r"negative", r"\-", r"neg", r"absent", r"non exprimé")


def has_contradiction(value_texts: list[str]) -> bool:
    """True if the lower-cased mention texts of ONE entity inside ONE document hold
    both a positive and a negative value."""
    has_pos = has_neg = False
    for txt in value_texts:
        has_pos = has_pos or any(re.search(p, txt) for p in VAL_POS)
        has_neg = has_neg or any(re.search(p, txt) for p in VAL_NEG)
        if has_pos and has_neg:
            return True
    return False


def derive_presets(decision_thresholds: dict, margin: float) -> dict:
    """Routing presets derived from the calibrated thresholds, never stored.

    - CALIBRATED: the calibrated thresholds themselves.
    - FRUGAL: every gate relaxed by `margin` toward the cheaper tier (lower Te/He
      gates, higher R tolerance, lower Feas bar → TBM before LLM).
    - QUALITY: every gate tightened by `margin` toward escalation.

    The routing graph is monotone in each threshold, so FRUGAL never routes an
    entity to a costlier tier than CALIBRATED, and QUALITY never to a cheaper one.
    Deriving them (instead of storing absolute values) keeps them consistent with
    every recalibration (`python main.py optimize_params`).
    """
    cal = {
        "Te": decision_thresholds["TE_HIGH"],
        "He": decision_thresholds["HE_HIGH"],
        "R": decision_thresholds["R_HIGH"],
        "Feas": decision_thresholds["FEAS_NER"],
    }
    # Direction that makes a threshold FRUGAL: R is a ceiling, the others are floors.
    frugal_sign = {"Te": -1, "He": -1, "R": +1, "Feas": -1}

    def _clip(v: float) -> float:
        return round(min(1.0, max(0.0, v)), 4)

    return {
        "CALIBRATED": {k: _clip(v) for k, v in cal.items()},
        "FRUGAL": {k: _clip(v + frugal_sign[k] * margin) for k, v in cal.items()},
        "QUALITY": {k: _clip(v - frugal_sign[k] * margin) for k, v in cal.items()},
    }


def _finalize(params: dict) -> dict:
    params["presets"] = derive_presets(
        params["decision_thresholds"], float(params.get("preset_margin", 0.10))
    )
    return params


def _find_params_file(start: str | PathLike[str] | None = None) -> Path | None:
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


def load_params(start: str | PathLike[str] | None = None) -> dict:
    """Return the merged params (DEFAULTS ← JSON) + derived presets. Never raises on missing file."""
    f = _find_params_file(start)
    if f is None:
        return _finalize(_deepcopy_defaults())
    try:
        user = json.loads(f.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        # Corrupt/unreadable params file: warn loudly (a reproducibility hazard)
        # instead of silently falling back to the built-in defaults.
        warnings.warn(
            f"Could not read params file {f} ({exc}); using built-in DEFAULTS. "
            "Calibrated thresholds will NOT be applied.",
            RuntimeWarning,
            stacklevel=2,
        )
        return _finalize(_deepcopy_defaults())
    # A stored "presets" block (older files) is ignored: presets are derived.
    user = {k: v for k, v in user.items() if not k.startswith("_") and k != "presets"}
    return _finalize(_merge(_deepcopy_defaults(), user))


PARAMS = load_params()
