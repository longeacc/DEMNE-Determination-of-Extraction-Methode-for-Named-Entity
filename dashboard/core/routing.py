"""DEMNE routing for the dashboard — thin adapter over the canonical graph.

The graph itself lives once, in :func:`demne.routing.route` (shared R− node between
the Te/He branch and the TF-IDF branch). This module only maps the dashboard's
metric/threshold dicts onto it, in two flavours:

* :func:`compute_routing_detail` — the full result, exactly as the CLI sees it
  (``{"method", "justification", "trace"}``);
* :func:`compute_routing` — the ``(method, justification)`` tuple the dashboard
  pages unpack.
"""

from demne.params import load_params
from demne.routing import route

# Default thresholds = data/demne_params.json (single source shared by CLI/scorers/dashboard)
_DT = load_params()["decision_thresholds"]


def compute_routing_detail(
    metrics: dict[str, float], thresholds: dict[str, float]
) -> dict[str, object]:
    """Route one entity and return ``{"method", "justification", "trace"}``.

    Args:
        metrics: {Te, He, R, Feas, tfidf_score (optional)} on scale [0-1]
            (lower-case keys are accepted too).
        thresholds: {Te, He, R, Feas, Y (optional)} on scale [0-1]; missing keys fall
            back to demne_params.json.
    """
    return route(
        te=metrics.get("Te", metrics.get("te", 0.0)),
        he=metrics.get("He", metrics.get("he", 0.0)),
        r=metrics.get("R", metrics.get("r", 0.0)),
        feas=metrics.get("Feas", metrics.get("feas", 0.0)),
        tfidf_score=metrics.get("tfidf_score"),
        te_high=thresholds.get("Te", _DT["TE_HIGH"]),
        he_high=thresholds.get("He", _DT["HE_HIGH"]),
        r_high=thresholds.get("R", _DT["R_HIGH"]),
        feas_ner=thresholds.get("Feas", _DT["FEAS_NER"]),
        tfidf_y=thresholds.get("Y", _DT.get("TFIDF_Y", 0.70)),
    )


def compute_routing(metrics: dict[str, float], thresholds: dict[str, float]) -> tuple[str, str]:
    """Same routing as :func:`compute_routing_detail`, as ``(method, justification)``."""
    result = compute_routing_detail(metrics, thresholds)
    return str(result["method"]), str(result["justification"])
