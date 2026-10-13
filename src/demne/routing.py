"""Canonical DEMNE routing graph — the single source of truth for tier selection.

Both the CLI decision-tree builder (:mod:`demne.E_creation_arbre_decision`) and the
Streamlit dashboard (``dashboard/core/routing.py``) call :func:`route`, so the graph
is implemented exactly once. Callers normalise their metrics first; ``route`` expects
metrics already on the [0, 1] scale.
"""

from __future__ import annotations


def _node_tfidf(tfidf_score, r_score, r_high, tfidf_y, trace):
    """TF-IDF node -> shared R- node -> RULES, else fall through to Feas."""
    if tfidf_score is None:
        trace.append("TF-IDF absent → Feas++ ?")
        return None
    routing_score = float(tfidf_score)
    trace.append(f"TF-IDF ? (recall={routing_score:.3f}, Y={tfidf_y})")
    if routing_score >= tfidf_y:
        trace.append("Yes → R− ? (shared node)")
        if r_score <= r_high:
            trace.append("Yes → [RULES] (conceptual synonymy + acceptable R)")
            return {
                "method": "RULES",
                "justification": (
                    f"TFIDF recall={routing_score:.3f}≥Y={tfidf_y} and "
                    f"R={r_score:.3f}≤{r_high} — "
                    "conceptual synonyms extractable by rule."
                ),
                "trace": trace,
            }
        trace.append(f"No (risk of conflict, R={r_score:.3f}) → Feas++ ?")
        return None
    trace.append(f"No (recall={routing_score:.3f} < Y={tfidf_y}) → Feas++ ?")
    return None


def route(*, te, he, r, feas, tfidf_score, te_high, he_high, r_high, feas_ner, tfidf_y):
    """Route one entity to ``RULES`` / ``TBM`` / ``LLM`` (shared R- node).

    Metrics and thresholds on the [0, 1] scale. Returns a dict with ``method``,
    ``justification`` and ``trace`` (the audit path).
    """
    r_score = r
    trace = []

    def _node_r(context_label):
        trace.append(f"R− ? (R={r_score:.3f} ≤ R_HIGH={r_high}) [{context_label}]")
        if r_score <= r_high:
            trace.append("Yes → [RULES]")
            return {
                "method": "RULES",
                "justification": (
                    f"{context_label}: R={r_score:.3f}≤{r_high} — " "acceptable contextual risk."
                ),
                "trace": trace,
            }
        trace.append(f"No (risk of conflict, R={r_score:.3f}) → Feas++ ?")
        return None

    trace.append(f"Te++ ? (Te={te:.3f}, TE_HIGH={te_high})")
    if te >= te_high:
        trace.append(f"Yes → He++ ? (He={he:.3f}, HE_HIGH={he_high})")
        if he >= he_high:
            trace.append("Yes → R− ?")
            result = _node_r(f"Te={te:.2f}≥{te_high}, He={he:.2f}≥{he_high}")
            if result:
                return result
        else:
            trace.append(f"No (He={he:.3f} < {he_high}) → TF-IDF ?")
            result = _node_tfidf(tfidf_score, r_score, r_high, tfidf_y, trace)
            if result:
                return result
    else:
        trace.append(f"No (Te={te:.3f} < {te_high}) → TF-IDF ?")
        result = _node_tfidf(tfidf_score, r_score, r_high, tfidf_y, trace)
        if result:
            return result

    trace.append(f"Feas++ ? (Feas={feas:.3f}, FEAS_NER={feas_ner})")
    if feas >= feas_ner:
        trace.append("Yes → [TBM]")
        return {
            "method": "TBM",
            "justification": f"Feas={feas:.3f}≥{feas_ner} — Transformer (DrBERT) feasible.",
            "trace": trace,
        }
    trace.append("No → [LLM]")
    return {
        "method": "LLM",
        "justification": f"Feas={feas:.3f}<{feas_ner} — LLM escalation required.",
        "trace": trace,
    }
