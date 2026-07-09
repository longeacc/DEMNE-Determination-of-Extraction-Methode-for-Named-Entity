class RESTDecisionBridge:
    """
    Cross-validation component.
    Compares Top-Down (tree) decisions vs Bottom-Up (REST annotation) decisions.
    """

    def compare(
        self,
        tree_config: dict,  # data/decision_config.json
        rest_reports: list,  # output of RESTEvaluator (list of RESTEntityReport)
    ) -> dict:
        """
        Compare decisions and generate a convergence report.
        """
        concordance_count = 0
        total_entities = 0
        divergences = []

        for report in rest_reports:
            entity = report.entity_type
            total_entities += 1

            # Top-Down decision (tree)
            top_down_entry = tree_config.get(entity, {})
            # Config may be structured differently (e.g. {'entities': {...}})
            if "entities" in tree_config:
                top_down_entry = tree_config["entities"].get(entity, {})

            tree_decision = top_down_entry.get("method", "Unknown")

            # Bottom-Up decision (empirical REST based on observed TE)
            report.recommended_method = self._decide_from_empirics(report)
            rest_decision = report.recommended_method

            # Comparison logic (simplified: RULES vs ML vs LLM)
            tree_normalized = self._normalize_method(tree_decision)
            rest_normalized = self._normalize_method(rest_decision)

            if tree_normalized == rest_normalized:
                concordance_count += 1
            else:
                divergences.append(
                    {
                        "entity": entity,
                        "tree_decision": tree_decision,
                        "rest_decision": rest_decision,
                        "metrics_delta": {
                            "empirical_te": report.empirical_te,
                            "theoretical_te": top_down_entry.get("metrics", {}).get("Te", "N/A"),
                        },
                    }
                )

        rate = (concordance_count / total_entities) if total_entities > 0 else 0.0

        return {
            "concordance_rate": rate,
            "n_divergences": len(divergences),
            "divergences": divergences,
        }

    def _normalize_method(self, method: str) -> str:
        """Normalise and align method names from the global decision tree and the empirical REST interface."""
        m = method.upper()
        if m in ["RULES", "REGLES"] or "RÈGLES" in m:
            return "RULES"
        if m in ["ML", "ML_NER", "ML_CRF", "TRANSFORMER"] or "ML" in m or "TRANSFORMER" in m:
            return "ML"
        if "LLM" in m:
            return "LLM"
        return "UNKNOWN"

    def _decide_from_empirics(self, report) -> str:
        """
        Bottom-Up decision logic.
        If the entity is highly repetitive (Te > 0.8) and stable enough (He > 0.7) -> RULES.
        Otherwise -> ML.
        """
        # Highly repetitive (observed Te > 0.8) -> Rules
        if report.empirical_te > 0.8:
            return "RULES"
        # Otherwise -> ML
        return "ML_NER"


if __name__ == "__main__":
    bridge = RESTDecisionBridge()
    print("REST Decision Bridge ready.")
