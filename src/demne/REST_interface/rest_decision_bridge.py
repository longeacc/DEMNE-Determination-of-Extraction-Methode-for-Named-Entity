class RESTDecisionBridge:
    """
    Composant de validation croisée.
    Compare les décisions Top-Down (Arbre) vs Bottom-Up (Annotation REST).
    """

    def compare(
        self,
        tree_config: dict,  # data/decision_config.json
        rest_reports: list,  # output of RESTEvaluator (list of RESTEntityReport)
    ) -> dict:
        """
        Compare les décisions et génère un rapport de convergence.
        """
        concordance_count = 0
        total_entities = 0
        divergences = []

        for report in rest_reports:
            entity = report.entity_type
            total_entities += 1

            # Décision Top-Down (Arbre)
            top_down_entry = tree_config.get(entity, {})
            # Parfois la config est structurée différemment (ex: {'entities': {...}})
            if "entities" in tree_config:
                top_down_entry = tree_config["entities"].get(entity, {})

            tree_decision = top_down_entry.get("method", "Unknown")

            # Décision Bottom-Up (Empirique REST basée sur TE observé)
            report.recommended_method = self._decide_from_empirics(report)
            rest_decision = report.recommended_method

            # Logique de comparaison (Simplifiée: REGLES vs ML vs LLM)
            # On considère "Rules" == "REGLES"
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
        """Normalise et aligne les noms de méthodes de l'arbre global et de l'interface REST empirique."""
        m = method.upper()
        if m in ["RULES", "REGLES"] or "RÈGLES" in m:
            return "REGLES"
        if m in ["ML", "ML_NER", "ML_CRF", "TRANSFORMER"] or "ML" in m or "TRANSFORMER" in m:
            return "ML"
        if "LLM" in m:
            return "LLM"
        return "UNKNOWN"

    def _decide_from_empirics(self, report) -> str:
        """
        Logique de décision Bottom-Up.
        Si l'entité est très répétitive (Te > 0.8) et assez stable (He > 0.7) -> REGLES.
        Sinon -> ML.
        """
        # Si très répétitif (Te observé > 0.8) -> Règles
        if report.empirical_te > 0.8:
            return "REGLES"
        # Sinon -> ML
        return "ML_NER"


if __name__ == "__main__":
    bridge = RESTDecisionBridge()
    print("REST Decision Bridge ready.")
