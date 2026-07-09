class RESTEntityReport:
    def __init__(self, entity_type, empirical_te, empirical_he, empirical_r, n_patterns):
        self.entity_type = entity_type
        self.empirical_te = empirical_te
        self.empirical_he = empirical_he
        self.empirical_r = empirical_r
        self.n_patterns = n_patterns
        self.recommended_method = "Unknown"

    def to_dict(self):
        return {
            "entity_type": self.entity_type,
            "metrics": {
                "Te_emp": self.empirical_te,
                "He_emp": self.empirical_he,
                "Risk_emp": self.empirical_r,
                "n_patterns": self.n_patterns,
            },
            "recommendation": self.recommended_method,
        }


class RESTEvaluator:
    """
    Empirical analysis of pilot annotations compared with theoretical metrics.
    Computes empirical proxies for Te, He, and R.
    """

    def evaluate_entity(
        self,
        entity_type: str,
        annotations: list,  # List of BratAnnotation objects
        context_window: int = 5,  # pylint: disable=unused-argument
    ) -> RESTEntityReport:
        """
        Analyse annotations for a given entity.
        """
        relevant_anns = [a for a in annotations if a.entity_type == entity_type]
        n_total = len(relevant_anns)

        if n_total == 0:
            return RESTEntityReport(entity_type, 0, 0, 0, 0)

        # 1. Empirical Te (Templateability)
        # Proxy: low unique-text ratio vs total — "Is the entity always phrased the same way?"
        unique_texts = set(a.text.lower().strip() for a in relevant_anns)
        n_unique = len(unique_texts)

        # Ratio : 1 - (unique / total). Si unique=total -> 0. Si unique=1 -> ~1.
        empirical_te = 1.0 - (n_unique / n_total)

        # 2. Empirical He (Homogeneity)
        # TTR (Token-Type Ratio) on the annotation corpus
        # "Is the vocabulary surrounding/within the entity consistent?"
        # Using entity text itself for internal lexical diversity
        empirical_he = self._calculate_ttr([a.text for a in relevant_anns])
        # Inverted: high TTR = heterogeneous, high He = homogeneous
        empirical_he = 1.0 - empirical_he

        # 3. Empirical R (Risk)
        # Analyse context surrounding the span
        # High context variability -> higher ambiguity risk -> R increases
        # Uses TTR of left/right contexts
        left_ctxs = [a.context_left for a in relevant_anns]
        right_ctxs = [a.context_right for a in relevant_anns]
        r_context = self._calculate_ttr(left_ctxs + right_ctxs)

        # Empirical Risk = r_context: higher context variability = riskier.
        # Stable context ("The patient has...") = lower false-positive risk.
        empirical_r = r_context

        return RESTEntityReport(
            entity_type,
            empirical_te=empirical_te,
            empirical_he=empirical_he,
            empirical_r=empirical_r,
            n_patterns=n_unique,
        )

    def _calculate_ttr(self, texts: list) -> float:
        """Calculates Type-Token Ratio."""
        if not texts:
            return 0.0

        all_tokens = []
        for text in texts:
            # Simple tokenization by space, lowercased
            tokens = text.lower().strip().split()
            all_tokens.extend(tokens)

        if not all_tokens:
            return 0.0

        unique_types = set(all_tokens)
        return len(unique_types) / len(all_tokens)


if __name__ == "__main__":
    evaluator = RESTEvaluator()
    print("REST Evaluator ready.")
