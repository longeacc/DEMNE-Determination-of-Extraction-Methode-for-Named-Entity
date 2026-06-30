import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from demne.REST_interface.rest_annotator import RESTAnnotator
from demne.REST_interface.rest_decision_bridge import RESTDecisionBridge
from demne.REST_interface.rest_evaluator import RESTEvaluator
from demne.REST_interface.yield_calculator import YieldCalculator


class RESTPipeline:
    """
    Main driver for the REST-interface (Rapid Empirical Semantic Tool).
    """

    def __init__(self):
        self.annotator = RESTAnnotator()
        self.evaluator = RESTEvaluator()
        self.bridge = RESTDecisionBridge()
        self.yield_calc = YieldCalculator()

    def run_pilot(self, documents: list):
        print("Starting REST Pilot Annotation...")
        annotations = self.annotator.annotate_batch(documents, mode="automated_test")

        print(f"Collected {len(annotations)} empirical annotations.")
        # Simulating Yield Calculation
        self.yield_calc.start_timing("Estrogen_receptor")
        # Simulating time passed
        self.yield_calc.stop_timing("Estrogen_receptor")

        report = self.evaluator.evaluate_entity("Estrogen_receptor", annotations)
        print(f"Empirical Evaluation done for 'Estrogen_receptor': Te={report.empirical_te}")
        return report


if __name__ == "__main__":
    p = RESTPipeline()
    p.run_pilot([("doc1", "Patient content here")])
