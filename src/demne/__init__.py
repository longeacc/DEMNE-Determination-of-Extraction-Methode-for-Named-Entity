"""
DEMNE / demne
=============
Main corpus analysis module for the DEMNE method.

Main exports:
- TemplatabilityScorer: compute the Te metric
- HomogeneityScorer: compute the He metric
- RiskContextScorer: compute the R metric
- FrequencyScorer: compute the Freq metric
- DecisionTree: decision tree routing RULES/TBM/LLM
"""

from .decision_tree_builder import DecisionTreeBuilder
from .frequency import FrequencyScorer
from .homogeneity import HomogeneityScorer
from .risk_context import RiskContextScorer
from .templatability import TemplatabilityScorer

DecisionTree = DecisionTreeBuilder

__version__ = "1.3.0"
__all__ = [
    "TemplatabilityScorer",
    "HomogeneityScorer",
    "RiskContextScorer",
    "FrequencyScorer",
    "DecisionTree",
]
