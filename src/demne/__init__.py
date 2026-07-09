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

from .E_creation_arbre_decision import DecisionTreeBuilder
from .E_frequency import FrequencyScorer
from .E_homogeneity import HomogeneityScorer
from .E_risk_context import RiskContextScorer
from .E_templatability import TemplatabilityScorer

DecisionTree = DecisionTreeBuilder

__version__ = "2.0.0"
__all__ = [
    "TemplatabilityScorer",
    "HomogeneityScorer",
    "RiskContextScorer",
    "FrequencyScorer",
    "DecisionTree",
]
