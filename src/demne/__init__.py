"""
DEMNE / demne
=============
Module principal d'analyse de corpus pour la méthode DEMNE.

Exports principaux :
- TemplatabilityScorer : calcul de la métrique Te
- HomogeneityScorer : calcul de la métrique He
- RiskContextScorer : calcul de la métrique R
- FrequencyScorer : calcul de la métrique Freq
- DecisionTree : arbre de décision RÈGLES/TBM/LLM
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
