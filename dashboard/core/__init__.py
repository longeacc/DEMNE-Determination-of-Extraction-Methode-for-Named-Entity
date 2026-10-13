import sys as _sys
from pathlib import Path as _Path

try:  # make the `demne` package importable when the dashboard runs from source
    import demne as _demne  # noqa: F401
except ModuleNotFoundError:  # pragma: no cover
    _src = _Path(__file__).resolve().parents[2] / "src"
    if _src.exists() and str(_src) not in _sys.path:
        _sys.path.insert(0, str(_src))

from .brat_parser import BratCorpusParser
from .metrics import DEMO_METRICS, MetricsCalculator
from .routing import compute_routing

__all__ = ["BratCorpusParser", "DEMO_METRICS", "MetricsCalculator", "compute_routing"]
