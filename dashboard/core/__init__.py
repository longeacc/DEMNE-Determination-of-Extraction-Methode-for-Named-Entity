try:
    import demne as _demne  # noqa: F401
except ModuleNotFoundError as exc:  # pragma: no cover
    raise ImportError(
        "The dashboard needs the demne package. From the repository root run:  pip install -e ."
    ) from exc

from .brat_parser import BratCorpusParser
from .metrics import DEMO_METRICS, MetricsCalculator
from .routing import compute_routing, compute_routing_detail

__all__ = [
    "BratCorpusParser",
    "DEMO_METRICS",
    "MetricsCalculator",
    "compute_routing",
    "compute_routing_detail",
]
