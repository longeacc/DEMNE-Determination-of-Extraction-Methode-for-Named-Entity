"""eco2ai energy tracking helpers -- single source of truth.

Every scorer (Te, He, Freq, R, Feas, tree) used to carry its own 20-line
import/start/stop block with subtle differences.  This module replaces all
of them with two calls: ``start_tracker`` and ``stop_tracker``.
"""

from __future__ import annotations

import os

try:
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        from eco2ai import Tracker, set_params  # type: ignore[import-untyped]

    HAS_ECO2AI = True
except ImportError:
    Tracker = None  # type: ignore[assignment,misc]
    set_params = None  # type: ignore[assignment]
    HAS_ECO2AI = False


def start_tracker(
    project_name: str,
    description: str,
    file_name: str = "data/energy_consumption_DEMNE.csv",
) -> Tracker | None:
    """Start eco2ai tracking if available and not disabled.

    Returns the running ``Tracker`` instance, or ``None`` if eco2ai is
    unavailable or disabled via the ``DISABLE_ECO2AI`` environment variable.
    """
    if not HAS_ECO2AI or os.environ.get("DISABLE_ECO2AI"):
        return None
    set_params(
        project_name=project_name,
        experiment_description=description,
        file_name=file_name,
    )
    tracker = Tracker()
    tracker.start()
    return tracker


def stop_tracker(tracker: Tracker | None) -> None:
    """Stop an eco2ai tracker safely (handles ``None`` and exceptions)."""
    if tracker is None:
        return
    try:
        tracker.stop()
    except Exception as exc:  # noqa: BLE001
        print(
            f"\nWarning: eco2ai tracking error: {exc}\n"
            "Carbon emission tracking data could not be saved, "
            "but analysis results are preserved."
        )
