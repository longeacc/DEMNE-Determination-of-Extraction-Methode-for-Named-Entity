"""`demne` console entry point: delegates to the repository's main.py."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


def main() -> None:
    # main.py drives repo-level assets (dashboard/, data/, Results/), so it only
    # exists next to an editable / source checkout (src/demne/cli.py → repo root).
    script = Path(__file__).resolve().parents[2] / "main.py"
    if not script.is_file():
        sys.exit(
            "The `demne` command needs a source checkout (main.py, dashboard/, data/).\n"
            "Clone the repository and run:  pip install -e ."
        )
    spec = importlib.util.spec_from_file_location("_demne_main", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.main()


if __name__ == "__main__":
    main()
