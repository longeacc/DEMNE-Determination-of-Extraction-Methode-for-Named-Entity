"""Unified BRAT corpus parser -- single source of truth.

Replaces the four divergent BRAT loaders that were scattered across
E_templatability, E_homogeneity, E_frequency, and E_risk_context.
Each scorer now calls ``load_brat_corpus`` with the options it needs.

The dashboard ``BratCorpusParser`` (dashboard/core/brat_parser.py) keeps
its own class because it handles Streamlit uploads, but it could delegate
line parsing to ``parse_ann_line`` in the future.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class BratAnnotation:
    """A single BRAT T-line annotation."""

    start: int
    end: int
    text: str
    entity_type: str
    file_id: str | None = None
    context: str = ""


def parse_ann_line(line: str) -> BratAnnotation | None:
    """Parse one BRAT ``.ann`` T-line into a :class:`BratAnnotation`.

    Returns ``None`` for non-T lines and malformed lines (silently skipped,
    matching previous behaviour).
    """
    line = line.strip()
    if not line.startswith("T"):
        return None
    parts = line.split("\t")
    if len(parts) < 3:
        return None
    try:
        info = parts[1].split()
        entity_type = info[0]
        start = int(info[1])
        # Handle discontinuous spans "10 20;30 40" -- take end of first span
        if ";" in parts[1]:
            end_str = parts[1].replace(";", " ").split()[2]
        else:
            end_str = info[-1]
        end = int(end_str)
        text = parts[2]
        return BratAnnotation(start=start, end=end, text=text, entity_type=entity_type)
    except (IndexError, ValueError):
        return None


def load_brat_corpus(
    data_dirs: list[str | Path],
    *,
    include_text: bool = False,
    context_window: int = 0,
) -> list[dict[str, Any]]:
    """Load BRAT ``.ann`` + ``.txt`` files into a corpus list.

    Parameters
    ----------
    data_dirs:
        Directories to scan for ``.ann`` files.
    include_text:
        If ``True``, load the full ``.txt`` content into each document dict.
    context_window:
        If > 0, extract *context_window* characters around each annotation
        and store it in ``annotation.context``.

    Returns
    -------
    list of dicts
        ``{"file_id": str, "text": str (optional), "annotations": list[BratAnnotation]}``
    """
    corpus: list[dict[str, Any]] = []
    processed: set[str] = set()

    for d in data_dirs:
        path = Path(d)
        if not path.exists():
            print(f"Warning: {path} does not exist.")
            continue

        for ann_file in sorted(path.glob("*.ann")):
            if ann_file.name in processed:
                continue
            processed.add(ann_file.name)

            text_content = ""
            txt_file = ann_file.with_suffix(".txt")
            if (include_text or context_window > 0) and txt_file.exists():
                with open(txt_file, encoding="utf-8") as fh:
                    text_content = fh.read()

            annotations: list[BratAnnotation] = []
            with open(ann_file, encoding="utf-8") as fh:
                for raw_line in fh:
                    ann = parse_ann_line(raw_line)
                    if ann is not None:
                        ann.file_id = ann_file.name
                        if context_window > 0 and text_content:
                            ctx_start = max(0, ann.start - context_window)
                            ctx_end = min(len(text_content), ann.end + context_window)
                            ann.context = text_content[ctx_start:ctx_end]
                        annotations.append(ann)

            doc: dict[str, Any] = {
                "file_id": ann_file.name,
                "annotations": annotations,
            }
            if include_text:
                doc["text"] = text_content
            corpus.append(doc)

    print(f"Loaded {len(corpus)} documents.")
    return corpus
