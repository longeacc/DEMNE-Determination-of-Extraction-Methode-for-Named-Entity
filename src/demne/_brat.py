"""Unified BRAT corpus parser -- single source of truth.

Every consumer parses T-lines with :func:`parse_ann_line` /
:func:`parse_ann_content`: the scorers (via ``load_brat_corpus``), the TF-IDF
coverage metric, the frequency scorer, the optimizer and the dashboard
``BratCorpusParser`` (which only adds Streamlit upload handling on top).
"""

from __future__ import annotations

from collections.abc import Sequence
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
    ann_id: str = ""


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
        return BratAnnotation(
            start=start, end=end, text=text, entity_type=entity_type, ann_id=parts[0]
        )
    except (IndexError, ValueError):
        return None


def parse_ann_content(
    ann_content: str, text: str = "", context_window: int = 0
) -> list[BratAnnotation]:
    """Parse the whole content of one ``.ann`` file.

    If ``context_window`` > 0 and ``text`` is given, each annotation gets
    ``context_window`` characters of ``text`` on both sides in ``.context``.
    """
    out: list[BratAnnotation] = []
    for raw_line in ann_content.splitlines():
        ann = parse_ann_line(raw_line)
        if ann is None:
            continue
        if context_window > 0 and text:
            ann.context = text[
                max(0, ann.start - context_window) : min(len(text), ann.end + context_window)
            ]
        out.append(ann)
    return out


def read_text(path: Path) -> str:
    """Read a UTF-8 file, falling back to latin-1 on decoding errors."""
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return path.read_text(encoding="latin-1", errors="replace")


def load_brat_corpus(
    data_dirs: Sequence[str | Path],
    *,
    include_text: bool = False,
    context_window: int = 0,
    recursive: bool = False,
    quiet: bool = False,
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
    recursive:
        If ``True``, also scan sub-folders.
    quiet:
        If ``True``, print neither the missing-folder warning nor the summary.

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
            if not quiet:
                print(f"Warning: {path} does not exist.")
            continue

        ann_files = path.rglob("*.ann") if recursive else path.glob("*.ann")
        for ann_file in sorted(ann_files):
            key = str(ann_file.relative_to(path)) if recursive else ann_file.name
            if key in processed:
                continue
            processed.add(key)

            text_content = ""
            txt_file = ann_file.with_suffix(".txt")
            if (include_text or context_window > 0) and txt_file.exists():
                text_content = read_text(txt_file)

            annotations = parse_ann_content(read_text(ann_file), text_content, context_window)
            for ann in annotations:
                ann.file_id = ann_file.name

            doc: dict[str, Any] = {
                "file_id": ann_file.name,
                "annotations": annotations,
            }
            if include_text:
                doc["text"] = text_content
            corpus.append(doc)

    if not quiet:
        print(f"Loaded {len(corpus)} documents.")
    return corpus
