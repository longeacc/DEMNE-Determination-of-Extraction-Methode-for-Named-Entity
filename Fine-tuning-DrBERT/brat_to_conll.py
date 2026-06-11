r"""BRAT standoff annotation → CoNLL BIO converter.

Reads paired .ann/.txt BRAT files and produces CoNLL-2003 formatted
output suitable for token-classification fine-tuning.

Handles:
    - Multi-word entity spans
    - Discontinuous annotations (fragmented spans)
    - Overlapping annotations (longest-span-wins strategy)
    - Sentence segmentation via regex heuristic tuned for French clinical text

Typical usage::

    python Fine-tuning-DrBERT/brat_to_conll.py --input_dir ..\Datasets\Emmanuelle_35_cantemist\Emmanuelle_35_cantemist
    --output_dir Fine-tuning-DrBERT/data/conll/Emmanuelle_35_cantemist
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass(frozen=True, order=True)
class BratAnnotation:
    """Single text-bound BRAT annotation (T-line).

    Attributes:
        start: Character offset start (inclusive).
        end: Character offset end (exclusive).
        entity_type: Entity label string.
        text: Surface form captured by the annotation.
        ann_id: Original BRAT annotation id (e.g. ``T1``).
    """

    start: int
    end: int
    entity_type: str
    text: str
    ann_id: str = field(compare=False, default="")


@dataclass
class Token:
    """A single whitespace-delimited token with character offsets.

    Attributes:
        text: Surface form of the token.
        start: Character offset start (inclusive).
        end: Character offset end (exclusive).
        label: BIO label assigned during alignment.
    """

    text: str
    start: int
    end: int
    label: str = "O"


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

def parse_ann_file(ann_path: Path) -> list[BratAnnotation]:
    """Parse a BRAT ``.ann`` file into a list of text-bound annotations.

    Only ``T``-type (text-bound) lines are parsed.  Attribute (``A``),
    relation (``R``), event (``E``) and normalisation (``N``) lines are
    silently skipped.

    Args:
        ann_path: Path to the ``.ann`` file.

    Returns:
        Sorted list of :class:`BratAnnotation` instances.

    Raises:
        FileNotFoundError: If *ann_path* does not exist.
    """

    if not ann_path.exists():
        raise FileNotFoundError(f"Annotation file not found: {ann_path}")

    annotations: list[BratAnnotation] = []

    with ann_path.open("r", encoding="utf-8") as fh:
        for line_no, raw_line in enumerate(fh, start=1):
            line = raw_line.strip()
            if not line or not line.startswith("T"):
                continue

            parts = line.split("\t")
            if len(parts) < 3:
                logger.warning(
                    "%s:%d — malformed T-line (expected ≥3 tab-separated fields): %s",
                    ann_path.name,
                    line_no,
                    line,
                )
                continue

            ann_id = parts[0]
            info = parts[1]
            surface_text = parts[2]

            # info format: "EntityType start end" or discontinuous
            # "EntityType start1 end1;start2 end2"
            info_tokens = info.split()
            if len(info_tokens) < 3:
                logger.warning(
                    "%s:%d — cannot parse annotation info: %s",
                    ann_path.name,
                    line_no,
                    info,
                )
                continue

            entity_type = info_tokens[0]

            # Handle discontinuous spans → merge into bounding span
            span_text = " ".join(info_tokens[1:])
            span_pairs = span_text.split(";")

            global_start: Optional[int] = None
            global_end: Optional[int] = None

            for pair in span_pairs:
                offsets = pair.strip().split()
                if len(offsets) != 2:
                    continue
                try:
                    s, e = int(offsets[0]), int(offsets[1])
                except ValueError:
                    continue
                if global_start is None or s < global_start:
                    global_start = s
                if global_end is None or e > global_end:
                    global_end = e

            if global_start is None or global_end is None:
                logger.warning(
                    "%s:%d — no valid offsets parsed: %s",
                    ann_path.name,
                    line_no,
                    info,
                )
                continue

            annotations.append(
                BratAnnotation(
                    start=global_start,
                    end=global_end,
                    entity_type=entity_type,
                    text=surface_text,
                    ann_id=ann_id,
                )
            )

    annotations.sort()
    return annotations


# ---------------------------------------------------------------------------
# Tokenization (whitespace + punctuation split)
# ---------------------------------------------------------------------------

_TOKEN_RE = re.compile(r"\S+")


def tokenize_text(text: str) -> list[Token]:
    """Whitespace-based tokenizer preserving character offsets.

    Splits on whitespace boundaries.  Punctuation attached to tokens is
    **not** further split — subword alignment during fine-tuning handles
    this via WordPiece.

    Args:
        text: Raw document text.

    Returns:
        Ordered list of :class:`Token` instances.
    """

    tokens: list[Token] = []
    for m in _TOKEN_RE.finditer(text):
        tokens.append(Token(text=m.group(), start=m.start(), end=m.end()))
    return tokens


# ---------------------------------------------------------------------------
# Sentence segmentation (heuristic for French clinical text)
# ---------------------------------------------------------------------------

_SENT_BOUNDARY_RE = re.compile(
    r"(?<=[.!?])\s+(?=[A-ZÀ-ÖÙ-Ý])"  # sentence-ending punct + uppercase
    r"|(?<=\n)\s*(?=\S)"  # newline boundaries
)


def segment_sentences(tokens: list[Token], text: str) -> list[list[Token]]:
    """Group tokens into sentence-like segments.

    Uses a heuristic regex on the raw text to find sentence boundaries,
    then assigns tokens to the nearest preceding boundary.  This is
    deliberately simple — the downstream Transformer tokenizer will
    handle context windows.

    Args:
        tokens: Ordered list of tokens.
        text: Original document text (needed for boundary detection).

    Returns:
        List of sentence-token lists.  Each inner list is non-empty.
    """

    if not tokens:
        return []

    # Collect boundary character positions
    boundaries: list[int] = [0]
    for m in _SENT_BOUNDARY_RE.finditer(text):
        boundaries.append(m.start())
    boundaries.append(len(text) + 1)
    boundaries = sorted(set(boundaries))

    sentences: list[list[Token]] = []
    current_sentence: list[Token] = []
    boundary_idx = 0

    for tok in tokens:
        # Advance boundary pointer
        while (
            boundary_idx + 1 < len(boundaries)
            and boundaries[boundary_idx + 1] <= tok.start
        ):
            boundary_idx += 1
            if current_sentence:
                sentences.append(current_sentence)
                current_sentence = []

        current_sentence.append(tok)

    if current_sentence:
        sentences.append(current_sentence)

    return sentences


# ---------------------------------------------------------------------------
# BIO label alignment
# ---------------------------------------------------------------------------

def align_labels(
    tokens: list[Token],
    annotations: list[BratAnnotation],
) -> None:
    """Assign BIO labels to tokens based on character-offset overlap.

    Mutates ``token.label`` in-place.  Uses a longest-span-wins strategy
    when annotations overlap.

    Args:
        tokens: Tokenized document (will be mutated).
        annotations: Sorted BRAT annotations for this document.
    """

    # Build a character → annotation mapping (longest span wins)
    char_to_ann: dict[int, BratAnnotation] = {}

    # Process longest annotations first so shorter overlapping ones
    # don't clobber them.
    sorted_anns = sorted(annotations, key=lambda a: -(a.end - a.start))

    for ann in sorted_anns:
        for c in range(ann.start, ann.end):
            if c not in char_to_ann:
                char_to_ann[c] = ann

    prev_ann: Optional[BratAnnotation] = None

    for tok in tokens:
        # Find the annotation that covers the majority of this token
        ann_votes: dict[str, int] = {}
        for c in range(tok.start, tok.end):
            if c in char_to_ann:
                ann_key = char_to_ann[c].ann_id
                ann_votes[ann_key] = ann_votes.get(ann_key, 0) + 1

        if not ann_votes:
            tok.label = "O"
            prev_ann = None
            continue

        # Pick annotation with most character overlap
        best_ann_id = max(ann_votes, key=lambda k: ann_votes[k])
        best_ann = next(
            a for a in annotations if a.ann_id == best_ann_id
        )

        # Determine B- vs I-
        if prev_ann is not None and prev_ann.ann_id == best_ann.ann_id:
            tok.label = f"I-{best_ann.entity_type}"
        else:
            tok.label = f"B-{best_ann.entity_type}"

        prev_ann = best_ann


# ---------------------------------------------------------------------------
# CoNLL output
# ---------------------------------------------------------------------------

def write_conll(
    sentences: list[list[Token]],
    output_path: Path,
    doc_id: str = "",
) -> int:
    """Write sentences in CoNLL-2003 format (token<TAB>label per line).

    Args:
        sentences: List of sentence-token lists with labels assigned.
        output_path: Destination ``.conll`` file path.
        doc_id: Optional document identifier written as a comment line.

    Returns:
        Total number of tokens written.
    """

    output_path.parent.mkdir(parents=True, exist_ok=True)
    total_tokens = 0

    with output_path.open("w", encoding="utf-8") as fh:
        if doc_id:
            fh.write(f"# doc_id = {doc_id}\n")

        for sent_idx, sentence in enumerate(sentences):
            for tok in sentence:
                fh.write(f"{tok.text}\t{tok.label}\n")
                total_tokens += 1
            fh.write("\n")  # blank line between sentences

    return total_tokens


# ---------------------------------------------------------------------------
# High-level conversion
# ---------------------------------------------------------------------------

def convert_document(
    txt_path: Path,
    ann_path: Path,
    output_path: Path,
) -> int:
    """Convert a single BRAT document pair to CoNLL format.

    Args:
        txt_path: Path to the ``.txt`` file.
        ann_path: Path to the ``.ann`` file.
        output_path: Destination ``.conll`` file.

    Returns:
        Number of tokens written.

    Raises:
        FileNotFoundError: If either input file is missing.
    """

    if not txt_path.exists():
        raise FileNotFoundError(f"Text file not found: {txt_path}")

    text = txt_path.read_text(encoding="utf-8")
    annotations = parse_ann_file(ann_path)
    tokens = tokenize_text(text)
    align_labels(tokens, annotations)
    sentences = segment_sentences(tokens, text)

    doc_id = txt_path.stem
    n_tokens = write_conll(sentences, output_path, doc_id=doc_id)

    n_entities = sum(1 for t in tokens if t.label.startswith("B-"))
    logger.info(
        "Converted %s: %d tokens, %d entities, %d sentences",
        doc_id,
        n_tokens,
        n_entities,
        len(sentences),
    )
    return n_tokens


def convert_corpus(
    input_dir: Path,
    output_dir: Path,
    corpus_name: str = "",
) -> list[Path]:
    """Convert all BRAT document pairs in a directory to CoNLL.

    Scans *input_dir* for ``.txt`` files that have a matching ``.ann``
    file.  Documents without annotations produce all-``O`` labels.

    Args:
        input_dir: Directory containing ``.txt`` and ``.ann`` files.
        output_dir: Destination directory for ``.conll`` files.
        corpus_name: Human-readable corpus name for logging.

    Returns:
        List of output ``.conll`` file paths.
    """

    if not input_dir.is_dir():
        raise FileNotFoundError(
            f"Corpus directory not found: {input_dir}"
        )

    txt_files = sorted(input_dir.glob("*.txt"))

    if not txt_files:
        logger.warning("No .txt files found in %s", input_dir)
        return []

    output_dir.mkdir(parents=True, exist_ok=True)
    output_paths: list[Path] = []
    total_tokens = 0

    for txt_path in txt_files:
        ann_path = txt_path.with_suffix(".ann")
        if not ann_path.exists():
            # Create an empty .ann so we still get O-labelled tokens
            logger.warning(
                "No .ann file for %s — all tokens will be labelled O",
                txt_path.name,
            )
            ann_path.write_text("", encoding="utf-8")

        out_path = output_dir / f"{txt_path.stem}.conll"
        n = convert_document(txt_path, ann_path, out_path)
        total_tokens += n
        output_paths.append(out_path)

    tag = corpus_name or input_dir.name
    logger.info(
        "[%s] Converted %d documents (%d total tokens)",
        tag,
        len(output_paths),
        total_tokens,
    )
    return output_paths


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    """Command-line entry point for batch BRAT→CoNLL conversion."""

    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(message)s",
    )

    parser = argparse.ArgumentParser(
        description="Convert BRAT standoff annotations to CoNLL BIO format."
    )
    parser.add_argument(
        "--input_dir",
        type=Path,
        required=True,
        help="Directory containing .txt and .ann BRAT files.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        required=True,
        help="Destination directory for .conll output files.",
    )
    parser.add_argument(
        "--corpus_name",
        type=str,
        default="",
        help="Human-readable corpus name for logging.",
    )
    args = parser.parse_args()

    paths = convert_corpus(args.input_dir, args.output_dir, args.corpus_name)
    logger.info("Done — %d files written to %s", len(paths), args.output_dir)


if __name__ == "__main__":
    main()
