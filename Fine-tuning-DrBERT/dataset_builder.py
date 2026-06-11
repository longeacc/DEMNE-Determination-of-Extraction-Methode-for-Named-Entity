"""Dataset builder for DEMNE fine-tuning.

Fuses CoNLL-formatted outputs from the three DEMNE corpora, discovers
the full label set, performs a stratified 80/20 train/test split that
preserves corpus-source proportions, and produces HuggingFace
:class:`datasets.DatasetDict` with tokenized + label-aligned examples.

Typical usage::

    python dataset_builder.py \\
        --corpus_dir /data/demne_corpora \\
        --output_dir /data/demne_hf_dataset \\
        --model_name Dr-BERT/DrBERT-7GB
"""

from __future__ import annotations

import json
import logging
import random
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import datasets
from transformers import AutoTokenizer, PreTrainedTokenizerBase

# Robust import of the BRAT→CoNLL converter whether this module is imported
# as a package (``from .brat_to_conll``) or as a flat top-level module
# (``python dataset_builder.py`` / ``from dataset_builder import …``).
try:
    from .brat_to_conll import convert_corpus
except ImportError:
    from brat_to_conll import convert_corpus

logger = logging.getLogger(__name__)

SEED = 42
TRAIN_RATIO = 0.80

# The three canonical sub-directory names expected under --corpus_dir.
CORPUS_SUBDIRS: dict[str, str] = {
    "cantemist": "cantemist",
    "redjdal": "redjdal",
    "rcp_esmo": "rcp_esmo",
}

# 7 priority biomarker entities (ICD-10 C50.*).
PRIORITY_ENTITIES: list[str] = [
    "Estrogen_receptor",
    "Progesterone_receptor",
    "HER2_status",
    "HER2_IHC",
    "Ki67",
    "HER2_FISH",
    "Genetic_mutation",
]


# ---------------------------------------------------------------------------
# CoNLL reading
# ---------------------------------------------------------------------------

@dataclass
class ConllSentence:
    """A single sentence read from a CoNLL file.

    Attributes:
        tokens: List of surface-form strings.
        labels: Corresponding BIO label strings.
        corpus: Source corpus identifier.
        doc_id: Originating document identifier.
    """

    tokens: list[str] = field(default_factory=list)
    labels: list[str] = field(default_factory=list)
    corpus: str = ""
    doc_id: str = ""


def read_conll_file(path: Path, corpus: str) -> list[ConllSentence]:
    """Read a single CoNLL file into a list of sentences.

    Args:
        path: Path to a ``.conll`` file.
        corpus: Corpus identifier string.

    Returns:
        List of :class:`ConllSentence` instances.
    """

    sentences: list[ConllSentence] = []
    current = ConllSentence(corpus=corpus, doc_id=path.stem)

    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            stripped = line.rstrip("\n")

            # Comment lines
            if stripped.startswith("#"):
                continue

            # Blank line → sentence boundary
            if not stripped:
                if current.tokens:
                    sentences.append(current)
                    current = ConllSentence(corpus=corpus, doc_id=path.stem)
                continue

            parts = stripped.split("\t")
            if len(parts) < 2:
                parts = stripped.split()
            if len(parts) < 2:
                logger.warning("Skipping malformed line in %s: %r", path.name, stripped)
                continue

            current.tokens.append(parts[0])
            current.labels.append(parts[1])

    # Flush last sentence
    if current.tokens:
        sentences.append(current)

    return sentences


def read_corpus_conll(conll_dir: Path, corpus: str) -> list[ConllSentence]:
    """Read all CoNLL files from a directory.

    Args:
        conll_dir: Directory containing ``.conll`` files.
        corpus: Corpus identifier string.

    Returns:
        Combined list of sentences from all files.
    """

    if not conll_dir.is_dir():
        raise FileNotFoundError(f"CoNLL directory not found: {conll_dir}")

    files = sorted(conll_dir.glob("*.conll"))
    if not files:
        logger.warning("No .conll files in %s", conll_dir)
        return []

    all_sentences: list[ConllSentence] = []
    for f in files:
        all_sentences.extend(read_conll_file(f, corpus))

    logger.info(
        "[%s] Loaded %d sentences from %d files",
        corpus,
        len(all_sentences),
        len(files),
    )
    return all_sentences


# ---------------------------------------------------------------------------
# Label registry
# ---------------------------------------------------------------------------

def build_label_map(
    sentences: list[ConllSentence],
) -> tuple[dict[str, int], dict[int, str]]:
    """Build label↔id mappings from observed labels.

    The ``O`` label is always assigned id 0.  ``B-`` and ``I-`` variants
    for the 7 priority biomarker entities are guaranteed to appear even if
    absent from the corpus (to ensure the classifier head has the correct
    dimensionality).

    Args:
        sentences: All fused sentences.

    Returns:
        Tuple of ``(label2id, id2label)`` dicts.
    """

    observed: set[str] = {"O"}
    for sent in sentences:
        observed.update(sent.labels)

    # Guarantee priority entity coverage
    for ent in PRIORITY_ENTITIES:
        observed.add(f"B-{ent}")
        observed.add(f"I-{ent}")

    # Deterministic ordering: O first, then B-/I- pairs alphabetically
    bio_labels = sorted(lbl for lbl in observed if lbl != "O")
    ordered = ["O"] + bio_labels

    label2id = {lbl: idx for idx, lbl in enumerate(ordered)}
    id2label = {idx: lbl for lbl, idx in label2id.items()}

    logger.info(
        "Label map: %d labels (%d entity types)",
        len(label2id),
        len([l for l in label2id if l.startswith("B-")]),
    )
    return label2id, id2label


# ---------------------------------------------------------------------------
# Stratified split
# ---------------------------------------------------------------------------

def stratified_split(
    sentences: list[ConllSentence],
    train_ratio: float = TRAIN_RATIO,
    seed: int = SEED,
) -> tuple[list[ConllSentence], list[ConllSentence]]:
    """Split sentences into train/test preserving corpus proportions.

    Within each corpus, sentences are shuffled and split at the
    *train_ratio* boundary, so every corpus contributes proportionally
    to both splits.

    Args:
        sentences: All fused sentences.
        train_ratio: Fraction of sentences used for training.
        seed: Random seed for reproducibility.

    Returns:
        Tuple of ``(train_sentences, test_sentences)``.
    """

    rng = random.Random(seed)

    # Group by corpus
    corpus_groups: dict[str, list[ConllSentence]] = {}
    for sent in sentences:
        corpus_groups.setdefault(sent.corpus, []).append(sent)

    train_sents: list[ConllSentence] = []
    test_sents: list[ConllSentence] = []

    for corpus, group in sorted(corpus_groups.items()):
        rng.shuffle(group)
        split_idx = max(1, int(len(group) * train_ratio))
        train_sents.extend(group[:split_idx])
        test_sents.extend(group[split_idx:])
        logger.info(
            "[%s] Split: %d train / %d test (ratio=%.2f)",
            corpus,
            split_idx,
            len(group) - split_idx,
            split_idx / len(group),
        )

    # Final shuffle within splits
    rng.shuffle(train_sents)
    rng.shuffle(test_sents)

    logger.info(
        "Total split: %d train / %d test",
        len(train_sents),
        len(test_sents),
    )
    return train_sents, test_sents


# ---------------------------------------------------------------------------
# Tokenization + subword label alignment
# ---------------------------------------------------------------------------

def tokenize_and_align(
    sentences: list[ConllSentence],
    tokenizer: PreTrainedTokenizerBase,
    label2id: dict[str, int],
    max_length: int = 512,
) -> dict[str, list[Any]]:
    """Tokenize sentences with WordPiece and align BIO labels.

    Implements the first-subtoken strategy:
    - The first sub-token of each word receives the original BIO label.
    - Continuation sub-tokens receive label id ``-100`` (ignored by
      cross-entropy loss in HuggingFace).
    - ``[CLS]`` and ``[SEP]`` special tokens receive ``-100``.

    Args:
        sentences: Sentences with pre-tokenized words and BIO labels.
        tokenizer: HuggingFace tokenizer.
        label2id: Label-to-id mapping.
        max_length: Maximum sequence length (truncation boundary).

    Returns:
        Dict with keys ``input_ids``, ``attention_mask``, ``labels``
        ready for :class:`datasets.Dataset.from_dict`.
    """

    all_input_ids: list[list[int]] = []
    all_attention_masks: list[list[int]] = []
    all_labels: list[list[int]] = []

    unknown_labels: Counter[str] = Counter()

    for sent in sentences:
        tokenized = tokenizer(
            sent.tokens,
            is_split_into_words=True,
            truncation=True,
            max_length=max_length,
            padding=False,  # dynamic padding via DataCollator
        )

        word_ids = tokenized.word_ids()
        aligned_labels: list[int] = []
        previous_word_id: int | None = None

        for word_id in word_ids:
            if word_id is None:
                # Special token ([CLS], [SEP])
                aligned_labels.append(-100)
            elif word_id != previous_word_id:
                # First sub-token of a new word
                raw_label = sent.labels[word_id]
                if raw_label in label2id:
                    aligned_labels.append(label2id[raw_label])
                else:
                    unknown_labels[raw_label] += 1
                    aligned_labels.append(label2id["O"])
            else:
                # Continuation sub-token → ignore
                aligned_labels.append(-100)

            previous_word_id = word_id

        all_input_ids.append(tokenized["input_ids"])
        all_attention_masks.append(tokenized["attention_mask"])
        all_labels.append(aligned_labels)

    if unknown_labels:
        logger.warning(
            "Encountered %d unknown labels (mapped to O): %s",
            sum(unknown_labels.values()),
            dict(unknown_labels.most_common(10)),
        )

    return {
        "input_ids": all_input_ids,
        "attention_mask": all_attention_masks,
        "labels": all_labels,
    }


# ---------------------------------------------------------------------------
# High-level builder
# ---------------------------------------------------------------------------

# Default CoNLL cache directory (next to this module), used when explicit
# corpus paths are supplied without a corpus_dir to anchor the cache.
_DEFAULT_CONLL_CACHE = Path(__file__).resolve().parent / "data" / "conll"


def resolve_corpus_dirs(
    corpus_dir: Path | None = None,
    corpus_paths: dict[str, Path] | None = None,
) -> dict[str, Path]:
    """Resolve the set of ``{corpus_name: brat_dir}`` to load.

    Two mutually-compatible modes:

    - ``corpus_paths``: explicit mapping of corpus name → BRAT directory
      (each containing ``.txt``/``.ann`` pairs). Takes priority.
    - ``corpus_dir``: legacy mode — a root holding the canonical
      sub-folders ``cantemist/``, ``redjdal/``, ``rcp_esmo/``.

    Args:
        corpus_dir: Optional legacy root directory.
        corpus_paths: Optional explicit name→path mapping.

    Returns:
        Ordered mapping of corpus name → existing BRAT directory.
    """

    resolved: dict[str, Path] = {}

    if corpus_paths:
        for name, path in corpus_paths.items():
            p = Path(path)
            if p.is_dir():
                resolved[name] = p
            else:
                logger.warning("Corpus path not found: %s — skipping", p)

    if corpus_dir is not None:
        for corpus_key, subdir_name in CORPUS_SUBDIRS.items():
            brat_dir = Path(corpus_dir) / subdir_name
            if brat_dir.is_dir() and corpus_key not in resolved:
                resolved[corpus_key] = brat_dir

    return resolved


def build_dataset(
    corpus_dir: Path | None = None,
    model_name: str = "Dr-BERT/DrBERT-7GB",
    max_length: int = 512,
    train_ratio: float = TRAIN_RATIO,
    seed: int = SEED,
    output_dir: Path | None = None,
    corpus_paths: dict[str, Path] | None = None,
    conll_cache: Path | None = None,
) -> tuple[datasets.DatasetDict, dict[str, int], dict[int, str]]:
    """End-to-end dataset construction pipeline.

    1. Converts BRAT → CoNLL for each sub-corpus (if not yet done).
    2. Reads CoNLL files.
    3. Builds the label map.
    4. Splits 80/20 stratified by corpus.
    5. Tokenizes + aligns labels.
    6. Returns a HuggingFace DatasetDict.

    Args:
        corpus_dir: Legacy root directory with sub-folders ``cantemist/``,
            ``redjdal/``, ``rcp_esmo/`` containing BRAT files. Optional if
            ``corpus_paths`` is given.
        model_name: HuggingFace model identifier for the tokenizer.
        max_length: Maximum sequence length.
        train_ratio: Train split fraction.
        seed: Random seed.
        output_dir: If provided, save the DatasetDict to disk.
        corpus_paths: Explicit mapping ``{corpus_name: brat_dir}``. Takes
            priority over ``corpus_dir``.
        conll_cache: Directory for cached ``.conll`` output. Defaults to a
            sub-folder of ``corpus_dir`` (legacy) or ``<module>/data/conll``.

    Returns:
        Tuple of ``(dataset_dict, label2id, id2label)``.
    """

    corpus_dirs = resolve_corpus_dirs(corpus_dir=corpus_dir, corpus_paths=corpus_paths)
    if not corpus_dirs:
        raise RuntimeError(
            "No corpus directories resolved. Provide --corpus_path name=dir "
            "entries, or a --corpus_dir containing cantemist/, redjdal/, "
            "rcp_esmo/ sub-folders."
        )

    # Step 1 — Convert BRAT → CoNLL (idempotent)
    if conll_cache is not None:
        conll_root = Path(conll_cache)
    elif corpus_dir is not None:
        conll_root = Path(corpus_dir) / "_conll_cache"
    else:
        conll_root = _DEFAULT_CONLL_CACHE

    all_sentences: list[ConllSentence] = []

    for corpus_key, brat_dir in corpus_dirs.items():
        conll_dir = conll_root / corpus_key
        if not any(conll_dir.glob("*.conll")):
            logger.info("Converting BRAT→CoNLL for [%s]…", corpus_key)
            convert_corpus(brat_dir, conll_dir, corpus_name=corpus_key)

        sents = read_corpus_conll(conll_dir, corpus=corpus_key)
        all_sentences.extend(sents)

    if not all_sentences:
        raise RuntimeError(
            f"No sentences loaded from corpora: {list(corpus_dirs)}. "
            "Check that each directory contains .txt/.ann pairs."
        )

    logger.info("Total sentences loaded: %d", len(all_sentences))

    # Step 2 — Label map
    label2id, id2label = build_label_map(all_sentences)

    # Step 3 — Stratified split (BEFORE tokenization → no data leakage)
    train_sents, test_sents = stratified_split(
        all_sentences, train_ratio=train_ratio, seed=seed
    )

    # Step 4 — Tokenize + align
    logger.info("Loading tokenizer: %s", model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    logger.info("Tokenizing train split (%d sentences)…", len(train_sents))
    train_data = tokenize_and_align(train_sents, tokenizer, label2id, max_length)

    logger.info("Tokenizing test split (%d sentences)…", len(test_sents))
    test_data = tokenize_and_align(test_sents, tokenizer, label2id, max_length)

    # Step 5 — HuggingFace Dataset
    train_ds = datasets.Dataset.from_dict(train_data)
    test_ds = datasets.Dataset.from_dict(test_data)

    ds = datasets.DatasetDict({"train": train_ds, "test": test_ds})

    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        ds.save_to_disk(str(output_dir))

        # Persist label map alongside dataset
        label_map_path = output_dir / "label_map.json"
        label_map_path.write_text(
            json.dumps({"label2id": label2id, "id2label": {str(k): v for k, v in id2label.items()}}, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        logger.info("Dataset saved to %s", output_dir)

    # Log corpus-level statistics
    _log_label_stats(all_sentences)

    return ds, label2id, id2label


def parse_corpus_paths(items: list[str]) -> dict[str, Path]:
    """Parse repeated ``NAME=DIR`` CLI strings into a ``{name: Path}`` dict.

    A bare path without ``=`` uses the directory's own name as the corpus
    key. Raises ``ValueError`` on duplicate names.

    Args:
        items: List of ``"name=path"`` (or bare ``"path"``) strings.

    Returns:
        Ordered mapping of corpus name → Path.
    """

    paths: dict[str, Path] = {}
    for item in items:
        if "=" in item:
            name, _, raw = item.partition("=")
            name = name.strip()
            path = Path(raw.strip())
        else:
            path = Path(item.strip())
            name = path.name
        if not name:
            raise ValueError(f"Empty corpus name in --corpus_path entry: {item!r}")
        if name in paths:
            raise ValueError(f"Duplicate corpus name: {name!r}")
        paths[name] = path
    return paths


def _log_label_stats(sentences: list[ConllSentence]) -> None:
    """Log entity label distribution for diagnostic purposes."""

    entity_counter: Counter[str] = Counter()
    for sent in sentences:
        for label in sent.labels:
            if label.startswith("B-"):
                entity_counter[label[2:]] += 1

    logger.info("--- Entity distribution (B- counts) ---")
    for ent, count in entity_counter.most_common():
        marker = " ★" if ent in PRIORITY_ENTITIES else ""
        logger.info("  %-35s %5d%s", ent, count, marker)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    """Command-line entry point."""

    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(message)s",
    )

    parser = argparse.ArgumentParser(
        description="Build HuggingFace NER dataset from DEMNE corpora."
    )
    parser.add_argument(
        "--corpus_dir",
        type=Path,
        default=None,
        help="Legacy root directory containing cantemist/, redjdal/, rcp_esmo/.",
    )
    parser.add_argument(
        "--corpus_path",
        action="append",
        default=[],
        metavar="NAME=DIR",
        help="Explicit corpus as name=path to a BRAT directory. Repeatable.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("data/demne_hf_dataset"),
        help="Destination for the HuggingFace DatasetDict.",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="Dr-BERT/DrBERT-7GB",
        help="HuggingFace model name for the tokenizer.",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=512,
        help="Maximum sequence length.",
    )

    args = parser.parse_args()

    corpus_paths = parse_corpus_paths(args.corpus_path)
    if not corpus_paths and args.corpus_dir is None:
        parser.error("Provide at least one --corpus_path NAME=DIR or --corpus_dir.")

    ds, label2id, id2label = build_dataset(
        corpus_dir=args.corpus_dir,
        corpus_paths=corpus_paths or None,
        model_name=args.model_name,
        max_length=args.max_length,
        output_dir=args.output_dir,
    )

    logger.info("Train: %d examples — Test: %d examples", len(ds["train"]), len(ds["test"]))
    logger.info("Labels: %d", len(label2id))


if __name__ == "__main__":
    main()
