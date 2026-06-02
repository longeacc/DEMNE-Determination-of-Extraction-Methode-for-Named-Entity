"""DEMNE — Standalone evaluation for a fine-tuned DrBERT NER checkpoint.

Loads a saved checkpoint and evaluates it against a held-out test set,
producing per-entity F1 scores, a full classification report, and a
CSV export.

Usage::

    python evaluate.py \\
        --checkpoint models/drbert_finetuned/best \\
        --corpus_dir /data/demne_corpora \\
        --device cuda
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    DataCollatorForTokenClassification,
    Trainer,
)

from fine_tuning.data.dataset_builder import (
    PRIORITY_ENTITIES,
    build_dataset,
)
from fine_tuning.train import export_metrics_csv, resolve_device

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Evaluation core
# ---------------------------------------------------------------------------

def evaluate_checkpoint(
    checkpoint_dir: Path,
    corpus_dir: Path,
    device: str = "auto",
    max_length: int = 512,
    batch_size: int = 32,
    results_dir: Path = Path("results"),
) -> dict[str, float]:
    """Evaluate a fine-tuned checkpoint on the DEMNE test split.

    Args:
        checkpoint_dir: Path to saved model directory (containing
            ``config.json``, ``model.safetensors``, ``label_map.json``).
        corpus_dir: Root BRAT corpus directory.
        device: Compute device.
        max_length: Maximum sequence length.
        batch_size: Evaluation batch size.
        results_dir: Directory for CSV output.

    Returns:
        Dictionary of evaluation metrics.
    """

    device = resolve_device(device)

    # ---- Load label map ----
    label_map_path = checkpoint_dir / "label_map.json"
    if not label_map_path.exists():
        raise FileNotFoundError(
            f"label_map.json not found in {checkpoint_dir}. "
            "Ensure the checkpoint was saved with train.py."
        )

    with label_map_path.open("r", encoding="utf-8") as fh:
        raw = json.load(fh)

    label2id: dict[str, int] = raw["label2id"]
    id2label: dict[int, str] = {int(k): v for k, v in raw["id2label"].items()}

    logger.info("Loaded label map: %d labels", len(label2id))

    # ---- Load model + tokenizer ----
    logger.info("Loading checkpoint: %s", checkpoint_dir)
    model = AutoModelForTokenClassification.from_pretrained(
        str(checkpoint_dir),
        num_labels=len(label2id),
        id2label=id2label,
        label2id=label2id,
    )
    tokenizer = AutoTokenizer.from_pretrained(str(checkpoint_dir))

    # ---- Build dataset (only need test split) ----
    model_name = str(checkpoint_dir)
    ds, _, _ = build_dataset(
        corpus_dir=corpus_dir,
        model_name=model_name,
        max_length=max_length,
    )

    test_ds = ds["test"]
    logger.info("Test split: %d examples", len(test_ds))

    # ---- seqeval metrics ----
    from seqeval.metrics import (
        classification_report,
        f1_score,
        precision_score,
        recall_score,
    )

    def compute_metrics(eval_pred: tuple[np.ndarray, np.ndarray]) -> dict[str, float]:
        logits, label_ids = eval_pred
        predictions = np.argmax(logits, axis=-1)

        true_labels: list[list[str]] = []
        pred_labels: list[list[str]] = []

        for pred_seq, label_seq in zip(predictions, label_ids):
            true_sent: list[str] = []
            pred_sent: list[str] = []
            for p, l in zip(pred_seq, label_seq):
                if l == -100:
                    continue
                true_sent.append(id2label.get(int(l), "O"))
                pred_sent.append(id2label.get(int(p), "O"))
            true_labels.append(true_sent)
            pred_labels.append(pred_sent)

        f1_macro = f1_score(true_labels, pred_labels, average="macro", zero_division=0)
        precision = precision_score(true_labels, pred_labels, average="macro", zero_division=0)
        recall = recall_score(true_labels, pred_labels, average="macro", zero_division=0)

        report_dict = classification_report(
            true_labels, pred_labels, output_dict=True, zero_division=0
        )

        # Full text report
        report_text = classification_report(
            true_labels, pred_labels, zero_division=0
        )
        logger.info("Classification report:\n%s", report_text)

        metrics: dict[str, float] = {
            "f1_macro": float(f1_macro),
            "precision_macro": float(precision),
            "recall_macro": float(recall),
        }

        # Per-entity metrics for all entities found
        for ent_key, ent_metrics in report_dict.items():
            if isinstance(ent_metrics, dict) and "f1-score" in ent_metrics:
                priority_flag = " ★" if ent_key in PRIORITY_ENTITIES else ""
                logger.info(
                    "  %-35s  F1=%.4f  P=%.4f  R=%.4f  n=%d%s",
                    ent_key,
                    ent_metrics["f1-score"],
                    ent_metrics["precision"],
                    ent_metrics["recall"],
                    ent_metrics["support"],
                    priority_flag,
                )
                metrics[f"f1_{ent_key}"] = float(ent_metrics["f1-score"])
                metrics[f"precision_{ent_key}"] = float(ent_metrics["precision"])
                metrics[f"recall_{ent_key}"] = float(ent_metrics["recall"])
                metrics[f"support_{ent_key}"] = float(ent_metrics["support"])

        return metrics

    # ---- Trainer for evaluation only ----
    data_collator = DataCollatorForTokenClassification(
        tokenizer=tokenizer,
        padding=True,
        max_length=max_length,
    )

    from transformers import TrainingArguments

    eval_args = TrainingArguments(
        output_dir=str(results_dir / "eval_tmp"),
        per_device_eval_batch_size=batch_size,
        no_cuda=(device == "cpu"),
        use_mps_device=(device == "mps"),
        report_to="none",
        seed=42,
    )

    trainer = Trainer(
        model=model,
        args=eval_args,
        eval_dataset=test_ds,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    # ---- Evaluate ----
    results = trainer.evaluate()

    # ---- Export ----
    export_metrics_csv(results, results_dir, tag="eval")

    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    """CLI entry point."""

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    parser = argparse.ArgumentParser(
        description="Evaluate a fine-tuned DrBERT NER checkpoint.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Path to the saved model checkpoint directory.",
    )
    parser.add_argument(
        "--corpus_dir",
        type=Path,
        required=True,
        help="Root BRAT corpus directory.",
    )
    parser.add_argument(
        "--device",
        type=str,
        choices=["cuda", "cpu", "mps", "auto"],
        default="auto",
        help="Compute device.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Evaluation batch size.",
    )
    parser.add_argument(
        "--results_dir",
        type=Path,
        default=Path("results"),
        help="Directory for metric CSV output.",
    )
    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("DEMNE — DrBERT NER Evaluation")
    logger.info("=" * 60)

    results = evaluate_checkpoint(
        checkpoint_dir=args.checkpoint,
        corpus_dir=args.corpus_dir,
        device=args.device,
        batch_size=args.batch_size,
        results_dir=args.results_dir,
    )

    logger.info("=" * 60)
    logger.info("F1 macro: %.4f", results.get("eval_f1_macro", 0.0))
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
