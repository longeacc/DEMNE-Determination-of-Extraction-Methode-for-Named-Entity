"""DEMNE — DrBERT NER fine-tuning script.

Fine-tunes DrBERT-7GB for token-classification (NER) on fused DEMNE
oncology corpora (Cantemist + Redjdal + RCP/ESMO) with:

- eco2ai carbon footprint tracking (kWh, kgCO2eq)
- Stratified 80/20 split by corpus source
- Early stopping on macro-F1
- seqeval metrics per epoch
- CSV metric export

Usage::

    python train.py --corpus_dir /data/demne_corpora
    python train.py --config config/drbert_ner_config.yaml --device cuda
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import datasets
import numpy as np
import torch
import yaml
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    DataCollatorForTokenClassification,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)

# Make sibling modules importable whether run as ``python train.py`` or
# ``python -m``. The script's own directory holds dataset_builder.py.
sys.path.insert(0, str(Path(__file__).resolve().parent))

try:
    from .dataset_builder import (
        PRIORITY_ENTITIES,
        build_dataset,
        parse_corpus_paths,
    )
except ImportError:
    from dataset_builder import (
        PRIORITY_ENTITIES,
        build_dataset,
        parse_corpus_paths,
    )

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

DEFAULTS: dict[str, Any] = {
    "model_name": "Dr-BERT/DrBERT-7GB",
    "max_length": 512,
    "learning_rate": 2e-5,
    "num_train_epochs": 10,
    "per_device_train_batch_size": 16,
    "per_device_eval_batch_size": 32,
    "warmup_ratio": 0.1,
    "weight_decay": 0.01,
    "lr_scheduler_type": "linear",
    "early_stopping_patience": 3,
    "seed": 42,
    "output_dir": "models/drbert_finetuned",
    "results_dir": "results",
    "logging_steps": 50,
    "fp16": True,
}


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------

def load_config(config_path: Path | None = None) -> dict[str, Any]:
    """Load training configuration from YAML, falling back to defaults.

    Args:
        config_path: Optional path to a YAML configuration file.

    Returns:
        Merged configuration dictionary.
    """

    cfg = dict(DEFAULTS)

    if config_path is not None and config_path.exists():
        with config_path.open("r", encoding="utf-8") as fh:
            user_cfg = yaml.safe_load(fh) or {}
        cfg.update(user_cfg)
        logger.info("Loaded config from %s (%d overrides)", config_path, len(user_cfg))

    return cfg


# ---------------------------------------------------------------------------
# Metrics computation
# ---------------------------------------------------------------------------

def build_compute_metrics(id2label: dict[int, str]):
    """Return a ``compute_metrics`` callable for the HF Trainer.

    Uses :mod:`seqeval` for entity-level F1/P/R.

    Args:
        id2label: Integer-to-label mapping.

    Returns:
        Callable compatible with ``Trainer(compute_metrics=…)``.
    """

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

        report = classification_report(
            true_labels, pred_labels, output_dict=True, zero_division=0
        )

        metrics: dict[str, float] = {
            "f1_macro": float(f1_macro),
            "precision_macro": float(precision),
            "recall_macro": float(recall),
        }

        # Per-entity F1 for priority biomarkers
        for ent in PRIORITY_ENTITIES:
            key = ent
            if key in report:
                metrics[f"f1_{ent}"] = float(report[key].get("f1-score", 0.0))
                metrics[f"support_{ent}"] = float(report[key].get("support", 0))

        return metrics

    return compute_metrics


# ---------------------------------------------------------------------------
# Device selection
# ---------------------------------------------------------------------------

def resolve_device(requested: str) -> str:
    """Resolve the requested device to an available one.

    Args:
        requested: One of ``"cuda"``, ``"mps"``, ``"cpu"``, or ``"auto"``.

    Returns:
        Validated device string.
    """

    if requested == "auto":
        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    if requested == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA requested but not available — falling back to CPU")
        return "cpu"

    if requested == "mps":
        if not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
            logger.warning("MPS requested but not available — falling back to CPU")
            return "cpu"

    return requested


# ---------------------------------------------------------------------------
# CSV metric export
# ---------------------------------------------------------------------------

def export_metrics_csv(
    metrics: dict[str, float],
    results_dir: Path,
    tag: str = "",
) -> Path:
    """Write evaluation metrics to a timestamped CSV.

    Args:
        metrics: Flat dict of metric name → value.
        results_dir: Directory for CSV output.
        tag: Optional tag appended to filename.

    Returns:
        Path to the written CSV file.
    """

    results_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    suffix = f"_{tag}" if tag else ""
    csv_path = results_dir / f"metrics_{ts}{suffix}.csv"

    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["metric", "value"])
        for k, v in sorted(metrics.items()):
            writer.writerow([k, f"{v:.6f}" if isinstance(v, float) else str(v)])

    logger.info("Metrics exported to %s", csv_path)
    return csv_path


# ---------------------------------------------------------------------------
# Main training routine
# ---------------------------------------------------------------------------

def train(
    corpus_dir: Path | None,
    config: dict[str, Any],
    device: str = "auto",
    corpus_paths: dict[str, Path] | None = None,
) -> Path:
    """Run the full training pipeline.

    Args:
        corpus_dir: Legacy root directory with BRAT sub-corpora (optional
            if ``corpus_paths`` is given).
        config: Training configuration dict.
        device: Target compute device.
        corpus_paths: Explicit ``{corpus_name: brat_dir}`` mapping.

    Returns:
        Path to the best checkpoint directory.
    """

    # ---- Reproducibility ----
    seed = int(config.get("seed", 42))
    torch.manual_seed(seed)
    np.random.seed(seed)

    # ---- Device ----
    device = resolve_device(device)
    use_fp16 = config.get("fp16", True) and device == "cuda"
    logger.info("Device: %s | FP16: %s", device, use_fp16)

    if device == "cuda":
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        logger.info("GPU: %s (%.1f GB)", gpu_name, gpu_mem)

    # ---- Dataset ----
    model_name = config.get("model_name", DEFAULTS["model_name"])
    max_length = int(config.get("max_length", DEFAULTS["max_length"]))
    output_dir = Path(config.get("output_dir", DEFAULTS["output_dir"]))
    results_dir = Path(config.get("results_dir", DEFAULTS["results_dir"]))
    results_dir.mkdir(parents=True, exist_ok=True)

    ds, label2id, id2label = build_dataset(
        corpus_dir=corpus_dir,
        corpus_paths=corpus_paths,
        model_name=model_name,
        max_length=max_length,
    )

    num_labels = len(label2id)
    logger.info("Dataset ready — %d labels", num_labels)

    # ---- Model ----
    logger.info("Loading model: %s", model_name)
    model = AutoModelForTokenClassification.from_pretrained(
        model_name,
        num_labels=num_labels,
        id2label=id2label,
        label2id=label2id,
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    data_collator = DataCollatorForTokenClassification(
        tokenizer=tokenizer,
        padding=True,
        max_length=max_length,
    )

    # ---- Training arguments ----
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        overwrite_output_dir=True,
        num_train_epochs=int(config.get("num_train_epochs", DEFAULTS["num_train_epochs"])),
        per_device_train_batch_size=int(config.get("per_device_train_batch_size", DEFAULTS["per_device_train_batch_size"])),
        per_device_eval_batch_size=int(config.get("per_device_eval_batch_size", DEFAULTS["per_device_eval_batch_size"])),
        learning_rate=float(config.get("learning_rate", DEFAULTS["learning_rate"])),
        warmup_ratio=float(config.get("warmup_ratio", DEFAULTS["warmup_ratio"])),
        weight_decay=float(config.get("weight_decay", DEFAULTS["weight_decay"])),
        lr_scheduler_type=str(config.get("lr_scheduler_type", DEFAULTS["lr_scheduler_type"])),
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        greater_is_better=True,
        save_total_limit=2,
        logging_steps=int(config.get("logging_steps", DEFAULTS["logging_steps"])),
        fp16=use_fp16,
        seed=seed,
        report_to="none",
        # Device mapping
        no_cuda=(device == "cpu"),
        use_mps_device=(device == "mps"),
    )

    # ---- Callbacks ----
    early_stop = EarlyStoppingCallback(
        early_stopping_patience=int(config.get("early_stopping_patience", DEFAULTS["early_stopping_patience"])),
    )

    # ---- Trainer ----
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=ds["train"],
        eval_dataset=ds["test"],
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=build_compute_metrics(id2label),
        callbacks=[early_stop],
    )

    # ---- eco2ai carbon tracking ----
    try:
        if os.environ.get("DISABLE_ECO2AI"):
            raise RuntimeError("eco2ai disabled via DISABLE_ECO2AI env var")
        import eco2ai

        tracker = eco2ai.Tracker(
            project_name="DEMNE-DrBERT-finetune",
            experiment_description=f"NER fine-tuning {model_name} on {num_labels} labels",
            file_name=str(results_dir / "emissions.csv"),
        )
        tracker.start()
        eco2ai_active = True
        logger.info("eco2ai tracker started")
    except ImportError:
        logger.warning(
            "eco2ai not installed — carbon tracking disabled. "
            "Install with: pip install eco2ai"
        )
        eco2ai_active = False
        tracker = None
    except Exception as exc:  # eco2ai/pandas version incompatibilities, etc.
        logger.warning("eco2ai tracker failed to start (%s) — carbon tracking disabled", exc)
        eco2ai_active = False
        tracker = None


    def _stop_tracker() -> None:
        """Stop eco2ai without ever letting its errors abort training/saving."""
        if eco2ai_active and tracker is not None:
            try:
                tracker.stop()
                logger.info("eco2ai tracker stopped — see %s", results_dir / "emissions.csv")
            except Exception as exc:  # noqa: BLE001 — carbon tracking must never be fatal
                logger.warning("eco2ai tracker.stop() failed (non-fatal): %s", exc)

    # ---- Train ----
    start_time = time.time()

    try:
        trainer.train()
    except RuntimeError as exc:
        if "out of memory" in str(exc).lower():
            logger.error(
                "CUDA OOM — reduce per_device_train_batch_size (currently %d) "
                "or max_length (currently %d). Try batch_size=8 or fp16=True.",
                training_args.per_device_train_batch_size,
                max_length,
            )
            _stop_tracker()
            raise
        raise

    elapsed = time.time() - start_time
    logger.info("Training completed in %.1f seconds (%.1f min)", elapsed, elapsed / 60)

    # ---- Stop eco2ai ----
    _stop_tracker()

    # ---- Final evaluation ----
    eval_results = trainer.evaluate()
    logger.info("Final evaluation: %s", eval_results)

    # ---- Save best model ----
    best_path = output_dir / "best"
    trainer.save_model(str(best_path))
    tokenizer.save_pretrained(str(best_path))

    # Save label map alongside model
    label_map_path = best_path / "label_map.json"
    label_map_path.write_text(
        json.dumps(
            {
                "label2id": label2id,
                "id2label": {str(k): v for k, v in id2label.items()},
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    logger.info("Best model saved to %s", best_path)

    # ---- Export metrics CSV ----
    export_metrics_csv(eval_results, results_dir, tag="final")

    return best_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(
        description="Fine-tune DrBERT for NER on DEMNE oncology corpora.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--corpus_dir",
        type=Path,
        default=None,
        help="Legacy root dir with sub-folders cantemist/, redjdal/, rcp_esmo/.",
    )
    parser.add_argument(
        "--corpus_path",
        action="append",
        default=[],
        metavar="NAME=DIR",
        help="Explicit corpus as name=path to a BRAT directory. Repeatable.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).resolve().parent / "drbert_ner_config.yaml",
        help="Path to YAML config file.",
    )
    parser.add_argument(
        "--device",
        type=str,
        choices=["cuda", "cpu", "mps", "auto"],
        default="auto",
        help="Compute device.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=None,
        help="Override checkpoint output directory.",
    )
    parser.add_argument(
        "--results_dir",
        type=Path,
        default=None,
        help="Override results/metrics directory.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Override num_train_epochs from the config.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=None,
        help="Override per_device_train_batch_size from the config.",
    )
    return parser.parse_args()


def main() -> None:
    """CLI entry point."""

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    args = parse_args()
    config = load_config(args.config)

    corpus_paths = parse_corpus_paths(args.corpus_path)
    if not corpus_paths and args.corpus_dir is None:
        raise SystemExit(
            "Provide at least one --corpus_path NAME=DIR or --corpus_dir."
        )

    # CLI overrides
    if args.output_dir is not None:
        config["output_dir"] = str(args.output_dir)
    if args.results_dir is not None:
        config["results_dir"] = str(args.results_dir)
    if args.epochs is not None:
        config["num_train_epochs"] = args.epochs
    if args.batch_size is not None:
        config["per_device_train_batch_size"] = args.batch_size

    logger.info("=" * 60)
    logger.info("DEMNE — DrBERT NER Fine-Tuning")
    logger.info("=" * 60)
    logger.info("Corpus dir : %s", args.corpus_dir)
    logger.info("Corpus paths: %s", corpus_paths or "(none)")
    logger.info("Config     : %s", args.config)
    logger.info("Device     : %s", args.device)
    logger.info("Output     : %s", config.get("output_dir"))

    best_path = train(
        corpus_dir=args.corpus_dir,
        corpus_paths=corpus_paths or None,
        config=config,
        device=args.device,
    )

    logger.info("=" * 60)
    logger.info("Done. Best checkpoint: %s", best_path)
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
