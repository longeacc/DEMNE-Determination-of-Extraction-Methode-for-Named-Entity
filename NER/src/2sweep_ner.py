# src/sweep_ner.py
# pip install "transformers>=4.41" datasets seqeval accelerate
import argparse
import csv
import json
import os

import numpy as np
from datasets import Dataset, DatasetDict
from seqeval.metrics import f1_score, precision_score, recall_score
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    DataCollatorForTokenClassification,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
    set_seed,
)


def read_conll(path: str) -> tuple[list[list[str]], list[list[str]]]:
    toks, tags, cur_t, cur_y = [], [], [], []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                if cur_t:
                    toks.append(cur_t)
                    tags.append(cur_y)
                    cur_t, cur_y = [], []
                continue
            parts = line.split()
            tok, tag = parts[0], parts[-1]
            cur_t.append(tok)
            cur_y.append(tag)
    if cur_t:
        toks.append(cur_t)
        tags.append(cur_y)
    return toks, tags


def build_dataset(data_dir: str) -> tuple[DatasetDict, list[str]]:
    train_t, train_y = read_conll(os.path.join(data_dir, "train.conll"))
    dev_t, dev_y = read_conll(os.path.join(data_dir, "dev.conll"))
    test_t, test_y = read_conll(os.path.join(data_dir, "test.conll"))
    labels = sorted(set(tag for seqs in [train_y, dev_y, test_y] for s in seqs for tag in s))
    ds = DatasetDict(
        {
            "train": Dataset.from_dict({"tokens": train_t, "ner_tags_str": train_y}),
            "dev": Dataset.from_dict({"tokens": dev_t, "ner_tags_str": dev_y}),
            "test": Dataset.from_dict({"tokens": test_t, "ner_tags_str": test_y}),
        }
    )
    return ds, labels


# -------------------------
# Tokenization + alignment
# -------------------------
def encode_with_labels(ds: DatasetDict, model_id: str, label2id: dict[str, int]):
    tok = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    id2label = {i: lbl for lbl, i in label2id.items()}

    def _encode(batch):
        enc = tok(batch["tokens"], is_split_into_words=True, truncation=True)
        all_labs = []
        for i, tags in enumerate(batch["ner_tags_str"]):
            word_ids = enc.word_ids(batch_index=i)
            prev = None
            labs = []
            for wid in word_ids:
                if wid is None:
                    labs.append(-100)
                elif wid != prev:
                    labs.append(label2id[tags[wid]])
                else:
                    labs.append(-100)
                prev = wid
            all_labs.append(labs)
        enc["labels"] = all_labs
        return enc

    enc = ds.map(_encode, batched=True, remove_columns=["tokens", "ner_tags_str"])
    return tok, enc, id2label


# -------------------------
# Freeze helper
# -------------------------
def freeze_bottom_layers(model, n_layers: int = 0):
    if n_layers <= 0:
        return
    base = getattr(model, model.base_model_prefix)  # e.g., "roberta", "bert", "electra"
    if hasattr(base, "embeddings"):
        for p in base.embeddings.parameters():
            p.requires_grad = False
    if hasattr(base, "encoder") and hasattr(base.encoder, "layer"):
        for layer in list(base.encoder.layer)[:n_layers]:
            for p in layer.parameters():
                p.requires_grad = False


# -------------------------
# Metrics
# -------------------------
def _norm_eval(metrics: dict) -> dict:
    """Normalize Trainer.evaluate() metrics to plain keys."""
    return {
        "f1": float(metrics.get("f1", metrics.get("eval_f1", 0.0))),
        "precision": float(metrics.get("precision", metrics.get("eval_precision", 0.0))),
        "recall": float(metrics.get("recall", metrics.get("eval_recall", 0.0))),
        # keep others if you like:
        "loss": float(metrics.get("loss", metrics.get("eval_loss", 0.0))),
    }


def seqeval_metrics(eval_pred, id2label):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=2)
    y_true, y_pred = [], []
    for pred, lab in zip(preds, labels, strict=False):
        t_seq, p_seq = [], []
        for p_i, l_i in zip(pred, lab, strict=False):
            if l_i == -100:
                continue
            t_seq.append(id2label[l_i])
            p_seq.append(id2label[p_i])
        y_true.append(t_seq)
        y_pred.append(p_seq)

    # zero_division=0 prevents warnings when a class is never predicted
    try:
        p = precision_score(y_true, y_pred, zero_division=0)
        r = recall_score(y_true, y_pred, zero_division=0)
        f = f1_score(y_true, y_pred, zero_division=0)
    except TypeError:
        # older seqeval without zero_division param
        p = precision_score(y_true, y_pred)
        r = recall_score(y_true, y_pred)
        f = f1_score(y_true, y_pred)

    return {"precision": p, "recall": r, "f1": f}


# -------------------------
# One training run
# -------------------------
def run_once(
    model_id: str,
    enc: DatasetDict,
    tok,
    id2label,
    args_dict,
    freeze_layers: int,
    seed: int,
):
    set_seed(seed)
    model = AutoModelForTokenClassification.from_pretrained(
        model_id,
        num_labels=len(id2label),
        id2label=id2label,
        label2id={v: k for k, v in id2label.items()},
        ignore_mismatched_sizes=True,
    )
    freeze_bottom_layers(model, freeze_layers)
    data_collator = DataCollatorForTokenClassification(tok)

    training_args = TrainingArguments(
        output_dir=args_dict["output_dir"],
        learning_rate=args_dict["learning_rate"],
        per_device_train_batch_size=args_dict["per_device_train_batch_size"],
        per_device_eval_batch_size=args_dict["per_device_eval_batch_size"],
        num_train_epochs=args_dict["num_train_epochs"],
        weight_decay=args_dict["weight_decay"],
        warmup_ratio=args_dict["warmup_ratio"],
        gradient_accumulation_steps=args_dict["gradient_accumulation_steps"],
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        logging_steps=10,
        report_to="none",
        seed=seed,
    )

    def compute_metrics(p):
        return seqeval_metrics(p, id2label)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=enc["train"],
        eval_dataset=enc["dev"],
        processing_class=tok,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )
    trainer.train()
    dev_metrics_raw = trainer.evaluate(enc["dev"])
    test_metrics_raw = trainer.evaluate(enc["test"])
    dev_metrics = _norm_eval(dev_metrics_raw)
    test_metrics = _norm_eval(test_metrics_raw)
    return dev_metrics, test_metrics


# -------------------------
# Sweep
# -------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="NER/data/conll")
    ap.add_argument("--output_dir_base", default="NER/models/sweeps")
    ap.add_argument("--results_csv", default="NER/sweep_results.csv")
    ap.add_argument("--seeds", nargs="*", type=int, default=[42])
    # Default model set: French biomed + a few oncology NER (English)
    ap.add_argument(
        "--models",
        nargs="*",
        default=[
            # French biomedical backbones
            "Dr-BERT/DrBERT-7GB",
            "almanach/camembert-bio-base",
            "quinten-datalab/AliBERT-7GB",
            # Oncology-specific NER (mostly English)
            "OpenMed/OpenMed-NER-OncologyDetect-PubMed-109M",
            "OpenMed/OpenMed-NER-OncologyDetect-BioMed-335M",
            "OpenMed/OpenMed-NER-OncologyDetect-SuperClinical-184M",
        ],
    )
    args = ap.parse_args()

    ds, label_list = build_dataset(args.data_dir)
    label2id = {lbl: i for i, lbl in enumerate(label_list)}

    # Small grid (safe for ~50 notes) - ADJUSTED FOR GTX 1650 (4GB VRAM)
    grid = {
        "learning_rate": [2e-5],
        "per_device_train_batch_size": [4],  # Reduced from 16 to 4 to prevent OOM
        "num_train_epochs": [3],
        "weight_decay": [0.01],
        "warmup_ratio": [0.1],
        "freeze_layers": [0, 2],
        "gradient_accumulation_steps": [4],  # Increased to 4 to keep effective batch size = 16
        "per_device_eval_batch_size": [4],  # Reduced from 16 to 4 to prevent OOM during evaluation
    }

    fieldnames = [
        "model_id",
        "seed",
        "learning_rate",
        "train_bs",
        "epochs",
        "weight_decay",
        "warmup_ratio",
        "freeze_layers",
        "dev_f1",
        "dev_p",
        "dev_r",
        "test_f1",
        "test_p",
        "test_r",
        "output_dir",
    ]
    with open(args.results_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

    best = None
    for model_id in args.models:
        tok, enc, id2label = encode_with_labels(ds, model_id, label2id)
        for seed in args.seeds:
            for lr in grid["learning_rate"]:
                for bs in grid["per_device_train_batch_size"]:
                    for ep in grid["num_train_epochs"]:
                        for wd in grid["weight_decay"]:
                            for wr in grid["warmup_ratio"]:
                                for frz in grid["freeze_layers"]:
                                    outdir = f"models/sweeps/{model_id.split('/')[-1]}_lr{lr}_bs{bs}_ep{ep}_wd{wd}_wr{wr}_frz{frz}_sd{seed}"
                                    args_dict = dict(
                                        output_dir=outdir,
                                        learning_rate=lr,
                                        per_device_train_batch_size=bs,
                                        per_device_eval_batch_size=grid[
                                            "per_device_eval_batch_size"
                                        ][0],
                                        num_train_epochs=ep,
                                        weight_decay=wd,
                                        warmup_ratio=wr,
                                        gradient_accumulation_steps=grid[
                                            "gradient_accumulation_steps"
                                        ][0],
                                    )
                                    dev_m, test_m = run_once(
                                        model_id,
                                        enc,
                                        tok,
                                        id2label,
                                        args_dict,
                                        frz,
                                        seed,
                                    )
                                    row = {
                                        "model_id": model_id,
                                        "seed": seed,
                                        "learning_rate": lr,
                                        "train_bs": bs,
                                        "epochs": ep,
                                        "weight_decay": wd,
                                        "warmup_ratio": wr,
                                        "freeze_layers": frz,
                                        "dev_f1": dev_m["f1"],
                                        "dev_p": dev_m["precision"],
                                        "dev_r": dev_m["recall"],
                                        "test_f1": test_m["f1"],
                                        "test_p": test_m["precision"],
                                        "test_r": test_m["recall"],
                                        "output_dir": outdir,
                                    }
                                    with open(
                                        args.results_csv,
                                        "a",
                                        newline="",
                                        encoding="utf-8",
                                    ) as f:
                                        csv.DictWriter(f, fieldnames=fieldnames).writerow(row)
                                    if (best is None) or (row["dev_f1"] > best["dev_f1"]):
                                        best = row
                                    print(
                                        f"[RUN] {model_id} sd{seed} lr={lr} bs={bs} ep={ep} wd={wd} wr={wr} frz={frz} "
                                        f"=> DEV F1={dev_m['f1']:.3f} | TEST F1={test_m['f1']:.3f}"
                                    )

    with open("best_run.json", "w", encoding="utf-8") as f:
        json.dump(best, f, indent=2, ensure_ascii=False)
    print("\n[BEST]\n", json.dumps(best, indent=2))


if __name__ == "__main__":
    main()
try:
    t = globals().get("tracker")
    if t:
        t.stop()
except Exception as e:
    print(
        f"\nWarning: Generalized error in Eco2AI tracking (likely 'N/A' vs float dtype issue): {e}"
    )
    print("Carbon emission tracking data could not be saved, but analysis results are preserved.")
