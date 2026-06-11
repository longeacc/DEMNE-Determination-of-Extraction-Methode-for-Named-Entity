# DEMNE — DrBERT NER Fine-Tuning

Fine-tuning [DrBERT-7GB](https://huggingface.co/Dr-BERT/DrBERT-7GB) for clinical Named Entity Recognition (NER) on French oncology texts, as part of the [DEMNE pipeline](https://github.com/longeacc/DEMNE-Determination-of-Extraction-Methode-for-Named-Entity).

## Overview

This module implements the **TBM (Transformer-Based Model)** layer of the DEMNE cascading pipeline. Entities routed to TBM by the DEMNE decision graph are extracted using a DrBERT model fine-tuned on 7 breast-cancer biomarkers (ER, PR, HER2\_status, HER2\_IHC, Ki67, HER2\_FISH, Genetic\_mutation).

Training includes:
- **eco2ai** carbon footprint tracking (kWh / kgCO₂eq)
- Stratified 80/20 split preserving corpus source proportions
- Early stopping on macro-F1
- Per-entity seqeval evaluation

## Prerequisites

- Python ≥ 3.10
- CUDA 11.8+ (recommended) or CPU / Apple Silicon (MPS)
- ~4 GB VRAM for batch\_size=16 with fp16

## Installation

```bash
cd Fine-tuning-DrBERT/
pip install -r requirements.txt
```

For DrBERT model download:

```bash
pip install huggingface_hub
huggingface-cli download Dr-BERT/DrBERT-7GB
```

## Pipeline order

The scripts form a dependency chain — each consumes the previous script's
output, so the order is **not** arbitrary:

```
BRAT (.txt/.ann)
      │
      ▼  ①  brat_to_conll.py        format → CoNLL BIO          (run once PER corpus)
   .conll
      │
      ▼  ②  dataset_builder.py      fusion + label map + 80/20 split + tokenisation
 DatasetDict + label_map.json                                   (optional: QA / inspection)
      │
      ▼  ③  train.py                DrBERT fine-tuning → best checkpoint
 models/drbert_finetuned/best/
      │
      ▼  ④  evaluate.py             per-entity seqeval report
 results/metrics_*_eval.csv
```

| Order | Script | Why |
|---|---|---|
| 1 | `brat_to_conll.py` | BRAT standoff (`.txt`/`.ann` char offsets) → CoNLL BIO, the format HuggingFace token-classification needs. Run **once per corpus**. |
| 2 | `dataset_builder.py` | Fuses corpora, builds the `label2id`/`id2label` map (sizes the classifier head), does the stratified 80/20 split *before* tokenisation (no leakage), tokenises + aligns labels. **Optional** as a standalone step — `train.py` calls `build_dataset()` internally; run it on its own only to inspect/validate the dataset. |
| 3 | `train.py` | Loads DrBERT-7GB, fine-tunes with early stopping on macro-F1, saves the best checkpoint + `label_map.json`. |
| 4 | `evaluate.py` | Reloads the saved checkpoint and produces the per-entity F1/P/R report on the test split. Needs the checkpoint from step 3. |

> **Minimal path** (skip the standalone QA step 2): `brat_to_conll.py` (×N corpora) → `train.py` → `evaluate.py`.

## Corpus input

Two equivalent ways to point at your BRAT corpora (paired `.txt` + `.ann`
files in standoff format):

**A — explicit paths (recommended)** — pass each corpus as `NAME=DIR`,
repeatable. `NAME` becomes the corpus key used for the stratified split and
the CoNLL cache folder:

```bash
--corpus_path cantemist=/path/to/Emmanuelle_35_cantemist \
--corpus_path rcp_esmo=/path/to/evaluation_set_breast_cancer_GS
```

**B — legacy single root** — a directory holding the canonical sub-folders:

```
corpus_dir/
├── cantemist/          # Cantemist-35 (BRAT .txt + .ann)
├── redjdal/            # Pr. Redjdal thesis corpus
└── rcp_esmo/           # RCP/ESMO 95-patient corpus
```

```bash
--corpus_dir /path/to/corpus_dir
```

## Step ① — BRAT → CoNLL

Run once per corpus (output goes under `data/conll/<name>/`, reused as cache):

```bash
python brat_to_conll.py \
    --input_dir /path/to/Emmanuelle_35_cantemist \
    --output_dir data/conll/Emmanuelle_35_cantemist \
    --corpus_name cantemist
```

## Step ② — Build dataset (optional QA)

```bash
python dataset_builder.py \
    --corpus_path cantemist=/path/to/Emmanuelle_35_cantemist \
    --corpus_path rcp_esmo=/path/to/evaluation_set_breast_cancer_GS \
    --output_dir data/demne_hf_dataset
```

## Step ③ — Training

```bash
python train.py \
    --corpus_path cantemist=/path/to/Emmanuelle_35_cantemist \
    --corpus_path rcp_esmo=/path/to/evaluation_set_breast_cancer_GS \
    --config drbert_ner_config.yaml \
    --device auto
```

### Key arguments

| Argument | Default | Description |
|---|---|---|
| `--corpus_path` | *(repeatable)* | Corpus as `NAME=DIR` (BRAT dir). Use this **or** `--corpus_dir`. |
| `--corpus_dir` | `None` | Legacy root with `cantemist/`, `redjdal/`, `rcp_esmo/`. |
| `--config` | `drbert_ner_config.yaml` | YAML config (next to the scripts) |
| `--device` | `auto` | `cuda`, `cpu`, `mps`, or `auto` |
| `--output_dir` | `models/drbert_finetuned` | Checkpoint output |
| `--results_dir` | `results` | Metrics CSV + eco2ai output |

> **Carbon tracking**: set `DISABLE_ECO2AI=1` to skip eco2ai (e.g. if it is
> incompatible with the installed `pandas` and floods the log). Tracking
> failures are non-fatal and never abort training or model saving.

### Outputs

```
models/drbert_finetuned/
├── best/                   # Best checkpoint (by F1 macro)
│   ├── config.json
│   ├── model.safetensors
│   ├── tokenizer_config.json
│   └── label_map.json      # label2id / id2label for DEMNE integration
└── checkpoint-*/            # Epoch checkpoints

results/
├── metrics_YYYYMMDD_HHMMSS_final.csv
└── emissions.csv            # eco2ai carbon report
```

## Evaluation

Standalone evaluation on a saved checkpoint:

```bash
python -m fine_tuning.evaluate \
    --checkpoint models/drbert_finetuned/best \
    --corpus_dir /path/to/corpus_dir \
    --device auto
```

## DEMNE Integration

The fine-tuned checkpoint is loaded by the TBM module of the DEMNE decision graph. The `label_map.json` file in the checkpoint directory provides the label↔id mapping needed for inference.

```python
from transformers import AutoModelForTokenClassification, AutoTokenizer
import json

ckpt = "models/drbert_finetuned/best"
model = AutoModelForTokenClassification.from_pretrained(ckpt)
tokenizer = AutoTokenizer.from_pretrained(ckpt)

with open(f"{ckpt}/label_map.json") as f:
    label_map = json.load(f)
```

## Configuration

All hyperparameters are externalised in `config/drbert_ner_config.yaml`. Edit this file to adjust learning rate, batch size, epochs, etc. without touching code.

## References

- Longeac, Redjdal & Kempf — *DuraXell: Le juste usage des LLM* (ESMO 2025)
- Dahl et al. (2025) — Systematic review of NLP methods for cancer IE
- Strubell et al. (2019) — Energy and policy considerations for NLP
- EU AI Act Art. 14 — Transparency requirements
- AFNOR SPEC 2314 — Frugal AI specification
