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
cd fine_tuning/
pip install -r requirements.txt
```

For DrBERT model download:

```bash
pip install huggingface_hub
huggingface-cli download Dr-BERT/DrBERT-7GB
```

## Corpus Layout

Organise your BRAT corpora under a single root directory:

```
corpus_dir/
├── cantemist/          # Cantemist-35 (BRAT .txt + .ann)
├── redjdal/            # Pr. Redjdal thesis corpus
└── rcp_esmo/           # RCP/ESMO 95-patient corpus
```

Each sub-directory contains paired `.txt` and `.ann` files in BRAT standoff format.

## Training

```bash
python -m fine_tuning.train \
    --corpus_dir /path/to/corpus_dir \
    --config fine_tuning/config/drbert_ner_config.yaml \
    --device auto
```

### Key arguments

| Argument | Default | Description |
|---|---|---|
| `--corpus_dir` | *(required)* | Root BRAT corpus directory |
| `--config` | `fine_tuning/config/drbert_ner_config.yaml` | YAML config |
| `--device` | `auto` | `cuda`, `cpu`, `mps`, or `auto` |
| `--output_dir` | `models/drbert_finetuned` | Checkpoint output |
| `--results_dir` | `results` | Metrics CSV + eco2ai output |

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
