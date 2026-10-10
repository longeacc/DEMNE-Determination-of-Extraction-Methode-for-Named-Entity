<div align="center">
   
# DEMNE

**Adaptive selection of the named-entity extraction method for frugal clinical NLP in oncology**

[![CI](https://github.com/longeacc/DEMNE-Determination-of-Extraction-Methode-for-Named-Entity/actions/workflows/ci.yml/badge.svg)](https://github.com/longeacc/DEMNE-Determination-of-Extraction-Methode-for-Named-Entity/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://python.org)

[Paper (in preparation)](#citation) · [Quick start](#quick-start) · [How it works](#how-it-works) · [Results](#results)

</div>

Not every entity in a French clinical report needs a transformer or an LLM. DEMNE measures each entity type in an annotated corpus with five metrics, then routes it to the **least costly extraction tier predicted to be good enough**: deterministic **rules**, a **transformer-based model (TBM)**, or an **LLM** as a last resort. It addresses the performance, explainability and frugality trilemma by matching method complexity to measured extraction difficulty.

|  **99.5 %**  |  **0.941 vs 0.358**  |  **0.866**  |  **59 · 92**  |
|:---:|:---:|:---:|:---:|
| of mentions routed to rules on an unseen AP-HP cohort (411/413) | mean F1, rules vs fine-tuned DrBERT, same entities | routing concordance on an unseen corpus (leave-one-corpus-out) | oncology entity types · entity-corpus pairs |

## How it works

<p align="center">
  <img src="https://github.com/user-attachments/assets/ce820f8d-2f83-4e7c-8798-d144e0e4660f" width="60%" alt="DEMNE routing decision graph">
</p>

1. **Characterize.** Five metrics are computed per entity type from BRAT annotations: templateability, lexical homogeneity, TF-IDF extractability, contextual risk (negation, uncertainty, contradiction), and a derived NER feasibility score.
2. **Route.** A deterministic decision graph with four nodes and calibrated thresholds assigns a tier. Rules are chosen only on positive evidence and a safe context; otherwise the entity escalates to a transformer when enough labelled examples exist, and to an LLM as a last resort.
3. **Calibrate.** The 17 free parameters are optimized with Optuna (NSGA-II) on routing concordance and an asymmetric cost that penalizes under-escalation twice as much as over-escalation.

Every decision traces back to explicit metric values and thresholds, which makes the routing auditable (EU AI Act, Art. 12 and 14).

## Results

| Setting | Concordance | Notes |
|---|---|---|
| In-domain, 59 oncology entities (Protocol B) | 0.954 | cost-score 0.947, pooled accuracy 0.915 (54/59), no over-escalation |
| Leave-one-corpus-out | 0.866 | expected performance on an unseen corpus of the same domain |
| Best fixed-tier baseline (always RULES) | 0.729 | always-LLM reaches 0.148 |
| Across domains (oncology, Quaero, MACCROBAT) | 0.45 to 0.73 | recalibrate for a new domain or language |

**External validation** on 300 AP-HP CT-scan reports (colorectal and head-and-neck cancers), frozen configuration, no retuning: 6 of 7 tumour-response entities routed to rules (F1 0.892 to 0.972, mean 0.941), 1 to the LLM tier. A fine-tuned DrBERT trained on five of these entities reaches a mean F1 of 0.358.

<p align="center">
   <img width="732" height="202" alt="RESULTS_CA_300" src="https://github.com/user-attachments/assets/f9f09cef-d591-45b5-b479-309d9f8682dd" />
</p>


**Limitations.** The reference standard is small (92 entity types, 59 in oncology), the LLM tier was not evaluated end to end on the external cohort, and the contextual-risk gate did not bind on the evaluated corpora.

## Quick start

```bash
git clone https://github.com/longeacc/DEMNE.git
cd DEMNE
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev,ner]"

# full pipeline report
python scripts/run_full_pipeline_report.py
```

The same pipeline is available as a notebook: `Reports/DuraXELL_Pipeline.ipynb`.

## Repository layout

| Path | Content |
|---|---|
| `src/demne/` | metrics, decision graph, calibration |
| `scripts/` | command-line entry points |
| `dashboard/` | Streamlit dashboard |
| `tests/` | unit tests (run by CI) |
| `docs/` | documentation |

## Data and ethics

The AP-HP data used for external validation are pseudonymized, were processed under the applicable institutional framework (AP-HP Health Data Warehouse, CSE-EDS approval CSE-220024) and are **not distributed** in this repository.

## Citation

```bibtex
@misc{demne2026,
  title  = {DEMNE: Adaptive Selection of Named Entity Extraction Method for Frugal Clinical NLP in Oncology},
  author = {Longeac, Clément et .al},
  year   = {2026},
  note   = {Manuscript in preparation},
  url    = {https://github.com/longeacc/DEMNE-Determination-of-Extraction-Methode-for-Named-Entity}
}
```

See also `CITATION.cff`. DEMNE builds on the REST tool (Rapid Entity Selection Tool) and on the DuraXELL project at the AP-HP Clinical Data Warehouse.

## Contributing and license

Issues and pull requests are welcome, see [CONTRIBUTING.md](CONTRIBUTING.md). Released under the [MIT license](LICENSE).
