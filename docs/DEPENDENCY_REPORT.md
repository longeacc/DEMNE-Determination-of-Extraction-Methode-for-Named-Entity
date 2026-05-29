# DuraXELL â€” Import Dependency Report

## ANALYSIS SUMMARY

---

### 1. MAIN ENTRY POINTS (files executed directly)

| Entry Point | Type | Description |
|---|---|---|
| `main.py` | CLI hub | Central CLI dispatcher (`extract`, `metrics`, `tree`, `rest`, `evaluate`) |
| `run_full_pipeline_report.py` | Pipeline | Full evaluation pipeline with figures |
| `src/duraxell/REST_interface/demo_rest.py` | Demo | REST interface demonstration |
| `src/duraxell/REST_interface/rest_pipeline.py` | Pipeline | REST annotation pipeline driver |
| `src/duraxell/E_creation_arbre_decision.py` | Script | Decision tree builder (also importable) |
| `src/duraxell/E_templatability.py` | Script | templatability scorer (also importable) |
| `src/duraxell/E_homogeneity.py` | Script | Homogeneity scorer (also importable) |
| `src/duraxell/E_risk_context.py` | Script | Risk context scorer (also importable) |
| `src/duraxell/E_frequency.py` | Script | Frequency scorer (also importable) |
| `src/duraxell/E_annotation_yield.py` | Script | Annotation yield scorer (also importable) |
| `src/duraxell/E_feasibility_NER.py` | Script | NER feasibility calculator |
| `src/duraxell/generate_*_report.py` (Ã—3) | Script | HTML report generators |
| `src/duraxell/Rules/src/Breast/lunch.py` | Script | Rule-based annotation runner |
| `NER/src/2bis_train_hf_ner.py` | Script | NER training |
| `NER/src/2sweep_ner.py` | Script | NER hyperparameter sweep |
| `NER/src/3infer.py` | Script | NER inference |
| `NER/src/4predict_to_brat.py` | Script | NER prediction to BRAT format |
| `NER/src/5evaluate_ner.py` | Script | NER evaluation |
| `NER/src/eval_best_model.py` | Script | Best model evaluation |

---

### 2. DEPENDENCY CLASSIFICATION

#### LEAF MODULES (imported by others, import no project-internal files)

These are the foundational modules with zero internal dependencies:

| Module | Imported By |
|---|---|
| `src/duraxell/structs.py` | `cascade_orchestrator`, `rules_cascade_connector`, `ner_cascade_connector`, tests |
| `src/duraxell/energy_tracker.py` | `__init__`, `run_full_pipeline_report`, tests |
| `src/duraxell/E_composite_scorer.py` | `__init__`, `run_full_pipeline_report`, tests |
| `src/duraxell/E_annotation_yield.py` | `__init__`, `E_creation_arbre_decision`, tests |
| `src/duraxell/E_frequency.py` | `__init__`, `debug_freq`, tests |
| `src/duraxell/E_homogeneity.py` | `__init__`, tests |
| `src/duraxell/E_risk_context.py` | `__init__`, tests |
| `src/duraxell/E_templatability.py` | `__init__`, tests |
| `src/duraxell/visualize_decision_tree.py` | `run_full_pipeline_report` |
| `src/duraxell/REST_interface/rest_annotator.py` | `__init__`, `rest_pipeline`, `demo_rest`, `run_full_pipeline_report` |
| `src/duraxell/REST_interface/rest_evaluator.py` | `__init__`, `rest_pipeline`, `demo_rest`, `run_full_pipeline_report` |
| `src/duraxell/REST_interface/rest_decision_bridge.py` | `__init__`, `rest_pipeline`, `demo_rest`, `run_full_pipeline_report` |
| `src/duraxell/REST_interface/convergence_analyzer.py` | `__init__`, `demo_rest` |
| `src/duraxell/REST_interface/yield_calculator.py` | `rest_pipeline` |
| `src/duraxell/Rules/src/Breast/biomarker_brat_annotator.py` | `rules_cascade_connector`, `lunch.py` |
| All `REST_modules/calculs/*.py` | via `calculs/__init__` |
| All `REST_modules/extraction/*.py` | via `extraction/__init__` |
| `REST_modules/visualization.py` | `__init__`, `ui.py` |

#### ROOT MODULES (import other project files, never imported by anyone)

These are the top-level entry points or standalone scripts:

| Module | Depends On |
|---|---|
| `main.py` | `cascade_orchestrator` (+ subprocess calls to many scripts) |
| `run_full_pipeline_report.py` | `E_composite_scorer`, `cascade_orchestrator`, `energy_tracker`, `visualize_decision_tree`, `sensitivity_analysis`, REST interface modules |
| `src/duraxell/REST_interface/demo_rest.py` | REST interface modules |
| `src/duraxell/REST_interface/rest_pipeline.py` | REST interface modules + `yield_calculator` |
| `src/duraxell/Rules/src/Breast/lunch.py` | `biomarker_brat_annotator` |
| `debug_freq.py` | `E_frequency` |
| All `src/duraxell/tests/*.py` | Various duraxell modules |
| All `NER/src/*.py` (except `ner_cascade_connector`) | External only |

#### HUB MODULES (both imported and import others)

| Module | Imports | Imported By |
|---|---|---|
| `src/duraxell/__init__.py` | 9 internal modules | All external `duraxell.*` imports |
| `src/duraxell/cascade_orchestrator.py` | `structs`, `rules_cascade_connector`, `ner_cascade_connector` | `__init__`, `main`, `run_full_pipeline_report`, tests |
| `src/duraxell/E_creation_arbre_decision.py` | `E_annotation_yield` | `__init__`, `sensitivity_analysis`, tests |
| `src/duraxell/sensitivity_analysis.py` | `E_creation_arbre_decision` | `run_full_pipeline_report` |
| `src/duraxell/Rules/src/Breast/rules_cascade_connector.py` | `biomarker_brat_annotator`, `structs` | `cascade_orchestrator` |
| `NER/src/ner_cascade_connector.py` | `structs` | `cascade_orchestrator` |
| `REST_modules/ui.py` | all other REST_modules | `__init__` |
| `REST_modules/loading.py` | `calculs`, `categorization`, `extraction` | `__init__`, `ui` |
| `REST_modules/categorization.py` | `calculs`, `extraction.normalisation` | `__init__`, `loading`, `ui` |
| `REST_modules/calculs/metrics.py` | `.regex` | `calculs/__init__` |
| `REST_modules/calculs/ngram.py` | `.tfidf` | `calculs/__init__` |

---

### 3. CIRCULAR DEPENDENCY ANALYSIS

**No true circular import dependencies found.**

The closest potential issue:
- `src/duraxell/__init__.py` imports `cascade_orchestrator`, which imports `rules_cascade_connector`, which imports `duraxell.structs`. But `structs` is loaded first by `__init__.py` order, so there's no cycle.
- `src/duraxell/__init__.py` imports both `E_creation_arbre_decision` and `E_annotation_yield`. `E_creation_arbre_decision` also imports `E_annotation_yield`, but this is a DAG, not a cycle.
- `REST_modules/__init__.py` â†’ `categorization` â†’ `.calculs` â†’ (submodules). No back-references.

**All dependency graphs are DAGs (Directed Acyclic Graphs). No circular imports exist.**

---

### 4. CORE DEPENDENCY CHAIN (Main Pipeline)

```
main.py
  â””â”€â”€ cascade_orchestrator.py
        â”œâ”€â”€ structs.py                    [LEAF]
        â”œâ”€â”€ rules_cascade_connector.py
        â”‚     â”œâ”€â”€ biomarker_brat_annotator.py  [LEAF]
        â”‚     â””â”€â”€ structs.py                    [LEAF]
        â””â”€â”€ ner_cascade_connector.py
              â””â”€â”€ structs.py                    [LEAF]

run_full_pipeline_report.py
  â”œâ”€â”€ E_composite_scorer.py               [LEAF]
  â”œâ”€â”€ cascade_orchestrator.py             (see above)
  â”œâ”€â”€ energy_tracker.py                   [LEAF]
  â”œâ”€â”€ visualize_decision_tree.py          [LEAF]
  â”œâ”€â”€ sensitivity_analysis.py
  â”‚     â””â”€â”€ E_creation_arbre_decision.py
  â”‚           â””â”€â”€ E_annotation_yield.py   [LEAF]
  â”œâ”€â”€ rest_annotator.py                   [LEAF]
  â”œâ”€â”€ rest_evaluator.py                   [LEAF]
  â””â”€â”€ rest_decision_bridge.py             [LEAF]
```

---

### 5. EXTERNAL DEPENDENCY SUMMARY

| Library | Used By (count) | Category |
|---|---|---|
| `pandas` | 15+ files | Data manipulation |
| `matplotlib` | 8+ files | Visualization |
| `numpy` | 7+ files | Numerics |
| `eco2ai` | 13 files | Energy tracking |
| `json` (stdlib) | 15+ files | Config/serialization |
| `re` (stdlib) | 12+ files | Regex |
| `torch` | 3 files (NER) | Deep learning |
| `transformers` | 5 files (NER) | HuggingFace models |
| `seqeval` | 3 files (NER) | NER evaluation |
| `datasets` | 3 files (NER) | HuggingFace datasets |
| `networkx` | 1 file | Graph visualization |
| `seaborn` | 2 files | Statistical plots |
| `ipywidgets` | 4 files (REST_modules) | Jupyter UI |
| `bqplot` | 3 files (REST_modules) | Jupyter plots |
| `ipydatagrid` | 4 files (REST_modules) | Jupyter tables |
| `plotly` | 2 files (REST_modules) | Interactive plots |
| `spacy` | 1 file (normalisation) | NLP |
| `Levenshtein` | 1 file (normalisation) | String distance |
| `inflect` | 1 file (regex) | Pluralization |
| `nltk` | 3 files (REST_modules) | NLP toolkit |
| `nbformat` | 1 file | Notebook creation |
| `psutil` | 1 file | Process monitoring |
| `unidecode` | 3 files (REST_modules) | Unicode normalization |

---

### 6. WARNINGS & OBSERVATIONS

1. **Hard eco2ai imports** in `generate_*_report.py` (Ã—3) and `biomarker_brat_annotator.py` and `lunch.py` â€” these will crash if `eco2ai` is not installed (no try/except guard).

2. **Inconsistent import styles** in tests: `test_homogeneity.py`, `test_risk_context.py`, and `test_templatability.py` use `sys.path.append` + bare module imports (`from E_homogeneity import ...`) instead of package-qualified imports (`from duraxell.E_homogeneity import ...`).

3. **REST_interface/__init__.py** uses bare module imports (`from convergence_analyzer import ...`) instead of relative imports (`from .convergence_analyzer import ...`), which will fail when imported as a package from outside the directory.

4. **graph_orchestrator.py** is an empty file â€” appears to be a placeholder.

5. **E_feasibility_NER.py** is a standalone script not imported by any other file.

6. **NER/src/** pipeline scripts (2bis, 2sweep, 3infer, 4predict, 5evaluate, eval_best) are fully standalone â€” they have no internal project imports and are never imported by others. Only `ner_cascade_connector.py` bridges into the main DuraXELL system.
