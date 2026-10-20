# Recalibrating DEMNE on your own corpora

`optimize_params` re-tunes the 17 free parameters of DEMNE (decision thresholds, metric
weights, negation scope, TF-IDF top-X and threshold) against routing labels you provide,
and writes the result to `demne_params.json`. This is the procedure that produced the
published parameters.

```bash
pip install -e ".[optimize]"          # or: pip install "demne[optimize]"
python main.py optimize_params --in_domain
python main.py optimize_params --all_corpora
python -m demne.optimize_params --in_domain   # same, without main.py
```

## Folder convention

Corpora are read from `data/` under the **current working directory**, not from the
installed package. Each category holds one sub-folder per corpus. Corpus names are free;
nothing is hard-coded.

```
data/
├── OPT_CORPORA/           required   target domain, used for optimization
│   ├── my_corpus_A/
│   └── my_corpus_B/
├── TEST_CORPORA/          required   target domain, held out (final report)
│   └── my_corpus_C/
├── OFF_DOMAIN_CORPORA/    optional   other domains or languages, --all_corpora only
│   └── other_domain/
└── ALL_CORPORA/           optional   final-fit corpora; if absent or empty,
                                      OPT_CORPORA ∪ TEST_CORPORA is used
```

A corpus name must be unique across the categories that end up in the same fit. Only
`ALL_CORPORA` may reuse names from `OPT_CORPORA` and `TEST_CORPORA`.

## What each corpus folder must contain

```
my_corpus_A/
├── doc001.txt                 BRAT text
├── doc001.ann                 BRAT annotations (T-lines); sub-folders are scanned too
├── ...
└── my_corpus_A_routing.csv    gold routing labels
```

- **BRAT pairs**: `*.ann` + `*.txt` with the same base name. Only entity lines
  (`T1<TAB>Type start end<TAB>text`) are used.
- **Gold file**: `*_routing.csv` (preferred), otherwise the first `*.csv` of the folder.
  It needs one row per entity type and two columns:

  | column | accepted headers | values |
  |---|---|---|
  | entity type | `entity` or `Entity` | the BRAT type, case-insensitive (`Ki67` = `ki67`) |
  | expected tier | `Label` or `label` | `RULES` (or `RÈGLES`/`REGLES`), `TBM`, `LLM` |

  ```csv
  entity,Label
  Stade_OMS_ECOG_Karnofsky,RULES
  Traitement_specifique_du_cancer,TBM
  Histologie_tumorale,LLM
  ```

Gold entities that never occur in the `.ann` files are ignored, and so are rows with any
other label. You are responsible for the licences of the corpora you use.

## What a run does

| Step | `--in_domain` | `--all_corpora` |
|---|---|---|
| 1. Optimize, then report before/after on held-out data | OPT → TEST | OPT → TEST |
| 2. Leave-one-corpus-out CV (expected score on an unseen corpus) | ALL | ALL + OFF_DOMAIN |
| 3. Final fit, written to `demne_params.json` | ALL | ALL + OFF_DOMAIN |

Optuna's NSGA-II sampler maximizes two objectives. **Concordance** is the share of entity
types routed to their gold tier. **Cost-score** is an ordinal score where routing to a tier
that is too weak costs twice as much as routing to one that is too strong. Every entity
weighs the same within its corpus, and every corpus weighs the same overall. The final
pick has the highest concordance, with cost-score as the tiebreak. Large corpora are split
into sub-samples of 40 documents. The run settings are fixed: 1000 trials for the final
fit, 300 for the held-out study and for each CV fold, and seed 42.

`--all_corpora` is meant for a new domain or language. Off-domain corpora pull the
thresholds away from the target domain. In the DEMNE study, French risk patterns never
fired on English text, so the R metric carried no signal there.

## Outputs

- **`demne_params.json`** is always rewritten after a complete run. This is the file
  that `demne.params.load_params()` reads, so `main.py`, the CLI scorers and the Streamlit
  dashboard use the new values right away. Only the tuned keys change. Every other key
  (`MIN_TE_SAMPLES`, `risk_weights.contradiction`, `risk_window_size`, `frequency`,
  `kfold`, `presets`, `TFIDF_SIM`, …) is kept as is.
  The routing presets follow automatically. They are derived from the calibrated
  thresholds, not stored: `CALIBRATED` is the thresholds themselves, and `FRUGAL` /
  `QUALITY` shift every gate by `preset_margin` (default 0.10) toward the cheaper or
  the stronger tier. The file is written atomically.
  **Two steps are left to you after a run:** re-run `python main.py evaluate --gs_dir
  <corpus>`, because the metric values and `data/decision_config.json` were computed
  with the old weights; and restart the Streamlit dashboard if it is open, because it
  reads the parameters at startup.
- **`Results/demne_params_before_<timestamp>.json`**: the previous file. Copy it back
  to undo a run.
- **`Results/optimize_params_<mode>_<timestamp>.json`**: corpora used, before/after
  scores, LOCO folds, and old and new parameters.
- **`Results/optuna_demne.db`**: the Optuna study. Each run gets its own study, so trials
  from different corpora are never mixed.

## Parameters that are not optimized

`TFIDF_SIM`, `TFIDF_WINDOW_TOKENS` and `TFIDF_USE_F1` are present in
`demne_params.json` but are not read by the routing code. They belong to the earlier
clustering-based TF-IDF metric, which the coverage-F1 metric (`demne.tfidf_coverage`)
replaced. Varying them would not change the objective, so they stay fixed.

## Published parameters

`Results/demne_params_published.json` is a copy of the parameters behind the results
reported in the README (in-domain concordance 0.954, leave-one-corpus-out 0.866). To check
those figures after recalibrating, copy it back over `src/demne/data/demne_params.json`.

## Consistency check

In a source checkout, each run first checks that the optimizer's fast feature path gives
the same Te, He, R, Freq and Feas as the dashboard's `MetricsCalculator`, parser included,
on the first OPT corpus (`drift-guard OK`). It also checks that the CLI scorers use the
same formulas and lexicons (`cli-scorer-guard OK`). After a `pip install`, `dashboard/` is
absent, so this check is skipped.
