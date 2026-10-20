# Changelog

## [Unreleased]
### Added

- `python main.py optimize_params --in_domain | --all_corpora`: Optuna recalibration of `demne_params.json` on user corpora (`data/*_CORPORA/`, extra `[optimize]`), see `docs/OPTIMIZE_PARAMS.md`
- Presets `CALIBRATED` / `FRUGAL` / `QUALITY` derived from the calibrated thresholds (`preset_margin`) instead of stored absolute values
- Extra `[energy]` for eco2ai
- Tests for the `main.py` sub-commands, the dashboard metrics/parser and the scorers' output files; coverage 70 % → 83 %, floor 75 %
- mypy `disallow_untyped_defs` on the package (all functions annotated) and public docstrings

### Changed

- Removed commands `tree`, `export-csv`, `rest` and `rest-config --import`
- `eco2ai` is now an optional dependency (`demne._tracking` is a no-op without it)
- One BRAT parser (`demne._brat`) for the scorers, frequency, TF-IDF coverage, optimizer and the dashboard parser
- One copy of the risk lexicons (`demne.params`) for the CLI scorer, the dashboard and the optimizer
- Dashboard R now includes the contradiction term, as the CLI scorer does (it was fixed to 0)
- Dashboard entity preselection comes from `data/decision_config.json` instead of a hardcoded list
- `main.py info` reads `demne.__version__`

### Fixed

- Annotation files (`*.ann`) could ship in the wheel/sdist: removed from `package-data`, excluded in `MANIFEST.in`

## [0.4.0] - 2026-05-29
### Changed

- Full HDR audit: removed 49 files (artefacts, duplicates, PDFs, caches)
- Encoding fix UTF-16 → UTF-8 on 5 files (ci.yml, Makefile, CONTRIBUTING.md, DATA_GOVERNANCE.md, reproducibility.yaml)
- Removed UTF-8 BOM from 4 files
- Fixed broken Python imports (3 tests + sensitivity_analysis + REST __init__)
- Replaced 6 absolute Windows paths with DEMNE_ESMO_DIR environment variable
- Rewrote .gitignore (90 duplicated lines → 35 clean lines)
- Rewrote Makefile (UTF-16 + broken target → UTF-8 + valid targets)

### Added

- pyproject.toml: 14 missing dependencies in 6 optional groups (rest, viz, notebook, reports, gui, ner[pycrysuite])
- config/example_env.sh: environment variable template
- Graph_decision.png in Results/figures/ (fixes README image)

### Fixed

- Broken README image (Graph_decision_biss.png → Graph_decision.png)
- Renamed lunch.py → launch.py (typo)
- README/CITATION.cff URLs

## [0.3.0] - 2026-04-19
### Changed

- Purged 182 ghost files from the Git index
- pyproject.toml: fixed build-backend (setuptools.build_meta)
- Removed requirements.txt (pyproject.toml is now the single source)
- README: CI/License/Python badges, corrected references

### Added

- CITATION.cff, CHANGELOG.md, data/README.md

### Fixed

- Regression from commit c0d5795 (4 files re-added)
