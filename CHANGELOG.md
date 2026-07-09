# Changelog

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
- README/CITATION.cff URLs (DuraXell → DEMNE-...)

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
