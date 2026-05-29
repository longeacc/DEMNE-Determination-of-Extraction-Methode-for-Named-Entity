# Changelog

## [0.4.0] - 2026-05-29
### Changed

- Audit HDR complet : suppression de 49 fichiers (artefacts, doublons, PDFs, caches)
- Correction encodage UTF-16 → UTF-8 sur 5 fichiers (ci.yml, Makefile, CONTRIBUTING.md, DATA_GOVERNANCE.md, reproducibility.yaml)
- Retrait BOM UTF-8 sur 4 fichiers
- Correction imports Python cassés (3 tests + sensitivity_analysis + REST __init__)
- Remplacement 6 chemins absolus Windows par variables d'environnement DEMNE_ESMO_DIR
- Réécriture .gitignore (90 lignes dupliquées → 35 lignes propres)
- Réécriture Makefile (UTF-16 + cible cassée → UTF-8 + cibles valides)

### Added

- pyproject.toml : 14 dépendances manquantes dans 6 groupes optionnels (rest, viz, notebook, reports, gui, ner[pycrfsuite])
- config/example_env.sh : template variables d'environnement
- Graph_decision.png dans Results/figures/ (corrige image README)

### Fixed

- Image README cassée (Graph_decision_biss.png → Graph_decision.png)
- Renommage lunch.py → launch.py (typo)
- URLs README/CITATION.cff (DuraXell → DEMNE-...)

## [0.3.0] - 2026-04-19
### Changed

- Purge de 182 fichiers fantômes de l'index Git
- pyproject.toml : build-backend corrigé (setuptools.build_meta)
- requirements.txt retiré (pyproject.toml = source unique)
- README : badges CI/License/Python, refs corrigées

### Added

- CITATION.cff, CHANGELOG.md, data/README.md

### Fixed

- Régression du commit c0d5795 (4 fichiers réajoutés)