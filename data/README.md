# Data

Ce dossier contient les données d'entrée et fichiers de configuration générés.

Les fichiers de données ne sont pas versionnés (voir `.gitignore`).
Pour obtenir les données, contactez l'auteur ou consultez la documentation.

## Corpus de recalibration (`python main.py optimize_params`)

    data/OPT_CORPORA/<corpus>/          domaine cible — optimisation (requis)
    data/TEST_CORPORA/<corpus>/         domaine cible — held-out (requis)
    data/OFF_DOMAIN_CORPORA/<corpus>/   hors domaine — avec --all_corpora uniquement
    data/ALL_CORPORA/<corpus>/          optionnel — par défaut OPT ∪ TEST

Chaque `<corpus>/` contient des paires BRAT `*.ann` + `*.txt` et un fichier
`*_routing.csv` (colonnes `entity`, `Label` ∈ RULES/TBM/LLM).
Détails : [docs/OPTIMIZE_PARAMS.md](../docs/OPTIMIZE_PARAMS.md).
