# DuraXELL : Sustainable Information Extraction for LLM en Cancérologie

[![CI](https://github.com/longeacc/DEMNE-Determination-of-Extraction-Methode-for-Named-Entity/actions/workflows/ci.yml/badge.svg)](https://github.com/longeacc/DEMNE-Determination-of-Extraction-Methode-for-Named-Entity/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://python.org)

DuraXELL est un pipeline d'extraction d'informations médicales (biomarqueurs) conçu pour optimiser le **Trilemme : Performance, Explicabilité, Frugalité**. Au lieu d'utiliser systématiquement des LLM très coûteux en énergie, DuraXELL utilise un arbre de décision pour router chaque entité vers la méthode la plus légère possible (Règles > ML > Transformer > LLM).

## Architecture arbre de décision

![Pipeline de Décision](Results/figures/Graph_decision.png)
*Arbre de Décision pour la Sélection Optimale de Méthodes d'Extraction d'Entités*

## Résultats Principaux (Front de Pareto)

![Exemple de résultats principaux](Results/figures/front_pareto_exemple.png)

## Installation et exécution reproductible

Suivez ces étapes pour obtenir un environnement reproductible (Linux/macOS/Windows). Les commandes Windows sont fournies quand elles diffèrent.

1. Cloner le dépôt :

```bash
git clone https://github.com/longeacc/DEMNE-Determination-of-Extraction-Methode-for-Named-Entity.git
cd DEMNE-Determination-of-Extraction-Methode-for-Named-Entity
```

2. Créer et activer un environnement Python (recommandé Python 3.10+) :

Linux/macOS
```bash
python -m venv .venv
source .venv/bin/activate
```

Windows (PowerShell)
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

3. Installer les dépendances via le packaging (extras utiles : `dev`, `ner`, `dashboard`) :

```bash
pip install --upgrade pip
pip install -e ".[dev,ner,dashboard]"
```

4. Configuration des variables d'environnement (reproductibilité des chemins)

Copiez le fichier d'exemple et adaptez les chemins locaux :

```bash
# Linux/macOS
cp config/example_env.sh .env
# Windows (PowerShell)
Copy-Item config\example_env.sh .env
# Ensuite, éditez .env pour ajuster les chemins (DEMNE_DATA_DIR, ...)
```

Le projet utilise aussi `config/reproducibility.yaml` pour fixer les seeds et configurations de répétabilité.

5. Exécuter le pipeline ou les composants

- Lancer le notebook principal (jupyter) :

```bash
jupyter notebook Reports/DuraXELL_Pipeline.ipynb
```

- Exécuter un script de génération de rapport (exemples) :

```bash
python scripts/run_full_pipeline_report.py
python scripts/run_sens.py
```

- Lancer le dashboard Streamlit (dans `dashboard/`) :

```bash
# à partir du répertoire racine du projet
python -m streamlit run dashboard/app.py
```

6. Tests et contribution

```bash
pytest -q
# Formatage/lint (si installé via extras dev)
black .
ruff check .
```

Notes
- Le packaging est géré par `pyproject.toml`. Les extras permettent d'installer uniquement ce dont vous avez besoin.
- Pour une reproduction complète sur CI, exportez et fixez la version des dépendances (pip freeze > requirements.txt) ou utilisez un environnement isolé (Docker/CI).

## Références et Citation

Ce travail s'appuie sur les recherches de **Akram REDJDAL et al. 2024** concernant l'extraction d'informations en oncologie et l'évaluation de la frugalité des modèles de langage.

**Citation :**
> Akram REDJDAL et al. 2024. *Le juste usage des LLM et méthode NLP en cancérologie : Vers une approche frugale et explicable*. ESIEE Paris.