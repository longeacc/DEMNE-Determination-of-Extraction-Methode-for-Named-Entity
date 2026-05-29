# DuraXELL : Sustainable Information Extraction for LLM en Cancérologie

[![CI](https://github.com/longeacc/DuraXell/actions/workflows/ci.yml/badge.svg)](https://github.com/longeacc/DuraXell/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://python.org)

DuraXELL est un pipeline d'extraction d'informations médicales (biomarqueurs) conçu pour optimiser le **Trilemme : Performance, Explicabilité, Frugalité**. Au lieu d'utiliser systématiquement des LLM très coûteux en énergie, DuraXELL utilise un arbre de décision pour router chaque entité vers la méthode la plus légère possible (Règles > ML > Transformer > LLM).

## Architecture arbre de décision

![Pipeline de Décision](Results/figures/Graph_decision.png)
*Arbre de Décision pour la Sélection Optimale de Méthodes d'Extraction d'Entités*

## Résultats Principaux (Front de Pareto)

![Exemple de résultats principaux](Results/figures/front_pareto_exemple.png)

## Installation et Exécution Reproductible

1. **Cloner le dépôt** :

   ```bash
   git clone https://github.com/longeacc/DuraXell.git
   cd DuraXELL
   ```

2. **Créer un environnement virtuel et installer les dépendances** :

   ```bash
   python -m venv .venv
   source .venv/bin/activate  # Sur Windows : .venv\Scripts\activate
   pip install -e .[dev,ner]
   ```

3. **Exécuter le pipeline complet** :
   Ouvrez et exécutez le notebook maître :

   ```bash
   jupyter notebook Reports/DuraXELL_Pipeline.ipynb
   ```

   Ou exécutez le script de rapport :

   ```bash
   python scripts/run_full_pipeline_report.py
   ```

## Références et Citation

Ce travail s'appuie sur les recherches de **Akram REDJDAL et al. 2024** concernant l'extraction d'informations en oncologie et l'évaluation de la frugalité des modèles de langage.

**Citation :**
> Akram REDJDAL et al. 2024. *Le juste usage des LLM et méthode NLP en cancérologie : Vers une approche frugale et explicable*. ESIEE Paris.