# RAPPORT DE COMPRÃ‰HENSION INTÃ‰GRALE â€” Projet DuraXELL

> **Auteur :** Profil Chercheur HDR (HabilitÃ© Ã  Diriger des Recherches)
> **Date de rÃ©daction :** 28 fÃ©vrier 2026
> **Objet :** *Le juste usage des LLM et des mÃ©thodes NLP en cancÃ©rologie*
> **Contexte :** ESMO 2025 â€” Extraction frugale d'information clinique dans les comptes-rendus d'anatomopathologie du cancer du sein

---

## Table des matiÃ¨res

1. [Contexte scientifique et problÃ¨me posÃ©](#1-contexte-scientifique-et-problÃ¨me-posÃ©)
2. [Philosophie architecturale : la Cascade Frugale](#2-philosophie-architecturale--la-cascade-frugale)
3. [L'Arbre de DÃ©cision : le cerveau de DuraXELL](#3-larbre-de-dÃ©cision--le-cerveau-de-duraxell)
4. [Architecture exhaustive du projet](#4-architecture-exhaustive-du-projet)
5. [Graphe de dÃ©pendances inter-fichiers](#5-graphe-de-dÃ©pendances-inter-fichiers)
6. [Flux de donnÃ©es (Data Flow) complet](#6-flux-de-donnÃ©es-data-flow-complet)
7. [Description dÃ©taillÃ©e de chaque composant](#7-description-dÃ©taillÃ©e-de-chaque-composant)
8. [Guide d'utilisation et commandes d'exÃ©cution](#8-guide-dutilisation-et-commandes-dexÃ©cution)
9. [Guide d'interprÃ©tation des rÃ©sultats](#9-guide-dinterprÃ©tation-des-rÃ©sultats)
10. [Seuils de l'arbre et justification scientifique](#10-seuils-de-larbre-et-justification-scientifique)
11. [DÃ©pendances logicielles](#11-dÃ©pendances-logicielles)
12. [ProblÃ¨mes connus et rÃ©solutions](#12-problÃ¨mes-connus-et-rÃ©solutions)
13. [Glossaire](#13-glossaire)

---

## 1. Contexte scientifique et problÃ¨me posÃ©

### 1.1 Le besoin clinique

L'oncologie moderne produit des volumes croissants de comptes-rendus textuels (chirurgie â€” CHIR, rÃ©union de concertation pluridisciplinaire â€” RCP) contenant des biomarqueurs critiques pour la prise de dÃ©cision thÃ©rapeutique : statut des rÃ©cepteurs hormonaux (ER, PR), statut HER2 (IHC, FISH), indice de prolifÃ©ration Ki67, grade histologique SBR, mutations gÃ©nÃ©tiques, etc.

L'extraction manuelle de ces donnÃ©es est chronophage, non reproductible et incompatible avec les volumes hospitaliers. L'extraction automatique par NLP s'impose, mais la question du **juste dimensionnement** du modÃ¨le se pose :

- Un **LLM** (GPT-4, Llama) peut extraire quasi n'importe quoi, mais Ã  un coÃ»t Ã©nergÃ©tique et financier prohibitif ;
- Un **modÃ¨le ML lÃ©ger** (BERT fine-tunÃ©, CRF) est plus frugal, mais nÃ©cessite des donnÃ©es d'entraÃ®nement annotÃ©es ;
- Des **rÃ¨gles Regex** sont quasi gratuites en Ã©nergie, mais ne fonctionnent que sur des entitÃ©s trÃ¨s structurÃ©es.

### 1.2 La question de recherche

> *Pour chaque biomarqueur, quelle est la mÃ©thode d'extraction la plus frugale garantissant une performance acceptable ?*

DuraXELL formalise cette question sous la forme d'un **arbre de dÃ©cision multicritÃ¨re** qui alloue dynamiquement chaque entitÃ© au modÃ¨le le moins coÃ»teux capable de l'extraire fidÃ¨lement.

---

## 2. Philosophie architecturale : la Cascade Frugale

DuraXELL structure l'infÃ©rence en **4 niveaux hiÃ©rarchiques ordonnÃ©s par coÃ»t Ã©nergÃ©tique croissant** :

| Niveau | MÃ©thode | Ã‰nergie estimÃ©e (kWh) | ExplicabilitÃ© | Cas d'usage |
|---|---|---|---|---|
| 1 | **RÃ¨gles (Regex)** | ~1e-6 | 1.0 (parfaite) | EntitÃ©s trÃ¨s structurÃ©es (ER, PR au format standard) |
| 2 | **ML lÃ©ger (CRF)** | ~1e-5 | 0.7 | EntitÃ©s semi-structurÃ©es Ã  frÃ©quence suffisante |
| 3 | **Transformer (BERT)** | ~1e-4 | 0.3 | EntitÃ©s nÃ©cessitant une comprÃ©hension contextuelle |
| 4 | **LLM (GPT/Llama)** | ~1e-2 | 0.1 (boÃ®te noire) | Dernier recours pour entitÃ©s complexes/rares |

Le principe de **cascade** implique qu'un niveau n n'est sollicitÃ© que si le niveau n-1 Ã©choue (confiance insuffisante). Le score composite pondÃ¨re trois axes :

```
C = Î± Â· F1 + Î² Â· ExplicabilitÃ© + Î³ Â· (1 - E_norm)
```

avec par dÃ©faut Î± = 0.4, Î² = 0.3, Î³ = 0.3.

---

## 3. L'Arbre de DÃ©cision : le cerveau de DuraXELL

### 3.1 Visualisation de l'arbre

L'arbre ci-dessous est le rÃ©sultat de l'exÃ©cution de `E_creation_arbre_decision.py`. Il route chaque biomarqueur vers la feuille appropriÃ©e en fonction de 7 mÃ©triques calculÃ©es sur le corpus :

![Arbre de DÃ©cision DuraXELL](Results/figures/decision_tree.png)

*Figure 1 â€” Arbre de dÃ©cision global. Les nÅ“uds internes reprÃ©sentent les tests sur les mÃ©triques (TemplatabilitÃ©, HomogÃ©nÃ©itÃ©, Risque...) ; les feuilles (rectangles colorÃ©s) indiquent la mÃ©thode d'extraction assignÃ©e.*

![Visualisation des assignations par entitÃ©](Results/figures/decision_tree_visualization.png)

*Figure 2 â€” Graphe d'assignation : chaque biomarqueur est reliÃ© Ã  sa feuille de dÃ©cision rÃ©sultante.*

### 3.2 Logique de l'arbre (pseudo-code)

```text
ENTRÃ‰E : entity, mÃ©triques {Te, He, R, Freq, Yield, Feas, DomainShift, LLM_Necessity}

SI Te >= TE_HIGH (70%) :
â”‚   SI He >= HE_HIGH (70%) :
â”‚   â”‚   SI R < RISK_HIGH (0.5) :
â”‚   â”‚   â”‚   â†’ FEUILLE NER Ã€ BASE DE RÃˆGLES       â† Regex pur
â”‚   â”‚   SINON :
â”‚   â”‚       â†’ continuer vers FaisabilitÃ© NER
â”‚   SINON :
â”‚       â†’ continuer vers TemplatabilitÃ© moyenne
â”‚
SI Te >= TE_MED (40%) :
â”‚   SI Freq >= FREQ_MIN (0.001) :
â”‚   â”‚   SI Yield >= YIELD_HIGH (0.75) :
â”‚   â”‚   â”‚   â†’ FEUILLE ML LÃ‰GER NER                â† CRF/BERT lÃ©ger
â”‚   â”‚   SINON :
â”‚   â”‚       â†’ continuer vers FaisabilitÃ© NER
â”‚   SINON :
â”‚       â†’ continuer vers FaisabilitÃ© NER
â”‚
SI Feas >= FEAS_NER (0.6) :
â”‚   SI DomainShift < DOMAIN_SHIFT_MAX (0.5) :
â”‚   â”‚   â†’ FEUILLE NER TRANSFORMER BIDIRECTIONNEL  â† BERT spÃ©cialisÃ©
â”‚   SINON :
â”‚       â†’ continuer vers NÃ©cessitÃ© LLM
â”‚
SI LLM_Necessity >= LLM_NEC_HIGH (0.5) :
â”‚   â†’ FEUILLE NER LLM                             â† GPT / Llama
SINON :
â”‚   SI Freq >= FREQ_MIN :
â”‚   â”‚   â†’ FEUILLE ML LÃ‰GER PAR DÃ‰FAUT             â† Backoff ML
â”‚   SINON :
â”‚       â†’ FEUILLE RÃˆGLES PAR DÃ‰FAUT               â† Backoff RÃ¨gles
```

### 3.3 RÃ©sultat actuel (Ã©tat du `decision_config.json`)

| Biomarqueur | Te | He | Freq | Feas | LLM_Nec | **Feuille assignÃ©e** |
|---|---|---|---|---|---|---|
| Estrogen_receptor | 43.8 | 97.4 | 0.0011 | 0.542 | 0.3 | **ML LÃ‰GER PAR DÃ‰FAUT** |
| Progesterone_receptor | 46.0 | 97.0 | 0.0009 | 0.475 | 0.6 | **NER LLM** |
| HER2_status | 41.1 | 97.5 | 0.0005 | 0.256 | 0.6 | **NER LLM** |
| HER2_IHC | 31.8 | 97.4 | 0.0006 | 0.299 | 0.6 | **NER LLM** |
| Ki67 | 40.6 | 98.1 | 0.0008 | 0.417 | 0.6 | **NER LLM** |
| HER2_FISH | 30.7 | 83.7 | 0.0001 | 0.070 | 0.6 | **NER LLM** |
| Genetic_mutation | 100.0 | 1.4 | 0.00001 | 0.005 | 0.6 | **NER LLM** |

---

## 4. Architecture exhaustive du projet

L'arborescence ci-dessous est **exhaustive** et reflÃ¨te la totalitÃ© des fichiers source, scripts utilitaires, donnÃ©es, rÃ©sultats et documentation. Seuls les fichiers binaires internes Ã  `.venv/` et `__pycache__/` sont exclus.

Les dossiers `evaluation_set_*` contiennent les corpus de textes cliniques anonymisÃ©s (des milliers de fichiers `.txt` et `.ann` au format BRAT). Les dossiers `checkpoint-*` et `sweeps/` contiennent les poids des modÃ¨les entraÃ®nÃ©s (fichiers `.safetensors`, `config.json`, `tokenizer.json`).

```text
DuraXELL/
â”‚
â”œâ”€â”€ main.py                                  # POINT D'ENTRÃ‰E CLI : commandes extract, tree, metrics, rest, evaluate, info
â”œâ”€â”€ decision_config.json                     # ORACLE CENTRAL : matrice JSON produite par l'arbre, lue par l'orchestrateur
â”œâ”€â”€ run_full_pipeline_report.py              # MACRO RUNNER : exÃ©cute les 6 blocs d'Ã©valuation et gÃ©nÃ¨re les figures
â”œâ”€â”€ requirements.txt                         # DÃ©pendances pip (torch, transformers, pandas, eco2ai, pytest, etc.)
â”‚
â”œâ”€â”€ fix_trackers.py                          # Patch : ajout try/except sur tracker.stop() pour contourner crash Eco2AI/Pandas
â”œâ”€â”€ fix_trackers2.py                         # Patch v2 : variante Ã©tendue du fix_trackers
â”œâ”€â”€ fix_bridge.py                            # Patch : mise Ã  jour de _normalize_method() dans rest_decision_bridge.py
â”œâ”€â”€ fix_demo.py                              # Patch : correction des labels mock dans demo_rest.py
â”œâ”€â”€ fix_orch.py                              # Patch : alignement des listes use_rules/use_ml dans cascade_orchestrator.py
â”œâ”€â”€ fix_scorer.py                            # Patch : ajout des labels FEUILLE dans EXPLAINABILITY_SCORES
â”œâ”€â”€ fix_tree.py                              # Patch : correction labels de feuilles dans E_creation_arbre_decision.py
â”œâ”€â”€ debug_freq.py                            # Script de dÃ©bogage pour le calcul de frÃ©quence
â”œâ”€â”€ create_notebook.py                       # GÃ©nÃ©rateur programmatique du notebook DuraXELL_Pipeline.ipynb
â”œâ”€â”€ test_psutil.py                           # VÃ©rification disponibilitÃ© psutil (monitoring systÃ¨me)
â”œâ”€â”€ test_run.py                              # Test rapide d'exÃ©cution du pipeline
â”œâ”€â”€ test_transformers.py                     # VÃ©rification import et chargement de HuggingFace Transformers
â”‚
â”œâ”€â”€ breast_cancer_biomarker_eval_summary.csv # Copie racine du bilan d'Ã©valuation biomarqueurs
â”œâ”€â”€ Consumtion_of_Duraxell.csv               # Log de consommation Ã©nergÃ©tique (Ã©crit par Eco2AI)
â”œâ”€â”€ dependency_tree_viewer.html              # Visualisation interactive HTML de l'arbre de dÃ©pendances
â”œâ”€â”€ DuraXELL_DependencyTree.jsx              # Composant React JSX de l'arbre de dÃ©pendances
â”‚
â”œâ”€â”€ compile_out.txt                          # Trace de compilation
â”œâ”€â”€ cuda_check.txt                           # VÃ©rification disponibilitÃ© CUDA
â”œâ”€â”€ nvidia_smi.txt                           # Sortie nvidia-smi (Ã©tat GPU)
â”œâ”€â”€ torch_info.txt                           # Informations PyTorch (version, device)
â”œâ”€â”€ out_demo_rest.txt                        # Sortie de demo_rest.py
â”œâ”€â”€ out_eval.txt                             # Sortie de run_full_pipeline_report.py
â”œâ”€â”€ out_eval_venv.txt                        # Sortie alternative (exÃ©cution via .venv)
â”œâ”€â”€ output_decision.txt                      # Rapport textuel gÃ©nÃ©rÃ© par E_creation_arbre_decision.py
â”œâ”€â”€ test_out.txt                             # Sortie de tests
â”œâ”€â”€ test_out2.txt                            # Sortie de tests (bis)
â”œâ”€â”€ test_result_decision.txt                 # RÃ©sultat du test de l'arbre de dÃ©cision
â”œâ”€â”€ test_result_freq.txt                     # RÃ©sultat du test de frÃ©quence
â”œâ”€â”€ test_result_yield.txt                    # RÃ©sultat du test de rendement
â”œâ”€â”€ project_files.txt                        # Index auto-gÃ©nÃ©rÃ© des fichiers du projet
â”‚
â”œâ”€â”€ ARCHITECTURE_RECAP.md                    # RÃ©capitulatif architectural (ancienne version)
â”œâ”€â”€ AUDIT_24FEV.md                           # Audit systÃ¨me du 24 fÃ©vrier
â”œâ”€â”€ BILAN_SEMAINE1.md                        # Bilan d'avancement semaine 1
â”œâ”€â”€ BILAN_SEMAINE2.md                        # Bilan d'avancement semaine 2
â”œâ”€â”€ DEPENDENCY_REPORT.md                     # Rapport de dÃ©pendances inter-modules
â”œâ”€â”€ planning_duraxell_v2_complet.md          # Planning complet v2
â”œâ”€â”€ planning_vacances_duraxell.md            # Planning vacances
â”œâ”€â”€ README.md                                # README principal du projet
â”œâ”€â”€ THRESHOLDS_JUSTIFICATION.md              # Justification scientifique des seuils de l'arbre
â”œâ”€â”€ RAPPORT_COMPREHENSION.md                 # CE DOCUMENT
â”‚
â”œâ”€â”€ commandes.ipynb                          # Notebook de commandes rapides
â”‚
â”‚
â”œâ”€â”€ src/duraxell/                                # â•â•â•â•â•â• MOTEUR SCIENTIFIQUE PRINCIPAL â•â•â•â•â•â•
â”‚   â”œâ”€â”€ __init__.py                          # Initialisation du package Python
â”‚   â”‚
â”‚   â”œâ”€â”€ structs.py                           # DataClass ExtractionResult (structure de donnÃ©es centrale)
â”‚   â”œâ”€â”€ cascade_orchestrator.py              # ORCHESTRATEUR : lit decision_config.json, route vers Rules/ML/LLM
â”‚   â”œâ”€â”€ energy_tracker.py                    # Profilage Ã©nergÃ©tique (encapsule Eco2AI avec mesure kWh/infÃ©rence)
â”‚   â”œâ”€â”€ graph_orchestrator.py                # Orchestrateur de graphes (placeholder pour extension future)
â”‚   â”œâ”€â”€ visualize_decision_tree.py           # GÃ©nÃ¨re les PNG de l'arbre via NetworkX + Matplotlib
â”‚   â”œâ”€â”€ sensitivity_analysis.py              # Analyse de sensibilitÃ© : perturbation des seuils +/-10-20%
â”‚   â”‚
â”‚   â”œâ”€â”€ E_creation_arbre_decision.py         # CONSTRUCTEUR DE L'ARBRE : gÃ©nÃ¨re decision_config.json
â”‚   â”œâ”€â”€ E_composite_scorer.py                # Score composite C = aF1 + bExpl + g(1-Enorm)
â”‚   â”œâ”€â”€ E_annotation_yield.py               # MÃ©trique : rendement d'annotation (F1 Rules vs Gold Standard)
â”‚   â”œâ”€â”€ E_feasibility_NER.py                  # MÃ©trique : faisabilitÃ© NER (capacitÃ© des modÃ¨les ML)
â”‚   â”œâ”€â”€ E_frequency.py                       # MÃ©trique : frÃ©quence d'apparition des entitÃ©s dans le corpus
â”‚   â”œâ”€â”€ E_homogeneity.py                     # MÃ©trique : homogÃ©nÃ©itÃ© lexicale (variabilitÃ© du vocabulaire)
â”‚   â”œâ”€â”€ E_risk_context.py                    # MÃ©trique : risque contextuel (nÃ©gations, ambiguÃ¯tÃ©s)
â”‚   â”œâ”€â”€ E_templatability.py                   # MÃ©trique : templatabilitÃ© (stabilitÃ© structurelle)
â”‚   â”‚
â”‚   â”œâ”€â”€ generate_homogeneity_report.py       # GÃ©nÃ¨re le rapport HTML d'homogÃ©nÃ©itÃ©
â”‚   â”œâ”€â”€ generate_risk_context_report.py      # GÃ©nÃ¨re le rapport HTML de risque contextuel
â”‚   â”œâ”€â”€ generate_templatability_report.py     # GÃ©nÃ¨re le rapport HTML de templatabilitÃ©
â”‚   â”‚
â”‚   â”œâ”€â”€ Consumtion_of_Duraxell.csv           # Log Eco2AI local au module duraxell
â”‚   â”‚
â”‚   â”œâ”€â”€ Breast/                              # â•â• CORPUS CLINIQUES (cancer du sein) â•â•
â”‚   â”‚   â”œâ”€â”€ CHIR/                            # Comptes-rendus de chirurgie
â”‚   â”‚   â”‚   â”œâ”€â”€ evaluation_set_breast_cancer_chir_GS/        # Gold Standard CHIR (.txt + .ann)
â”‚   â”‚   â”‚   â””â”€â”€ evaluation_set_breast_cancer_chir_pred_rules/ # PrÃ©dictions RÃ¨gles CHIR
â”‚   â”‚   â””â”€â”€ RCP/                             # Comptes-rendus RCP
â”‚   â”‚       â”œâ”€â”€ evaluation_set_breast_cancer_GS/              # Gold Standard RCP
â”‚   â”‚       â”œâ”€â”€ evaluation_set_breast_cancer_pred_ner/        # PrÃ©dictions NER (Transformers)
â”‚   â”‚       â”œâ”€â”€ evaluation_set_breast_cancer_pred_rules/      # PrÃ©dictions RÃ¨gles RCP
â”‚   â”‚       â””â”€â”€ training_set_breast_cancer/                   # Set d'entraÃ®nement
â”‚   â”‚
â”‚   â”œâ”€â”€ REST_interface/                      # â•â• COUCHE API / VALIDATION CROISÃ‰E â•â•
â”‚   â”‚   â”œâ”€â”€ __init__.py
â”‚   â”‚   â”œâ”€â”€ rest_pipeline.py                 # Ordonnancement du pipeline REST complet
â”‚   â”‚   â”œâ”€â”€ rest_annotator.py                # Service d'annotation clinique par lot
â”‚   â”‚   â”œâ”€â”€ rest_decision_bridge.py          # PONT : compare dÃ©cision arbre (Top-Down) vs empirique (Bottom-Up)
â”‚   â”‚   â”œâ”€â”€ rest_evaluator.py                # Ã‰valuateur F1/PrÃ©cision sur les sorties REST
â”‚   â”‚   â”œâ”€â”€ demo_rest.py                     # DÃ©mo locale (mock) du pont REST
â”‚   â”‚   â”œâ”€â”€ convergence_analyzer.py          # Analyse de convergence des rÃ©ponses dans le temps
â”‚   â”‚   â”œâ”€â”€ yield_calculator.py              # Calcul du rendement de traitement
â”‚   â”‚   â”œâ”€â”€ logo.png                         # Logo du module REST
â”‚   â”‚   â”œâ”€â”€ README.md                        # Documentation spÃ©cifique REST
â”‚   â”‚   â”‚
â”‚   â”‚   â”œâ”€â”€ REST_modules/                    # Briques utilitaires
â”‚   â”‚   â”‚   â”œâ”€â”€ __init__.py
â”‚   â”‚   â”‚   â”œâ”€â”€ categorization.py            # Classification des requÃªtes par type
â”‚   â”‚   â”‚   â”œâ”€â”€ initialization.py            # Initialisation de l'environnement API
â”‚   â”‚   â”‚   â”œâ”€â”€ loading.py                   # Chargement et parsing des entrÃ©es
â”‚   â”‚   â”‚   â”œâ”€â”€ ui.py                        # Interface console (Rich/Formatage)
â”‚   â”‚   â”‚   â”œâ”€â”€ visualization.py             # GÃ©nÃ©ration de graphiques REST
â”‚   â”‚   â”‚   â”‚
â”‚   â”‚   â”‚   â”œâ”€â”€ calculs/                     # Sous-routines mathÃ©matiques
â”‚   â”‚   â”‚   â”‚   â”œâ”€â”€ __init__.py
â”‚   â”‚   â”‚   â”‚   â”œâ”€â”€ bootstrap.py             # Intervalles de confiance par bootstrap
â”‚   â”‚   â”‚   â”‚   â”œâ”€â”€ concordancer.py          # Concordance inter-annotateurs
â”‚   â”‚   â”‚   â”‚   â”œâ”€â”€ metrics.py               # Calcul F1, PrÃ©cision, Rappel
â”‚   â”‚   â”‚   â”‚   â”œâ”€â”€ ngram.py                 # Analyse N-grammes
â”‚   â”‚   â”‚   â”‚   â”œâ”€â”€ recommendations.py       # Moteur de recommandation
â”‚   â”‚   â”‚   â”‚   â”œâ”€â”€ regex.py                 # BibliothÃ¨que de patterns regex
â”‚   â”‚   â”‚   â”‚   â”œâ”€â”€ results.py               # AgrÃ©gation des rÃ©sultats
â”‚   â”‚   â”‚   â”‚   â””â”€â”€ tfidf.py                 # PondÃ©ration TF-IDF
â”‚   â”‚   â”‚   â”‚
â”‚   â”‚   â”‚   â””â”€â”€ extraction/                  # Sous-routines d'extraction
â”‚   â”‚   â”‚       â”œâ”€â”€ __init__.py
â”‚   â”‚   â”‚       â”œâ”€â”€ brat.py                  # Parseur du format BRAT (.ann)
â”‚   â”‚   â”‚       â”œâ”€â”€ normalisation.py         # Normalisation des entitÃ©s extraites
â”‚   â”‚   â”‚       â””â”€â”€ saving.py                # Sauvegarde des rÃ©sultats
â”‚   â”‚   â”‚
â”‚   â”‚   â””â”€â”€ templates/                       # Templates HTML pour l'interface web
â”‚   â”‚       â”œâ”€â”€ annotation_template.html     # Formulaire d'annotation
â”‚   â”‚       â”œâ”€â”€ evaluation_form.html         # Formulaire d'Ã©valuation
â”‚   â”‚       â””â”€â”€ results_dashboard.html       # Dashboard des rÃ©sultats
â”‚   â”‚
â”‚   â”œâ”€â”€ Rules/                               # â•â• MOTEUR RÃˆGLES (NIVEAU 1) â•â•
â”‚   â”‚   â”œâ”€â”€ src/
â”‚   â”‚   â”‚   â””â”€â”€ Breast/
â”‚   â”‚   â”‚       â”œâ”€â”€ rules_cascade_connector.py    # Connecteur : interface vers CascadeOrchestrator
â”‚   â”‚   â”‚       â”œâ”€â”€ biomarker_brat_annotator.py   # Export au format BRAT
â”‚   â”‚   â”‚       â”œâ”€â”€ lunch.py                      # Lanceur du module Breast
â”‚   â”‚   â”‚       â””â”€â”€ breast_cancer_biomarker_eval_summary.csv
â”‚   â”‚   â””â”€â”€ Results/                         # RÃ©sultats intermÃ©diaires du module RÃ¨gles
â”‚   â”‚       â”œâ”€â”€ frequency_analysis.csv       # FrÃ©quences calculÃ©es par E_frequency.py
â”‚   â”‚       â”œâ”€â”€ homogeneity_analysis.csv     # Scores He par E_homogeneity.py
â”‚   â”‚       â”œâ”€â”€ ner_feasibility_analysis.csv # FaisabilitÃ© par E_feasibility_NER.py
â”‚   â”‚       â”œâ”€â”€ risk_context_analysis.csv    # Scores R par E_risk_context.py
â”‚   â”‚       â”œâ”€â”€ templatability_analysis.csv   # Scores Te (CSV) par E_templatability.py
â”‚   â”‚       â””â”€â”€ templatability_analysis.json  # Scores Te (JSON dÃ©taillÃ©)
â”‚   â”‚
â”‚   â””â”€â”€ tests/                               # â•â• TESTS UNITAIRES & INTÃ‰GRATION (Pytest) â•â•
â”‚       â”œâ”€â”€ conftest.py                      # Fixtures partagÃ©es Pytest
â”‚       â”œâ”€â”€ test_simple.py                   # Smoke test minimal
â”‚       â”œâ”€â”€ test_cascade.py                  # Tests du CascadeOrchestrator
â”‚       â”œâ”€â”€ test_composite_scorer.py         # Tests du CompositeScorer
â”‚       â”œâ”€â”€ test_decision_tree.py            # Tests de l'arbre de dÃ©cision
â”‚       â”œâ”€â”€ test_energy_tracker.py           # Tests du tracker Ã©nergÃ©tique
â”‚       â”œâ”€â”€ test_annotation_yield.py         # Tests du rendement d'annotation
â”‚       â”œâ”€â”€ test_frequency.py                # Tests du calcul de frÃ©quence
â”‚       â”œâ”€â”€ test_homogeneity.py              # Tests de l'homogÃ©nÃ©itÃ©
â”‚       â”œâ”€â”€ test_risk_context.py             # Tests du contexte de risque
â”‚       â”œâ”€â”€ test_templatability.py            # Tests de la templatabilitÃ©
â”‚       â””â”€â”€ test_pipeline_integration.py     # Tests d'intÃ©gration bout-en-bout
â”‚
â”‚
â”œâ”€â”€ NER/                                     # â•â•â•â•â•â• MOTEUR ML / TRANSFORMERS (NIVEAU 2-3) â•â•â•â•â•â•
â”‚   â”œâ”€â”€ src/
â”‚   â”‚   â”œâ”€â”€ ner_cascade_connector.py         # Connecteur : interface vers CascadeOrchestrator
â”‚   â”‚   â”œâ”€â”€ 2bis_train_hf_ner.py             # EntraÃ®nement NER HuggingFace (PyTorch Trainer)
â”‚   â”‚   â”œâ”€â”€ 2sweep_ner.py                    # Recherche d'hyperparamÃ¨tres (sweep grid)
â”‚   â”‚   â”œâ”€â”€ 3infer.py                        # InfÃ©rence batchÃ©e sur corpus
â”‚   â”‚   â”œâ”€â”€ 4predict_to_brat.py              # Conversion prÃ©dictions tensorielles vers format BRAT
â”‚   â”‚   â”œâ”€â”€ 5evaluate_ner.py                 # Ã‰valuation F1/PrÃ©cision sur les prÃ©dictions NER
â”‚   â”‚   â””â”€â”€ eval_best_model.py               # Ã‰valuation du meilleur checkpoint
â”‚   â”‚
â”‚   â”œâ”€â”€ data/                                # DonnÃ©es d'entraÃ®nement
â”‚   â”‚   â””â”€â”€ Breast/test/
â”‚   â”‚       â””â”€â”€ combined_brat_files.txt      # Fichier BRAT combinÃ© pour le test
â”‚   â”‚
â”‚   â”œâ”€â”€ models/                              # ModÃ¨les entraÃ®nÃ©s
â”‚   â”‚   â”œâ”€â”€ output/bc_ner/                   # ModÃ¨le NER cancer du sein
â”‚   â”‚   â”‚   â”œâ”€â”€ best/                        # Meilleur checkpoint (config, tokenizer, vocab, weights)
â”‚   â”‚   â”‚   â””â”€â”€ checkpoint-*/                # Checkpoints intermÃ©diaires (131, 151, 262, ..., 786)
â”‚   â”‚   â””â”€â”€ sweeps/                          # Checkpoints des sweeps d'hyperparamÃ¨tres
â”‚   â”‚
â”‚   â”œâ”€â”€ sweep_results.csv                    # RÃ©sultats du sweep principal
â”‚   â”œâ”€â”€ sweep_results_new.csv                # RÃ©sultats mis Ã  jour
â”‚   â””â”€â”€ sweep_results (copie).csv            # Backup des rÃ©sultats
â”‚
â”‚
â”œâ”€â”€ Evaluation/                              # â•â•â•â•â•â• DONNÃ‰ES D'Ã‰VALUATION â•â•â•â•â•â•
â”‚   â””â”€â”€ REST_Annotations/                    # Annotations REST pour la validation croisÃ©e
â”‚
â”‚
â”œâ”€â”€ models/                                  # â•â•â•â•â•â• MODÃˆLES SWEEP RACINE â•â•â•â•â•â•
â”‚   â””â”€â”€ sweeps/
â”‚       â””â”€â”€ BiomedNLP-PubMedBERT-*/          # Checkpoints PubMedBERT (lr2e-05, bs4, ep3, wd0.01...)
â”‚
â”‚
â”œâ”€â”€ Reports/                                 # â•â•â•â•â•â• RAPPORTS & PUBLICATIONS â•â•â•â•â•â•
â”‚   â”œâ”€â”€ DuraXELL_Pipeline.ipynb              # Notebook Jupyter : dÃ©monstration complÃ¨te du pipeline
â”‚   â”œâ”€â”€ BILAN_VACANCES.md                    # Point d'Ã©tape vacances
â”‚   â””â”€â”€ conference_frugalite_abstract.md     # RÃ©sumÃ© soumis pour confÃ©rence sur la frugalitÃ© NLP
â”‚
â”‚
â””â”€â”€ Results/                                 # â•â•â•â•â•â• RÃ‰SULTATS FINAUX & FIGURES â•â•â•â•â•â•
    â”œâ”€â”€ benchmark_performance.csv            # F1-Score par mÃ©thode x biomarqueur
    â”œâ”€â”€ benchmark_explainability.csv         # Score d'explicabilitÃ© par mÃ©thode
    â”œâ”€â”€ benchmark_energy.csv                 # Consommation kWh par mÃ©thode
    â”œâ”€â”€ benchmark_pareto.csv                 # Score composite (front de Pareto)
    â”œâ”€â”€ breast_cancer_biomarker_eval_summary.csv  # Bilan global biomarqueurs
    â”œâ”€â”€ energy_summary.json                  # RÃ©sumÃ© JSON de la consommation
    â”œâ”€â”€ explainability_summary.json          # RÃ©sumÃ© JSON de l'explicabilitÃ©
    â”œâ”€â”€ risk_context_analysis_report.html    # Rapport HTML interactif du risque
    â”œâ”€â”€ risk_context_full.json               # DonnÃ©es brutes risque contextuel
    â”œâ”€â”€ risk_context_summary.csv             # RÃ©sumÃ© CSV du risque
    â”‚
    â”œâ”€â”€ REST_results/
    â”‚   â””â”€â”€ rest_validation_summary.csv      # Concordance arbre vs empirique REST
    â”‚
    â””â”€â”€ figures/                             # Images gÃ©nÃ©rÃ©es
        â”œâ”€â”€ decision_tree.png                # Arbre de dÃ©cision (NetworkX)
        â”œâ”€â”€ decision_tree_visualization.png  # Graphe d'assignation entitÃ©s vers feuilles
        â”œâ”€â”€ fig1_performance_heatmap.png     # Heatmap F1-Score
        â”œâ”€â”€ fig2_explainability_barplot.png  # Barplot explicabilitÃ©
        â”œâ”€â”€ fig3_energy_log_barplot.png      # Barplot Ã©nergie (Ã©chelle log)
        â”œâ”€â”€ fig4_pareto_scatter.png          # Scatter plot Pareto (F1 vs Ã‰nergie)
        â”œâ”€â”€ fig5_trilemma_radar.png          # Radar chart trilemme (Perf/Expl/Ã‰nergie)
        â”œâ”€â”€ CHIR_F1_per_entity.png           # F1 par entitÃ© (corpus CHIR)
        â”œâ”€â”€ CHIR_heatmap.png                 # Heatmap CHIR
        â”œâ”€â”€ RCP_F1_per_entity.png            # F1 par entitÃ© (corpus RCP)
        â”œâ”€â”€ RCP_HeatMap.png                  # Heatmap RCP
        â”œâ”€â”€ confusion_matrix.png             # Matrice de confusion
        â”œâ”€â”€ convergence_pie_chart.png        # Convergence arbre/empirique
        â”œâ”€â”€ metrics_heatmap.png              # Heatmap des mÃ©triques
        â”œâ”€â”€ OverallF1.png                    # F1 global
        â””â”€â”€ Figure_1.png                     # Figure complÃ©mentaire
```

---

## 5. Graphe de dÃ©pendances inter-fichiers

### 5.1 Classification des modules

Les fichiers du projet se classent en trois catÃ©gories structurelles :

**Modules Feuilles** (importÃ©s par d'autres, mais n'importent rien du projet) :
- `structs.py` â€” la pierre angulaire : dÃ©finit `ExtractionResult`
- `energy_tracker.py` â€” autonome (n'importe que `eco2ai`)
- Tous les `E_*.py` (mÃ©triques) â€” chacun est un module indÃ©pendant
- Tous les fichiers `REST_modules/calculs/*.py` et `REST_modules/extraction/*.py`

**Modules Hub** (Ã  la fois importÃ©s et importants) :
- `cascade_orchestrator.py` â€” le noeud le plus connectÃ© du projet
- `rules_cascade_connector.py` â€” passerelle RÃ¨gles vers Orchestrateur
- `ner_cascade_connector.py` â€” passerelle NER vers Orchestrateur
- `E_composite_scorer.py` â€” utilisÃ© par l'Ã©valuation et le report

**Modules Racine** (points d'entrÃ©e, jamais importÃ©s) :
- `main.py`, `run_full_pipeline_report.py`
- Tous les `test_*.py`, tous les `fix_*.py`
- Les scripts NER (`2bis_train_hf_ner.py`, `3infer.py`, `5evaluate_ner.py`)

### 5.2 Diagramme de dÃ©pendances (Mermaid)

```mermaid
graph TD
    subgraph "POINTS D ENTRÃ‰E"
        MAIN["main.py"]
        REPORT["run_full_pipeline_report.py"]
        TREE_SCRIPT["E_creation_arbre_decision.py"]
    end

    subgraph "ORCHESTRATION"
        ORCH["cascade_orchestrator.py"]
        STRUCTS["structs.py â€” ExtractionResult"]
        CONFIG["decision_config.json"]
        TRACKER["energy_tracker.py"]
    end

    subgraph "CONNECTEURS"
        RULES_CONN["rules_cascade_connector.py"]
        NER_CONN["ner_cascade_connector.py"]
    end

    subgraph "MÃ‰TRIQUES"
        TE["E_templatability.py"]
        HE["E_homogeneity.py"]
        RC["E_risk_context.py"]
        FR["E_frequency.py"]
        AY["E_annotation_yield.py"]
        FE["E_feasibility_NER.py"]
    end

    subgraph "Ã‰VALUATION"
        SCORER["E_composite_scorer.py"]
        SENS["sensitivity_analysis.py"]
        VIZ["visualize_decision_tree.py"]
    end

    subgraph "REST INTERFACE"
        BRIDGE["rest_decision_bridge.py"]
        EVAL_REST["rest_evaluator.py"]
        ANNOT["rest_annotator.py"]
        PIPE_REST["rest_pipeline.py"]
    end

    MAIN -->|"import"| ORCH
    MAIN -->|"subprocess"| TREE_SCRIPT
    MAIN -->|"subprocess"| VIZ

    REPORT -->|"import"| ORCH
    REPORT -->|"import"| SCORER
    REPORT -->|"import"| TRACKER
    REPORT -->|"import"| SENS
    REPORT -->|"import"| VIZ
    REPORT -->|"import"| ANNOT
    REPORT -->|"import"| EVAL_REST
    REPORT -->|"import"| BRIDGE

    ORCH -->|"import"| STRUCTS
    ORCH -->|"try import"| RULES_CONN
    ORCH -->|"try import"| NER_CONN
    ORCH -->|"lit"| CONFIG

    TREE_SCRIPT -->|"import"| AY
    TREE_SCRIPT -->|"lit CSVs"| TE
    TREE_SCRIPT -->|"lit CSVs"| HE
    TREE_SCRIPT -->|"lit CSVs"| RC
    TREE_SCRIPT -->|"lit CSVs"| FR
    TREE_SCRIPT -->|"lit CSVs"| FE
    TREE_SCRIPT -->|"Ã©crit"| CONFIG

    SENS -->|"import"| TREE_SCRIPT

    RULES_CONN -->|"import"| STRUCTS
    NER_CONN -->|"import"| STRUCTS
```

### 5.3 DÃ©pendances intramodulaires dÃ©taillÃ©es

| Fichier source | Importe (interne au projet) | Est importÃ© par |
|---|---|---|
| `structs.py` | *(aucun)* | `cascade_orchestrator`, `rules_cascade_connector`, `ner_cascade_connector`, tests |
| `cascade_orchestrator.py` | `structs`, `rules_cascade_connector`, `ner_cascade_connector` | `main`, `run_full_pipeline_report`, tests |
| `E_creation_arbre_decision.py` | `E_annotation_yield` | `main` (subprocess), `sensitivity_analysis` |
| `E_composite_scorer.py` | *(aucun)* | `run_full_pipeline_report`, tests |
| `energy_tracker.py` | *(aucun)* | `run_full_pipeline_report`, tests |
| `sensitivity_analysis.py` | `E_creation_arbre_decision` | `run_full_pipeline_report` |
| `visualize_decision_tree.py` | *(aucun â€” lit le JSON)* | `main` (subprocess), `run_full_pipeline_report` |
| `rest_decision_bridge.py` | *(aucun)* | `run_full_pipeline_report` |
| `rest_evaluator.py` | *(aucun)* | `run_full_pipeline_report` |
| `rules_cascade_connector.py` | `structs` | `cascade_orchestrator` |
| `ner_cascade_connector.py` | `structs` | `cascade_orchestrator` |
| `run_full_pipeline_report.py` | `E_composite_scorer`, `cascade_orchestrator`, `energy_tracker`, `visualize_decision_tree`, `sensitivity_analysis`, `rest_annotator`, `rest_evaluator`, `rest_decision_bridge` | *(point d'entrÃ©e)* |

---

## 6. Flux de donnÃ©es (Data Flow) complet

### 6.1 Flux de construction de l'arbre (offline, phase de calibration)

```mermaid
graph LR
    subgraph "CORPUS"
        GS["Gold Standard â€” .txt + .ann"]
        PRED_R["PrÃ©dictions RÃ¨gles"]
        PRED_NER["PrÃ©dictions NER"]
    end

    subgraph "CALCUL DES MÃ‰TRIQUES"
        TE["E_templatability.py â€” Te score"]
        HE["E_homogeneity.py â€” He score"]
        RC["E_risk_context.py â€” R score"]
        FR["E_frequency.py â€” Freq"]
        AY["E_annotation_yield.py â€” Yield"]
        FE["E_feasibility_NER.py â€” Feas, DomainShift"]
    end

    subgraph "RÃ‰SULTATS INTERMÃ‰DIAIRES"
        CSV["Rules/Results/*.csv + *.json"]
    end

    subgraph "CONSTRUCTION"
        BUILDER["E_creation_arbre_decision.py â€” DecisionTreeBuilder"]
        JSON["decision_config.json"]
        TXT["output_decision.txt"]
    end

    GS --> TE & HE & RC & FR & AY & FE
    PRED_R --> AY
    PRED_NER --> FE

    TE & HE & RC & FR --> CSV
    CSV --> BUILDER
    AY --> BUILDER
    FE --> CSV

    BUILDER -->|"Arbre logique if/else"| JSON
    BUILDER --> TXT
```

### 6.2 Flux d'infÃ©rence (online, phase de production)

```mermaid
graph TD
    INPUT["Texte clinique + EntitÃ© cible"]
    CLI["main.py extract"]
    ORCH["CascadeOrchestrator"]
    CONFIG["decision_config.json"]

    INPUT --> CLI --> ORCH
    ORCH -->|"Lecture"| CONFIG
    CONFIG -->|"ex: FEUILLE NER Ã€ BASE DE RÃˆGLES"| ORCH

    ORCH -->|"Niveau 1"| RULES["_try_rules â€” RulesCascadeConnector"]
    ORCH -->|"Niveau 2-3"| ML["_try_transformer â€” NERCascadeConnector"]
    ORCH -->|"Niveau 4"| LLM["_try_llm â€” LLMClient"]

    RULES -->|"confidence >= 0.7 ?"| CHECK1{Confiance OK ?}
    CHECK1 -->|"Oui"| RESULT
    CHECK1 -->|"Non â€” fallback"| ML

    ML -->|"confidence >= 0.7 ?"| CHECK2{Confiance OK ?}
    CHECK2 -->|"Oui"| RESULT
    CHECK2 -->|"Non â€” fallback"| LLM

    LLM --> RESULT["ExtractionResult â€” value, confidence, method_used, energy_kwh, cascade_level"]
```

### 6.3 Flux d'Ã©valuation complÃ¨te (run_full_pipeline_report.py)

Ce script exÃ©cute **6 blocs sÃ©quentiels** qui produisent l'ensemble des livrables scientifiques :

| Bloc | Nom | EntrÃ©e | Sortie |
|---|---|---|---|
| 1 | Performance (F1) | Benchmarks simulÃ©s | `Results/benchmark_performance.csv` |
| 2 | ExplicabilitÃ© | `EXPLAINABILITY_SCORES` dict | `Results/benchmark_explainability.csv` |
| 3 | Ã‰nergie | CoÃ»ts de rÃ©fÃ©rence | `Results/benchmark_energy.csv` |
| 4 | Pareto (Score Composite) | Blocs 1+2+3 | `Results/benchmark_pareto.csv` |
| 5 | Validation REST | Concordance arbre/empirique | `Results/REST_results/rest_validation_summary.csv` |
| 6 | Figures | Toutes les donnÃ©es | `Results/figures/fig1..fig5.png` |

---

## 7. Description dÃ©taillÃ©e de chaque composant

### 7.1 `structs.py` â€” La structure de donnÃ©es centrale

DÃ©finit la `dataclass` `ExtractionResult`, objet universel retournÃ© par chaque extracteur :

```python
@dataclass
class ExtractionResult:
    entity_type: str            # EntitÃ© ciblÃ©e (ex: "Estrogen_receptor")
    value: Optional[str]        # Valeur extraite (ex: "positive (100%)")
    method_used: str            # MÃ©thode ayant produit le rÃ©sultat ("Rules", "Transformer", "LLM")
    confidence: float           # Score de confiance [0.0 - 1.0]
    energy_kwh: float           # CoÃ»t Ã©nergÃ©tique de cette infÃ©rence
    cascade_level: int          # Niveau dans la cascade (1=RÃ¨gles, 2=ML, 3=Transformer, 4=LLM)
    span: Optional[Tuple]       # Positions [dÃ©but, fin] dans le texte
    metadata: Dict              # MÃ©tadonnÃ©es additionnelles
    execution_time_ms: float    # Temps d'exÃ©cution en millisecondes
```

### 7.2 `cascade_orchestrator.py` â€” Le coeur du systÃ¨me

**Classe `CascadeOrchestrator`** â€” ResponsabilitÃ©s :
- Charger `decision_config.json` au dÃ©marrage
- Pour chaque appel `extract(document, entity_type)` :
  1. Lire la mÃ©thode recommandÃ©e par l'arbre
  2. Mapper le label franÃ§ais (ex: `"FEUILLE NER Ã€ BASE DE RÃˆGLES"`) vers le connecteur technique correspondant
  3. ExÃ©cuter le connecteur ; si la confiance est insuffisante (< 0.7), cascader vers le niveau suivant
  4. Retourner un `ExtractionResult` enrichi du temps et de l'Ã©nergie

**Seuils de confiance internes :**
- `HIGH` = 0.9 â€” confiance trÃ¨s Ã©levÃ©e
- `MEDIUM` = 0.7 â€” seuil de validation (au-dessus : on accepte le rÃ©sultat)
- `LOW` = 0.4 â€” en dessous : escalade vers LLM

**Mapping des feuilles vers les connecteurs :**

```python
# RÃ¨gles
use_rules = recommended_method in [
    "FEUILLE NER Ã€ BASE DE RÃˆGLES",
    "FEUILLE RÃˆGLES PAR DÃ‰FAUT",
    "REGLES", "Rules"
]

# ML / Transformer
use_ml_transformer = recommended_method in [
    "FEUILLE ML LÃ‰GER NER",
    "FEUILLE ML LÃ‰GER PAR DÃ‰FAUT",
    "FEUILLE NER TRANSFORMER BIDIRECTIONNEL",
    "ML_CRF", "Transformer"
]

# LLM
use_llm = recommended_method in [
    "FEUILLE NER LLM",
    "LLM", "GPT"
]
```

### 7.3 `E_creation_arbre_decision.py` â€” Le constructeur de l'arbre

**Classe `DecisionTreeBuilder`** â€” ResponsabilitÃ©s :
- Charger les mÃ©triques depuis `Rules/Results/*.csv` et `*.json`
- Pour chaque biomarqueur, parcourir l'arbre logique et dÃ©terminer la feuille
- Sauvegarder le rÃ©sultat dans `decision_config.json` et `output_decision.txt`

**Fonction `load_metrics_from_csv()`** â€” AgrÃ¨ge les CSV de mÃ©triques :
- `templatability_analysis.json` â†’ Te
- `homogeneity_analysis.csv` â†’ He
- `risk_context_summary.csv` â†’ R
- `frequency_analysis.csv` â†’ Freq
- `ner_feasibility_analysis.csv` â†’ Feas, DomainShift, LLM_Necessity

### 7.4 `E_composite_scorer.py` â€” Le scoring multicritÃ¨re

**Classe `CompositeScorer`** â€” Fournit :
- Le dictionnaire `EXPLAINABILITY_SCORES` (associant chaque mÃ©thode Ã  un score d'explicabilitÃ©)
- La mÃ©thode `compute(f1, method, energy_kwh)` â†’ score composite
- La mÃ©thode `pareto_analysis(results_df)` â†’ identification des configurations Pareto-optimales

**Dictionnaire `EXPLAINABILITY_SCORES` :**

```python
EXPLAINABILITY_SCORES = {
    "Rules": 1.0,     "REGLES": 1.0,
    "CRF": 0.7,       "ML_CRF": 0.7,
    "Transformer": 0.3, "BERT": 0.3,
    "LLM": 0.1,       "GPT": 0.1,
    "FEUILLE NER Ã€ BASE DE RÃˆGLES": 1.0,
    "FEUILLE RÃˆGLES PAR DÃ‰FAUT": 1.0,
    "FEUILLE ML LÃ‰GER NER": 0.7,
    "FEUILLE ML LÃ‰GER PAR DÃ‰FAUT": 0.7,
    "FEUILLE NER TRANSFORMER BIDIRECTIONNEL": 0.3,
    "FEUILLE NER LLM": 0.1,
}
```

### 7.5 `energy_tracker.py` â€” Le profileur de frugalitÃ©

**Classe `EnergyTracker`** â€” Offre un context manager `measure(method, entity)` qui :
- Lance un tracker Eco2AI en arriÃ¨re-plan (si la librairie est prÃ©sente)
- Mesure le kWh consommÃ© pendant l'exÃ©cution d'un bloc de code
- Fallback sur des estimations tabulÃ©es si Eco2AI est absent ou en environnement de test

**CoÃ»ts de rÃ©fÃ©rence tabulÃ©s :**

```python
REFERENCE_COSTS_KWH = {
    "Rules":       1e-6,
    "CRF":         1e-5,
    "Transformer": 1e-4,
    "LLM":         1e-3,
    "LLM_API":     1e-2,
}
```

### 7.6 Les connecteurs (`rules_cascade_connector.py`, `ner_cascade_connector.py`)

Ces deux fichiers implÃ©mentent la mÃªme interface `.predict(text, entity_type) â†’ ExtractionResult` mais avec des backends diffÃ©rents :
- **Rules** : applique des expressions rÃ©guliÃ¨res spÃ©cialisÃ©es cancer du sein
- **NER** : charge un modÃ¨le HuggingFace (PubMedBERT fine-tunÃ©) et exÃ©cute l'infÃ©rence PyTorch

### 7.7 `rest_decision_bridge.py` â€” La validation croisÃ©e Top-Down / Bottom-Up

Compare les dÃ©cisions de l'arbre (thÃ©oriques, basÃ©es sur les mÃ©triques) avec les observations empiriques du pipeline REST (basÃ©es sur les performances rÃ©elles des annotations). Un taux de concordance Ã©levÃ© (>90%) valide la cohÃ©rence de l'arbre.

**MÃ©thode `compare(entity, tree_method, empirical_te)`** :
1. Normalise les deux mÃ©thodes (`_normalize_method()`)
2. Compare les catÃ©gories rÃ©sultantes (Rules/ML/LLM)
3. Retourne un dict avec concordance, justification, et recommandation

### 7.8 Pipeline NER complet (`NER/src/`)

Les scripts sont numÃ©rotÃ©s sÃ©quentiellement dans l'ordre d'exÃ©cution :

| Script | Ã‰tape | Description |
|---|---|---|
| `2bis_train_hf_ner.py` | EntraÃ®nement | Fine-tuning de PubMedBERT sur le corpus BRAT annotÃ© |
| `2sweep_ner.py` | Optimisation | Grid search sur lr, batch_size, epochs, warmup, weight_decay |
| `3infer.py` | InfÃ©rence | PrÃ©diction batchÃ©e sur le corpus de test |
| `4predict_to_brat.py` | Conversion | Transformation des tenseurs en fichiers `.ann` BRAT |
| `5evaluate_ner.py` | Ã‰valuation | Calcul F1/PrÃ©cision/Rappel par entitÃ© vs Gold Standard |
| `eval_best_model.py` | SÃ©lection | Ã‰valuation finale du meilleur checkpoint |

### 7.9 Les scripts `fix_*.py` â€” Patchs de maintenance

| Script | Ce qu'il corrige | Fichier cible |
|---|---|---|
| `fix_trackers.py` | Encapsule `tracker.stop()` dans try/except | `energy_tracker.py`, `E_creation_arbre_decision.py` |
| `fix_trackers2.py` | Extension de fix_trackers avec plus de fichiers | Idem + autres |
| `fix_orch.py` | Ajoute labels FEUILLE dans les listes de matching | `cascade_orchestrator.py` |
| `fix_bridge.py` | Met Ã  jour `_normalize_method()` pour labels FR | `rest_decision_bridge.py` |
| `fix_demo.py` | Corrige les donnÃ©es mock pour la dÃ©mo | `demo_rest.py` |
| `fix_scorer.py` | Ajoute labels FEUILLE dans EXPLAINABILITY_SCORES | `E_composite_scorer.py` |
| `fix_tree.py` | Renomme les feuilles en labels franÃ§ais | `E_creation_arbre_decision.py` |

### 7.10 Modules REST auxiliaires

| Module | RÃ´le |
|---|---|
| `REST_modules/categorization.py` | Classe les requÃªtes par type de biomarqueur |
| `REST_modules/initialization.py` | Initialise les chemins, les configs et l'environnement |
| `REST_modules/loading.py` | Lit et parse les fichiers BRAT d'entrÃ©e |
| `REST_modules/ui.py` | Affichage formatÃ© (barres de progression, tableaux) |
| `REST_modules/visualization.py` | GÃ©nÃ¨re des graphiques Matplotlib pour l'interface REST |
| `REST_modules/calculs/bootstrap.py` | Intervalles de confiance par rÃ©Ã©chantillonnage |
| `REST_modules/calculs/concordancer.py` | Mesure l'accord inter-annotateurs |
| `REST_modules/calculs/metrics.py` | Calcul F1, PrÃ©cision, Rappel Ã  partir de TP/FP/FN |
| `REST_modules/calculs/ngram.py` | Extraction de n-grammes pour l'analyse de frÃ©quence |
| `REST_modules/calculs/recommendations.py` | Recommandations automatiques basÃ©es sur les rÃ©sultats |
| `REST_modules/calculs/regex.py` | BibliothÃ¨que de patterns regex biomÃ©dicaux |
| `REST_modules/calculs/results.py` | AgrÃ©gation et formatage des rÃ©sultats |
| `REST_modules/calculs/tfidf.py` | PondÃ©ration TF-IDF pour l'analyse lexicale |
| `REST_modules/extraction/brat.py` | Parseur du format BRAT (.ann) en objets Python |
| `REST_modules/extraction/normalisation.py` | Normalisation des entitÃ©s (casse, accents, synonymes) |
| `REST_modules/extraction/saving.py` | Ã‰criture des rÃ©sultats sur disque |

---

## 8. Guide d'utilisation et commandes d'exÃ©cution

### 8.1 PrÃ©requis

```powershell
# SystÃ¨me : Python 3.10+, Windows 10/11, GPU NVIDIA recommandÃ© (CUDA)
# Activation de l'environnement
.venv\Scripts\Activate.ps1

# Installation des dÃ©pendances
pip install -r requirements.txt
```

### 8.2 Commandes principales (via `main.py`)

Le point d'entrÃ©e unique est `main.py`. Voici la **totalitÃ©** des sous-commandes disponibles :

| Commande | Description |
|---|---|
| `extract` | Extraire **une** entitÃ© depuis un texte clinique |
| `extract-all` | Extraire **toutes** les entitÃ©s connues (7 biomarqueurs) depuis un texte |
| `batch` | Extraction par lot sur un dossier de fichiers `.txt` (Ã— toutes les entitÃ©s) |
| `tree` | RÃ©gÃ©nÃ©rer l'arbre de dÃ©cision + visualisation |
| `metrics` | Calculer les mÃ©triques Te (templatability) et He (homogeneity) |
| `rest` | Lancer la dÃ©mo REST (mock local) |
| `evaluate` | Ã‰valuation complÃ¨te multi-critÃ¨res (rapport + figures) |
| `serve` | DÃ©marrer un serveur HTTP local (WIP) |
| `info` | Afficher les informations systÃ¨me et composants chargÃ©s |

**EntitÃ©s disponibles** (configurÃ©es dans `decision_config.json`) :
`Estrogen_receptor`, `Progesterone_receptor`, `HER2_status`, `HER2_IHC`, `Ki67`, `HER2_FISH`, `Genetic_mutation`

---

#### Commandes principales dÃ©taillÃ©es

```powershell
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# 1. EXTRACTION UNITAIRE : une entitÃ©, un texte
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
python main.py extract --doc "ER positif, PR nÃ©gatif." --entity "Estrogen_receptor"
# Sortie :
#   Result: positif | Method: Rules | Confidence: 1.00 | Energy: 0.000000 kWh

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# 2. EXTRACTION COMPLÃˆTE : toutes les entitÃ©s sur un texte  â˜… RECOMMANDÃ‰
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
python main.py extract-all --doc "Tumor phenotype: ER positif. PR nÃ©gatif. HER2 score 2+. Ki67 30%."
# Sortie : tableau des 7 entitÃ©s avec valeur, mÃ©thode, confiance, Ã©nergie
#   Estrogen_receptor         => positif    | Rules  | conf=1.00
#   Progesterone_receptor     => negatif    | Rules  | conf=1.00
#   HER2_status               => positif    | Rules  | conf=1.00
#   HER2_IHC                  => 2+         | Rules  | conf=1.00
#   Ki67                      => 30%        | Rules  | conf=1.00
#   HER2_FISH                 => NON TROUVÃ‰ | None   | conf=0.00
#   Genetic_mutation          => NON TROUVÃ‰ | None   | conf=0.00

# Variante : sous-ensemble d'entitÃ©s spÃ©cifiques
python main.py extract-all --doc "ER positif" --entities "Estrogen_receptor,HER2_status"

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# 3. EXTRACTION PAR LOT : dossier de fichiers Ã— toutes les entitÃ©s
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
python main.py batch --input_dir src/duraxell/Breast/RCP/evaluation_set_breast_cancer_GS
# Traite tous les .txt du dossier, extrait les 7 entitÃ©s pour chacun

# Variante : lot restreint Ã  certaines entitÃ©s
python main.py batch --input_dir src/duraxell/Breast/RCP --entities "Estrogen_receptor,Ki67"

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# 4. ARBRE DE DÃ‰CISION : rÃ©gÃ©nÃ©ration + visualisation
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
python main.py tree
# â†’ decision_config.json, output_decision.txt, Results/figures/decision_tree*.png

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# 5. MÃ‰TRIQUES : Te (templatability) et He (homogeneity)
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
python main.py metrics

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# 6. DÃ‰MO REST (mock local)
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
python main.py rest

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# 7. Ã‰VALUATION COMPLÃˆTE (6 blocs + figures)
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
python main.py evaluate

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# 8. SERVEUR HTTP LOCAL (WIP)
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
python main.py serve

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# 9. INFORMATIONS SYSTÃˆME
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
python main.py info
```

### 8.3 ExÃ©cution directe des scripts spÃ©cialisÃ©s

```powershell
# --- GÃ©nÃ©ration isolÃ©e de rapports HTML ---
python src/duraxell/generate_homogeneity_report.py
python src/duraxell/generate_risk_context_report.py
python src/duraxell/generate_templatability_report.py

# --- Ã‰valuation complÃ¨te avec toutes les figures ---
python run_full_pipeline_report.py
# Produit 5 figures dans Results/figures/ et 4 CSV dans Results/

# --- Pipeline NER complet (nÃ©cessite GPU) ---
python NER/src/2bis_train_hf_ner.py       # EntraÃ®nement (~30min sur GPU)
python NER/src/2sweep_ner.py              # Sweep hyperparamÃ¨tres
python NER/src/3infer.py                  # InfÃ©rence
python NER/src/4predict_to_brat.py        # Export BRAT
python NER/src/5evaluate_ner.py           # Ã‰valuation F1
```

### 8.4 ExÃ©cution des tests

```powershell
# Variable d'environnement nÃ©cessaire pour la rÃ©solution des imports
$env:PYTHONPATH="."

# Lancement de la suite complÃ¨te (25 tests)
pytest -v src/duraxell/tests/

# Lancement ciblÃ©
pytest -v src/duraxell/tests/test_cascade.py          # Teste l'orchestrateur
pytest -v src/duraxell/tests/test_decision_tree.py     # Teste la logique de l'arbre
pytest -v src/duraxell/tests/test_composite_scorer.py  # Teste le scoring
pytest -v src/duraxell/tests/test_pipeline_integration.py  # Tests d'intÃ©gration
```

---

## 9. Guide d'interprÃ©tation des rÃ©sultats

### 9.1 Lire un `ExtractionResult`

Lorsque vous exÃ©cutez une commande `extract`, le terminal affiche :

```
Result: positive (100%) | Method: Rules | Confidence: 0.95 | Energy: 0.000001 kWh
```

| Champ | InterprÃ©tation |
|---|---|
| `Result` | La valeur clinique extraite. `None` signifie qu'aucune extraction n'a abouti. |
| `Method` | La source qui a rÃ©pondu. RÃ¨gles = niveau 1, Transformer = niveau 2-3, LLM = niveau 4. |
| `Confidence` | FiabilitÃ© auto-Ã©valuÃ©e. >= 0.7 = acceptÃ© directement ; < 0.7 = dÃ©clenche un fallback cascade. |
| `Energy` | Empreinte Ã©nergÃ©tique en kWh. Comparer : RÃ¨gles ~ 1e-6 vs LLM ~ 1e-2 (ratio x10 000). |

### 9.2 Lire le `decision_config.json`

Chaque entrÃ©e du fichier contient :
- **`method`** : La feuille assignÃ©e par l'arbre (ex: `"FEUILLE ML LÃ‰GER PAR DÃ‰FAUT"`)
- **`metrics`** : Les scores bruts ayant conduit Ã  cette dÃ©cision
- **`justification`** : Explication en langage naturel
- **`trace`** : Le chemin exact parcouru dans l'arbre (liste de noeuds traversÃ©s)

Exemple d'entrÃ©e :

```json
{
  "Estrogen_receptor": {
    "metrics": {
      "Te": 43.75,
      "He": 97.44,
      "Freq": 0.0011,
      "Feas": 0.542,
      "DomainShift": 0.0,
      "LLM_Necessity": 0.3
    },
    "method": "FEUILLE ML LÃ‰GER PAR DÃ‰FAUT",
    "justification": "Te moyen (43.8%), Freq >= seuil mais Yield insuffisant...",
    "trace": [
      "Te < TE_HIGH (70.0)",
      "Te >= TE_MED (40.0)",
      "Freq >= FREQ_MIN (0.001)",
      "Yield < YIELD_HIGH (0.75)",
      "Feas < FEAS_NER (0.6)",
      "LLM_Necessity < LLM_NEC_HIGH (0.5)",
      "Freq >= FREQ_MIN -> FEUILLE ML LÃ‰GER PAR DÃ‰FAUT"
    ]
  }
}
```

### 9.3 Lire les figures de `Results/figures/`

| Figure | Ce qu'elle montre | Comment la lire |
|---|---|---|
| `fig1_performance_heatmap.png` | Matrice F1 par biomarqueur x mÃ©thode | Plus le bleu est foncÃ©, meilleur est le F1 |
| `fig2_explainability_barplot.png` | Score d'explicabilitÃ© par mÃ©thode | RÃ¨gles=1.0, LLM=0.1 |
| `fig3_energy_log_barplot.png` | Ã‰nergie par mÃ©thode (Ã©chelle log) | Chaque barre montre le coÃ»t kWh ; l'Ã©chelle log rÃ©vÃ¨le les ordres de grandeur |
| `fig4_pareto_scatter.png` | Front de Pareto F1 vs Ã‰nergie | Les points en haut Ã  gauche sont Pareto-optimaux (haute perf, basse Ã©nergie) |
| `fig5_trilemma_radar.png` | Trilemme Performance/ExplicabilitÃ©/FrugalitÃ© | L'aire couverte par chaque mÃ©thode visualise le compromis global |
| `decision_tree.png` | L'arbre de dÃ©cision global | Les noeuds = tests sur les mÃ©triques ; les feuilles = mÃ©thode assignÃ©e |
| `decision_tree_visualization.png` | Graphe d'assignation entitÃ©s vers feuilles | Chaque biomarqueur est reliÃ© Ã  sa feuille rÃ©sultante |
| `CHIR_F1_per_entity.png` | F1 par entitÃ© sur corpus chirurgical | Identifie les entitÃ©s difficiles pour le sous-corpus CHIR |
| `RCP_F1_per_entity.png` | F1 par entitÃ© sur corpus RCP | MÃªme chose pour le sous-corpus RCP |
| `confusion_matrix.png` | Matrice de confusion NER | Vraies vs prÃ©dites pour chaque label d'entitÃ© |
| `convergence_pie_chart.png` | Concordance arbre/empirique | Le % de vert = entitÃ©s oÃ¹ l'arbre et REST sont d'accord |
| `metrics_heatmap.png` | Heatmap des 7 mÃ©triques de l'arbre | Vue synthÃ©tique de toutes les mÃ©triques par biomarqueur |

### 9.4 InterprÃ©ter la concordance REST

Le fichier `Results/REST_results/rest_validation_summary.csv` donne le taux de concordance entre :
- **Top-Down** (dÃ©cision thÃ©orique de l'arbre) : basÃ©e sur les mÃ©triques calculÃ©es
- **Bottom-Up** (dÃ©cision empirique REST) : basÃ©e sur les performances rÃ©elles observÃ©es

Un taux > 90% signifie que l'arbre est bien calibrÃ©. Un taux plus bas pointe les entitÃ©s nÃ©cessitant un recalibrage des seuils.

### 9.5 InterprÃ©ter le score composite

Le score composite `C` normalise trois dimensions sur [0, 1] :

```
C = 0.4 * F1 + 0.3 * ExplicabilitÃ© + 0.3 * (1 - E_norm)
```

- Un `C` proche de 1.0 indique une mÃ©thode trÃ¨s performante, explicable et frugale (utopique)
- En pratique, les RÃ¨gles obtiennent un C Ã©levÃ© pour les entitÃ©s structurÃ©es (F1 bon + Expl=1 + E~0)
- Les LLM obtiennent un C plus bas (F1 excellent mais Expl=0.1 et E Ã©levÃ©)
- Le front de Pareto (`fig4_pareto_scatter.png`) visualise les solutions non-dominÃ©es

---

## 10. Seuils de l'arbre et justification scientifique

Les seuils suivants sont dÃ©finis dans `E_creation_arbre_decision.py` et justifiÃ©s dans `THRESHOLDS_JUSTIFICATION.md` :

| Seuil | Valeur | RÃ´le dans l'arbre | Justification |
|---|---|---|---|
| `TE_HIGH` | 70.0 | Te >= 70% : structure stable pour Regex | >70% des occurrences suivent un nombre restreint de motifs ; les Regex capturent le gros du volume |
| `TE_MED` | 40.0 | 40% <= Te < 70% : zone intermÃ©diaire ML | Structure partiellement prÃ©visible ; les modÃ¨les ML lÃ©gers compensent la variabilitÃ© |
| `HE_HIGH` | 70.0 | He >= 70% : vocabulaire restreint | Confirme que le lexique est maintenable manuellement |
| `RISK_HIGH` | 0.5 | R >= 0.5 : plus de 50% de contextes ambigus | Les nÃ©gations et incertitudes rendent les Regex dangereuses (faux positifs) |
| `FREQ_MIN` | 0.001 | Freq >= 0.1% : volume suffisant pour ML | En dessous, le surapprentissage est quasi certain |
| `YIELD_HIGH` | 0.75 | Yield >= 75% : RÃ¨gles dÃ©jÃ  performantes | Au-dessus de ce seuil, le coÃ»t d'un modÃ¨le ML n'est pas justifiÃ© |
| `FEAS_NER` | 0.6 | Feas >= 0.6 : NER techniquement praticable | Score agrÃ©gÃ© de compatibilitÃ© des donnÃ©es avec un pipeline NER |
| `DOMAIN_SHIFT_MAX` | 0.5 | DomainShift < 0.5 : pas de dÃ©rive de domaine | Au-dessus : les modÃ¨les prÃ©-entraÃ®nÃ©s ne transfÃ¨rent pas bien |
| `LLM_NEC_HIGH` | 0.5 | LLM_Necessity >= 0.5 : complexitÃ© justifie un LLM | Score composite intÃ©grant raretÃ©, ambiguÃ¯tÃ© et hÃ©tÃ©rogÃ©nÃ©itÃ© |

---

## 11. DÃ©pendances logicielles

Le fichier `requirements.txt` dÃ©finit l'environnement reproductible :

| CatÃ©gorie | Packages | Usage dans DuraXELL |
|---|---|---|
| **Deep Learning** | `torch==2.2.1`, `transformers==4.38.2` | EntraÃ®nement et infÃ©rence NER (PubMedBERT) |
| **Data Science** | `pandas==2.2.1`, `numpy==1.26.4`, `scipy==1.12.0`, `scikit-learn==1.4.1` | Manipulation des mÃ©triques, calculs statistiques |
| **Visualisation** | `matplotlib==3.8.3`, `seaborn==0.13.2`, `plotly==5.19.0` | GÃ©nÃ©ration des figures (heatmaps, radar, scatter) |
| **NLP** | `nltk==3.8.1`, `fr_core_news_sm` (spaCy), `python-Levenshtein==0.25.0`, `Unidecode==1.3.8` | Tokenisation, distance d'Ã©dition, normalisation |
| **NER spÃ©cifique** | `datasets==2.18.0`, `seqeval==1.2.2` | Chargement donnÃ©es HuggingFace, Ã©valuation sÃ©quentielle |
| **FrugalitÃ©** | `eco2ai==0.3.8` | Mesure de la consommation Ã©nergÃ©tique (kWh, CO2) |
| **QualitÃ©** | `pytest==8.1.1`, `black==24.2.0`, `isort==5.13.2` | Tests, formatage, tri des imports |
| **Notebooks** | `nbformat==5.9.2`, `ipywidgets==8.1.2`, `ipython==8.22.2` | Support Jupyter |
| **API** | `système==1.13.3` | Client optionnel pour LLM via API |
| **Graphes** | `networkx==3.2.1` | Visualisation de l'arbre de dÃ©cision |

---

## 12. ProblÃ¨mes connus et rÃ©solutions

### 12.1 Crash Eco2AI / Pandas (`LossySetitemError`)

**SymptÃ´me :** `tracker.stop()` lÃ¨ve une `TypeError` ou `LossySetitemError` car Eco2AI insÃ¨re des valeurs `'N/A'` (type `str`) dans des colonnes numÃ©riques d'un DataFrame Pandas.

**Solution dÃ©ployÃ©e :** Les fichiers `fix_trackers.py` et `fix_trackers2.py` encapsulent systÃ©matiquement `tracker.stop()` dans un bloc `try/except` :

```python
if HAS_ECO2AI:
    try:
        tracker.stop()
    except Exception:
        pass  # La tÃ©lÃ©mÃ©trie ne doit jamais bloquer le pipeline clinique
```

### 12.2 DÃ©calage vocabulaire Arbre / Orchestrateur

**SymptÃ´me :** L'arbre produit des labels comme `"FEUILLE NER Ã€ BASE DE RÃˆGLES"` mais l'orchestrateur attendait `"REGLES"`. Aucun modÃ¨le ne se dÃ©clenche correctement.

**Solution dÃ©ployÃ©e :** Les fichiers `fix_orch.py`, `fix_bridge.py`, `fix_demo.py`, `fix_scorer.py` ont alignÃ© toutes les listes de mapping dans `cascade_orchestrator.py`, `rest_decision_bridge.py`, `demo_rest.py` et `E_composite_scorer.py` pour reconnaÃ®tre les deux vocabulaires (ancien et nouveau).

### 12.3 Corruption de l'environnement `.venv`

**SymptÃ´me :** `ModuleNotFoundError: No module named 'matplotlib.backends.registry'` empÃªchant l'exÃ©cution des tests.

**Solution :** Purge des reliquats d'installation (`~atplotlib*/`) et rÃ©installation forcÃ©e :

```powershell
Remove-Item -Recurse -Force ".venv\Lib\site-packages\~*"
pip install matplotlib pandas numpy --force-reinstall
```

---

## 13. Glossaire

| Terme | DÃ©finition |
|---|---|
| **BRAT** | Outil d'annotation textuelle ; format `.ann` associant des spans de texte Ã  des labels d'entitÃ©s |
| **Cascade** | MÃ©canisme d'escalade : si le niveau n Ã©choue, on passe au n+1 (plus puissant mais plus coÃ»teux) |
| **CHIR** | Comptes-rendus de chirurgie (sous-corpus du cancer du sein) |
| **CRF** | Conditional Random Field â€” modÃ¨le sÃ©quentiel lÃ©ger pour l'Ã©tiquetage de sÃ©quences |
| **Eco2AI** | Librairie Python de mesure de la consommation Ã©nergÃ©tique (kWh, CO2) des calculs |
| **ER** | Estrogen Receptor â€” rÃ©cepteur aux oestrogÃ¨nes (biomarqueur clÃ© du cancer du sein) |
| **ExtractionResult** | DataClass Python contenant le rÃ©sultat d'une extraction (valeur, confiance, Ã©nergie, mÃ©thode) |
| **F1-Score** | Moyenne harmonique de la prÃ©cision et du rappel ; mÃ©trique standard en NER |
| **Feuille** | Noeud terminal de l'arbre de dÃ©cision, associÃ© Ã  une mÃ©thode d'extraction |
| **Gold Standard (GS)** | Annotations de rÃ©fÃ©rence produites par des experts humains |
| **HER2** | Human Epidermal Growth Factor Receptor 2 â€” biomarqueur oncologique |
| **IHC** | Immunohistochimie â€” technique de laboratoire pour caractÃ©riser HER2 |
| **FISH** | Fluorescence In Situ Hybridization â€” technique complÃ©mentaire pour HER2 |
| **Ki67** | Indice de prolifÃ©ration cellulaire |
| **LLM** | Large Language Model (GPT, Llama, etc.) |
| **NER** | Named Entity Recognition â€” extraction d'entitÃ©s nommÃ©es |
| **Pareto** | Front de Pareto : ensemble des solutions non dominÃ©es dans un espace multicritÃ¨re |
| **PR** | Progesterone Receptor â€” rÃ©cepteur Ã  la progestÃ©rone |
| **PubMedBERT** | ModÃ¨le Transformer prÃ©-entraÃ®nÃ© sur la littÃ©rature biomÃ©dicale |
| **RCP** | RÃ©union de Concertation Pluridisciplinaire (sous-corpus du cancer du sein) |
| **REST** | Interface de validation croisÃ©e (Representational State Transfer) |
| **Sweep** | Recherche systÃ©matique d'hyperparamÃ¨tres sur une grille |
| **Te** | TemplatabilitÃ© : stabilitÃ© structurelle d'une entitÃ© (% de motifs rÃ©currents) |
| **He** | HomogÃ©nÃ©itÃ© : restriction du vocabulaire utilisÃ© pour une entitÃ© |
| **R** | Risque contextuel : probabilitÃ© de contextes ambigus (nÃ©gations, incertitudes) |
| **Freq** | FrÃ©quence d'apparition de l'entitÃ© dans le corpus |
| **Yield** | Rendement d'annotation : F1 des rÃ¨gles vs Gold Standard |
| **Feas** | FaisabilitÃ© NER : capacitÃ© d'un modÃ¨le ML Ã  traiter une entitÃ© donnÃ©e |

---

> *Ce rapport constitue une documentation intÃ©grale et autosuffisante du projet DuraXELL. Chaque composant, chaque fichier, chaque flux de donnÃ©es a Ã©tÃ© documentÃ© avec une rigueur compatible avec les exigences d'une habilitation Ã  diriger des recherches (HDR).*
>
> *Fin du rapport â€” FÃ©vrier 2026*
