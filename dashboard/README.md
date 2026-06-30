# demne Dashboard

Ce dossier contient l'interface graphique interactive de supervision NLP pour le projet de recherche **demne*.
L'application Streamlit permet d'analyser les métriques de second niveau (L2) et d'agir sur le routage en cascade (Rules -> Transformers -> LLM) selon les scores calculés ou modifiables par l'utilisateur.

---

## Table des matières & Interfaces

1. ** Dashboard Métriques** (1_Dashboard_Metriques.py)
   - Vue analytique de l'algorithme "Cascade Frugale".
   - **Édition Temps Réel** : Le tableau des métriques permet à l'utilisateur de cliquer et de modifier manuellement les scores (Te, He, R, Yield, etc.). Sitôt modifiés, les graphiques (Radar, Nuages de point) et la décision du modèle (Rules, LLM, etc.) sont recalculés.
   - Manipulation des Seuils (thresholds) par sliders pour comparer les modes de frugalité d'énergie.

2. ** Console CLI** (2_Console_CLI.py)
   - Simulateur et wrapper de l'application Terminal main.py.
   - Permet de lancer directement les commandes extract, batch, metrics et d'apercevoir les logs du terminal à même l'interface web.

3. ** REST Integration & Config** (3_REST_Integration.py)
   - Page dédiée à la gestion d'exports de configuration L2 Json.
   - Permet de filtrer / exclure des entités spécifiques du Dashboard sans les perdre.

4. ** Serveur REST & Jupyter** (4_Notebook_REST.py)
   - Panneau de commande de l'API de traitement de rapport clinique (DuraXell REST).
   - Intègre l'ordinateur Jupyter (REST.ipynb) directement au sein d'une IFrame pour des tests interactifs croisés sur le même serveur web qu'un développeur. 

---

##  Lancement Optimal

1. **Activation de l'environnement virtuel** :
   Veillez à ce que vos libraires soient à jour via :
   \\\dash
   pip install -r dashboard/requirements.txt
   \\\
   *Astuce:* Installez Jupyter (pip install jupyter) si vous comptez lancer l'IFrame du Notebook interactif.

2. **Démarrage de l'Application** :
   Ouvrez un terminal dans le dossier \dashboard\ :
   \\\dash
   cd dashboard    
   streamlit run app.py
   \\\

3. **Port par défaut** : L'interface sera déployée à l'adresse \http://localhost:8501\.

---

## Modifier les métriques interactives (Demo/Playground)

Sur la page ** Dashboard Métriques L2 **, cherchez le composant **Résumé analytique (Éditable)**.
Vous pouvez cliquer sur n'importe quelle cellule du tableau (à part la colonne d'Entité et de Décision) :
- Passez par exemple la valeur \Yield\ ou le \R\ d'un biomarqueur (Ex: Ki67). 
- Validez l'entrée (Touche "Entrée" du clavier) et vous verrez immédiatement l'entité être rétrogradée au niveau LLM ou passer au niveau Rule. Les graphiques inférieurs capteront l'itération.

Pour l'API et le Notebook, allez en **Serveur REST & Jupyter**, pressez "Lancer API" et/ou "Lancer Jupyter", l'Iframe se chargera alors interactivement de vous afficher l'environnement.
