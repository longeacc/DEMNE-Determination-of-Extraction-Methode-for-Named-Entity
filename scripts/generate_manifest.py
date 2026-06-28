"""Generate the DEMNE feature manifest as a PDF (docs/DEMNE_Manifest.pdf)."""

from __future__ import annotations

from datetime import date
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "docs" / "DEMNE_Manifest.pdf"
OUT.parent.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Styles
# ---------------------------------------------------------------------------
styles = getSampleStyleSheet()
H1 = ParagraphStyle("H1", parent=styles["Heading1"], fontSize=20, spaceAfter=12,
                    textColor=colors.HexColor("#1F3A93"))
H2 = ParagraphStyle("H2", parent=styles["Heading2"], fontSize=14, spaceBefore=14,
                    spaceAfter=6, textColor=colors.HexColor("#26577C"))
H3 = ParagraphStyle("H3", parent=styles["Heading3"], fontSize=12, spaceBefore=8,
                    spaceAfter=4, textColor=colors.HexColor("#444"))
BODY = ParagraphStyle("Body", parent=styles["BodyText"], fontSize=10, leading=14,
                      alignment=TA_LEFT)
CODE = ParagraphStyle("Code", parent=styles["Code"], fontSize=8.5, leading=11,
                      backColor=colors.HexColor("#F4F4F4"),
                      borderColor=colors.HexColor("#D0D0D0"), borderWidth=0.5,
                      borderPadding=4, leftIndent=4, rightIndent=4,
                      spaceBefore=4, spaceAfter=6)
NOTE = ParagraphStyle("Note", parent=BODY, fontSize=9,
                      textColor=colors.HexColor("#555"), leftIndent=10)


def P(text: str) -> Paragraph:
    return Paragraph(text, BODY)


def code(text: str) -> Paragraph:
    # ReportLab Paragraph treats <br/> and uses XML — escape & and <
    safe = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    safe = safe.replace("\n", "<br/>")
    return Paragraph(f"<font face='Courier'>{safe}</font>", CODE)


def make_table(rows, col_widths=None, header=True):
    t = Table(rows, colWidths=col_widths, repeatRows=1 if header else 0)
    style = [
        ("FONT", (0, 0), (-1, -1), "Helvetica", 9),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#888")),
        ("INNERGRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#CCC")),
        ("LEFTPADDING", (0, 0), (-1, -1), 5),
        ("RIGHTPADDING", (0, 0), (-1, -1), 5),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]
    if header:
        style += [
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1F3A93")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONT", (0, 0), (-1, 0), "Helvetica-Bold", 9),
        ]
    t.setStyle(TableStyle(style))
    return t


# ---------------------------------------------------------------------------
# Content
# ---------------------------------------------------------------------------
story = []

# Cover
story += [
    Paragraph("DEMNE — Manifeste des fonctionnalités", H1),
    P("<b>Determination of Extraction Method for Named Entity</b> — pipeline "
      "NLP cascadé Règles → ML → LLM pour l'extraction de biomarqueurs "
      "oncologiques sur comptes-rendus cliniques."),
    Spacer(1, 6),
    P(f"<b>Version</b> : DuraXELL 2.0   |   <b>Date</b> : {date.today().isoformat()}   "
      f"|   <b>Auteur</b> : Clément Longeac (ESIEE Paris)"),
    Spacer(1, 12),
]

# Section 1 — Architecture
story += [Paragraph("1. Architecture du dépôt", H2)]
story += [code(
    "DEMNE-.../\n"
    "├── main.py                       # Point d'entrée CLI unifié\n"
    "├── data/\n"
    "│   ├── decision_config.json      # Routage entité → méthode (généré)\n"
    "│   └── ESMO2025/                 # Corpus BRAT (CHIR, RCP, sein, prostate)\n"
    "├── src/duraxell/\n"
    "│   ├── E_templatability.py       # Score Te (templatabilité, JSON)\n"
    "│   ├── E_homogeneity.py          # Score He (homogénéité, CSV)\n"
    "│   ├── E_frequency.py            # Score Freq (fréquence, CSV)\n"
    "│   ├── E_risk_context.py         # Score R (risque contextuel, CSV)\n"
    "│   ├── E_feasibility_NER.py      # Score Feas (CSV)\n"
    "│   ├── E_creation_arbre_decision.py  # Arbre de décision, écrit\n"
    "│   │                              decision_config.json\n"
    "│   ├── visualize_decision_tree.py\n"
    "│   └── REST_interface/\n"
    "│        ├── REST.ipynb           # Notebook démo Voila\n"
    "│        └── demo_rest.py         # Serveur FastAPI démo\n"
    "├── dashboard/                    # Application Streamlit (4 pages)\n"
    "│   ├── app.py\n"
    "│   ├── pages/1_Dashboard_Metriques.py\n"
    "│   ├── pages/2_Analyse_Corpus.py\n"
    "│   ├── pages/3_REST_Integration.py\n"
    "│   ├── pages/4_Notebook_REST.py\n"
    "│   └── core/                     # Brat parser, metrics, routing\n"
    "├── scripts/\n"
    "│   └── generate_manifest.py\n"
    "├── tests/                        # Tests unitaires\n"
    "├── notebooks/commandes.ipynb     # Notebook d'exploration\n"
    "├── Results/                      # Sorties : *.csv, *.json\n"
    "│   ├── templatability_analysis.json\n"
    "│   ├── homogeneity_analysis.csv\n"
    "│   ├── frequency_analysis.csv\n"
    "│   ├── risk_context_analysis.csv\n"
    "│   ├── ner_feasibility_analysis.csv\n"
    "│   └── decision_summary.csv      # Résumé final (TSV)\n"
    "├── logs/output_decision.txt      # Trace humaine arbre\n"
    "├── .venv/                        # Python 3.12 + pandas, torch, voila…\n"
    "└── pyproject.toml")]

# Section 2 — Quickstart
story += [PageBreak(), Paragraph("2. Quickstart", H2)]
story += [P("<b>Environnement recommandé</b> : Python 3.12 via le venv embarqué "
            "(<font face='Courier'>.venv/Scripts/python.exe</font>) qui contient "
            "<font face='Courier'>pandas</font>, <font face='Courier'>torch</font>, "
            "<font face='Courier'>voila</font>, <font face='Courier'>eco2ai</font>, "
            "<font face='Courier'>transformers</font>.")]
story += [code(
    "# Aide générale\n"
    "python main.py --help\n\n"
    "# Pipeline complet (métriques + arbre + CSV résumé)\n"
    "python main.py evaluate\n\n"
    "# Pour les commandes nécessitant pandas/torch/voila :\n"
    "./.venv/Scripts/python.exe main.py notebook --port 8888")]

# Section 3 — Subcommands table
story += [Paragraph("3. Carte des sous-commandes", H2)]
sub_rows = [
    ["Sous-commande", "Rôle", "Dépendances clés"],
    ["info", "Diagnostic environnement (version, chemins, config)", "stdlib"],
    ["metrics", "Lance E_templatability/Homogeneity/Frequency/Risk/Feasibility", "stdlib"],
    ["tree", "Construit l'arbre de décision + export decision_summary.csv", "stdlib"],
    ["evaluate", "Pipeline complet : metrics → tree → CSV", "stdlib"],
    ["dashboard", "Mirror page Streamlit 1 : preset FRUGAL/QUALITY + routage", "stdlib"],
    ["corpus", "Mirror page Streamlit 2 : stats corpus BRAT", "stdlib (st stub)"],
    ["rest-config", "Mirror page Streamlit 3 : export/import config JSON", "stdlib"],
    ["notebook", "Mirror page Streamlit 4 : lance Voila sur REST.ipynb + API", "voila"],
    ["rest", "Lance uniquement demo_rest.py (legacy)", "fastapi"],
    ["export-csv", "Régénère decision_summary.csv depuis decision_config.json", "stdlib"],
]
story += [make_table(sub_rows, col_widths=[3.2*cm, 9*cm, 4.5*cm])]

# Section 4 — Detailed examples per command
story += [PageBreak(), Paragraph("4. Référence détaillée des commandes", H2)]

CMD_DOC = [
    ("info", "Affiche la version, la racine du projet, les chemins GS/Pred par défaut, "
              "la liste des entités et l'état du fichier de configuration.",
     "python main.py info"),
    ("metrics", "Lance en séquence les 5 scorers : Te, He, Freq, R, Feas. "
                 "Chaque script écrit son CSV/JSON dans Results/. "
                 "--gs_dir / --pred_dir surchargent les chemins ESMO2025 par défaut.",
     "python main.py metrics                            # corpus par défaut\n"
     "python main.py metrics \\\n"
     "  --gs_dir   data/ESMO2025/Breast/RCP/evaluation_set_breast_cancer_GS \\\n"
     "  --pred_dir data/ESMO2025/Breast/RCP/evaluation_set_breast_cancer_pred_rules"),
    ("tree", "Lit les CSVs de Results/, lance la k-fold de stabilité (k=3) puis "
              "écrit data/decision_config.json + logs/output_decision.txt. "
              "Régénère également Results/decision_summary.csv (TSV).",
     "python main.py tree                  # avec visualisation matplotlib\n"
     "python main.py tree --no-visualize   # batch / serveur sans X"),
    ("evaluate", "Raccourci : metrics + tree.",
     "python main.py evaluate --no-visualize"),
    ("dashboard", "Applique un preset de seuils (FRUGAL ou QUALITY) ou des seuils "
                   "personnalisés et ré-évalue l'arbre sur la config courante. "
                   "Mirror exact de la page Streamlit 1.",
     "python main.py dashboard --preset FRUGAL\n"
     "python main.py dashboard --preset QUALITY\n"
     "python main.py dashboard --te 0.15 --he 0.80 --r 0.20 --feas 0.55"),
    ("corpus", "Parse un dossier BRAT et affiche par entité : nombre d'occurrences "
                "et top-3 valeurs uniques.",
     "python main.py corpus\n"
     "python main.py corpus --path data/ESMO2025/Breast/CHIR/training_set_chir_GS"),
    ("rest-config", "Sérialise (ou recharge) la liste d'entités sélectionnées, les "
                     "seuils globaux et le routage par entité — format JSON commun "
                     "avec la page Streamlit 3.",
     "python main.py rest-config --export Results/config_export.json\n"
     "python main.py rest-config --import Results/config_export.json"),
    ("notebook", "Lance Voila sur src/duraxell/REST_interface/REST.ipynb (port 8888 "
                  "par défaut) et le serveur FastAPI demo_rest.py en parallèle. "
                  "Mirror exact de la page Streamlit 4.",
     "python main.py notebook                       # Voila + API\n"
     "python main.py notebook --notebook-only        # uniquement Voila\n"
     "python main.py notebook --api-only --port 9000"),
    ("rest", "Lance uniquement demo_rest.py (compatibilité avec versions antérieures).",
     "python main.py rest"),
    ("export-csv", "Régénère Results/decision_summary.csv sans relancer le pipeline. "
                    "Utile après édition manuelle de decision_config.json.",
     "python main.py export-csv"),
]
for name, desc, ex in CMD_DOC:
    story += [Paragraph(f"4.{CMD_DOC.index((name, desc, ex))+1} <font face='Courier'>{name}</font>", H3)]
    story += [P(desc)]
    story += [code(ex)]

# Section 5 — Metrics
story += [PageBreak(), Paragraph("5. Les 5 métriques de l'arbre", H2)]
metric_rows = [
    ["Métrique", "Sens", "Fichier source", "Formule clé"],
    ["Te (Templatability)",
     "Fraction d'occurrences capturables par une regex template",
     "E_templatability.py",
     "matches_template / total"],
    ["He (Homogeneity)",
     "Cohérence syntaxique du contexte gauche/droit d'une entité",
     "E_homogeneity.py",
     "1 − entropy(n-grams)"],
    ["Freq (Frequency)",
     "Densité de l'entité dans le corpus (occurrences / tokens)",
     "E_frequency.py",
     "count / corpus_tokens"],
    ["R (Risk Context)",
     "Présence de négations, incertitudes, contradictions autour de l'entité",
     "E_risk_context.py",
     "0.1·f_neg + 0.3·f_unc + 0.6·f_cont"],
    ["Feas (NER Feasibility)",
     "Faisabilité d'un modèle NER (volume + homogénéité)",
     "E_feasibility_NER.py",
     "0.2·min(1,Freq) + 0.2·He"],
]
story += [make_table(metric_rows, col_widths=[3.5*cm, 6.5*cm, 4*cm, 4*cm])]
story += [Spacer(1, 6),
          P("<b>Arbre de décision (E_creation_arbre_decision.py)</b> :")]
story += [code(
    "Te ≥ 0.10  ?\n"
    "  ├─ oui →  He ≥ 0.85  ?\n"
    "  │           ├─ oui →  R ≤ 0.25  ?\n"
    "  │           │           ├─ oui → RÈGLES\n"
    "  │           │           └─ non → Feas ≥ 0.20  ?\n"
    "  │           └─ non →  Feas ≥ 0.20  ?\n"
    "  └─ non →  Feas ≥ 0.20  ?\n"
    "                            ├─ oui → TBM (DrBERT)\n"
    "                            └─ non → LLM\n")]

# Section 6 — Presets dashboard
story += [Paragraph("6. Presets de seuils (commande dashboard)", H2)]
preset_rows = [
    ["Preset", "Te", "He", "R", "Feas", "Objectif"],
    ["FRUGAL", "0.10", "0.85", "0.25", "0.20",
     "Maximiser RÈGLES → énergie minimale, explicabilité maximale"],
    ["QUALITY", "0.25", "0.55", "0.15", "0.70",
     "Resserrer RÈGLES → recours plus fréquent à TBM/LLM"],
]
story += [make_table(preset_rows, col_widths=[2*cm, 1.3*cm, 1.3*cm, 1.3*cm, 1.3*cm, 7.7*cm])]

# Section 7 — Sorties
story += [Paragraph("7. Sorties produites", H2)]
out_rows = [
    ["Fichier", "Producteur", "Schéma / contenu"],
    ["Results/templatability_analysis.json", "E_templatability.py",
     "{entité: {templatability_score, count}}"],
    ["Results/homogeneity_analysis.csv", "E_homogeneity.py",
     "Entity, He_Score_Percent, …"],
    ["Results/frequency_analysis.csv", "E_frequency.py",
     "Entity, Frequency, Count, Per_1k_tokens, Strategy_Hint"],
    ["Results/risk_context_analysis.csv", "E_risk_context.py",
     "Entity, R_Score, Negation_Rate, Uncertainty_Rate, "
     "Contradiction_Rate, Count"],
    ["Results/ner_feasibility_analysis.csv", "E_feasibility_NER.py",
     "Entity, Feas_Score"],
    ["data/decision_config.json", "E_creation_arbre_decision.py",
     "{version, global_thresholds, entities: {…, method, justification, trace}}"],
    ["Results/decision_summary.csv", "main.py export-csv",
     "TSV : entity, Te, He, R, Freq, Feas, Method, Justification"],
    ["logs/output_decision.txt", "E_creation_arbre_decision.py",
     "Rapport humain : seuils + méthode + trace par entité"],
    ["Consumtion_of_Duraxell.csv", "eco2ai", "Empreinte CO₂ / énergie cumulée"],
]
story += [make_table(out_rows, col_widths=[6*cm, 4.5*cm, 7.5*cm])]

# Section 8 — Routage par défaut sur ESMO2025 / Breast / RCP
story += [PageBreak(), Paragraph("8. Routage de référence — Breast/RCP/GS", H2)]
story += [P("Résultat actuel du pipeline sur "
            "<font face='Courier'>evaluation_set_breast_cancer_GS</font> (95 docs, 55 643 mots) :")]
rout_rows = [
    ["Entité", "Te", "He", "R", "Freq", "Feas", "Méthode"],
    ["Estrogen_receptor",      "0.3040", "0.978", "0.059", "0.0028", "0.991", "RÈGLES"],
    ["Progesterone_receptor",  "0.2940", "0.964", "0.138", "0.0023", "0.986", "RÈGLES"],
    ["HER2_status",            "0.2410", "0.969", "0.076", "0.0015", "0.880", "RÈGLES"],
    ["HER2_IHC",               "0.1760", "0.958", "0.160", "0.0014", "0.839", "RÈGLES"],
    ["Ki67",                   "0.2690", "0.971", "0.064", "0.0021", "0.988", "RÈGLES"],
    ["HER2_FISH",              "0.1000", "0.500", "0.274", "0.0003", "0.296", "LLM"],
    ["Genetic_mutation",       "0.0000", "0.014", "0.000", "0.0000", "0.018", "LLM"],
]
story += [make_table(rout_rows, col_widths=[3.6*cm, 1.5*cm, 1.5*cm, 1.5*cm, 1.6*cm, 1.5*cm, 2*cm])]

# Section 9 — Parité dashboard ↔ CLI
story += [Paragraph("9. Parité Streamlit ↔ CLI", H2)]
par_rows = [
    ["Page Streamlit", "Sous-commande CLI équivalente"],
    ["1 — Dashboard Métriques (sliders + presets)", "main.py dashboard --preset FRUGAL|QUALITY"],
    ["2 — Analyse Corpus BRAT", "main.py corpus --path <dir>"],
    ["3 — REST Integration (export/import JSON)", "main.py rest-config --export|--import"],
    ["4 — Notebook & API REST (Voila)", "main.py notebook [--port 8888]"],
]
story += [make_table(par_rows, col_widths=[8*cm, 9*cm])]

# Section 10 — Notes opérationnelles
story += [Paragraph("10. Notes opérationnelles", H2)]
notes = [
    "Le venv embarqué (Python 3.12) contient pandas, torch 2.6+cu124, voila, transformers, "
    "eco2ai. Le Python système 3.14 suffit pour metrics/tree/evaluate/dashboard/corpus/"
    "rest-config (les imports lourds sont contournés par importlib).",
    "main.py force PYTHONIOENCODING=utf-8 et PYTHONUTF8=1 dans les sous-process → "
    "fin du UnicodeEncodeError cp1252 sous Windows.",
    "E_feasibility_NER.py calcule le score Feas à partir de Freq et He.",
    "L'arbre est calibré sur le corpus Breast/RCP. Pour valider la stabilité sur un autre "
    "corpus, lancer `evaluate` avec --gs_dir/--pred_dir adaptés.",
    "decision_config.json est ré-écrit à chaque `tree`/`evaluate`. Un export figé doit "
    "passer par `rest-config --export`.",
    "Suivi énergie : chaque script E_*.py démarre un Tracker eco2ai qui appondit dans "
    "Consumtion_of_Duraxell.csv.",
]
for n in notes:
    story += [Paragraph(f"• {n}", BODY), Spacer(1, 3)]

# Build
doc = SimpleDocTemplate(
    str(OUT), pagesize=A4,
    leftMargin=1.8*cm, rightMargin=1.8*cm,
    topMargin=1.8*cm, bottomMargin=1.8*cm,
    title="DEMNE — Manifeste des fonctionnalités",
    author="Clément Longeac",
)
doc.build(story)
print(f"PDF généré : {OUT}")
