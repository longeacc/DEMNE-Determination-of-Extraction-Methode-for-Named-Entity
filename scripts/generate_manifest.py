"""Generate the DEMNE feature manifest as a PDF (docs/DEMNE_Manifest.pdf)."""

from __future__ import annotations

from datetime import date
from pathlib import Path

try:
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
except ImportError as e:
    raise ImportError(
        "reportlab est requis pour gÃ©nÃ©rer le PDF : pip install DEMNE[reports]"
    ) from e

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "docs" / "DEMNE_Manifest.pdf"
OUT.parent.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Styles
# ---------------------------------------------------------------------------
styles = getSampleStyleSheet()
H1 = ParagraphStyle(
    "H1",
    parent=styles["Heading1"],
    fontSize=20,
    spaceAfter=12,
    textColor=colors.HexColor("#1F3A93"),
)
H2 = ParagraphStyle(
    "H2",
    parent=styles["Heading2"],
    fontSize=14,
    spaceBefore=14,
    spaceAfter=6,
    textColor=colors.HexColor("#26577C"),
)
H3 = ParagraphStyle(
    "H3",
    parent=styles["Heading3"],
    fontSize=12,
    spaceBefore=8,
    spaceAfter=4,
    textColor=colors.HexColor("#444"),
)
BODY = ParagraphStyle("Body", parent=styles["BodyText"], fontSize=10, leading=14, alignment=TA_LEFT)
CODE = ParagraphStyle(
    "Code",
    parent=styles["Code"],
    fontSize=8.5,
    leading=11,
    backColor=colors.HexColor("#F4F4F4"),
    borderColor=colors.HexColor("#D0D0D0"),
    borderWidth=0.5,
    borderPadding=4,
    leftIndent=4,
    rightIndent=4,
    spaceBefore=4,
    spaceAfter=6,
)
NOTE = ParagraphStyle(
    "Note", parent=BODY, fontSize=9, textColor=colors.HexColor("#555"), leftIndent=10
)


def paragraph(text: str) -> Paragraph:
    return Paragraph(text, BODY)


def code(text: str) -> Paragraph:
    # ReportLab Paragraph treats <br/> and uses XML â€” escape & and <
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
    Paragraph("DEMNE â€” Manifeste des fonctionnalitÃ©s", H1),
    paragraph(
        "<b>Determination of Extraction Method for Named Entity</b> â€” pipeline "
        "NLP cascadÃ© RÃ¨gles â†’ ML â†’ LLM pour l'extraction de biomarqueurs "
        "oncologiques sur comptes-rendus cliniques."
    ),
    Spacer(1, 6),
    paragraph(
        f"<b>Version</b> : DuraXELL 2.0   |   <b>Date</b> : {date.today().isoformat()}   "
        f"|   <b>Auteur</b> : ClÃ©ment Longeac (ESIEE Paris)"
    ),
    Spacer(1, 12),
]

# Section 1 â€” Architecture
story += [Paragraph("1. Architecture du dÃ©pÃ´t", H2)]
story += [
    code(
        "DEMNE-.../\n"
        "â”œâ”€â”€ main.py                       # Point d'entrÃ©e CLI unifiÃ©\n"
        "â”œâ”€â”€ data/\n"
        "â”‚   â”œâ”€â”€ decision_config.json      # Routage entitÃ© â†’ mÃ©thode (gÃ©nÃ©rÃ©)\n"
        "â”‚   â””â”€â”€ ESMO2025/                 # Corpus BRAT (CHIR, RCP, sein, prostate)\n"
        "â”œâ”€â”€ src/demne/\n"
        "â”‚   â”œâ”€â”€ E_templatability.py       # Score Te (templatabilitÃ©, JSON)\n"
        "â”‚   â”œâ”€â”€ E_homogeneity.py          # Score He (homogÃ©nÃ©itÃ©, CSV)\n"
        "â”‚   â”œâ”€â”€ E_frequency.py            # Score Freq (frÃ©quence, CSV)\n"
        "â”‚   â”œâ”€â”€ E_risk_context.py         # Score R (risque contextuel, CSV)\n"
        "â”‚   â”œâ”€â”€ E_feasibility_NER.py      # Score Feas (CSV)\n"
        "â”‚   â”œâ”€â”€ E_creation_arbre_decision.py  # Arbre de dÃ©cision, Ã©crit\n"
        "â”‚   â”‚                              decision_config.json\n"
        "â”‚   â”œâ”€â”€ visualize_decision_tree.py\n"
        "â”‚   â””â”€â”€ REST_interface/\n"
        "â”‚        â”œâ”€â”€ REST.ipynb           # Notebook dÃ©mo Voila\n"
        "â”‚        â””â”€â”€ demo_rest.py         # Serveur FastAPI dÃ©mo\n"
        "â”œâ”€â”€ dashboard/                    # Application Streamlit (4 pages)\n"
        "â”‚   â”œâ”€â”€ app.py\n"
        "â”‚   â”œâ”€â”€ pages/1_Dashboard_Metriques.py\n"
        "â”‚   â”œâ”€â”€ pages/2_Analyse_Corpus.py\n"
        "â”‚   â”œâ”€â”€ pages/3_REST_Integration.py\n"
        "â”‚   â”œâ”€â”€ pages/4_Notebook_REST.py\n"
        "â”‚   â””â”€â”€ core/                     # Brat parser, metrics, routing\n"
        "â”œâ”€â”€ scripts/\n"
        "â”‚   â””â”€â”€ generate_manifest.py\n"
        "â”œâ”€â”€ tests/                        # Tests unitaires\n"
        "â”œâ”€â”€ notebooks/commandes.ipynb     # Notebook d'exploration\n"
        "â”œâ”€â”€ Results/                      # Sorties : *.csv, *.json\n"
        "â”‚   â”œâ”€â”€ templatability_analysis.json\n"
        "â”‚   â”œâ”€â”€ homogeneity_analysis.csv\n"
        "â”‚   â”œâ”€â”€ frequency_analysis.csv\n"
        "â”‚   â”œâ”€â”€ risk_context_analysis.csv\n"
        "â”‚   â”œâ”€â”€ ner_feasibility_analysis.csv\n"
        "â”‚   â””â”€â”€ decision_summary.csv      # RÃ©sumÃ© final (TSV)\n"
        "â”œâ”€â”€ logs/output_decision.txt      # Trace humaine arbre\n"
        "â”œâ”€â”€ .venv/                        # Python 3.12 + pandas, torch, voilaâ€¦\n"
        "â””â”€â”€ pyproject.toml"
    )
]

# Section 2 â€” Quickstart
story += [PageBreak(), Paragraph("2. Quickstart", H2)]
story += [
    paragraph(
        "<b>Environnement recommandÃ©</b> : Python 3.12 via le venv embarquÃ© "
        "(<font face='Courier'>.venv/Scripts/python.exe</font>) qui contient "
        "<font face='Courier'>pandas</font>, <font face='Courier'>torch</font>, "
        "<font face='Courier'>voila</font>, <font face='Courier'>eco2ai</font>, "
        "<font face='Courier'>transformers</font>."
    )
]
story += [
    code(
        "# Aide gÃ©nÃ©rale\n"
        "python main.py --help\n\n"
        "# Pipeline complet (mÃ©triques + arbre + CSV rÃ©sumÃ©)\n"
        "python main.py evaluate\n\n"
        "# Pour les commandes nÃ©cessitant pandas/torch/voila :\n"
        "./.venv/Scripts/python.exe main.py notebook --port 8888"
    )
]

# Section 3 â€” Subcommands table
story += [Paragraph("3. Carte des sous-commandes", H2)]
sub_rows = [
    ["Sous-commande", "RÃ´le", "DÃ©pendances clÃ©s"],
    ["info", "Diagnostic environnement (version, chemins, config)", "stdlib"],
    ["metrics", "Lance E_templatability/Homogeneity/Frequency/Risk/Feasibility", "stdlib"],
    ["tree", "Construit l'arbre de dÃ©cision + export decision_summary.csv", "stdlib"],
    ["evaluate", "Pipeline complet : metrics â†’ tree â†’ CSV", "stdlib"],
    ["dashboard", "Mirror page Streamlit 1 : preset FRUGAL/QUALITY + routage", "stdlib"],
    ["corpus", "Mirror page Streamlit 2 : stats corpus BRAT", "stdlib (st stub)"],
    ["rest-config", "Mirror page Streamlit 3 : export/import config JSON", "stdlib"],
    ["notebook", "Mirror page Streamlit 4 : lance Voila sur REST.ipynb + API", "voila"],
    ["rest", "Lance uniquement demo_rest.py (legacy)", "fastapi"],
    ["export-csv", "RÃ©gÃ©nÃ¨re decision_summary.csv depuis decision_config.json", "stdlib"],
]
story += [make_table(sub_rows, col_widths=[3.2 * cm, 9 * cm, 4.5 * cm])]

# Section 4 â€” Detailed examples per command
story += [PageBreak(), Paragraph("4. RÃ©fÃ©rence dÃ©taillÃ©e des commandes", H2)]

CMD_DOC = [
    (
        "info",
        "Affiche la version, la racine du projet, les chemins GS/Pred par dÃ©faut, "
        "la liste des entitÃ©s et l'Ã©tat du fichier de configuration.",
        "python main.py info",
    ),
    (
        "metrics",
        "Lance en sÃ©quence les 5 scorers : Te, He, Freq, R, Feas. "
        "Chaque script Ã©crit son CSV/JSON dans Results/. "
        "--gs_dir / --pred_dir surchargent les chemins ESMO2025 par dÃ©faut.",
        "python main.py metrics                            # corpus par dÃ©faut\n"
        "python main.py metrics \\\n"
        "  --gs_dir   data/ESMO2025/Breast/RCP/evaluation_set_breast_cancer_GS \\\n"
        "  --pred_dir data/ESMO2025/Breast/RCP/evaluation_set_breast_cancer_pred_rules",
    ),
    (
        "tree",
        "Lit les CSVs de Results/, lance la k-fold de stabilitÃ© (k=3) puis "
        "Ã©crit data/decision_config.json + logs/output_decision.txt. "
        "RÃ©gÃ©nÃ¨re Ã©galement Results/decision_summary.csv (TSV).",
        "python main.py tree                  # avec visualisation matplotlib\n"
        "python main.py tree --no-visualize   # batch / serveur sans X",
    ),
    ("evaluate", "Raccourci : metrics + tree.", "python main.py evaluate --no-visualize"),
    (
        "dashboard",
        "Applique un preset de seuils (FRUGAL ou QUALITY) ou des seuils "
        "personnalisÃ©s et rÃ©-Ã©value l'arbre sur la config courante. "
        "Mirror exact de la page Streamlit 1.",
        "python main.py dashboard --preset FRUGAL\n"
        "python main.py dashboard --preset QUALITY\n"
        "python main.py dashboard --te 0.15 --he 0.80 --r 0.20 --feas 0.55",
    ),
    (
        "corpus",
        "Parse un dossier BRAT et affiche par entitÃ© : nombre d'occurrences "
        "et top-3 valeurs uniques.",
        "python main.py corpus\n"
        "python main.py corpus --path data/ESMO2025/Breast/CHIR/training_set_chir_GS",
    ),
    (
        "rest-config",
        "SÃ©rialise (ou recharge) la liste d'entitÃ©s sÃ©lectionnÃ©es, les "
        "seuils globaux et le routage par entitÃ© â€” format JSON commun "
        "avec la page Streamlit 3.",
        "python main.py rest-config --export Results/config_export.json\n"
        "python main.py rest-config --import Results/config_export.json",
    ),
    (
        "notebook",
        "Lance Voila sur src/demne/REST_interface/REST.ipynb (port 8888 "
        "par dÃ©faut) et le serveur FastAPI demo_rest.py en parallÃ¨le. "
        "Mirror exact de la page Streamlit 4.",
        "python main.py notebook                       # Voila + API\n"
        "python main.py notebook --notebook-only        # uniquement Voila\n"
        "python main.py notebook --api-only --port 9000",
    ),
    (
        "rest",
        "Lance uniquement demo_rest.py (compatibilitÃ© avec versions antÃ©rieures).",
        "python main.py rest",
    ),
    (
        "export-csv",
        "RÃ©gÃ©nÃ¨re Results/decision_summary.csv sans relancer le pipeline. "
        "Utile aprÃ¨s Ã©dition manuelle de decision_config.json.",
        "python main.py export-csv",
    ),
]
for name, desc, ex in CMD_DOC:
    story += [
        Paragraph(f"4.{CMD_DOC.index((name, desc, ex))+1} <font face='Courier'>{name}</font>", H3)
    ]
    story += [paragraph(desc)]
    story += [code(ex)]

# Section 5 â€” Metrics
story += [PageBreak(), Paragraph("5. Les 5 mÃ©triques de l'arbre", H2)]
metric_rows = [
    ["MÃ©trique", "Sens", "Fichier source", "Formule clÃ©"],
    [
        "Te (Templatability)",
        "Fraction d'occurrences capturables par une regex template",
        "E_templatability.py",
        "matches_template / total",
    ],
    [
        "He (Homogeneity)",
        "CohÃ©rence syntaxique du contexte gauche/droit d'une entitÃ©",
        "E_homogeneity.py",
        "1 âˆ’ entropy(n-grams)",
    ],
    [
        "Freq (Frequency)",
        "DensitÃ© de l'entitÃ© dans le corpus (occurrences / tokens)",
        "E_frequency.py",
        "count / corpus_tokens",
    ],
    [
        "R (Risk Context)",
        "PrÃ©sence de nÃ©gations, incertitudes, contradictions autour de l'entitÃ©",
        "E_risk_context.py",
        "0.1Â·f_neg + 0.3Â·f_unc + 0.6Â·f_cont",
    ],
    [
        "Feas (NER Feasibility)",
        "FaisabilitÃ© d'un modÃ¨le NER (volume + homogÃ©nÃ©itÃ©)",
        "E_feasibility_NER.py",
        "0.2Â·min(1,Freq) + 0.2Â·He",
    ],
]
story += [make_table(metric_rows, col_widths=[3.5 * cm, 6.5 * cm, 4 * cm, 4 * cm])]
story += [Spacer(1, 6), paragraph("<b>Arbre de dÃ©cision (E_creation_arbre_decision.py)</b> :")]
story += [
    code(
        "Te â‰¥ 0.10  ?\n"
        "  â”œâ”€ oui â†’  He â‰¥ 0.85  ?\n"
        "  â”‚           â”œâ”€ oui â†’  R â‰¤ 0.25  ?\n"
        "  â”‚           â”‚           â”œâ”€ oui â†’ RÃˆGLES\n"
        "  â”‚           â”‚           â””â”€ non â†’ Feas â‰¥ 0.20  ?\n"
        "  â”‚           â””â”€ non â†’  Feas â‰¥ 0.20  ?\n"
        "  â””â”€ non â†’  Feas â‰¥ 0.20  ?\n"
        "                            â”œâ”€ oui â†’ TBM (DrBERT)\n"
        "                            â””â”€ non â†’ LLM\n"
    )
]

# Section 6 â€” Presets dashboard
story += [Paragraph("6. Presets de seuils (commande dashboard)", H2)]
preset_rows = [
    ["Preset", "Te", "He", "R", "Feas", "Objectif"],
    [
        "FRUGAL",
        "0.10",
        "0.85",
        "0.25",
        "0.20",
        "Maximiser RÃˆGLES â†’ Ã©nergie minimale, explicabilitÃ© maximale",
    ],
    [
        "QUALITY",
        "0.25",
        "0.55",
        "0.15",
        "0.70",
        "Resserrer RÃˆGLES â†’ recours plus frÃ©quent Ã  TBM/LLM",
    ],
]
story += [
    make_table(preset_rows, col_widths=[2 * cm, 1.3 * cm, 1.3 * cm, 1.3 * cm, 1.3 * cm, 7.7 * cm])
]

# Section 7 â€” Sorties
story += [Paragraph("7. Sorties produites", H2)]
out_rows = [
    ["Fichier", "Producteur", "SchÃ©ma / contenu"],
    [
        "Results/templatability_analysis.json",
        "E_templatability.py",
        "{entitÃ©: {templatability_score, count}}",
    ],
    ["Results/homogeneity_analysis.csv", "E_homogeneity.py", "Entity, He_Score_Percent, â€¦"],
    [
        "Results/frequency_analysis.csv",
        "E_frequency.py",
        "Entity, Frequency, Count, Per_1k_tokens, Strategy_Hint",
    ],
    [
        "Results/risk_context_analysis.csv",
        "E_risk_context.py",
        "Entity, R_Score, Negation_Rate, Uncertainty_Rate, " "Contradiction_Rate, Count",
    ],
    ["Results/ner_feasibility_analysis.csv", "E_feasibility_NER.py", "Entity, Feas_Score"],
    [
        "data/decision_config.json",
        "E_creation_arbre_decision.py",
        "{version, global_thresholds, entities: {â€¦, method, justification, trace}}",
    ],
    [
        "Results/decision_summary.csv",
        "main.py export-csv",
        "TSV : entity, Te, He, R, Freq, Feas, Method, Justification",
    ],
    [
        "logs/output_decision.txt",
        "E_creation_arbre_decision.py",
        "Rapport humain : seuils + mÃ©thode + trace par entitÃ©",
    ],
    ["Consumtion_of_Duraxell.csv", "eco2ai", "Empreinte COâ‚‚ / Ã©nergie cumulÃ©e"],
]
story += [make_table(out_rows, col_widths=[6 * cm, 4.5 * cm, 7.5 * cm])]

# Section 8 â€” Routage par dÃ©faut sur ESMO2025 / Breast / RCP
story += [PageBreak(), Paragraph("8. Routage de rÃ©fÃ©rence â€” Breast/RCP/GS", H2)]
story += [
    paragraph(
        "RÃ©sultat actuel du pipeline sur "
        "<font face='Courier'>evaluation_set_breast_cancer_GS</font> (95 docs, 55 643 mots) :"
    )
]
rout_rows = [
    ["EntitÃ©", "Te", "He", "R", "Freq", "Feas", "MÃ©thode"],
    ["Estrogen_receptor", "0.3040", "0.978", "0.059", "0.0028", "0.991", "RÃˆGLES"],
    ["Progesterone_receptor", "0.2940", "0.964", "0.138", "0.0023", "0.986", "RÃˆGLES"],
    ["HER2_status", "0.2410", "0.969", "0.076", "0.0015", "0.880", "RÃˆGLES"],
    ["HER2_IHC", "0.1760", "0.958", "0.160", "0.0014", "0.839", "RÃˆGLES"],
    ["Ki67", "0.2690", "0.971", "0.064", "0.0021", "0.988", "RÃˆGLES"],
    ["HER2_FISH", "0.1000", "0.500", "0.274", "0.0003", "0.296", "LLM"],
    ["Genetic_mutation", "0.0000", "0.014", "0.000", "0.0000", "0.018", "LLM"],
]
story += [
    make_table(
        rout_rows, col_widths=[3.6 * cm, 1.5 * cm, 1.5 * cm, 1.5 * cm, 1.6 * cm, 1.5 * cm, 2 * cm]
    )
]

# Section 9 â€” ParitÃ© dashboard â†” CLI
story += [Paragraph("9. ParitÃ© Streamlit â†” CLI", H2)]
par_rows = [
    ["Page Streamlit", "Sous-commande CLI Ã©quivalente"],
    ["1 â€” Dashboard MÃ©triques (sliders + presets)", "main.py dashboard --preset FRUGAL|QUALITY"],
    ["2 â€” Analyse Corpus BRAT", "main.py corpus --path <dir>"],
    ["3 â€” REST Integration (export/import JSON)", "main.py rest-config --export|--import"],
    ["4 â€” Notebook & API REST (Voila)", "main.py notebook [--port 8888]"],
]
story += [make_table(par_rows, col_widths=[8 * cm, 9 * cm])]

# Section 10 â€” Notes opÃ©rationnelles
story += [Paragraph("10. Notes opÃ©rationnelles", H2)]
notes = [
    "Le venv embarquÃ© (Python 3.12) contient pandas, torch 2.6+cu124, voila, transformers, "
    "eco2ai. Le Python systÃ¨me 3.14 suffit pour metrics/tree/evaluate/dashboard/corpus/"
    "rest-config (les imports lourds sont contournÃ©s par importlib).",
    "main.py force PYTHONIOENCODING=utf-8 et PYTHONUTF8=1 dans les sous-process â†’ "
    "fin du UnicodeEncodeError cp1252 sous Windows.",
    "E_feasibility_NER.py calcule le score Feas Ã  partir de Freq et He.",
    "L'arbre est calibrÃ© sur le corpus Breast/RCP. Pour valider la stabilitÃ© sur un autre "
    "corpus, lancer `evaluate` avec --gs_dir/--pred_dir adaptÃ©s.",
    "decision_config.json est rÃ©-Ã©crit Ã  chaque `tree`/`evaluate`. Un export figÃ© doit "
    "passer par `rest-config --export`.",
    "Suivi Ã©nergie : chaque script E_*.py dÃ©marre un Tracker eco2ai qui appondit dans "
    "Consumtion_of_Duraxell.csv.",
]
for n in notes:
    story += [Paragraph(f"â€¢ {n}", BODY), Spacer(1, 3)]

# Build
doc = SimpleDocTemplate(
    str(OUT),
    pagesize=A4,
    leftMargin=1.8 * cm,
    rightMargin=1.8 * cm,
    topMargin=1.8 * cm,
    bottomMargin=1.8 * cm,
    title="DEMNE â€” Manifeste des fonctionnalitÃ©s",
    author="ClÃ©ment Longeac",
)
doc.build(story)
print(f"PDF gÃ©nÃ©rÃ© : {OUT}")
