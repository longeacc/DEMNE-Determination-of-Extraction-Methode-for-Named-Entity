"""
Générer un rapport HTML pour l'analyse de templeabilité.
"""

import json
import os
from pathlib import Path


def generate_html_report(json_file: str, csv_file: str, output_file: str):
    """
    Générer un rapport HTML à partir des résultats d'analyse de templeabilité.

    Args:
        json_file: Chemin vers le fichier de rapport JSON
        csv_file: Chemin vers le fichier de rapport CSV
        output_file: Chemin pour sauvegarder le rapport HTML
    """

    # Chargement des données
    try:
        with open(json_file, encoding="utf-8") as f:
            raw_data = json.load(f)

        # Conversion JSON->Liste pour éviter les problèmes de parsing CSV
        data_list = []
        for entity, stats in raw_data.items():
            stats["Entity_Type"] = entity
            stats["Templatability_Score"] = stats.get("templatability_score", 0)
            # Récupération sécurisée des listes
            stats["Sample_Values_List"] = stats.get("sample_values", [])
            stats["Abstract_Patterns_List"] = stats.get("top_abstract_patterns", [])

            # Restauration des métriques essentielles demandées
            stats["Top_Pattern_Coverage"] = stats.get("top_3_pattern_coverage", 0)
            stats["Diversity_Ratio"] = stats.get("diversity_ratio", 0)
            stats["Total_Occurrences"] = stats.get("count", 0)
            stats["Presence_Ratio"] = stats.get("presence_ratio", 0)
            stats["Unique_Values"] = stats.get("unique_values_count", 0)

            data_list.append(stats)

        # Tri par score de templeabilité
        data_list.sort(key=lambda x: x["Templatability_Score"], reverse=True)

    except FileNotFoundError:
        print(f"Erreur: Fichiers de données non trouvés ({json_file})")
        return

    # Création du HTML
    html_content = """
    <!DOCTYPE html>
    <html lang="fr">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Rapport d'Analyse de Templeabilité</title>
        <style>
            * {
                margin: 0;
                padding: 0;
                box-sizing: border-box;
            }

            body {
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                min-height: 100vh;
                padding: 20px;
                color: #333;
            }

            .container {
                max-width: 1400px;
                margin: 0 auto;
                background: white;
                border-radius: 10px;
                box-shadow: 0 10px 40px rgba(0,0,0,0.2);
                overflow: hidden;
            }

            .header {
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                padding: 40px;
                text-align: center;
            }

            .header h1 {
                font-size: 2.5em;
                margin-bottom: 10px;
            }

            .header p {
                font-size: 1.1em;
                opacity: 0.9;
            }

            .content {
                padding: 40px;
            }

            .method-box {
                background: #f1f5f9;
                padding: 25px;
                border-radius: 8px;
                margin-bottom: 40px;
                border-left: 5px solid #667eea;
                color: #334155;
            }

            .method-box h3 {
                color: #4c1d95;
                margin-bottom: 15px;
                font-size: 1.4em;
            }

            .method-box h4 {
                color: #5b21b6;
                margin-top: 15px;
                margin-bottom: 8px;
                font-size: 1.1em;
            }

            .method-box ul {
                padding-left: 20px;
                margin-bottom: 10px;
            }

            .method-box li {
                margin-bottom: 5px;
            }

            .formula {
                background: #e2e8f0;
                padding: 10px;
                border-radius: 4px;
                font-family: 'Courier New', monospace;
                margin: 10px 0;
                font-weight: bold;
                color: #1e293b;
            }

            .tech-details {
                margin-top: 15px;
                padding-top: 15px;
                border-top: 1px solid #cbd5e1;
                font-size: 0.95em;
            }

            .metrics {
               display: grid;
               grid-template-columns: 1fr 1fr;
               gap: 10px;
               background: #f8fafc;
               padding: 10px;
               border-radius: 6px;
               margin-bottom: 10px;
            }

            .metric {
                display: flex;
                flex-direction: column;
                font-size: 0.85em;
                color: #555;
            }

            .metric-label { font-weight: normal; margin-bottom: 2px; }
            .metric-value { font-weight: bold; color: #333; font-size: 1.1em; }

            .summary {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
                gap: 20px;
                margin-bottom: 40px;
            }

            .summary-card {
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                padding: 20px;
                border-radius: 8px;
                text-align: center;
            }

            .summary-card h3 {
                font-size: 0.9em;
                opacity: 0.9;
                margin-bottom: 10px;
            }

            .summary-card .value {
                font-size: 2em;
                font-weight: bold;
            }

            .entity-grid {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(400px, 1fr));
                gap: 20px;
                margin-bottom: 40px;
            }

            .entity-card {
                border: 2px solid #ddd;
                border-radius: 8px;
                padding: 20px;
                transition: all 0.3s ease;
                background: white;
            }

            .entity-card:hover {
                box-shadow: 0 5px 15px rgba(102, 126, 234, 0.2);
                border-color: #667eea;
            }

            .entity-card.high { border-left: 5px solid #22c55e; }
            .entity-card.medium { border-left: 5px solid #f59e0b; }
            .entity-card.low { border-left: 5px solid #ef4444; }

            .entity-header {
                display: flex;
                justify-content: space-between;
                align-items: center;
                margin-bottom: 15px;
            }

            .entity-name {
                font-size: 1.3em;
                font-weight: bold;
                color: #667eea;
            }

            .status-badge {
                padding: 4px 8px;
                border-radius: 12px;
                font-size: 0.75em;
                color: white;
                font-weight: bold;
            }
            .status-badge.high { background: #22c55e; }
            .status-badge.medium { background: #f59e0b; }
            .status-badge.low { background: #ef4444; }

            .score-bar {
                display: flex;
                align-items: center;
                margin: 15px 0;
            }

            .score-bar-fill {
                flex: 1;
                height: 15px;
                background: #e5e7eb;
                border-radius: 4px;
                overflow: hidden;
                margin-right: 10px;
            }

            .score-bar-value {
                height: 100%;
                background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
            }

            .metric {
                display: flex;
                justify-content: space-between;
                margin: 5px 0;
                font-size: 0.9em;
                color: #555;
            }

            .metric-value {
                font-weight: bold;
                color: #333;
            }

            .patterns-section {
                margin-top: 15px;
                border-top: 1px solid #eee;
                padding-top: 10px;
            }

            .pattern-tag {
                display: inline-block;
                background: #f1f5f9;
                color: #475569;
                padding: 2px 6px;
                border-radius: 4px;
                font-family: monospace;
                font-size: 0.85em;
                margin: 2px;
                border: 1px solid #cbd5e1;
            }

            .sample-values {
                margin-top: 10px;
                font-size: 0.85em;
                color: #666;
                background: #f8fafc;
                padding: 8px;
                border-radius: 4px;
                font-family: monospace;
            }

            .table-wrapper {
                margin-top: 40px;
                overflow-x: auto;
            }

            table {
                width: 100%;
                border-collapse: collapse;
                margin-bottom: 20px;
            }

            th, td {
                padding: 12px;
                text-align: left;
                border-bottom: 1px solid #ddd;
            }

            th {
                background-color: #f8f9fa;
                color: #667eea;
            }

            .footer {
                margin-top: 50px;
                text-align: center;
                color: #aaa;
                font-size: 0.9em;
                padding-bottom: 20px;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>Rapport d'Analyse de Templeabilité</h1>
                <p>Analyse de la structure et de la régularité des entités biomédicales</p>
                <p style="margin-top:5px; font-size:0.9em">Jeu de données: Cancer du Sein (ESMO 2025)</p>
            </div>

            <div class="content">
                <div class="method-box">
                    <h3>Méthodologie & Définitions Complètes</h3>

                    <h4>1. Templeabilité (Te)</h4>
                    <p>La <strong>Templeabilité</strong> mesure à quel point les valeurs d'une entité suivent une structure rigide et prévisible (Regex/Templates).</p>
                    <ul>
                        <li><strong>Motif Abstrait :</strong> Conversion (Chiffres='D', Maj='X', Min='x'). Ex: <em>"HER2 3+"</em> → <em>"XXXD D+"</em>.</li>
                        <li><strong>Couverture (P_cov) :</strong> % des valeurs couvertes par les 3 motifs dominants.</li>
                        <li><strong>Diversité :</strong> Rapport (Valeurs Uniques / Total) mesurant la variabilité.</li>
                    </ul>
                    <div class="formula">Te = (0.7 * P_cov) + S_bonus + P_bonus</div>

                    <div class="tech-details">
                        <h4>2. Métriques Connexes (Références)</h4>
                        <p><strong>Contexte de Risque (R) :</strong> Ambiguïté des mentions. Score 0 = "Auto-Décrit" (haute qualité, non ambigu).</p>
                        <p><strong>Homogénéité (He) :</strong> Redondance lexicale. <code>He = (TotalMots - MotsUniques) / TotalMots</code>.</p>
                    </div>
                </div>

                <div class="summary">
    """

    # Statistiques globales
    scores = [d["Templatability_Score"] for d in data_list]
    avg_score = sum(scores) / len(scores) if scores else 0
    high_temp = sum(1 for s in scores if s >= 0.7)
    total_ents = len(data_list)

    html_content += f"""
                    <div class="summary-card">
                        <h3>Score Moyen</h3>
                        <div class="value">{avg_score:.2f}</div>
                    </div>
                    <div class="summary-card">
                        <h3>Entités Analysées</h3>
                        <div class="value">{total_ents}</div>
                    </div>
                    <div class="summary-card">
                        <h3>Haute Templeabilité (>0.7)</h3>
                        <div class="value">{high_temp}</div>
                    </div>
                </div>

                <h2 style="color: #667eea; margin-bottom: 20px;">Détail par Entité</h2>
                <div class="entity-grid">
    """

    for row in data_list:
        score = row["Templatability_Score"]
        entity = row["Entity_Type"]

        if score >= 0.7:
            cls, status = "high", "ÉLEVÉE"
        elif score >= 0.4:
            cls, status = "medium", "MOYENNE"
        else:
            cls, status = "low", "FAIBLE"

        width_pct = min(100, max(5, int(score * 100)))

        # Listes sont déjà des listes dans le JSON
        patterns = row.get("Abstract_Patterns_List", [])
        samples = row.get("Sample_Values_List", [])

        # Limiter l'affichage
        patterns_display = "".join(
            [f'<span class="pattern-tag">{p[:30]}</span>' for p in patterns[:3]]
        )
        # Afficher plus d'échantillons en liste
        samples_html = "".join(
            [
                f'<div style="border-bottom:1px dashed #ddd; padding:2px;">{s}</div>'
                for s in samples[:6]
            ]
        )

        html_content += f"""
                    <div class="entity-card {cls}">
                        <div class="entity-header">
                            <div class="entity-name">{entity}</div>
                            <span class="status-badge {cls}">{status}</span>
                        </div>

                        <div class="score-bar">
                            <div class="score-bar-fill">
                                <div class="score-bar-value" style="width: {width_pct}%"></div>
                            </div>
                            <strong style="color: #667eea;">{score:.2f}</strong>
                        </div>

                        <div class="metrics">
                            <div class="metric">
                                <span class="metric-label">Couverture Motif</span>
                                <span class="metric-value">{row.get("Top_Pattern_Coverage", 0) * 100:.1f}%</span>
                            </div>
                            <div class="metric">
                                <span class="metric-label">Diversité</span>
                                <span class="metric-value">{row.get("Diversity_Ratio", 0):.2f}</span>
                            </div>
                            <div class="metric">
                                <span class="metric-label">Fréquence Rel.</span>
                                <span class="metric-value">{row.get("Presence_Ratio", 0) * 100:.1f}%</span>
                            </div>
                             <div class="metric">
                                <span class="metric-label">Occurrences</span>
                                <span class="metric-value">{row.get("Total_Occurrences", 0)}</span>
                            </div>
                        </div>

                        <div class="patterns-section">
                            <div style="font-size:0.85em; font-weight:bold; color:#666; margin-bottom:5px;">Motifs Dominants:</div>
                            {patterns_display}
                        </div>

                        <div class="sample-values">
                            <strong>Échantillons:</strong>
                            <div style="max-height:80px; overflow-y:auto; margin-top:5px; font-size:0.85em; font-family:monospace; color:#333;">
                                {samples_html}
                            </div>
                        </div>
                    </div>
        """

    html_content += """
                </div>

                <h2 style="color: #667eea; margin-top: 40px;">Tableau Récapitulatif</h2>
                <div class="table-wrapper">
                    <table>
                        <thead>
                            <tr>
                                <th>Entité</th>
                                <th>Score</th>
                                <th>Couverture Motif</th>
                                <th>Diversité</th>
                                <th>Fréquence</th>
                            </tr>
                        </thead>
                        <tbody>
    """

    for row in data_list:
        html_content += f"""
                            <tr>
                                <td><strong>{row["Entity_Type"]}</strong></td>
                                <td><strong>{row["Templatability_Score"]:.3f}</strong></td>
                                <td>{row.get("Top_Pattern_Coverage", 0) * 100:.1f}%</td>
                                <td>{row.get("Diversity_Ratio", 0):.3f}</td>
                                <td>{row.get("Total_Occurrences", 0)}</td>
                            </tr>
        """

    html_content += """
                        </tbody>
                    </table>
                </div>

                <div class="footer">
                    Généré par le module d'Analyse NLP - ESMO 2025
                </div>
            </div>
        </div>
    </body>
    </html>
    """

    with open(output_file, "w", encoding="utf-8") as f:
        f.write(html_content)

    print(f"Rapport HTML généré avec succès : {output_file}")


if __name__ == "__main__":
    # Détection automatique de l'environnement
    base_dir = Path(__file__).parent / "Rules/src/Results"

    # Fallback via variable d'environnement
    if not base_dir.exists():
        env_dir = os.environ.get("DEMNE_RESULTS_DIR")
        if env_dir and Path(env_dir).exists():
            base_dir = Path(env_dir)

    # Chemins des fichiers
    json_path = base_dir / "templatability_analysis.json"
    csv_path = base_dir / "templatability_analysis.csv"
    html_path = base_dir / "templatability_analysis_report.html"

    print(f"Recherche des données dans : {base_dir}")

    if json_path.exists() and csv_path.exists():
        generate_html_report(str(json_path), str(csv_path), str(html_path))
    else:
        print("ATTENTION: Fichiers de données (JSON/CSV) introuvables.")
        print("Veuillez d'abord exécuter le script d'analyse 'templatability.py'.")
try:
    t = globals().get("tracker")
    if t:
        t.stop()
except Exception as e:
    print(
        f"\nWarning: Generalized error in Eco2AI tracking (likely 'N/A' vs float dtype issue): {e}"
    )
    print("Carbon emission tracking data could not be saved, but analysis results are preserved.")
