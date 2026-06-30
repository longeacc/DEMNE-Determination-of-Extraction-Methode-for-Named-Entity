import os

import matplotlib.pyplot as plt


class ConvergenceAnalyzer:
    """
    Outil de visualisation des résultats de concordance Arbre vs REST.
    """

    def __init__(self, output_dir="Results/figures"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def analyze_convergence(self, bridge_report: dict) -> None:
        """
        Génère un rapport texte et des graphiques simples.
        """
        print("\n--- RAPPORT DE CONVERGENCE DURAXELL ---")
        print(f"Taux de concordance global : {bridge_report['concordance_rate'] * 100:.1f}%")

        divs = bridge_report.get("divergences", [])
        print(f"Nombre de divergences : {len(divs)}")

        if divs:
            print("\nDétail des divergences (Top-Down vs Bottom-Up):")
            print(f"{'ENTITE':<25} | {'ARBRE':<15} | {'REST (EMPIRIQUE)':<15} | {'DELTA Te'}")
            print("-" * 75)
            for div in divs:
                ent = div["entity"]
                tree = div["tree_decision"]
                rest = div["rest_decision"]

                # Accès sécurisé aux métriques avec fallback
                metrics = div.get("metrics_delta", {})
                te_emp = metrics.get("empirical_te", 0.0)
                # Vérification type pour theoretical_te qui peut être 'N/A'
                te_theo = metrics.get("theoretical_te", "N/A")

                delta_str = f"E:{te_emp:.2f} vs T:{te_theo}"
                print(f"{ent:<25} | {tree:<15} | {rest:<15} | {delta_str}")

        self._plot_concordance_pie(bridge_report)

    def _plot_concordance_pie(self, bridge_report):
        """Camembert concordance vs divergence."""
        n_ok = bridge_report["concordance_rate"] * 100
        n_ko = 100 - n_ok

        labels = ["Concordance", "Divergence"]
        sizes = [n_ok, n_ko]
        colors = ["#4CAF50", "#F44336"]  # Green, Red

        plt.figure(figsize=(6, 6))
        plt.pie(sizes, labels=labels, colors=colors, autopct="%1.1f%%", startangle=140)
        plt.title("Convergence Arbre de Décision vs Annotation REST")

        path = os.path.join(self.output_dir, "convergence_pie_chart.png")
        plt.savefig(path)
        plt.close()
        print(f"\nGraphique sauvegardé : {path}")


if __name__ == "__main__":
    # Test simple
    report_mock = {"concordance_rate": 0.85, "n_divergences": 2, "divergences": []}
    analyzer = ConvergenceAnalyzer()
    analyzer.analyze_convergence(report_mock)
