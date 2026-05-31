"""
Calculate 'Risk Context' (R) Score.
Measures the complexity of the linguistic context surrounding an entity.

R Score ranges from 0.0 (Simple) to 1.0 (Complex/Dangerous).
High R indicates the entity is surrounded by:
- Negations (simple adjustment needed)
- Uncertainty (probabilistic language)
- Contradictions (conflicting values in same doc)
"""

import csv
import os
import re
from collections import defaultdict
from pathlib import Path

# Eco2AI for energy tracking
try:
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        from eco2ai import Tracker, set_params

    HAS_ECO2AI = True
except ImportError:
    HAS_ECO2AI = False

if __name__ == "__main__" and HAS_ECO2AI and not os.environ.get("DISABLE_ECO2AI"):
    set_params(
        project_name="Consumtion_of_E_risk_context.py",
        experiment_description="Calculating Contextual Risk",
        file_name="data/Consumtion_of_Duraxell.csv",
    )
    tracker = Tracker()
    tracker.start()


class RiskContextScorer:
    """
    Calcule le score de Risque Contextuel (R).
    Analyse le texte autour de l'entité (fenêtre de tokens) pour détecter :
    1. Négation (faible augmentation de R)
    2. Incertitude (forte augmentation de R)
    3. Contradiction (R maximal)
    """

    def __init__(self, data_dirs: list[Path] = None):
        self.data_dirs = data_dirs or []
        self.document_data = defaultdict(list)

        # --- CONFIGURATION DES RADARS ---

        # Fenêtre d'analyse : nombre de caractères autour de l'entité à lire
        self.WINDOW_SIZE = 50

        # 1. Patterns de Négation (Augmente R un peu)
        # Patterns volontairement restrictifs : on évite "pas", "ni", "non" seuls qui
        # sur-détectent (ex. "non surexprimé" est déjà un état clinique attendu).
        # "négatif" / "negatif" volontairement exclus : c'est un état clinique
        # attendu (PR-, HER2-), pas un signal de risque contextuel.
        self.NEGATION_PATTERNS = [
            r"\bne\s+pas\b",
            r"\babsent\b",
            r"\babsence\b",
            r"\baucun\b",
            r"\bsans\b",
        ]

        # 2. Patterns d'Incertitude (Augmente R beaucoup)
        self.UNCERTAINTY_PATTERNS = [
            r"\bprobable\b",
            r"\bpossible\b",
            r"\bà confirmer\b",
            r"\ba confirmer\b",
            r"\bsuspecté\b",
            r"\bsuspecte\b",
            r"\béquivoque\b",
            r"\bequivoque\b",
            r"\bdiscuté\b",
            r"\bincertain\b",
            r"\bhypothèse\b",
            r"\?",
            r"\bdiscordant\b",
            r"\bdiscordance\b",
        ]

        # 3. Termes spécifiques pour détecter les contradictions (Positif vs Négatif)
        self.VAL_POS = {r"positif", r"positive", r"\+", r"pos", r"exprimé", r"present"}
        self.VAL_NEG = {
            r"négatif",
            r"negative",
            r"\-",
            r"neg",
            r"absent",
            r"non exprimé",
        }

        # R(E) = min(1, α_R · f_neg + β_R · f_unc + γ_R · f_fam)
        self.ALPHA_R = 0.1
        self.BETA_R = 0.3
        self.GAMMA_R = 0.6
        self.weights = {
            "negation": self.ALPHA_R,
            "uncertainty": self.BETA_R,
            "contradiction": self.GAMMA_R,
        }

    def _learn_weights(self, annotated_data: list[tuple[int, int, int, int]]):
        """
        Apprend les poids R via Régression Logistique sur un ensemble de validation
        (Chapman et al., 2001 - approche type NegEx pondéré).
        annotated_data : list de tuples (has_neg, has_uncert, has_contradiction, is_risky_ground_truth)
        """
        try:
            import numpy as np
            from sklearn.linear_model import LogisticRegression

            x = np.array([[d[0], d[1], d[2]] for d in annotated_data])
            y = np.array([d[3] for d in annotated_data])

            # Contrainte de poids positifs
            clf = LogisticRegression(fit_intercept=False, positive=True)
            clf.fit(x, y)

            self.weights["negation"] = float(clf.coef_[0][0])
            self.weights["uncertainty"] = float(clf.coef_[0][1])
            self.weights["contradiction"] = float(clf.coef_[0][2])
            print(f"Poids R recalibrés via RL : {self.weights}")
        except ImportError:
            print("scikit-learn non disponible pour la R.L., utilisation des heuristiques.")
        except Exception as e:
            print(f"Erreur lors de l'apprentissage des poids : {e}")

    def has_negation(self, text: str, entity_type: str = "") -> bool:
        """Vérifie si le texte contient une négation."""
        text = text.lower()
        return any(re.search(pat, text) for pat in self.NEGATION_PATTERNS)

    def has_uncertainty(self, text: str, entity_type: str = "") -> bool:
        """Vérifie si le texte contient une incertitude."""
        text = text.lower()
        return any(re.search(pat, text) for pat in self.UNCERTAINTY_PATTERNS)

    def compute_score_from_stats(
        self,
        negated_count: int,
        uncertain_count: int,
        total_count: int,
        contradicted_rate: float = 0.0,
    ) -> float:
        """
        Méthode unique pour calculer le score R à partir des statistiques de base.
        Garantit la cohérence de la formule partout.
        """
        if total_count == 0:
            return 0.0

        f_neg = negated_count / total_count
        f_unc = uncertain_count / total_count

        # R(E) = min(1, α_R · f_neg + β_R · f_unc + γ_R · f_fam)
        raw_risk = (
            (self.ALPHA_R * f_neg)
            + (self.BETA_R * f_unc)
            + (self.GAMMA_R * contradicted_rate)
        )
        return min(1.0, raw_risk)

    def compute_score(self, texts: list[str], entity_type: str) -> float:
        """
        Calcule un score R sur une liste de courtes phrases (sans analyse document-level).
        Fait désormais appel à compute_score_from_stats.
        """
        if not texts:
            return 0.0

        total = len(texts)
        negated = sum(1 for t in texts if self.has_negation(t))
        uncertain = sum(1 for t in texts if self.has_uncertainty(t))

        return self.compute_score_from_stats(negated, uncertain, total, contradicted_rate=0.0)

    def _load_data(self):
        """Lit les fichiers .ann ET .txt pour avoir le contexte."""
        print("Chargement des données (Annotations + Texte)...")
        for d in self.data_dirs:
            if not d.exists():
                continue

            # Pour chaque fichier .ann, on cherche le .txt correspondant
            for ann_file in d.glob("*.ann"):
                txt_file = ann_file.with_suffix(".txt")
                if not txt_file.exists():
                    continue

                try:
                    # Lire le texte complet
                    with open(txt_file, encoding="utf-8") as f:
                        full_text = f.read()

                    # Lire les annotations
                    with open(ann_file, encoding="utf-8") as f:
                        for line in f:
                            if line.startswith("T"):
                                parts = line.strip().split("\t")
                                if len(parts) >= 3:
                                    # Parse: T1  Status 10 15  Her2+
                                    meta = parts[1].split()
                                    etype = meta[0]
                                    start = int(meta[1])
                                    end = int(
                                        meta[-1]
                                    )  # Parfois "10 15;20 25" -> on prend les extrêmes simplifiés

                                    # Extraire le contexte (fenêtre)
                                    ctx_start = max(0, start - self.WINDOW_SIZE)
                                    ctx_end = min(len(full_text), end + self.WINDOW_SIZE)
                                    context = full_text[
                                        ctx_start:ctx_end
                                    ].lower()  # Contexte normalisé

                                    self.document_data[ann_file.name].append(
                                        {
                                            "type": etype,
                                            "value_text": parts[2].lower(),
                                            "context": context,
                                        }
                                    )
                except Exception:
                    # print(f"Erreur lecture {ann_file}: {e}")
                    pass

    def _check_contradiction(self, entries: list[dict]) -> bool:
        """Détecte si une entité a des valeurs contradictoires dans le MEME document."""
        has_pos = False
        has_neg = False

        for e in entries:
            txt = e["value_text"]
            # Check POS
            if any(re.search(p, txt) for p in self.VAL_POS):
                has_pos = True
            # Check NEG
            if any(re.search(p, txt) for p in self.VAL_NEG):
                has_neg = True

        return has_pos and has_neg

    def compute_all(self) -> list[dict]:
        """Calcule le score R agrégé par Type d'Entité."""
        self._load_data()

        # Regrouper tout par type d'entité pour stats globales
        entity_stats = defaultdict(
            lambda: {"total": 0, "negated": 0, "uncertain": 0, "contradicted_docs": 0}
        )
        entity_docs = defaultdict(lambda: defaultdict(list))  # type -> doc -> [entries]

        # 1. Analyse Locale (Négation / Incertitude) pour chaque occurrence
        for filename, entries in self.document_data.items():
            for entry in entries:
                etype = entry["type"]
                ctx = entry["context"]

                entity_stats[etype]["total"] += 1
                entity_docs[etype][filename].append(entry)

                # Check Négation
                if any(re.search(pat, ctx) for pat in self.NEGATION_PATTERNS):
                    entity_stats[etype]["negated"] += 1

                # Check Incertitude
                if any(re.search(pat, ctx) for pat in self.UNCERTAINTY_PATTERNS):
                    entity_stats[etype]["uncertain"] += 1

        # 2. Analyse Globale (Contradiction) par document
        for etype, docs in entity_docs.items():
            for _filename, entries in docs.items():
                if self._check_contradiction(entries):
                    entity_stats[etype]["contradicted_docs"] += 1

        # 3. Calcul du Score R Final
        results = []
        for etype, stats in entity_stats.items():
            n_total = stats["total"]
            if n_total == 0:
                continue

            # Fréquences relatives
            f_neg = stats["negated"] / n_total
            f_unc = stats["uncertain"] / n_total

            # Pour la contradiction, c'est le ratio de documents contradictoires
            # On approxime le nombre de docs total pour cette entité comme len(entity_docs[etype])
            n_docs = len(entity_docs[etype])
            f_cont = stats["contradicted_docs"] / n_docs if n_docs > 0 else 0

            # Utilisation de la méthode unifiée
            risk_score = self.compute_score_from_stats(
                negated_count=stats["negated"],
                uncertain_count=stats["uncertain"],
                total_count=n_total,
                contradicted_rate=f_cont,
            )

            results.append(
                {
                    "Entity": etype,
                    "R_Score": round(risk_score, 4),
                    "Negation_Rate": round(f_neg, 2),
                    "Uncertainty_Rate": round(f_unc, 2),
                    "Contradiction_Rate": round(f_cont, 2),
                    "Count": n_total,
                }
            )

        return sorted(results, key=lambda x: x["R_Score"], reverse=True)

    def to_csv(self, output_path: Path):
        data = self.compute_all()
        with open(output_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "Entity",
                    "R_Score",
                    "Negation_Rate",
                    "Uncertainty_Rate",
                    "Contradiction_Rate",
                    "Count",
                ],
            )
            writer.writeheader()
            writer.writerows(data)
        try:
            rel = os.path.relpath(str(output_path))
        except ValueError:
            rel = str(output_path)
        print(f"Sauvegardé dans {rel}")


# ==================================================================================
# MAIN EXECUTION
# ==================================================================================
def main(learn_weights=False):
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--gs_dir", type=str, default=None)
    parser.add_argument("--pred_dir", type=str, default=None)
    parser.add_argument("--learn_weights", action="store_true")
    args, unknown = parser.parse_known_args()

    learn_weights = args.learn_weights or learn_weights

    # RELATIVE PATHS
    script_dir = Path(__file__).parent
    root_dir = script_dir.parent.parent

    if args.gs_dir:
        data_dirs = [Path(args.gs_dir)]
    else:
        data_dirs = [
            root_dir / "src/duraxell/NER/data/Breast/train",
            root_dir / "src/duraxell/NER/data/Breast/val",
            root_dir / "src/duraxell/NER/data/Breast/test",
        ]

    corpus_name = Path(args.gs_dir).parent.name if args.gs_dir else "Breast"
    output_file = root_dir / "Results" / f"risk_context_analysis_{corpus_name}.csv"
    output_file.parent.mkdir(parents=True, exist_ok=True)

    print("=== Démarrage de l'analyse Risk Context (R) ===")
    scorer = RiskContextScorer(data_dirs)

    if learn_weights:
        print("--- Mode apprentissage des poids (Calibration RL) ---")
        import numpy as np

        entities = list(scorer.entities_stats.keys())
        if entities:
            x = []
            for ent in entities:
                stats = scorer.entities_stats[ent]
                x.append([stats["f_neg"], stats["f_unc"], stats["f_cont"]])
            y = np.random.randint(0, 2, size=len(x))  # mockup labels
            scorer._learn_weights(np.array(x), y)
            print(f"Nouveaux poids appris : {scorer.weights}")

    scorer.to_csv(output_file)

    # === TESTS CRITIQUES (Demandés par l'utilisateur) ===
    print("\n=== VERIFICATION TESTS CRITIQUES ===")

    # Simulation de cas artificiels pour valider la logique
    # On crée une instance vide et on injecte des données fake
    test_scorer = RiskContextScorer([])

    # Test 1: "HER2 non surexprimé" (Négation simple)
    # Contexte: ... le statut est HER2 non surexprimé sur la lame ...
    test_scorer.document_data["test1.txt"] = [
        {
            "type": "TEST_NEG",
            "value_text": "her2",
            "context": "le statut est her2 non surexprimé sur la lame",
        }
    ]

    # Test 2: "statut HER2 discordant entre biopsie et pièce opératoire" (Contradiction/Conflit explicite)
    # Ici simulons une contradiction logique: dans le même doc, une valeur POS et une valeur NEG
    test_scorer.document_data["test2.txt"] = [
        {
            "type": "TEST_CONTRA",
            "value_text": "her2 positif",
            "context": "biopsie montre her2 positif",
        },
        {
            "type": "TEST_CONTRA",
            "value_text": "her2 negatif",
            "context": "piece operatoire montre her2 negatif",
        },
    ]

    # Test 3: "Incertitude"
    test_scorer.document_data["test3.txt"] = [
        {
            "type": "TEST_UNCERT",
            "value_text": "tumeur",
            "context": "origine probable de la tumeur a confirmer",
        }
    ]

    res = test_scorer.compute_all()
    for r in res:
        print(
            f"Entité Test: {r['Entity']:<15} | Score R: {r['R_Score']:.2f} | (Neg={r['Negation_Rate']}, Unc={r['Uncertainty_Rate']}, Contra={r['Contradiction_Rate']})"
        )

    if HAS_ECO2AI and not os.environ.get("DISABLE_ECO2AI"):
        tracker.stop()


if __name__ == "__main__":
    main()
