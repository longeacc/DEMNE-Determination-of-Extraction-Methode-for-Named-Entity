import os
import re
import time
from datetime import datetime


class BratAnnotation:
    def __init__(
        self,
        doc_id: str,
        entity_type: str,
        start: int,
        end: int,
        text: str,
        context_left: str = "",
        context_right: str = "",
    ):
        self.doc_id = doc_id
        self.entity_type = entity_type
        self.start = start
        self.end = end
        self.text = text
        self.context_left = context_left
        self.context_right = context_right

    def to_dict(self):
        return {
            "doc_id": self.doc_id,
            "entity_type": self.entity_type,
            "start": self.start,
            "end": self.end,
            "text": self.text,
            "context_left": self.context_left,
            "context_right": self.context_right,
        }


class RESTAnnotator:
    """
    Outil d'annotation pilote selon la méthodologie REST (Rapid Expert Supervision Tool).
    Objectif : réduire le temps d'annotation de 15-20 min -> 3-5 min.
    """

    def __init__(self, output_dir="Evaluation/REST_Annotations"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        # Patterns simples pour le mode automatisé (simulation expert)
        self.AUTO_PATTERNS = {
            "Estrogen_receptor": [
                r"(ER\s*(\+|positive|\>\d+%))",
                r"(Estrogen Receptor\s*(\+|positive))",
            ],
            "Progesterone_receptor": [
                r"(PR\s*(\+|positive|\>\d+%))",
                r"(Progesterone Receptor\s*(\+|positive))",
            ],
            "HER2": [r"(HER2\s*(\-|negative|positive|\d\+))"],
            "Ki67": [r"(Ki67\s*(\<|\>)?\s*\d+%)"],
        }

    def annotate_batch(
        self,
        documents: list[tuple[str, str]],  # List of (doc_id, text) tuples
        entity_types: list[str] = None,
        mode: str = "highlighting",
    ) -> list[BratAnnotation]:
        """
        Simulation d'une session d'annotation.

        Modes:
        - 'manual': Interactif (console input)
        - 'automated_test': Utilise des regex prédéfinies pour simuler un expert rapide
        - 'highlighting': (Futur) Interface graphique
        """
        if entity_types is None:
            entity_types = ["Estrogen_receptor", "Progesterone_receptor", "HER2", "Ki67"]
        annotations = []

        print(f"--- Démarrage Session REST ({mode}) ---")
        print(f"Nombre de documents : {len(documents)}")

        for doc_id, text in documents:
            start_time = time.time()
            doc_anns = []

            print(f"\nDocument {doc_id} analysis...")

            if mode == "automated_test":
                # Simulation expert: trouve les entités via regex
                doc_anns = self._auto_annotate(doc_id, text, entity_types)
                # Simulation temps de lecture humain rapide (0.1s par entité trouvée + base)
                time.sleep(0.05 + 0.01 * len(doc_anns))

            elif mode == "manual":
                # Mode interactif console (simplifié)
                print(text[:300] + "...")
                print("Entrez 'type:start-end' (ex: HER2:10-15) ou 'n' pour suivant.")
                # user_input = input("Annot > ") # Commenté pour éviter blocage
                pass

            duration = time.time() - start_time
            annotations.extend(doc_anns)

            # Log de performance de l'annotateur
            {
                "doc_id": doc_id,
                "n_annotations": len(doc_anns),
                "duration_sec": duration,
                "mode": mode,
                "timestamp": datetime.now().isoformat(),
            }
            # Enregistrer log_entry (optionnel, ici print)
            print(f"  > {len(doc_anns)} annotations found in {duration:.4f}s")

        self.export_to_brat(annotations)
        return annotations

    def _auto_annotate(
        self, doc_id: str, text: str, entity_types: list[str]
    ) -> list[BratAnnotation]:
        """Méthode interne pour simuler l'annotation via patterns regex."""
        anns = []
        for ent_type in entity_types:
            patterns = self.AUTO_PATTERNS.get(ent_type, [])
            for pat in patterns:
                for match in re.finditer(pat, text, re.IGNORECASE):
                    start, end = match.span()
                    span_text = match.group()
                    # Capture contexte (50 chars avant/après)
                    ctx_left = text[max(0, start - 50) : start]
                    ctx_right = text[end : min(len(text), end + 50)]

                    anns.append(
                        BratAnnotation(doc_id, ent_type, start, end, span_text, ctx_left, ctx_right)
                    )
        return anns

    def export_to_brat(self, annotations: list[BratAnnotation], output_dir: str = None) -> None:
        """Exporte en format .ann (BRAT)."""
        out = output_dir or self.output_dir

        # Group by document
        docs = {}
        for ann in annotations:
            if ann.doc_id not in docs:
                docs[ann.doc_id] = []
            docs[ann.doc_id].append(ann)

        for doc_id, anns in docs.items():
            file_path = os.path.join(out, f"{doc_id}.ann")
            # Create corresponding .txt file if not exists (required by BRAT)
            # (Skipped here as we assume texts are managed elsewhere, but nice to have)

            with open(file_path, "w", encoding="utf-8") as f:
                for i, ann in enumerate(anns):
                    # Format BRAT: T1  Entity Start End  Text
                    # Note: BRAT entities cannot have spaces in type name (usually)
                    # We ensure tabs are correct
                    line = f"T{i + 1}\t{ann.entity_type} {ann.start} {ann.end}\t{ann.text}\n"
                    f.write(line)


if __name__ == "__main__":
    annotator = RESTAnnotator()
    print("REST Annotator ready.")
