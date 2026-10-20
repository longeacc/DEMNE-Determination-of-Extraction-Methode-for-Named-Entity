import logging
import os
from dataclasses import dataclass
from typing import Any

from demne._brat import parse_ann_content

# Characters of text kept on each side of a mention (= risk_window_size default).
CONTEXT_WINDOW = 50


@dataclass
class BratAnnotation:
    id: str
    entity_type: str
    start: int
    end: int
    value: str
    context: str = ""


@dataclass
class BratDocument:
    filename: str
    text: str
    annotations: list[BratAnnotation]


class BratCorpusParser:
    def __init__(self) -> None:
        self.logger = logging.getLogger(__name__)

    def _report_error(self, message: str) -> None:
        """Show `message` in the Streamlit UI when running under it, else just log it.

        Streamlit is imported lazily so this module (and the parser) work and can be
        tested without the UI stack installed.
        """
        self.logger.error(message)
        try:
            import streamlit as st
        except ImportError:
            return
        st.error(message)

    def _parse_ann_content(self, ann_content: str, text_content: str) -> list[BratAnnotation]:
        # Line parsing and context window live in demne._brat (single parser).
        return [
            BratAnnotation(
                id=a.ann_id,
                entity_type=a.entity_type,
                start=a.start,
                end=a.end,
                value=a.text,
                context=a.context,
            )
            for a in parse_ann_content(ann_content, text_content, CONTEXT_WINDOW)
        ]

    def parse_uploaded_files(self, uploaded_files: list[Any]) -> list[BratDocument]:
        documents = []
        files_by_base = {}
        for f in uploaded_files:
            base_name, ext = os.path.splitext(f.name)
            if base_name not in files_by_base:
                files_by_base[base_name] = {}
            try:
                content = f.read().decode("utf-8")
            except UnicodeDecodeError:
                f.seek(0)
                content = f.read().decode("latin-1", errors="replace")

            files_by_base[base_name][ext.lower()] = content

        for base_name, parts in files_by_base.items():
            if ".ann" in parts:
                ann_content = parts[".ann"]
                text_content = parts.get(".txt", "")
                annotations = self._parse_ann_content(ann_content, text_content)
                documents.append(
                    BratDocument(filename=base_name, text=text_content, annotations=annotations)
                )

        return documents

    def parse_directory(self, path: str) -> list[BratDocument]:
        documents = []
        if not os.path.isdir(path):
            self._report_error(f"Le dossier spécifié n'existe pas : {path}")
            return documents

        files_by_base = {}
        for root, _, files in os.walk(path):
            for file in files:
                base_name, ext = os.path.splitext(file)
                if ext.lower() in [".txt", ".ann"]:
                    if base_name not in files_by_base:
                        files_by_base[base_name] = {}

                    file_path = os.path.join(root, file)
                    try:
                        with open(file_path, encoding="utf-8") as f:
                            content = f.read()
                    except UnicodeDecodeError:
                        with open(file_path, encoding="latin-1", errors="replace") as f:
                            content = f.read()

                    files_by_base[base_name][ext.lower()] = content

        for base_name, parts in files_by_base.items():
            if ".ann" in parts:
                ann_content = parts[".ann"]
                text_content = parts.get(".txt", "")
                annotations = self._parse_ann_content(ann_content, text_content)
                documents.append(
                    BratDocument(filename=base_name, text=text_content, annotations=annotations)
                )

        return documents

    def get_entity_statistics(self, documents: list[BratDocument]) -> dict[str, dict[str, Any]]:
        stats: dict[str, dict[str, Any]] = {}

        for doc in documents:
            for ann in doc.annotations:
                if ann.entity_type not in stats:
                    stats[ann.entity_type] = {"count": 0, "value_distribution": {}}

                stats[ann.entity_type]["count"] += 1
                val_lower = ann.value.lower()

                if val_lower not in stats[ann.entity_type]["value_distribution"]:
                    stats[ann.entity_type]["value_distribution"][val_lower] = 0
                stats[ann.entity_type]["value_distribution"][val_lower] += 1

        return stats
