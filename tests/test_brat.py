"""Tests for demne._brat -- unified BRAT corpus parser."""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from demne._brat import BratAnnotation, load_brat_corpus, parse_ann_line


# ---------------------------------------------------------------------------
# parse_ann_line
# ---------------------------------------------------------------------------
class TestParseAnnLine:
    """Unit tests for parse_ann_line."""

    def test_simple_t_line(self):
        line = "T1\tStatus 10 15\tHER2+"
        ann = parse_ann_line(line)
        assert ann is not None
        assert ann.entity_type == "Status"
        assert ann.start == 10
        assert ann.end == 15
        assert ann.text == "HER2+"

    def test_discontinuous_span(self):
        """Discontinuous span 'Type 10 20;30 40' → end of first span = 20."""
        line = "T2\tBiomarker 10 20;30 40\tmulti span"
        ann = parse_ann_line(line)
        assert ann is not None
        assert ann.start == 10
        assert ann.end == 20  # end of first span
        assert ann.entity_type == "Biomarker"

    def test_non_t_line_returns_none(self):
        assert parse_ann_line("A1\tAnnotation arg1:T1 arg2:T2") is None
        assert parse_ann_line("R1\tRelation Arg1:T1 Arg2:T2") is None
        assert parse_ann_line("#1\tAnnotatorNotes T1\tSome note") is None

    def test_empty_line(self):
        assert parse_ann_line("") is None
        assert parse_ann_line("   ") is None

    def test_malformed_line(self):
        """Malformed T-line with insufficient tabs → None."""
        assert parse_ann_line("T1") is None
        assert parse_ann_line("T1\tStatus") is None

    def test_invalid_offsets(self):
        """Non-numeric offsets → None."""
        assert parse_ann_line("T1\tStatus abc def\ttext") is None

    def test_whitespace_stripped(self):
        line = "  T3\tDrug 100 110\taspirin  \n"
        ann = parse_ann_line(line)
        assert ann is not None
        assert ann.text == "aspirin"


# ---------------------------------------------------------------------------
# BratAnnotation dataclass
# ---------------------------------------------------------------------------
class TestBratAnnotation:
    def test_defaults(self):
        ann = BratAnnotation(start=0, end=5, text="test", entity_type="E")
        assert ann.file_id is None
        assert ann.context == ""

    def test_equality(self):
        a = BratAnnotation(start=0, end=5, text="x", entity_type="E")
        b = BratAnnotation(start=0, end=5, text="x", entity_type="E")
        assert a == b


# ---------------------------------------------------------------------------
# load_brat_corpus
# ---------------------------------------------------------------------------
class TestLoadBratCorpus:
    """Integration tests for load_brat_corpus using tmp_path fixtures."""

    @pytest.fixture()
    def corpus_dir(self, tmp_path: Path) -> Path:
        """Create a minimal BRAT corpus with 2 documents."""
        d = tmp_path / "corpus"
        d.mkdir()

        # doc1
        (d / "doc1.txt").write_text("HER2 positif sur la lame", encoding="utf-8")
        (d / "doc1.ann").write_text(
            textwrap.dedent("""\
                T1\tStatus 0 4\tHER2
                T2\tResult 5 12\tpositif
            """),
            encoding="utf-8",
        )

        # doc2
        (d / "doc2.txt").write_text("ER négatif", encoding="utf-8")
        (d / "doc2.ann").write_text("T1\tStatus 0 2\tER\n", encoding="utf-8")

        return d

    def test_loads_documents(self, corpus_dir: Path):
        corpus = load_brat_corpus([corpus_dir])
        assert len(corpus) == 2

    def test_annotation_count(self, corpus_dir: Path):
        corpus = load_brat_corpus([corpus_dir])
        # doc1 has 2 annotations, doc2 has 1
        total = sum(len(doc["annotations"]) for doc in corpus)
        assert total == 3

    def test_annotations_are_brat_objects(self, corpus_dir: Path):
        corpus = load_brat_corpus([corpus_dir])
        for doc in corpus:
            for ann in doc["annotations"]:
                assert isinstance(ann, BratAnnotation)

    def test_include_text(self, corpus_dir: Path):
        corpus = load_brat_corpus([corpus_dir], include_text=True)
        texts = {doc["file_id"]: doc.get("text", "") for doc in corpus}
        assert "HER2 positif sur la lame" in texts.get("doc1.ann", "")

    def test_text_not_included_by_default(self, corpus_dir: Path):
        corpus = load_brat_corpus([corpus_dir])
        for doc in corpus:
            assert "text" not in doc

    def test_context_window(self, corpus_dir: Path):
        corpus = load_brat_corpus([corpus_dir], context_window=5)
        # Find the HER2 annotation (start=0, end=4)
        doc1 = next(d for d in corpus if d["file_id"] == "doc1.ann")
        her2_ann = next(a for a in doc1["annotations"] if a.text == "HER2")
        # Context should include some characters after "HER2" (up to 5)
        assert len(her2_ann.context) > 0
        assert "HER2" in her2_ann.context

    def test_missing_directory_prints_warning(self, tmp_path: Path, capsys):
        corpus = load_brat_corpus([tmp_path / "nonexistent"])
        assert corpus == []
        assert "does not exist" in capsys.readouterr().out

    def test_duplicate_filenames_across_dirs(self, corpus_dir: Path, tmp_path: Path):
        """Same filename in two dirs → loaded only once (dedup by name)."""
        d2 = tmp_path / "corpus2"
        d2.mkdir()
        (d2 / "doc1.ann").write_text("T1\tOther 0 3\tFoo\n", encoding="utf-8")
        (d2 / "doc1.txt").write_text("Foo bar", encoding="utf-8")

        corpus = load_brat_corpus([corpus_dir, d2])
        doc1_files = [d for d in corpus if d["file_id"] == "doc1.ann"]
        assert len(doc1_files) == 1  # dedup

    def test_ann_without_txt(self, tmp_path: Path):
        """An .ann file without a matching .txt should still load annotations."""
        d = tmp_path / "notxt"
        d.mkdir()
        (d / "orphan.ann").write_text("T1\tEntity 0 5\thello\n", encoding="utf-8")

        corpus = load_brat_corpus([d])
        assert len(corpus) == 1
        assert len(corpus[0]["annotations"]) == 1
