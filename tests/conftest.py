"""
Shared pytest fixtures for the DEMNE test suite.

Why a conftest.py?
------------------
pytest automatically discovers this file and makes every fixture defined here
available to ALL test modules in the same directory — no import needed.
Without it, each test file had to re-declare its own corpus, BRAT files and
eco2ai mock, leading to duplication and drift.

Design principle: fixtures build on each other from the bottom up:
  eco2ai_mock               (session-scoped: mocked once for the whole run)
      └─ brat_corpus_dir    (function-scoped: fresh tmp dir per test)
              └─ brat_corpus (list of dicts ready for scorers)
              └─ synthetic_metrics (dict ready for DecisionTreeBuilder)
"""

import sys
from unittest.mock import MagicMock

import pytest

# ---------------------------------------------------------------------------
# eco2ai mock — must be installed BEFORE any demne import so that the
# module-level `from eco2ai import Tracker` inside each E_*.py never fails.
# session scope = installed once, shared across all tests in the run.
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session", autouse=True)
def eco2ai_mock():
    """
    Intercept eco2ai at import time for the whole test session.

    autouse=True means every test gets this automatically — no need to
    request it explicitly. This replaces the per-file `sys.modules["eco2ai"]
    = MagicMock()` lines that were scattered across test files.
    """
    sys.modules.setdefault("eco2ai", MagicMock())


# ---------------------------------------------------------------------------
# BRAT corpus on disk — reusable across FrequencyScorer, TemplatabilityScorer,
# HomogeneityScorer, RiskContextScorer and the integration test.
# function scope = a fresh tmp_path per test (pytest's tmp_path fixture
# creates an isolated directory under the OS temp folder).
# ---------------------------------------------------------------------------

# Three synthetic oncology documents covering the main entity types used in
# DEMNE: HER2_Status (structurally rigid), ER_Status (homogeneous),
# Pays_Origine (heterogeneous, for TF-IDF contrast).
_BRAT_DOCS = [
    {
        "name": "doc_her2",
        "txt": (
            "Statut HER2 : 3+ (FISH positif). "
            "Le récepteur HER2 est surexprimé. "
            "Conclusion : HER2 positif confirmé."
        ),
        "ann": (
            "T1\tHER2_Status 7 9\t3+\n"
            "T2\tHER2_Status 54 62\tpositif\n"
            "T3\tHER2_Status 97 104\tpositif\n"
        ),
    },
    {
        "name": "doc_er",
        "txt": (
            "Récepteurs estrogéniques ER : positif 90%. "
            "ER positif à 80%. "
            "Absence de récepteurs progestéroniques PR."
        ),
        "ann": (
            "T1\tER_Status 29 36\tpositif\n"
            "T2\tER_Status 42 49\tpositif\n"
            "T3\tPR_Status 84 90\tabsent\n"
        ),
    },
    {
        "name": "doc_risk",
        "txt": (
            "Le statut HER2 est possible amplifié, à confirmer. "
            "Absence de surexpression PR. "
            "ER : probable positif."
        ),
        "ann": (
            "T1\tHER2_Status 15 23\tamplifié\n"
            "T2\tPR_Status 52 58\tabsent\n"
            "T3\tER_Status 67 75\tpositif\n"
        ),
    },
]


@pytest.fixture
def brat_corpus_dir(tmp_path):
    """
    Write _BRAT_DOCS to a temporary directory as .txt / .ann file pairs.

    tmp_path is pytest's built-in fixture: each test gets a unique directory
    under the OS temp folder, automatically deleted after the test.

    Returns the Path to the directory (usable by file-based scorers).
    """
    for doc in _BRAT_DOCS:
        (tmp_path / f"{doc['name']}.txt").write_text(doc["txt"], encoding="utf-8")
        (tmp_path / f"{doc['name']}.ann").write_text(doc["ann"], encoding="utf-8")
    return tmp_path


@pytest.fixture
def brat_corpus():
    """
    Return the same documents as a list-of-dicts corpus (in-memory format).

    Used by scorers that accept a corpus directly (TemplatabilityScorer,
    HomogeneityScorer) without needing to touch the filesystem.
    Each annotation is a dict with keys 'entity_type' and 'text', which both
    scorers accept alongside BratAnnotation objects.
    """
    corpus = []
    for doc in _BRAT_DOCS:
        annotations = []
        for line in doc["ann"].splitlines():
            if not line.startswith("T"):
                continue
            parts = line.split("\t")
            if len(parts) >= 3:
                etype = parts[1].split()[0]
                annotations.append({"entity_type": etype, "text": parts[2]})
        corpus.append({"file_id": doc["name"], "text": doc["txt"], "annotations": annotations})
    return corpus


@pytest.fixture
def synthetic_metrics():
    """
    Pre-built metrics dict ready for DecisionTreeBuilder.analyze_entity().

    Keys are entity names; values are the metric dicts expected by the tree.
    Covers all four DEMNE routing outcomes (RULES, TBM, LLM, plus TFIDF rescue)
    so integration tests can verify each branch without recomputing scorers.
    """
    return {
        # Te+He high + R low → RULES (classical branch)
        "HER2_Status": {"Te": 90.0, "Te_count": 20, "He": 0.95, "R": 0.05, "Feas": 0.6},
        # Te low + Feas high → TBM
        "Pays_Origine": {"Te": 5.0, "Te_count": 20, "He": 0.10, "R": 0.10, "Feas": 0.8},
        # Te low + Feas low → LLM
        "Contexte_Social": {"Te": 5.0, "Te_count": 20, "He": 0.10, "R": 0.20, "Feas": 0.1},
        # Te low + TFIDF high + R low → RULES (TFIDF rescue branch)
        "Evolution_Tumorale": {
            "Te": 5.0,
            "Te_count": 20,
            "He": 0.10,
            "R": 0.05,
            "Feas": 0.8,
            "tfidf_score": 0.92,
            "f1_score": 0.92,
        },
    }
