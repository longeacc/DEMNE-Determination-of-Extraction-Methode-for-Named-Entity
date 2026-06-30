import sys
from unittest.mock import MagicMock

# Mock eco2ai BEFORE importing E_frequency to avoid any import-time side effects or delays
sys.modules["eco2ai"] = MagicMock()

from demne.E_frequency import FrequencyScorer


def test_frequency_calculation():
    # Setup dummy scorer (no real path needed for unit test of ingestion)
    scorer = FrequencyScorer([])

    # Text: 10 words. Annotation: 1 entity "Gene"
    text1 = "This is a sample text with ten words exactly count them."
    anns1 = ["T1\tGene 0 4\tThis"]

    scorer.ingest_document(text1, anns1)

    assert scorer.total_docs == 1
    # text1 has 11 words tokens based on spaces?
    # "This" "is" "a" "sample" "text" "with" "ten" "words" "exactly" "count" "them."
    # 1 2 3 4 5 6 7 8 9 10 11.
    # Let's adjust expectation or input.
    # "one two three four five six seven eight nine ten"
    text1 = "one two three four five six seven eight nine ten"
    scorer.total_tokens = 0  # reset
    scorer.entity_counts.clear()  # reset
    scorer.total_docs = 0  # reset
    scorer.ingest_document(text1, anns1)

    assert scorer.total_tokens == 10
    assert scorer.entity_counts["Gene"] == 1

    # Text: 5 words. Annotation: 2 entities "Protein"
    text2 = "one two three four five"
    anns2 = ["T1\tProtein 0 7\tAnother", "T2\tProtein 8 13\tshort"]

    scorer.ingest_document(text2, anns2)

    assert scorer.total_docs == 2
    assert scorer.total_tokens == 15  # 10 + 5
    assert scorer.entity_counts["Gene"] == 1
    assert scorer.entity_counts["Protein"] == 2


def test_frequency_stats():
    scorer = FrequencyScorer([])

    # 1000 tokens total
    # 10 occurrences of "Common" -> Freq = 10/1000 = 0.01
    # 1 occurrence of "Rare" -> Freq = 1/1000 = 0.001

    # Mocking internal state directly for speed
    scorer.total_tokens = 1000
    scorer.entity_counts["Common"] = 100  # > RARE_THRESHOLD (10)
    scorer.entity_counts["Rare"] = 2  # < RARE_THRESHOLD (10)

    stats = scorer.get_stats()

    # Expect sorted by Frequency DESC
    assert stats[0]["Entity"] == "Common"
    assert stats[0]["Frequency"] == 0.1  # Wait, 100/1000 = 0.1
    assert "FREQUENT" in stats[0]["Strategy_Hint"]

    assert stats[1]["Entity"] == "Rare"
    assert stats[1]["Frequency"] == 0.002
    assert "RARE" in stats[1]["Strategy_Hint"]


def test_empty_corpus():
    scorer = FrequencyScorer([])
    stats = scorer.get_stats()
    assert stats == []
