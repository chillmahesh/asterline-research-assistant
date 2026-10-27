"""
Deterministic evaluation metrics for Asterline Research Assistant.
Measures retrieval recall, citation precision, conflict resolution,
groundedness, and memory lifecycle operations.
"""

from typing import List, Set, Dict, Any


def calculate_retrieval_recall(expected_chunks: List[str], retrieved_chunk_ids: List[str]) -> float:
    """Calculates recall: fraction of expected ground-truth chunks retrieved."""
    if not expected_chunks:
        return 1.0  # If none expected (e.g. unanswerable), perfect recall
    matched = set(expected_chunks).intersection(set(retrieved_chunk_ids))
    return len(matched) / len(expected_chunks)


def calculate_citation_precision(
    cited_chunk_ids: List[str],
    retrieved_chunk_ids: List[str],
    valid_corpus_chunk_ids: Set[str]
) -> float:
    """
    Calculates citation precision:
    1. Cited chunk must exist in valid corpus (no hallucinations).
    2. Cited chunk must have actually been retrieved by the research agent.
    """
    if not cited_chunk_ids:
        return 1.0
    valid_retrieved = set(retrieved_chunk_ids).intersection(valid_corpus_chunk_ids)
    correct_citations = [cid for cid in cited_chunk_ids if cid in valid_retrieved]
    return len(correct_citations) / len(cited_chunk_ids)


def calculate_groundedness(expected_entities: List[str], answer: str) -> float:
    """Fraction of expected ground-truth entities present in the final answer."""
    if not expected_entities:
        return 1.0
    answer_lower = answer.lower()
    found = [e for e in expected_entities if e.lower() in answer_lower]
    return len(found) / len(expected_entities)


def calculate_conflict_resolution(answer: str, current_fact: str, superseded_fact: str) -> float:
    """
    Evaluates whether the answer correctly prioritizes the current fact
    over the superseded historical fact and mentions supersession/replacement.
    """
    answer_lower = answer.lower()
    has_current = current_fact.lower() in answer_lower
    has_superseded = superseded_fact.lower() in answer_lower
    has_supersession_keyword = any(k in answer_lower for k in ["supersede", "replace", "effective", "update"])

    if has_current and has_supersession_keyword:
        return 1.0
    elif has_current:
        return 0.7
    return 0.0


def calculate_unanswerable_honesty(answer: str) -> float:
    """Evaluates whether an unanswerable question was honestly rejected."""
    answer_lower = answer.lower()
    phrases = ["cannot be answered", "insufficient evidence", "not mentioned", "not contain", "does not reference"]
    return 1.0 if any(p in answer_lower for p in phrases) else 0.0
