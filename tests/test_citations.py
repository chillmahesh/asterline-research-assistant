"""
Tests for citation accuracy and ground-truth validation.
Verifies that only real, retrieved corpus chunks are cited with correct chunk IDs and titles.
"""

import re
import json
from pathlib import Path
import pytest

from src.retrieval import DocumentRetriever
from src.agents.llm_client import LLMClient
from src.graph import AsterlineGraph

CORPUS_PATH = Path(__file__).resolve().parent.parent / "corpus.jsonl"


@pytest.fixture(scope="module")
def corpus_data():
    with open(CORPUS_PATH, "r", encoding="utf-8") as f:
        chunks = [json.loads(line) for line in f if line.strip()]
    return {c["chunk_id"]: c for c in chunks}


@pytest.fixture(scope="module")
def graph():
    return AsterlineGraph()


def test_pricing_citations_are_grounded(graph, corpus_data):
    state = graph.invoke(
        user_id="test_user",
        conversation_id="conv_cite_1",
        user_message="What is the pricing and record allowance for the Launch and Team plans?"
    )

    answer = state["final_answer"]
    assert answer, "Final answer must not be empty"

    # Extract all citations [Source: chunk_id — title]
    citations = re.findall(r"\[Source:\s*([^\s—]+)\s*—\s*([^\]]+)\]", answer)
    assert len(citations) > 0, "Response must include citations"

    for cid, title in citations:
        cid_clean = cid.strip()
        title_clean = title.strip()
        assert cid_clean in corpus_data, f"Hallucinated chunk ID detected: {cid_clean}"
        assert corpus_data[cid_clean]["title"] == title_clean, (
            f"Title mismatch for chunk {cid_clean}. "
            f"Expected '{corpus_data[cid_clean]['title']}', got '{title_clean}'"
        )


def test_missing_evidence_does_not_hallucinate_citations(graph):
    state = graph.invoke(
        user_id="test_user",
        conversation_id="conv_cite_2",
        user_message="What quantum computing algorithms does Asterline use to encrypt backups?"
    )

    answer = state["final_answer"]
    assert "cannot be answered" in answer.lower() or "not contain" in answer.lower() or "insufficient" in answer.lower()
    
    # Must not fabricate quantum citations
    citations = re.findall(r"\[Source:\s*([^\s—]+)\s*—\s*([^\]]+)\]", answer)
    assert len(citations) == 0, f"Unanswerable question must not invent citations: {citations}"
