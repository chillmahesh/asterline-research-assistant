"""
Tests for local document ingestion and retrieval.
Validates metadata preservation, relevance, and formatting of documentary evidence.
"""

import pytest
from src.retrieval import DocumentRetriever
from src.ingestion import load_corpus_chunks


@pytest.fixture
def retriever():
    return DocumentRetriever()


def test_load_all_75_chunks():
    chunks = load_corpus_chunks()
    assert len(chunks) == 75
    chunk_ids = [c.chunk_id for c in chunks]
    assert len(set(chunk_ids)) == 75


def test_retrieval_finds_pricing_evidence(retriever):
    results = retriever.search("Launch and Team plan pricing run records", top_k=5)
    assert len(results) > 0
    top_ids = [r.chunk.chunk_id for r in results]
    assert "pricing:0" in top_ids or "pricing:1" in top_ids
    
    # Verify metadata fields are preserved
    pricing_res = next(r for r in results if r.chunk.chunk_id in ("pricing:0", "pricing:1"))
    assert pricing_res.chunk.title == "Plans and billing"
    assert pricing_res.chunk.category == "pricing"
    assert pricing_res.chunk.owner == "Asterline Documentation"
    assert pricing_res.chunk.source_url == "kb://asterline/pricing/pricing"
    assert pricing_res.score > 0.0


def test_retrieval_finds_oncall_stipend_conflict(retriever):
    results = retriever.search("weekly on-call stipend operational runbook replacement", top_k=5)
    top_ids = [r.chunk.chunk_id for r in results]
    assert "handbook:1" in top_ids or "runbook:0" in top_ids
    
    # Format evidence verification
    formatted = retriever.format_evidence_for_prompt(results)
    assert "[Source: handbook:1 — Employee handbook]" in formatted or "[Source: runbook:0 — Operational runbook]" in formatted


def test_retrieval_finds_cold_storage_retention(retriever):
    results = retriever.search("raw run records cold recovery storage retention period", top_k=5)
    top_ids = [r.chunk.chunk_id for r in results]
    assert "retention:0" in top_ids or "security:1" in top_ids


def test_retrieval_finds_incidents(retriever):
    results_2026 = retriever.search("INC-2026-03 delayed signals alerts patch 2.3.1", top_k=5)
    top_ids_2026 = [r.chunk.chunk_id for r in results_2026]
    assert any("incident-03" in cid or "alerts:2" in cid or "release-23:1" in cid for cid in top_ids_2026)

    results_2025 = retriever.search("INC-2025-12 duplicate Warehouse Export rows Heron deduplication", top_k=5)
    top_ids_2025 = [r.chunk.chunk_id for r in results_2025]
    assert any("incident-12" in cid or "export:1" in cid or "release-21:2" in cid for cid in top_ids_2025)
