"""
Automated regression tests for Asterline Plans and Billing (Pricing).
Validates that the assistant accurately retrieves, synthesizes, and cites
authentic pricing, seat counts, allowances, retention, overages, and seat policies
from corpus chunks pricing:0, pricing:1, and pricing:2 without inventing values.
"""

import pytest
from src.graph import AsterlineGraph
from src.retrieval import DocumentRetriever
from src.evaluation.metrics import (
    calculate_retrieval_recall,
    calculate_citation_precision,
    calculate_groundedness,
)


def test_pricing_query_retrieval():
    """Verify that retrieval accurately surfaces pricing:0 and pricing:1 with exact text."""
    retriever = DocumentRetriever()
    results = retriever.search(
        "What is the pricing and record allowance for the Launch and Team plans?",
        top_k=5
    )
    retrieved_ids = [r.chunk.chunk_id for r in results]
    
    assert "pricing:0" in retrieved_ids, "pricing:0 must be retrieved"
    assert "pricing:1" in retrieved_ids, "pricing:1 must be retrieved"
    
    p0 = next(r.chunk for r in results if r.chunk.chunk_id == "pricing:0")
    p1 = next(r.chunk for r in results if r.chunk.chunk_id == "pricing:1")
    
    # Verify exact corpus text
    assert "$59 per month" in p0.text
    assert "4 seats" in p0.text
    assert "1 million run records" in p0.text
    assert "$449 per month" in p0.text
    assert "20 seats" in p0.text
    assert "15 million run records" in p0.text
    assert "12% discount" in p0.text
    
    assert "21 days" in p1.text
    assert "180 days" in p1.text
    assert "$1.20 per 10,000 records" in p1.text
    assert "$0.70 per 10,000" in p1.text
    assert "$16 per seat per month" in p1.text
    assert "Launch cannot buy extra seats" in p1.text


def test_pricing_pipeline_all_expected_values():
    """
    End-to-end regression test for the pricing query.
    Checks all expected values from the original assignment dataset:
    - Launch: $59/month, 4 seats, 1 million run records/month.
    - Team: $449/month, 20 seats, 15 million run records/month.
    - Annual billing discount: 12% on Launch and Team base subscriptions.
    - Launch searchable retention: 21 days.
    - Team searchable retention: 180 days.
    - Launch overage: $1.20 per 10,000 records.
    - Team overage: $0.70 per 10,000 records.
    - Extra Team seats: $16 per seat/month.
    - Launch cannot buy extra seats.
    - Citations: pricing:0 and pricing:1
    """
    graph = AsterlineGraph()
    state = graph.invoke(
        user_id="regression_user",
        conversation_id="conv_pricing_regression_full",
        user_message="What is the pricing, seat count, monthly record allowance, searchable retention, overage rate, and extra seat policy for Launch and Team plans?"
    )

    answer = state.get("final_answer", "")
    citations = state.get("citations", [])
    cited_ids = [c["chunk_id"] for c in citations]
    answer_lower = answer.lower()

    # 1. Base pricing and seat counts
    assert "$59" in answer, "Must cite Launch pricing of $59"
    assert "4 seats" in answer_lower, "Must cite Launch seat count of 4 seats"
    assert ("1 million" in answer_lower or "1,000,000" in answer_lower), "Must cite Launch record allowance of 1 million"

    assert "$449" in answer, "Must cite Team pricing of $449"
    assert "20 seats" in answer_lower, "Must cite Team seat count of 20 seats"
    assert ("15 million" in answer_lower or "15,000,000" in answer_lower), "Must cite Team record allowance of 15 million"

    # 2. Annual discount
    assert "12%" in answer, "Must cite 12% annual billing discount"

    # 3. Searchable retention
    assert "21 days" in answer_lower, "Must cite Launch retention of 21 days"
    assert "180 days" in answer_lower, "Must cite Team retention of 180 days"

    # 4. Overages & extra seat policy
    assert "$1.20" in answer, "Must cite Launch overage rate of $1.20 per 10,000"
    assert "$0.70" in answer, "Must cite Team overage rate of $0.70 per 10,000"
    assert "$16" in answer, "Must cite Extra Team seats rate of $16"
    assert ("cannot buy extra seats" in answer_lower or "launch cannot" in answer_lower), "Must state Launch cannot buy extra seats"

    # 5. Hallucination guards - ensure outdated/invented mock values are NEVER present
    assert "$49" not in answer, "Must not hallucinate outdated $49 price"
    assert "$249" not in answer, "Must not hallucinate outdated $249 price"
    assert "100,000 run records" not in answer_lower, "Must not hallucinate 100k runs for Launch"
    assert "$18" not in answer, "Must not hallucinate $18 extra seats"
    assert "$24" not in answer, "Must not hallucinate $24 extra seats"

    # 6. Citations
    assert "pricing:0" in cited_ids, "Must cite pricing:0"
    assert "pricing:1" in cited_ids, "Must cite pricing:1"
    for c in citations:
        assert c["title"] == "Plans and billing"


def test_pricing_case_a_evaluation():
    """Verify that case_a_pricing in evaluation suite passes with authentic expectations."""
    graph = AsterlineGraph()
    state = graph.invoke(
        user_id="eval_user",
        conversation_id="conv_eval_case_a",
        user_message="What is the pricing and record allowance for the Launch and Team plans?"
    )

    answer = state.get("final_answer", "")
    retrieved_ids = [c["chunk_id"] for c in state.get("retrieved_chunks", [])]
    cited_ids = [c["chunk_id"] for c in state.get("citations", [])]

    # Verify retrieval recall for pricing:0 and pricing:1
    recall = calculate_retrieval_recall(["pricing:0", "pricing:1"], retrieved_ids)
    assert recall == 1.0, f"Expected perfect recall 1.0, got {recall}"

    # Verify citation precision
    prec = calculate_citation_precision(cited_ids, retrieved_ids, set(retrieved_ids))
    assert prec == 1.0, f"Expected citation precision 1.0, got {prec}"

    # Verify groundedness with authentic figures
    expected_entities = ["$59", "$449", "1 million", "15 million", "21 days", "180 days"]
    groundedness = calculate_groundedness(expected_entities, answer)
    assert groundedness >= 0.8, f"Expected groundedness >= 0.8, got {groundedness}"
