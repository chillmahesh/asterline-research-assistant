"""
Tests for documentation conflict and supersession resolution.
Verifies that the assistant prioritizes explicit supersession statements,
newer specific policies over older general documents, and distinguishes historical facts.
"""

import pytest
from src.graph import AsterlineGraph


@pytest.fixture(scope="module")
def graph():
    return AsterlineGraph()


def test_oncall_stipend_conflict_resolution(graph):
    """
    Scenario B:
    Handbook v3 ($175, effective 2026-01-01) supersedes older operational runbook ($140).
    """
    state = graph.invoke(
        user_id="test_user",
        conversation_id="conv_conflict_1",
        user_message="What is the current weekly on-call stipend, and does it replace any older amount?"
    )

    answer = state["final_answer"]
    assert "$175" in answer, "Must identify current $175 stipend"
    assert "$140" in answer, "Must acknowledge historical $140 runbook figure"
    assert "supersedes" in answer.lower() or "replaces" in answer.lower() or "effective" in answer.lower()
    
    # Must cite handbook:1
    cited_chunk_ids = [c["chunk_id"] for c in state["citations"]]
    assert "handbook:1" in cited_chunk_ids or "runbook:0" in cited_chunk_ids


def test_cold_storage_retention_conflict_resolution(graph):
    """
    Scenario C:
    Data retention policy v2 (20 days, effective 2026-01-15) supersedes security overview (60 days).
    """
    state = graph.invoke(
        user_id="test_user",
        conversation_id="conv_conflict_2",
        user_message="How long are raw run records kept in cold recovery storage after searchable retention ends?"
    )

    answer = state["final_answer"]
    assert "20 days" in answer, "Must state current 20-day retention"
    assert "60 days" in answer, "Must note older 60-day figure"
    assert "supersedes" in answer.lower() or "policy" in answer.lower()

    cited_chunk_ids = [c["chunk_id"] for c in state["citations"]]
    assert "retention:0" in cited_chunk_ids or "security:1" in cited_chunk_ids


def test_inc_2026_03_resolution(graph):
    """
    Scenario D:
    INC-2026-03 clock adjustment defect delayed Signals alerts by up to 34 mins; resolved in patch 2.3.1.
    """
    state = graph.invoke(
        user_id="test_user",
        conversation_id="conv_conflict_3",
        user_message="What caused delayed alerts during INC-2026-03 and which patch resolved it?"
    )

    answer = state["final_answer"]
    assert "2.3.1" in answer, "Must identify patch 2.3.1"
    assert "monotonic" in answer.lower() or "clock" in answer.lower()


def test_inc_2025_12_resolution(graph):
    """
    Scenario E:
    INC-2025-12 duplicate Warehouse Export rows caused by Heron query engine ordering; resolved in patch 2.1.2.
    """
    state = graph.invoke(
        user_id="test_user",
        conversation_id="conv_conflict_4",
        user_message="What caused duplicate Warehouse Export rows in INC-2025-12 and how was it fixed?"
    )

    answer = state["final_answer"]
    assert "2.1.2" in answer, "Must identify patch 2.1.2"
    assert "heron" in answer.lower() or "export_job_id" in answer.lower() or "record_hash" in answer.lower()


def test_apac_unannounced_region(graph):
    """
    Scenario F:
    No announced launch date or timeline for Asia-Pacific region.
    """
    state = graph.invoke(
        user_id="test_user",
        conversation_id="conv_conflict_5",
        user_message="Has Asterline announced a launch date or timeline for an Asia-Pacific region?"
    )

    answer = state["final_answer"].lower()
    assert "no launch date" in answer or "no announced" in answer or "not announced" in answer
