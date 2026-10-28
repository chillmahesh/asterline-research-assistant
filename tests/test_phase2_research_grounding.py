"""
Automated Test Suite for Phase 2: Research and Grounded-Answer Tests (Tests A - I).
"""

import pytest
from src.graph import AsterlineGraph


@pytest.fixture(scope="module")
def graph():
    return AsterlineGraph()


def test_phase2_test_a_basic_pricing(graph):
    state = graph.invoke(
        user_id="user_test_a",
        conversation_id="conv_test_a",
        user_message="What are the monthly prices of the Launch and Team plans, and what does each include?"
    )
    answer = state["final_answer"]
    answer_lower = answer.lower()
    citations = [c["chunk_id"] for c in state["citations"]]

    assert "$59" in answer, "Launch must be $59"
    assert "4 seats" in answer_lower, "Launch includes 4 seats"
    assert "1 million" in answer_lower or "1,000,000" in answer_lower, "Launch includes 1M records"
    assert "$449" in answer, "Team must be $449"
    assert "20 seats" in answer_lower, "Team includes 20 seats"
    assert "15 million" in answer_lower or "15,000,000" in answer_lower, "Team includes 15M records"
    assert "12%" in answer, "Must cite 12% annual billing discount"
    assert "pricing:0" in citations, "Must cite pricing:0"


def test_phase2_test_b_plan_limits(graph):
    state = graph.invoke(
        user_id="user_test_b",
        conversation_id="conv_test_b",
        user_message="Compare searchable retention, overage rates, extra seats, alert-rule limits, and Warehouse Export for Launch and Team."
    )
    answer = state["final_answer"]
    answer_lower = answer.lower()
    citations = [c["chunk_id"] for c in state["citations"]]

    assert "21 days" in answer_lower, "Launch retention 21 days"
    assert "180 days" in answer_lower, "Team retention 180 days"
    assert "$1.20" in answer, "Launch overage $1.20 per 10,000"
    assert "$0.70" in answer, "Team overage $0.70 per 10,000"
    assert "$16" in answer, "Team extra seats $16/seat/month"
    assert "cannot buy extra seats" in answer_lower or "cannot purchase" in answer_lower or "launch cannot" in answer_lower
    assert "6" in answer, "Launch 6 alert rules"
    assert "75" in answer, "Team 75 alert rules"
    assert "unavailable" in answer_lower or "no warehouse export" in answer_lower or "not available" in answer_lower
    assert "included" in answer_lower or "team" in answer_lower
    assert "pricing:1" in citations or "pricing:2" in citations


def test_phase2_test_c_conflicting_stipend(graph):
    state = graph.invoke(
        user_id="user_test_c",
        conversation_id="conv_test_c",
        user_message="The old runbook says the weekly on-call stipend is $140, but the employee handbook says $175. Which is current?"
    )
    answer = state["final_answer"]
    citations = [c["chunk_id"] for c in state["citations"]]

    assert "$175" in answer, "Current stipend is $175"
    assert "2026-01-01" in answer or "january 1, 2026" in answer.lower(), "Effective date 2026-01-01"
    assert "supersedes" in answer.lower() or "replacing" in answer.lower() or "replaces" in answer.lower()
    assert "handbook:1" in citations, "Must cite handbook:1"


def test_phase2_test_d_conflicting_retention(graph):
    state = graph.invoke(
        user_id="user_test_d",
        conversation_id="conv_test_d",
        user_message="Security documentation says 60 days of cold storage, but the newer retention policy says 20 days. Which applies? Does cold storage extend searchable retention?"
    )
    answer = state["final_answer"]
    answer_lower = answer.lower()
    citations = [c["chunk_id"] for c in state["citations"]]

    assert "20 days" in answer_lower, "Must state 20 days"
    assert "supersedes" in answer_lower or "newer policy" in answer_lower or "replaces" in answer_lower
    assert "not" in answer_lower and ("searchable" in answer_lower or "aql" in answer_lower), "Cold storage is not searchable through AQL"
    assert "retention:0" in citations or "retention:1" in citations or "security:1" in citations


def test_phase2_test_e_incident_investigation(graph):
    state = graph.invoke(
        user_id="user_test_e",
        conversation_id="conv_test_e",
        user_message="Investigate INC-2026-03. What happened, what caused it, how was it fixed, and were alerts lost?"
    )
    answer = state["final_answer"]
    answer_lower = answer.lower()
    citations = [c["chunk_id"] for c in state["citations"]]

    assert "34 minutes" in answer_lower or "34 min" in answer_lower, "Delayed up to 34 minutes"
    assert "sev-3" in answer_lower, "Classified SEV-3"
    assert "clock" in answer_lower, "Wall clock backward / time sync"
    assert "2.3.1" in answer, "Fixed in Patch 2.3.1"
    assert "monotonic" in answer_lower, "Monotonic clock"
    assert "no alerts were lost" in answer_lower or "no alerts lost" in answer_lower, "No alerts lost"
    assert any(c in citations for c in ["incident-03:0", "incident-03:1", "release-23:1"])


def test_phase2_test_f_release_notes_v23(graph):
    state = graph.invoke(
        user_id="user_test_f",
        conversation_id="conv_test_f",
        user_message="What changed in version 2.3, and what happened to the AQL timeout?"
    )
    answer = state["final_answer"]
    answer_lower = answer.lower()
    citations = [c["chunk_id"] for c in state["citations"]]

    assert "pagerduty" in answer_lower, "PagerDuty added as Signals destination"
    assert "50" in answer, "AQL timeout increased to 50 seconds"
    assert "25" in answer, "Previous AQL timeout was 25 seconds"
    assert "unchanged" in answer_lower or "same" in answer_lower, "Query row / concurrency limits unchanged"
    assert "release-23:0" in citations or "query:0" in citations


def test_phase2_test_g_missing_information_apac(graph):
    state = graph.invoke(
        user_id="user_test_g",
        conversation_id="conv_test_g",
        user_message="Give me the confirmed launch date for Asterline's Asia-Pacific region."
    )
    answer = state["final_answer"]
    answer_lower = answer.lower()
    citations = [c["chunk_id"] for c in state["citations"]]

    assert "no launch" in answer_lower or "no announced" in answer_lower or "not announced" in answer_lower
    assert any(c in citations for c in ["faq:1", "regions:2"])


def test_phase2_test_h_unsupported_certifications(graph):
    state = graph.invoke(
        user_id="user_test_h",
        conversation_id="conv_test_h",
        user_message="List Asterline's healthcare-specific certifications."
    )
    answer = state["final_answer"]
    answer_lower = answer.lower()
    citations = [c["chunk_id"] for c in state["citations"]]

    assert "no claims" in answer_lower or "does not establish" in answer_lower or "not list" in answer_lower or "no healthcare" in answer_lower
    assert "security:2" in citations


def test_phase2_test_i_complex_multi_source_comparison(graph):
    state = graph.invoke(
        user_id="user_test_i",
        conversation_id="conv_test_i",
        user_message="Compare Launch and Team for a customer expecting 10 million run records per month. Include base price, included volume, overage rate, retention, and Warehouse Export. Explain assumptions."
    )
    answer = state["final_answer"]
    answer_lower = answer.lower()
    citations = [c["chunk_id"] for c in state["citations"]]

    assert "pricing:0" in citations, "Must cite pricing:0"
    assert "pricing:1" in citations, "Must cite pricing:1"
    assert "pricing:2" in citations, "Must cite pricing:2"

    assert "15 million" in answer_lower or "15m" in answer_lower
    assert "$449" in answer
    assert "$59" in answer
    assert "9 million" in answer_lower or "9,000,000" in answer_lower
    assert "$1,080" in answer or "1080" in answer
    assert "$1,139" in answer or "1139" in answer
    assert "assum" in answer_lower
    assert "invoice" in answer_lower or "quote" in answer_lower or "commercial" in answer_lower
