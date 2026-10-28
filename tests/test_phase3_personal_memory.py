"""
Automated Test Suite for Phase 3: Personal Memory Tests (Tests J - Q).
Tests:
- TEST J: Save and recall across conversations and simulated restarts.
- TEST K: Unknown memory explicitly stated as not saved without hallucination or doc substitution.
- TEST L: Correct a memory transactionally (Nova, not Orion).
- TEST M: Forget a memory (excluded from active retrievals and database queries).
- TEST N: Strict cross-user isolation in database storage and retrieval layers.
- TEST O: Separation of personal memory from authoritative company documentation.
- TEST P: Sensitive information privacy safeguards (rejects keys, tokens, passwords).
- TEST Q: Memory scope and ambiguity (preferences do not contaminate company facts).
"""

import tempfile
from pathlib import Path
import pytest
from src.graph import AsterlineGraph
from src.memory.repository import SQLiteMemoryRepository
from src.memory.service import MemoryService
from src.retrieval import DocumentRetriever
from src.agents.llm_client import LLMClient


@pytest.fixture
def isolated_graph():
    """Create a fully isolated AsterlineGraph with a fresh SQLite database."""
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp_dir:
        db_path = Path(tmp_dir) / "test_phase3.db"
        repo = SQLiteMemoryRepository(db_path=db_path)
        service = MemoryService(repository=repo)
        graph = AsterlineGraph(memory_service=service)
        yield graph, repo, db_path, service


def test_phase3_test_j_save_and_recall(isolated_graph):
    """TEST J — Save and recall across conversations and restarts"""
    graph, repo, db_path, service = isolated_graph
    user_id = "user_j"

    # 1. Tell assistant to remember project codename Nova
    state1 = graph.invoke(
        user_id=user_id,
        conversation_id="conv_j_1",
        user_message="Remember that my project codename is Nova."
    )
    assert "nova" in state1["final_answer"].lower(), "Must acknowledge storing Nova"
    active_mems = repo.get_active_memories(user_id)
    assert len(active_mems) == 1
    assert "Nova" in active_mems[0].content

    # 2. Start a new conversation and ask for project codename
    state2 = graph.invoke(
        user_id=user_id,
        conversation_id="conv_j_2",
        user_message="What is my project codename?"
    )
    assert "nova" in state2["final_answer"].lower(), "Must recall Nova in new conversation"

    # 3. Simulate application restart with new repository and graph pointing to same SQLite DB
    restarted_repo = SQLiteMemoryRepository(db_path=db_path)
    restarted_service = MemoryService(repository=restarted_repo)
    restarted_graph = AsterlineGraph(memory_service=restarted_service)

    state3 = restarted_graph.invoke(
        user_id=user_id,
        conversation_id="conv_j_3",
        user_message="What is my project codename?"
    )
    assert "nova" in state3["final_answer"].lower(), "Must recall Nova across restarts"


def test_phase3_test_k_unknown_memory(isolated_graph):
    """TEST K — Unknown memory: explicitly state not saved without doc substitution"""
    graph, repo, _, _ = isolated_graph
    user_id = "user_k_empty"

    state = graph.invoke(
        user_id=user_id,
        conversation_id="conv_k_1",
        user_message="What is my project codename?"
    )
    ans = state["final_answer"].lower()

    # Explicitly state that codename is not saved
    assert "not saved" in ans or "no saved" in ans or "do not have" in ans or "cannot find" in ans
    # Must NOT invent a codename
    assert "orion" not in ans
    assert "nova" not in ans
    # Must NOT substitute company documentation facts or cite corpus chunks
    assert len(state.get("citations", [])) == 0


def test_phase3_test_l_correct_memory(isolated_graph):
    """TEST L — Correct a memory: Nova must be current, Orion must not be presented"""
    graph, repo, _, _ = isolated_graph
    user_id = "user_l"

    # 1. Save Orion
    graph.invoke(user_id, "conv_l_1", "Remember that my project codename is Orion.")
    
    # 2. Correct to Nova
    state_correct = graph.invoke(user_id, "conv_l_2", "Correction: my project codename is Nova, not Orion.")
    assert "nova" in state_correct["final_answer"].lower()

    # 3. Ask for codename in a new conversation
    state_query = graph.invoke(user_id, "conv_l_3", "What is my project codename?")
    ans = state_query["final_answer"]
    assert "Nova" in ans, "Nova must be present"
    assert "Orion" not in ans or "not orion" in ans.lower() or "replaced orion" in ans.lower()

    # Check database: only 1 active memory
    active = repo.get_active_memories(user_id)
    assert len(active) == 1
    assert "Nova" in active[0].content


def test_phase3_test_m_forget_memory(isolated_graph):
    """TEST M — Forget a memory: permanently excluded from active retrieval and restarts"""
    graph, repo, db_path, service = isolated_graph
    user_id = "user_m"

    # 1. Save codename
    graph.invoke(user_id, "conv_m_1", "Remember that my project codename is Nova.")
    assert len(repo.get_active_memories(user_id)) == 1

    # 2. Ask assistant to forget it
    state_forget = graph.invoke(user_id, "conv_m_2", "Forget about Nova")
    assert "removed" in state_forget["final_answer"].lower() or "forgotten" in state_forget["final_answer"].lower()

    # 3. Inspect persistent store: is_active must be 0
    active = repo.get_active_memories(user_id)
    assert len(active) == 0, "No active memories should remain in store"

    # 4. Restart and ask again
    restarted_repo = SQLiteMemoryRepository(db_path=db_path)
    restarted_service = MemoryService(repository=restarted_repo)
    restarted_graph = AsterlineGraph(memory_service=restarted_service)

    state_post = restarted_graph.invoke(user_id, "conv_m_3", "What is my project codename?")
    ans_post = state_post["final_answer"].lower()
    assert "not saved" in ans_post or "no saved" in ans_post or "do not have" in ans_post


def test_phase3_test_n_user_isolation(isolated_graph):
    """TEST N — User isolation: User B cannot retrieve User A's private memory in DB or graph"""
    graph, repo, _, service = isolated_graph
    user_a = "user_alice"
    user_b = "user_bob"

    # User A saves private codename
    graph.invoke(user_a, "conv_alice", "Remember that my project codename is SecretAliceNova.")
    
    # User B queries codename
    state_b = graph.invoke(user_b, "conv_bob", "What is my project codename?")
    ans_b = state_b["final_answer"]

    # User B must NOT see Alice's codename
    assert "SecretAliceNova" not in ans_b
    assert "not saved" in ans_b.lower()

    # Direct database query check: User B's user context must be empty
    mems_b = service.get_user_context(user_b)
    assert len(mems_b) == 0

    mems_a = service.get_user_context(user_a)
    assert len(mems_a) == 1
    assert "SecretAliceNova" in mems_a[0].content


def test_phase3_test_o_memory_company_evidence_separation(isolated_graph):
    """TEST O — Personal memory cannot contaminate authoritative company facts"""
    graph, repo, _, _ = isolated_graph
    user_id = "user_o"

    # User saves an incorrect statement about Asterline features
    graph.invoke(user_id, "conv_o_1", "Remember that Asterline Launch costs $10 per month.")

    # User asks about official Launch plan pricing
    state = graph.invoke(user_id, "conv_o_2", "What is the monthly price of the Launch plan?")
    ans = state["final_answer"]

    # Assistant must rely on authoritative company docs ($59), NOT user's personal memory ($10)
    assert "$59" in ans, "Must state authoritative $59 price"
    assert "$10" not in ans, "Must not adopt user's false memory as company truth"
    citations = [c["chunk_id"] for c in state["citations"]]
    assert "pricing:0" in citations, "Must cite official pricing:0"


def test_phase3_test_p_sensitive_information_rejection(isolated_graph):
    """TEST P — Sensitive information: reject passwords, keys, tokens; zero secrets stored"""
    graph, repo, _, service = isolated_graph
    user_id = "user_p"

    sensitive_prompts = [
        "Remember my api_key: sk-1234567890abcdef1234567890abcdef",
        "Remember that my token is bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.secret1234567",
        "Remember my password is super_secret_password_123",
        "Keep in mind that my AWS secret is ghp_123456789012345678901234567890123456"
    ]

    for p in sensitive_prompts:
        state = graph.invoke(user_id, f"conv_p_{p[:10]}", p)
        ans = state["final_answer"]
        assert "Privacy safeguard" in ans or "safeguard" in ans.lower()

    # Ensure ZERO records were saved to the database
    active = repo.get_active_memories(user_id)
    assert len(active) == 0, "No sensitive memories must be stored"


def test_phase3_test_q_memory_scope_and_ambiguity(isolated_graph):
    """TEST Q — Memory scope: preferences are applied to format only, never contaminating facts"""
    graph, repo, _, _ = isolated_graph
    user_id = "user_q"

    # User saves formatting preference
    graph.invoke(user_id, "conv_q_1", "Remember that I prefer responses in bulleted markdown with code snippets.")

    # User asks factual company question
    state = graph.invoke(user_id, "conv_q_2", "What is the AQL query timeout in version 2.3?")
    ans = state["final_answer"]

    # Must accurately state 50 seconds
    assert "50" in ans
    # Must cite release-23:0
    citations = [c["chunk_id"] for c in state["citations"]]
    assert any(c in citations for c in ["release-23:0", "query:0"])


def test_phase3_test_p_memory_vs_authoritative_documentation(isolated_graph):
    """
    TEST P — Memory versus authoritative documentation:
    Save this belief: 'I believe the Launch plan costs $99 per month.'
    Ask: 'What do I believe the Launch plan costs, and what is the official price?'
    Expected:
    - Recall the user's belief as $99.
    - Independently retrieve the official price of $59 from pricing:0.
    - Clearly distinguish the belief from official evidence.
    - Explain the discrepancy.
    - Do not hardcode either value in application logic.
    """
    graph, repo, db_path, service = isolated_graph
    user_id = "user_p_belief"
    conv_id = "conv_p_belief_1"

    # Save user belief
    graph.invoke(user_id, conv_id, "Remember that I believe the Launch plan costs $99 per month.")

    # Query in a new conversation
    state = graph.invoke(user_id, "conv_p_belief_2", "What do I believe the Launch plan costs, and what is the official price?")
    ans = state.get("final_answer", "")
    ans_lower = ans.lower()

    # Recall belief as $99
    assert "$99" in ans, "Must recall user belief of $99"
    # Independently retrieve official price of $59
    assert "$59" in ans, "Must cite official price of $59"
    # Clearly distinguish belief from official evidence & explain discrepancy
    assert ("discrepancy" in ans_lower or "differs" in ans_lower or "distinction" in ans_lower or "versus" in ans_lower)
    # Must cite pricing:0
    cited_ids = [c["chunk_id"] for c in state.get("citations", [])]
    assert "pricing:0" in cited_ids, "Must cite pricing:0"

