"""
Tests for persistent user memory, corrections, forgetting, and privacy safeguards.
"""

import tempfile
from pathlib import Path
import pytest
from src.memory.repository import SQLiteMemoryRepository
from src.memory.service import MemoryService
from src.models import MemoryOperation


@pytest.fixture
def temp_service():
    """Create a temporary SQLite memory repository and service for isolated testing."""
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp_dir:
        db_path = Path(tmp_dir) / "test_memory.db"
        repo = SQLiteMemoryRepository(db_path=db_path)
        service = MemoryService(repository=repo)
        yield service, db_path, repo


def test_memory_creation_and_persistence_across_restarts(temp_service):
    service, db_path, repo = temp_service
    user_id = "user_alice"

    # Store a memory
    op = MemoryOperation(action="store", content="User prefers concise markdown tables", category="preference")
    res = service.execute_memory_operation(user_id, op)
    assert res["success"] is True

    # Check active memories
    mems = service.get_user_context(user_id)
    assert len(mems) == 1
    assert "concise markdown tables" in mems[0].content

    # Simulate full application restart by creating a new repository instance pointing to the same DB file
    new_repo = SQLiteMemoryRepository(db_path=db_path)
    new_service = MemoryService(repository=new_repo)

    restarted_mems = new_service.get_user_context(user_id)
    assert len(restarted_mems) == 1
    assert restarted_mems[0].content == mems[0].content
    assert restarted_mems[0].id == mems[0].id


def test_memory_correction(temp_service):
    service, db_path, repo = temp_service
    user_id = "user_alice"

    # Create initial memory
    op1 = MemoryOperation(action="store", content="Focusing on Warehouse Export project", category="project")
    res1 = service.execute_memory_operation(user_id, op1)
    mem_id = res1["memory"]["id"]

    # Correct memory
    op2 = MemoryOperation(action="correct", memory_id=mem_id, content="Switched focus to Signals Alerting team", category="project")
    res2 = service.execute_memory_operation(user_id, op2)
    assert res2["success"] is True
    assert res2["memory"]["content"] == "Switched focus to Signals Alerting team"

    # Verify only updated memory is active
    active = service.get_user_context(user_id)
    assert len(active) == 1
    assert active[0].content == "Switched focus to Signals Alerting team"


def test_memory_explicit_forgetting(temp_service):
    service, db_path, repo = temp_service
    user_id = "user_alice"

    # Store two memories
    service.execute_memory_operation(user_id, MemoryOperation(action="store", content="Needs weekly status summaries", category="preference"))
    res2 = service.execute_memory_operation(user_id, MemoryOperation(action="store", content="Debugging INC-2026-03", category="project"))
    inc_mem_id = res2["memory"]["id"]

    assert len(service.get_user_context(user_id)) == 2

    # Forget INC memory
    res_forget = service.execute_memory_operation(user_id, MemoryOperation(action="forget", memory_id=inc_mem_id))
    assert res_forget["success"] is True

    # Check that forgotten memory is excluded from future retrievals
    active = service.get_user_context(user_id)
    assert len(active) == 1
    assert active[0].content == "Needs weekly status summaries"
    assert not any("INC-2026-03" in m.content for m in active)


def test_privacy_safeguard_refuses_secrets(temp_service):
    service, db_path, repo = temp_service
    user_id = "user_alice"

    # Attempt to store sensitive credential
    secret_payload = "My secret token is bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.sensitive_payload_data"
    op = MemoryOperation(action="store", content=secret_payload, category="general")
    res = service.execute_memory_operation(user_id, op)

    assert res["success"] is False
    assert res["action"] == "store_rejected"
    assert "Privacy safeguard" in res["reason"]

    # Ensure nothing was stored
    assert len(service.get_user_context(user_id)) == 0


def test_intent_detection(temp_service):
    service, _, _ = temp_service

    op1 = service.parse_memory_intent("Please remember that I prefer short bullet points")
    assert op1 is not None
    assert op1.action == "store"
    assert "short bullet points" in op1.content
    assert op1.category == "preference"

    op2 = service.parse_memory_intent("Forget about the legacy warehouse project")
    assert op2 is not None
    assert op2.action == "forget"
    assert "legacy warehouse project" in op2.content

    op3 = service.parse_memory_intent("Actually, my role is Lead Reliability Engineer")
    assert op3 is not None
    assert op3.action == "correct"
    assert "Lead Reliability Engineer" in op3.content
