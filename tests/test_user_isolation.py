"""
Tests for strict multi-user memory and conversation isolation.
Validates that synthetic users cannot inspect, overwrite, or access each other's memories.
"""

import tempfile
from pathlib import Path
import pytest
from src.memory.repository import SQLiteMemoryRepository
from src.memory.service import MemoryService
from src.models import MemoryOperation


@pytest.fixture
def temp_service():
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp_dir:
        db_path = Path(tmp_dir) / "test_isolation.db"
        repo = SQLiteMemoryRepository(db_path=db_path)
        service = MemoryService(repository=repo)
        yield service, repo


def test_cross_user_memory_isolation(temp_service):
    service = temp_service[0]
    repo = temp_service[-1]
    user_a = "user_alice"
    user_b = "user_bob"

    # Alice saves a preference
    op_a = MemoryOperation(action="store", content="Alice prefers Python and pytest", category="preference")
    res_a = service.execute_memory_operation(user_a, op_a)
    assert res_a["success"] is True
    alice_mem_id = res_a["memory"]["id"]

    # Bob saves an opposing preference
    op_b = MemoryOperation(action="store", content="Bob prefers Java and JUnit", category="preference")
    res_b = service.execute_memory_operation(user_b, op_b)
    assert res_b["success"] is True
    bob_mem_id = res_b["memory"]["id"]

    # Retrieve Alice's memories
    alice_mems = service.get_user_context(user_a)
    assert len(alice_mems) == 1
    assert "Alice prefers Python" in alice_mems[0].content
    assert "Java" not in alice_mems[0].content

    # Retrieve Bob's memories
    bob_mems = service.get_user_context(user_b)
    assert len(bob_mems) == 1
    assert "Bob prefers Java" in bob_mems[0].content
    assert "Python" not in bob_mems[0].content

    # Bob attempts to read Alice's memory by ID
    bob_reads_alice = repo.get_memory_by_id(user_id=user_b, memory_id=alice_mem_id)
    assert bob_reads_alice is None, "Bob must not be able to read Alice's memory by ID"

    # Bob attempts to update Alice's memory
    bob_updates_alice = repo.update_memory(user_id=user_b, memory_id=alice_mem_id, new_content="Hacked content")
    assert bob_updates_alice is None, "Bob must not be able to update Alice's memory"

    # Bob attempts to forget Alice's memory
    bob_forgets_alice = repo.forget_memory(user_id=user_b, memory_id=alice_mem_id)
    assert bob_forgets_alice is False, "Bob must not be able to forget Alice's memory"

    # Alice's memory must remain completely unaltered
    alice_check = repo.get_memory_by_id(user_id=user_a, memory_id=alice_mem_id)
    assert alice_check is not None
    assert "Alice prefers Python" in alice_check.content


def test_cross_user_conversation_isolation(temp_service):
    service = temp_service[0]
    repo = temp_service[-1]
    user_a = "user_alice"
    user_b = "user_bob"

    conv_a = "conv_alice_1"
    conv_b = "conv_bob_1"

    # Save messages for Alice
    repo.save_message(user_id=user_a, conversation_id=conv_a, role="user", content="Secret project discussion for Alice")
    repo.save_message(user_id=user_a, conversation_id=conv_a, role="assistant", content="Here is Alice's answer")

    # Save messages for Bob
    repo.save_message(user_id=user_b, conversation_id=conv_b, role="user", content="Bob's public query")

    # Bob attempts to fetch Alice's conversation messages
    bob_sees_alice = repo.get_messages(user_id=user_b, conversation_id=conv_a)
    assert len(bob_sees_alice) == 0, "Bob must see 0 messages when requesting Alice's conversation ID"

    # Alice fetches her own messages
    alice_msgs = repo.get_messages(user_id=user_a, conversation_id=conv_a)
    assert len(alice_msgs) == 2
    assert "Secret project discussion for Alice" in alice_msgs[0]["content"]

    # Bob lists conversations
    bob_convs = repo.list_conversations(user_id=user_b)
    assert len(bob_convs) == 1
    assert bob_convs[0]["id"] == conv_b
