"""
SQLite-backed persistent memory and conversation storage.
Enforces strict user isolation at the SQL storage layer for all CRUD operations.
"""

import sqlite3
import uuid
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional, Dict, Any
from src.config import settings
from src.models import MemoryRecord


class SQLiteMemoryRepository:
    """
    Direct SQLite repository for user memories and conversation history.
    Every single query requires and enforces the user_id parameter.
    """

    def __init__(self, db_path: Optional[Path] = None):
        self.db_path = db_path or settings.sqlite_db_path
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON;")
        return conn

    def _init_db(self):
        """Create tables for conversations, messages, and user-isolated memories."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS conversations (
                    id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    title TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
            """)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS messages (
                    id TEXT PRIMARY KEY,
                    conversation_id TEXT NOT NULL,
                    user_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    citations_json TEXT,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY (conversation_id) REFERENCES conversations(id) ON DELETE CASCADE
                )
            """)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS memories (
                    id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    content TEXT NOT NULL,
                    category TEXT NOT NULL DEFAULT 'general',
                    confidence REAL NOT NULL DEFAULT 1.0,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    is_active INTEGER NOT NULL DEFAULT 1,
                    superseded_by TEXT
                )
            """)
            # Indices for rapid user-scoped queries
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_memories_user ON memories(user_id, is_active);")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_conversations_user ON conversations(user_id);")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_messages_conv ON messages(conversation_id, user_id);")
            conn.commit()

    @staticmethod
    def _now() -> str:
        return datetime.now(timezone.utc).isoformat()

    # =========================================================================
    # User Memory Operations (Strict user_id isolation)
    # =========================================================================

    def create_memory(
        self,
        user_id: str,
        content: str,
        category: str = "general",
        confidence: float = 1.0
    ) -> MemoryRecord:
        """Create a new active memory scoped to user_id."""
        cleaned_content = content.strip()
        if not cleaned_content:
            raise ValueError("Memory content cannot be empty")

        mem_id = str(uuid.uuid4())
        now = self._now()

        with self._get_connection() as conn:
            conn.execute(
                """
                INSERT INTO memories (id, user_id, content, category, confidence, created_at, updated_at, is_active)
                VALUES (?, ?, ?, ?, ?, ?, ?, 1)
                """,
                (mem_id, user_id, cleaned_content, category, confidence, now, now)
            )
            conn.commit()

        return MemoryRecord(
            id=mem_id,
            user_id=user_id,
            content=cleaned_content,
            category=category,
            confidence=confidence,
            created_at=now,
            updated_at=now,
            is_active=True
        )

    def get_active_memories(self, user_id: str) -> List[MemoryRecord]:
        """Retrieve all currently active memories for the specified user ONLY."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT id, user_id, content, category, confidence, created_at, updated_at, is_active
                FROM memories
                WHERE user_id = ? AND is_active = 1
                ORDER BY created_at DESC
                """,
                (user_id,)
            )
            rows = cursor.fetchall()
            return [
                MemoryRecord(
                    id=row["id"],
                    user_id=row["user_id"],
                    content=row["content"],
                    category=row["category"],
                    confidence=row["confidence"],
                    created_at=row["created_at"],
                    updated_at=row["updated_at"],
                    is_active=bool(row["is_active"])
                )
                for row in rows
            ]

    def get_memory_by_id(self, user_id: str, memory_id: str) -> Optional[MemoryRecord]:
        """Fetch a specific memory ensuring it belongs to user_id."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT id, user_id, content, category, confidence, created_at, updated_at, is_active
                FROM memories
                WHERE user_id = ? AND id = ?
                """,
                (user_id, memory_id)
            )
            row = cursor.fetchone()
            if not row:
                return None
            return MemoryRecord(
                id=row["id"],
                user_id=row["user_id"],
                content=row["content"],
                category=row["category"],
                confidence=row["confidence"],
                created_at=row["created_at"],
                updated_at=row["updated_at"],
                is_active=bool(row["is_active"])
            )

    def update_memory(
        self,
        user_id: str,
        memory_id: str,
        new_content: str,
        new_category: Optional[str] = None
    ) -> Optional[MemoryRecord]:
        """
        Transactionally update/correct an existing memory for user_id.
        Ensures atomic operation so no conflicting intermediate state is retained.
        """
        cleaned_content = new_content.strip()
        if not cleaned_content:
            raise ValueError("Updated content cannot be empty")

        now = self._now()
        with self._get_connection() as conn:
            cursor = conn.cursor()
            # Verify memory exists and belongs to user
            cursor.execute(
                "SELECT id, category FROM memories WHERE user_id = ? AND id = ? AND is_active = 1",
                (user_id, memory_id)
            )
            row = cursor.fetchone()
            if not row:
                return None

            category = new_category or row["category"]

            cursor.execute(
                """
                UPDATE memories
                SET content = ?, category = ?, updated_at = ?
                WHERE user_id = ? AND id = ?
                """,
                (cleaned_content, category, now, user_id, memory_id)
            )
            conn.commit()

        return self.get_memory_by_id(user_id, memory_id)

    def forget_memory(self, user_id: str, memory_id: str) -> bool:
        """
        Transactionally deactivates/forgets a memory so it never appears in future retrievals.
        Scoped strictly to user_id.
        """
        now = self._now()
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                UPDATE memories
                SET is_active = 0, updated_at = ?
                WHERE user_id = ? AND id = ? AND is_active = 1
                """,
                (now, user_id, memory_id)
            )
            conn.commit()
            return cursor.rowcount > 0

    def forget_by_topic(self, user_id: str, topic_keyword: str) -> int:
        """
        Forgets all active memories for user_id containing the topic keyword (case-insensitive).
        """
        now = self._now()
        pattern = f"%{topic_keyword.strip()}%"
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                UPDATE memories
                SET is_active = 0, updated_at = ?
                WHERE user_id = ? AND is_active = 1 AND content LIKE ?
                """,
                (now, user_id, pattern)
            )
            conn.commit()
            return cursor.rowcount

    # =========================================================================
    # Conversation & Message History Operations (Strict user_id isolation)
    # =========================================================================

    def get_or_create_conversation(self, user_id: str, conversation_id: str, title: str = "New Conversation") -> Dict[str, Any]:
        """Fetch or create a conversation record scoped to user_id."""
        now = self._now()
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT id, user_id, title, created_at, updated_at FROM conversations WHERE id = ? AND user_id = ?", (conversation_id, user_id))
            row = cursor.fetchone()
            if row:
                return dict(row)

            cursor.execute(
                "INSERT INTO conversations (id, user_id, title, created_at, updated_at) VALUES (?, ?, ?, ?, ?)",
                (conversation_id, user_id, title, now, now)
            )
            conn.commit()
            return {"id": conversation_id, "user_id": user_id, "title": title, "created_at": now, "updated_at": now}

    def list_conversations(self, user_id: str) -> List[Dict[str, Any]]:
        """List all conversations for user_id in descending chronological order."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT id, user_id, title, created_at, updated_at FROM conversations WHERE user_id = ? ORDER BY updated_at DESC",
                (user_id,)
            )
            return [dict(r) for r in cursor.fetchall()]

    def save_message(
        self,
        user_id: str,
        conversation_id: str,
        role: str,
        content: str,
        citations: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[str, Any]:
        """Save a message to the conversation."""
        msg_id = str(uuid.uuid4())
        now = self._now()
        citations_json = json.dumps(citations) if citations else None

        with self._get_connection() as conn:
            cursor = conn.cursor()
            # Ensure conversation exists and belongs to user
            cursor.execute("SELECT id FROM conversations WHERE id = ? AND user_id = ?", (conversation_id, user_id))
            if not cursor.fetchone():
                cursor.execute(
                    "INSERT INTO conversations (id, user_id, title, created_at, updated_at) VALUES (?, ?, ?, ?, ?)",
                    (conversation_id, user_id, content[:40] if content else "Conversation", now, now)
                )

            cursor.execute(
                """
                INSERT INTO messages (id, conversation_id, user_id, role, content, citations_json, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (msg_id, conversation_id, user_id, role, content, citations_json, now)
            )
            cursor.execute("UPDATE conversations SET updated_at = ? WHERE id = ? AND user_id = ?", (now, conversation_id, user_id))
            conn.commit()

        return {
            "id": msg_id,
            "conversation_id": conversation_id,
            "user_id": user_id,
            "role": role,
            "content": content,
            "citations": citations,
            "created_at": now
        }

    def get_messages(self, user_id: str, conversation_id: str) -> List[Dict[str, Any]]:
        """Retrieve all messages in a conversation, verifying user ownership."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT id, conversation_id, user_id, role, content, citations_json, created_at
                FROM messages
                WHERE conversation_id = ? AND user_id = ?
                ORDER BY created_at ASC
                """,
                (conversation_id, user_id)
            )
            rows = cursor.fetchall()
            results = []
            for r in rows:
                item = dict(r)
                if item.get("citations_json"):
                    try:
                        item["citations"] = json.loads(item["citations_json"])
                    except Exception:
                        item["citations"] = []
                else:
                    item["citations"] = []
                results.append(item)
            return results

    def clear_conversation(self, user_id: str, conversation_id: str) -> bool:
        """Clear all messages in a conversation for user_id."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM messages WHERE user_id = ? AND conversation_id = ?", (user_id, conversation_id))
            conn.commit()
            return cursor.rowcount > 0
