"""
Memory service providing high-level memory operations, privacy checks,
intent detection, and strict user isolation.
"""

import re
from typing import List, Dict, Any, Optional, Tuple
from src.models import MemoryRecord, MemoryOperation
from src.memory.repository import SQLiteMemoryRepository


# Patterns to detect potential secrets / credentials that must NEVER be stored
SENSITIVE_PATTERNS = [
    re.compile(r"(?i)(api[_-]?key|apikey|secret|token|password|passwd|auth[_-]?token)\s*(?:[:=]|\bis\b)\s*['\"]?[\w\-\.]{8,}['\"]?"),
    re.compile(r"(?i)bearer\s+[a-zA-Z0-9_\-\.]{15,}"),
    re.compile(r"ghp_[a-zA-Z0-9]{36}"),             # GitHub PAT
    re.compile(r"sk-[a-zA-Z0-9]{32,}"),             # OpenAI key pattern
    re.compile(r"lsv2_[a-zA-Z0-9_]{32,}"),          # LangSmith key pattern
    re.compile(r"AIza[0-9A-Za-z\-_]{35}"),          # Google AI API key pattern
    re.compile(r"AQ\.[a-zA-Z0-9_\-]{30,}"),         # Gemini token pattern
    re.compile(r"\b(?:\d{3}-\d{2}-\d{4}|\d{9})\b"), # SSN
]

# Patterns for explicit user memory commands
REMEMBER_PATTERNS = [
    re.compile(r"(?i)^(?:please\s+)?remember\s+(?:that\s+)?(.+)$"),
    re.compile(r"(?i)^(?:keep\s+in\s+mind|note\s+that|save\s+to\s+memory)\s+(?:that\s+)?(.+)$"),
    re.compile(r"(?i)^my\s+(?:role|team|focus|project|preference|stack)\s+(?:is|are)\s+(.+)$"),
]

FORGET_PATTERNS = [
    re.compile(r"(?i)^(?:please\s+)?forget\s+(?:about\s+|that\s+)?(.+)$"),
    re.compile(r"(?i)^(?:delete|remove|erase|clear)\s+(?:my\s+)?(?:memory\s+about\s+|memories\s+about\s+|preference\s+about\s+)?(.+)$"),
]

CORRECT_PATTERNS = [
    re.compile(r"(?i)^(?:actually|correction|update|change)\s*[:,\s]+(?:my\s+)?(.+)$"),
]


class MemoryService:
    """
    Coordinates persistent user-scoped memory with privacy safeguards,
    intent classification, and transactional updates.
    """

    def __init__(self, repository: Optional[SQLiteMemoryRepository] = None):
        self.repo = repository or SQLiteMemoryRepository()

    @staticmethod
    def contains_sensitive_data(text: str) -> Tuple[bool, str]:
        """Check if the text contains credentials, keys, or sensitive PII."""
        for pattern in SENSITIVE_PATTERNS:
            match = pattern.search(text)
            if match:
                return True, f"Detected potential secret or credential: '{match.group(0)[:12]}...'"
        return False, ""

    def get_user_context(self, user_id: str, query: Optional[str] = None, limit: int = 10) -> List[MemoryRecord]:
        """
        Retrieve active memories belonging strictly to user_id.
        If a query is provided, ranks memories that mention terms in the query higher.
        """
        all_memories = self.repo.get_active_memories(user_id)
        if not query:
            return all_memories[:limit]

        query_terms = set(re.findall(r"\w+", query.lower()))
        
        def score_memory(mem: MemoryRecord) -> int:
            mem_terms = set(re.findall(r"\w+", mem.content.lower()))
            return len(query_terms.intersection(mem_terms))

        ranked = sorted(all_memories, key=score_memory, reverse=True)
        return ranked[:limit]

    @staticmethod
    def is_hybrid_query(text: str) -> bool:
        """
        Determines whether the query asks both about user memory/belief AND official company evidence.
        Example: 'What do I believe the Launch plan costs, and what is the official price?'
        """
        lower = text.lower().strip()
        has_memory_aspect = any(k in lower for k in [
            "what do i believe", "what did i think", "what was my belief", "my belief", 
            "what did i say", "what do i think", "my estimate", "what did i believe",
            "do i believe", "what do i remember", "what is my belief"
        ])
        has_official_aspect = any(k in lower for k in [
            "official", "actual", "company", "documentation", "handbook", "policy", 
            "real price", "asterline", "corpus", "authoritative", "what is the price"
        ])
        return has_memory_aspect and has_official_aspect

    @staticmethod
    def is_personal_memory_query(text: str) -> Tuple[bool, Optional[str]]:
        """
        Determines whether the user query is asking about personal memory/context
        (such as project codename, team, role, preferences, or saved items).
        Returns (is_personal_memory, attribute_topic).
        """
        if MemoryService.is_hybrid_query(text):
            return False, None

        lower = text.lower().strip()
        patterns = [
            (r"\b(?:what(?:\s+is|\'s)?|tell\s+me|show\s+me|do\s+you\s+know|do\s+you\s+remember)\s+(?:my\s+)?(?:project\s+)?codename\b", "codename"),
            (r"\b(?:what(?:\s+is|\'s)?|tell\s+me)\s+my\s+project\b", "project"),
            (r"\b(?:what(?:\s+is|\'s)?|tell\s+me)\s+my\s+team\b", "team"),
            (r"\b(?:what(?:\s+is|\'s)?|tell\s+me)\s+my\s+role\b", "role"),
            (r"\b(?:what(?:\s+is|\'s)?|tell\s+me)\s+my\s+preference\b", "preference"),
            (r"\b(?:what\s+do\s+you\s+remember\s+about\s+me|what\s+memories\s+do\s+you\s+have)\b", "all"),
            (r"\b(?:what\s+did\s+i\s+ask\s+you\s+to\s+remember)\b", "all"),
        ]
        for pat, attr in patterns:
            if re.search(pat, lower):
                return True, attr

        if "my project codename" in lower or "my codename" in lower:
            return True, "codename"

        return False, None

    def parse_memory_intent(self, user_message: str) -> Optional[MemoryOperation]:
        """
        Detect whether the user is explicitly directing the assistant to
        store, update/correct, or forget memory.
        """
        cleaned = user_message.strip()

        # Check for FORGET intent
        for pat in FORGET_PATTERNS:
            m = pat.match(cleaned)
            if m:
                target = m.group(1).strip().rstrip(".!?")
                return MemoryOperation(action="forget", content=target, reason="User requested forgetting")

        # Check for CORRECT intent
        for pat in CORRECT_PATTERNS:
            m = pat.match(cleaned)
            if m:
                new_info = m.group(1).strip().rstrip(".!?")
                return MemoryOperation(action="correct", content=new_info, reason="User offered correction")

        # Check for REMEMBER intent
        for pat in REMEMBER_PATTERNS:
            m = pat.match(cleaned)
            if m:
                content = m.group(1).strip().rstrip(".!?")
                # Infer category
                cat = "general"
                lower = content.lower()
                if any(k in lower for k in ["prefer", "format", "concise", "detailed", "bullet"]):
                    cat = "preference"
                elif any(k in lower for k in ["project", "working on", "repo", "building"]):
                    cat = "project"
                elif any(k in lower for k in ["role", "engineer", "lead", "architect", "manager", "team"]):
                    cat = "role"
                elif any(k in lower for k in ["constraint", "must not", "only use", "avoid"]):
                    cat = "constraint"
                return MemoryOperation(action="store", content=content, category=cat, reason="User requested remembering")

        return None

    def execute_memory_operation(self, user_id: str, op: MemoryOperation) -> Dict[str, Any]:
        """
        Executes a memory operation for user_id with privacy validation.
        Guarantees that credentials are not saved.
        """
        if op.action == "forget":
            if op.memory_id:
                success = self.repo.forget_memory(user_id, op.memory_id)
                return {"success": success, "action": "forget", "memory_id": op.memory_id}
            else:
                # Forget by topic or keyword
                count = self.repo.forget_by_topic(user_id, op.content)
                return {"success": count > 0, "action": "forget_topic", "topic": op.content, "count": count}

        elif op.action == "store":
            # Privacy check
            is_sensitive, reason = self.contains_sensitive_data(op.content)
            if is_sensitive:
                return {
                    "success": False,
                    "action": "store_rejected",
                    "reason": f"Privacy safeguard: {reason}. Credentials and secrets are never stored.",
                }

            # Avoid storing near-duplicates
            existing = self.repo.get_active_memories(user_id)
            for mem in existing:
                if mem.content.lower() == op.content.lower():
                    return {"success": True, "action": "already_exists", "memory_id": mem.id}

            created = self.repo.create_memory(
                user_id=user_id,
                content=op.content,
                category=op.category,
                confidence=1.0
            )
            return {"success": True, "action": "stored", "memory": created.model_dump()}

        elif op.action == "correct":
            is_sensitive, reason = self.contains_sensitive_data(op.content)
            if is_sensitive:
                return {
                    "success": False,
                    "action": "correct_rejected",
                    "reason": f"Privacy safeguard: {reason}",
                }

            if op.memory_id:
                updated = self.repo.update_memory(user_id, op.memory_id, op.content, op.category)
                return {"success": updated is not None, "action": "corrected", "memory": updated.model_dump() if updated else None}
            else:
                # Find most relevant active memory to correct
                existing = self.get_user_context(user_id, op.content, limit=1)
                if existing:
                    updated = self.repo.update_memory(user_id, existing[0].id, op.content, op.category)
                    return {"success": True, "action": "corrected", "memory": updated.model_dump() if updated else None}
                else:
                    # Treat as new memory if no prior matching memory
                    created = self.repo.create_memory(user_id, op.content, op.category)
                    return {"success": True, "action": "stored_new", "memory": created.model_dump()}

        return {"success": False, "error": f"Unknown memory action: {op.action}"}
