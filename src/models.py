"""
Core data structures and types for Asterline Research Assistant.
Provides seamless Pydantic BaseModel support with transparent dataclass fallback.
"""

from typing import List, Dict, Any, Optional, Union
from typing_extensions import TypedDict

try:
    from pydantic import BaseModel, Field
    HAS_PYDANTIC = True
except ImportError:
    HAS_PYDANTIC = False

    class BaseModel:
        def __init__(self, **kwargs):
            for k, v in kwargs.items():
                setattr(self, k, v)

        def model_dump(self) -> Dict[str, Any]:
            res = {}
            for k, v in self.__dict__.items():
                if hasattr(v, "model_dump"):
                    res[k] = v.model_dump()
                elif isinstance(v, list):
                    res[k] = [item.model_dump() if hasattr(item, "model_dump") else item for item in v]
                else:
                    res[k] = v
            return res

        def __repr__(self):
            return f"{self.__class__.__name__}({self.__dict__})"

    def Field(default=..., **kwargs):
        return default


class CorpusChunk(BaseModel):
    """Represents an authoritative documentary chunk from corpus.jsonl."""
    chunk_id: str
    doc_id: str
    title: str
    category: str
    owner: str
    source_url: str
    published: str
    version: Union[int, str]
    text: str

    def format_citation(self) -> str:
        """Standard citation string: [Source: chunk_id — title]"""
        return f"[Source: {self.chunk_id} — {self.title}]"


class RetrievalResult(BaseModel):
    """Result of a semantic search query against the local vector store."""
    chunk: CorpusChunk
    score: float = 1.0
    distance: Optional[float] = None


class Citation(BaseModel):
    """Structured citation referencing verified documentary evidence."""
    chunk_id: str
    title: str
    source_url: str
    published: str
    version: Union[int, str]
    excerpt: Optional[str] = None

    @property
    def label(self) -> str:
        return f"[Source: {self.chunk_id} — {self.title}]"


class MemoryRecord(BaseModel):
    """Persistent, user-isolated memory record."""
    id: str
    user_id: str
    content: str
    category: str = "general"
    confidence: float = 1.0
    created_at: str
    updated_at: str
    is_active: bool = True


class MemoryOperation(BaseModel):
    """Requested memory mutation (create, correct/update, forget)."""
    action: str = "store"
    content: str = ""
    memory_id: Optional[str] = None
    category: str = "general"
    reason: str = ""


class AgentTraceStep(BaseModel):
    """Observability trace record for agent actions."""
    agent: str
    action: str
    details: Dict[str, Any]
    timestamp: str


class AgentState(TypedDict):
    """Shared typed state propagated across LangGraph agents."""
    user_id: str
    conversation_id: str
    user_message: str
    
    # User Context (isolated from company facts)
    user_memories: List[Dict[str, Any]]
    
    # Research Agent State
    search_queries: List[str]
    retrieved_chunks: List[Dict[str, Any]]
    research_summary: str
    needs_secondary_retrieval: bool
    retrieval_iterations: int
    
    # Verification & Synthesis Agent State
    verification_verdict: Dict[str, Any]
    conflict_analysis: Dict[str, Any]
    missing_evidence_detected: bool
    final_answer: str
    citations: List[Dict[str, Any]]
    
    # Memory Agent State
    suggested_memory_ops: List[Dict[str, Any]]
    executed_memory_ops: List[Dict[str, Any]]
    
    # Memory Intent & Routing State
    is_memory_command: bool
    is_personal_memory_query: bool
    is_hybrid_query: bool
    personal_memory_attribute: Optional[str]
    memory_command_result: Optional[Dict[str, Any]]

    # Observability & Trace
    agent_traces: List[Dict[str, Any]]
    error: Optional[str]
