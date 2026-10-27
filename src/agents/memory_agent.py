"""
Memory Agent (Agent C) for Asterline Research Assistant.
Manages user-isolated contextual memory: retrieval, selective creation,
corrections, explicit forgetting, and privacy filters against credentials.
"""

import logging
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional

from src.models import AgentState, MemoryOperation
from src.memory.service import MemoryService

logger = logging.getLogger("asterline.agents.memory")


class MemoryAgent:
    """
    Agent C: Memory Agent
    - Retrieves active memories scoped strictly to the current user_id.
    - Decides whether the user's message contains context worth remembering.
    - Executes corrections and explicit forgetting transactionally.
    - Shields system from storing credentials, keys, or sensitive secrets.
    - Keeps user memories strictly isolated from company document evidence.
    """

    def __init__(self, memory_service: MemoryService):
        self.service = memory_service

    def pre_process(self, state: AgentState) -> AgentState:
        """
        Runs before research: retrieves relevant user context scoped strictly to user_id.
        """
        user_id = state["user_id"]
        query = state["user_message"]
        traces = state.get("agent_traces", [])

        # Retrieve user-isolated memories
        user_memories = self.service.get_user_context(user_id=user_id, query=query, limit=5)
        memory_dicts = [m.model_dump() for m in user_memories]

        step_trace = {
            "agent": "MemoryAgent",
            "action": "pre_process_retrieve_memories",
            "user_id": user_id,
            "active_memory_count": len(memory_dicts),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        traces.append(step_trace)

        state["user_memories"] = memory_dicts
        state["agent_traces"] = traces
        return state

    def post_process(self, state: AgentState) -> AgentState:
        """
        Runs after synthesis: evaluates whether to store, update, or forget memory
        based on the user's message.
        """
        user_id = state["user_id"]
        message = state["user_message"]
        traces = state.get("agent_traces", [])

        suggested_ops: List[Dict[str, Any]] = []
        executed_ops: List[Dict[str, Any]] = []

        # Parse memory intent
        detected_op = self.service.parse_memory_intent(message)
        if detected_op:
            suggested_ops.append(detected_op.model_dump())
            # Execute operation
            res = self.service.execute_memory_operation(user_id=user_id, op=detected_op)
            executed_ops.append(res)

        # Refresh active memories for user
        active_memories = self.service.get_user_context(user_id=user_id)
        refreshed_mems = [m.model_dump() for m in active_memories]

        step_trace = {
            "agent": "MemoryAgent",
            "action": "post_process_memory_lifecycle",
            "user_id": user_id,
            "suggested_operations": suggested_ops,
            "executed_operations": executed_ops,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        traces.append(step_trace)

        state["suggested_memory_ops"] = suggested_ops
        state["executed_memory_ops"] = executed_ops
        state["user_memories"] = refreshed_mems
        state["agent_traces"] = traces
        return state
