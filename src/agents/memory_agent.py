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
        Runs before research: retrieves relevant user context scoped strictly to user_id,
        detects memory mutation commands, and routes personal-memory queries.
        """
        user_id = state["user_id"]
        query = state["user_message"]
        traces = state.get("agent_traces", [])

        # Step 1: Check if user is issuing an explicit memory mutation command (store, correct, forget)
        detected_op = self.service.parse_memory_intent(query)
        if detected_op:
            res = self.service.execute_memory_operation(user_id=user_id, op=detected_op)
            state["is_memory_command"] = True
            state["memory_command_result"] = res
            state["suggested_memory_ops"] = [detected_op.model_dump()]
            state["executed_memory_ops"] = [res]

            if res.get("action") == "store_rejected":
                state["final_answer"] = f"Privacy safeguard: {res.get('reason', 'Credentials rejected')}. Credentials, passwords, API keys, and customer secrets are never stored in personal memory."
            elif res.get("action") == "stored":
                state["final_answer"] = f"I've saved that to your personal memory: \"{detected_op.content}\"."
            elif res.get("action") == "already_exists":
                state["final_answer"] = f"That fact is already saved in your active personal memory: \"{detected_op.content}\"."
            elif res.get("action") == "corrected":
                state["final_answer"] = f"I've updated your personal memory: \"{detected_op.content}\"."
            elif res.get("action") in ("forget", "forget_topic"):
                count = res.get("count", 1)
                state["final_answer"] = f"I've removed \"{detected_op.content}\" from your active personal memory."
            else:
                state["final_answer"] = "Memory operation processed."

        # Step 2: Check if user is asking a personal memory question (e.g. project codename)
        is_personal, attr = self.service.is_personal_memory_query(query)
        state["is_personal_memory_query"] = is_personal
        state["personal_memory_attribute"] = attr

        # Step 3: Retrieve user-isolated memories
        user_memories = self.service.get_user_context(user_id=user_id, query=query, limit=10)
        memory_dicts = [m.model_dump() for m in user_memories]

        step_trace = {
            "agent": "MemoryAgent",
            "action": "pre_process_retrieve_memories",
            "user_id": user_id,
            "is_memory_command": bool(detected_op),
            "is_personal_memory_query": is_personal,
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
