"""
Multi-Agent Orchestration Graph coordinating Research, Verification, and Memory Agents.
Supports LangGraph StateGraph coordination and LangSmith tracing.
"""

import logging
from typing import Dict, Any, Optional
from datetime import datetime, timezone

from src.models import AgentState
from src.retrieval import DocumentRetriever
from src.agents.llm_client import LLMClient
from src.agents.research_agent import ResearchAgent
from src.agents.verification_agent import VerificationAgent
from src.agents.memory_agent import MemoryAgent
from src.memory.service import MemoryService
from src.config import settings

logger = logging.getLogger("asterline.graph")


class AsterlineGraph:
    """
    Multi-Agent Workflow Orchestrator coordinating:
    1. Memory Agent (pre-retrieval context)
    2. Research Agent (vector search & bounded iterative retrieval)
    3. Verification Agent (grounded synthesis, conflict & supersession resolution)
    4. Memory Agent (post-process memory lifecycle & privacy safeguarding)
    """

    def __init__(
        self,
        retriever: Optional[DocumentRetriever] = None,
        llm_client: Optional[LLMClient] = None,
        memory_service: Optional[MemoryService] = None,
    ):
        self.retriever = retriever or DocumentRetriever()
        self.llm = llm_client or LLMClient()
        self.memory_service = memory_service or MemoryService()

        self.research_agent = ResearchAgent(self.retriever, self.llm)
        self.verification_agent = VerificationAgent(self.llm)
        self.memory_agent = MemoryAgent(self.memory_service)
        
        self._compiled_graph = None
        self._build_langgraph()

    def _build_langgraph(self):
        """Builds LangGraph StateGraph if langgraph is available."""
        try:
            from langgraph.graph import StateGraph, END

            workflow = StateGraph(AgentState)

            # Define Nodes
            workflow.add_node("memory_pre", self.memory_agent.pre_process)
            workflow.add_node("research", self.research_agent.run)
            workflow.add_node("verification", self.verification_agent.run)
            workflow.add_node("memory_post", self.memory_agent.post_process)

            # Define Edges
            workflow.set_entry_point("memory_pre")

            def route_after_memory_pre(state: AgentState) -> str:
                if state.get("is_memory_command") or state.get("is_personal_memory_query"):
                    return "verification"
                return "research"

            workflow.add_conditional_edges(
                "memory_pre",
                route_after_memory_pre,
                {
                    "verification": "verification",
                    "research": "research"
                }
            )

            def should_retrieve_more(state: AgentState) -> str:
                if state.get("needs_secondary_retrieval") and state.get("retrieval_iterations", 0) < 2:
                    return "research"
                return "verification"

            workflow.add_conditional_edges(
                "research",
                should_retrieve_more,
                {
                    "research": "research",
                    "verification": "verification"
                }
            )

            workflow.add_edge("verification", "memory_post")
            workflow.add_edge("memory_post", END)

            self._compiled_graph = workflow.compile()
            logger.info("LangGraph multi-agent graph compiled successfully.")
        except Exception as e:
            logger.warning(f"LangGraph compile warning: {e}. Utilizing built-in resilient orchestrator.")
            self._compiled_graph = None

    def invoke(self, user_id: str, conversation_id: str, user_message: str) -> AgentState:
        """
        Executes the multi-agent workflow for a given user message.
        Ensures strict user isolation and complete agent execution traces.
        """
        initial_state: AgentState = {
            "user_id": user_id,
            "conversation_id": conversation_id,
            "user_message": user_message,
            "user_memories": [],
            "search_queries": [],
            "retrieved_chunks": [],
            "research_summary": "",
            "needs_secondary_retrieval": False,
            "retrieval_iterations": 0,
            "verification_verdict": {},
            "conflict_analysis": {},
            "missing_evidence_detected": False,
            "final_answer": "",
            "citations": [],
            "suggested_memory_ops": [],
            "executed_memory_ops": [],
            "is_memory_command": False,
            "is_personal_memory_query": False,
            "personal_memory_attribute": None,
            "memory_command_result": None,
            "agent_traces": [],
            "error": None,
        }

        # If LangGraph compiled graph is ready, invoke with tracing
        if self._compiled_graph:
            try:
                config = {}
                if settings.is_langsmith_configured():
                    config["metadata"] = {
                        "user_id": user_id,
                        "conversation_id": conversation_id,
                        "project": settings.langchain_project,
                    }
                return self._compiled_graph.invoke(initial_state, config=config)
            except Exception as e:
                logger.error(f"LangGraph invocation error: {e}. Executing sequential pipeline.")

        # Sequential multi-agent pipeline
        state = self.memory_agent.pre_process(initial_state)
        state = self.research_agent.run(state)

        # Bounded secondary retrieval loop
        if state.get("needs_secondary_retrieval") and state.get("retrieval_iterations", 0) < 2:
            state = self.research_agent.run(state)

        state = self.verification_agent.run(state)
        state = self.memory_agent.post_process(state)
        return state
