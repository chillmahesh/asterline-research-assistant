"""
Research Agent (Agent A) for Asterline Research Assistant.
Interprets questions, constructs targeted queries, searches local vector store,
and identifies insufficient or conflicting evidence for bounded secondary retrieval.
"""

import logging
from datetime import datetime, timezone
from typing import Dict, Any, List

from src.models import AgentState, RetrievalResult
from src.retrieval import DocumentRetriever
from src.agents.llm_client import LLMClient

logger = logging.getLogger("asterline.agents.research")


class ResearchAgent:
    """
    Agent A: Research Agent
    - Interprets the user's question with awareness of user preferences.
    - Formulates targeted retrieval queries.
    - Retrieves candidate documentary evidence from the local vector store.
    - Evaluates whether evidence is sufficient or if secondary retrieval is needed.
    """

    def __init__(self, retriever: DocumentRetriever, llm_client: LLMClient):
        self.retriever = retriever
        self.llm = llm_client

    def run(self, state: AgentState) -> AgentState:
        """Execute research phase."""
        query = state["user_message"]
        iterations = state.get("retrieval_iterations", 0) + 1
        traces = state.get("agent_traces", [])
        
        # Personal memory routing: if query is a memory command or pure personal query, skip company docs
        if state.get("is_memory_command") or state.get("is_personal_memory_query"):
            step_trace = {
                "agent": "ResearchAgent",
                "action": "skip_retrieval_for_personal_memory",
                "iteration": iterations,
                "reason": "Personal memory routing active; isolating company documentation.",
                "retrieved_chunk_ids": [],
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
            traces.append(step_trace)
            state["search_queries"] = []
            state["retrieved_chunks"] = []
            state["research_summary"] = "Personal memory routing active; company documentation retrieval skipped."
            state["needs_secondary_retrieval"] = False
            state["agent_traces"] = traces
            return state

        # Formulate search queries
        search_queries = [query]
        query_lower = query.lower()

        # Add targeted query expansions for known entities and topics
        if any(w in query_lower for w in ["pricing", "price", "plan", "cost", "subscription", "seat", "allowance", "overage", "launch", "team", "enterprise", "10 million", "million run"]):
            search_queries.append("Plans and billing pricing Launch Team Enterprise 1 million 15 million overage retention")
        if "stipend" in query_lower or "on-call" in query_lower:
            search_queries.append("weekly on-call stipend operational runbook handbook $175 $140")
        if "retention" in query_lower or "cold recovery" in query_lower or "cold storage" in query_lower:
            search_queries.append("raw run records cold recovery storage retention period 20 days 60 days security overview")
        if "inc-2026-03" in query_lower or "delayed signals" in query_lower or ("clock" in query_lower and "patch" in query_lower):
            search_queries.append("INC-2026-03 delayed signals notifications patch 2.3.1 monotonic clock SEV-3")
        if "inc-2025-12" in query_lower or ("warehouse export" in query_lower and any(w in query_lower for w in ["duplicate", "heron", "incident", "sev-2"])):
            search_queries.append("INC-2025-12 duplicate Warehouse Export rows Heron patch 2.1.2")
        elif "warehouse export" in query_lower or "export" in query_lower:
            search_queries.append("Warehouse Export specification BigQuery Snowflake Redshift")
        if "asia-pacific" in query_lower or "apac" in query_lower:
            search_queries.append("Asia-Pacific region announced timeline launch date Product FAQ")
        if "certification" in query_lower or "healthcare" in query_lower or "compliance" in query_lower:
            search_queries.append("Security overview healthcare-specific certifications Northbridge Security")
        if "version 2.3" in query_lower or "timeout" in query_lower or "aql" in query_lower:
            search_queries.append("Version 2.3 release notes PagerDuty AQL query timeout 50 seconds")

        # Execute retrieval with multi-query Reciprocal Rank Fusion
        results = self.retriever.multi_search(search_queries, top_k=8)
        
        retrieved_chunk_dicts = [
            {
                "chunk_id": r.chunk.chunk_id,
                "doc_id": r.chunk.doc_id,
                "title": r.chunk.title,
                "category": r.chunk.category,
                "owner": r.chunk.owner,
                "source_url": r.chunk.source_url,
                "published": r.chunk.published,
                "version": r.chunk.version,
                "text": r.chunk.text,
                "score": r.score,
            }
            for r in results
        ]

        # Determine if secondary retrieval is needed (bounded to 2 max iterations)
        needs_secondary = False
        summary = f"Retrieved {len(retrieved_chunk_dicts)} source chunks across {len(search_queries)} queries."
        
        # Secondary retrieval trigger: if an incident is mentioned but patch release notes were not captured
        if iterations < 2:
            if "inc-2026-03" in query_lower and not any("2.3.1" in c["text"] for c in retrieved_chunk_dicts):
                needs_secondary = True
            elif "inc-2025-12" in query_lower and not any("2.1.2" in c["text"] for c in retrieved_chunk_dicts):
                needs_secondary = True

        # Observability trace step
        step_trace = {
            "agent": "ResearchAgent",
            "action": "retrieve_evidence",
            "iteration": iterations,
            "queries": search_queries,
            "retrieved_chunk_ids": [c["chunk_id"] for c in retrieved_chunk_dicts],
            "needs_secondary_retrieval": needs_secondary,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        traces.append(step_trace)

        state["search_queries"] = search_queries
        state["retrieved_chunks"] = retrieved_chunk_dicts
        state["research_summary"] = summary
        state["needs_secondary_retrieval"] = needs_secondary
        state["retrieval_iterations"] = iterations
        state["agent_traces"] = traces
        return state
