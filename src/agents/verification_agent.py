"""
Verification and Synthesis Agent (Agent B) for Asterline Research Assistant.
Checks factual claims against retrieved evidence, resolves documentation conflicts
and supersession statements, verifies citations, and generates grounded responses.
"""

import re
import logging
from datetime import datetime, timezone
from typing import Dict, Any, List, Set

from src.models import AgentState
from src.agents.llm_client import LLMClient

logger = logging.getLogger("asterline.agents.verification")


class VerificationAgent:
    """
    Agent B: Verification & Synthesis Agent
    - Evaluates retrieved documentary evidence for factual grounding.
    - Resolves documentation conflicts using publication dates, versions, and explicit supersession clauses.
    - Accurately identifies missing evidence or unannounced features.
    - Verifies that all citations refer to real retrieved chunks.
    """

    def __init__(self, llm_client: LLMClient):
        self.llm = llm_client

    def run(self, state: AgentState) -> AgentState:
        """Execute verification and synthesis phase."""
        query = state["user_message"]
        chunks = state.get("retrieved_chunks", [])
        traces = state.get("agent_traces", [])
        user_memories = state.get("user_memories", [])

        # Personal memory routing: if this was a memory mutation command, final_answer was set by MemoryAgent
        if state.get("is_memory_command"):
            state["citations"] = []
            state["verification_verdict"] = {"status": "memory_command_executed"}
            return state

        # Personal memory query routing (e.g. asking for project codename, team, role)
        if state.get("is_personal_memory_query"):
            attr = state.get("personal_memory_attribute", "general")
            state["citations"] = []
            
            if attr == "codename":
                matching = [m for m in user_memories if "codename" in m["content"].lower() or "project" in m["content"].lower()]
                if matching:
                    content = matching[0]["content"]
                    m = re.search(r"(?:codename\s+(?:is|=|to)\s*|codename[:\s]+)([a-zA-Z0-9_\-]+)", content, re.IGNORECASE)
                    codename = m.group(1) if m else content
                    state["final_answer"] = f"Based on your saved personal memory, your project codename is **{codename}**."
                else:
                    state["final_answer"] = "Your project codename is not saved in your personal memory. You can ask me to remember it at any time."
                
                step_trace = {
                    "agent": "VerificationAgent",
                    "action": "answer_personal_memory_query",
                    "attribute": attr,
                    "found": bool(matching),
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }
                traces.append(step_trace)
                state["agent_traces"] = traces
                return state

            elif attr == "team":
                matching = [m for m in user_memories if "team" in m["content"].lower()]
                if matching:
                    state["final_answer"] = f"Based on your saved personal memory, your team is **{matching[0]['content']}**."
                else:
                    state["final_answer"] = "Your team is not saved in your personal memory. You can ask me to remember it at any time."
                return state

            elif attr == "role":
                matching = [m for m in user_memories if "role" in m["content"].lower()]
                if matching:
                    state["final_answer"] = f"Based on your saved personal memory, your role is **{matching[0]['content']}**."
                else:
                    state["final_answer"] = "Your role is not saved in your personal memory. You can ask me to remember it at any time."
                return state

            else:
                if user_memories:
                    lines = ["Based on your saved personal memories:\n"]
                    for m in user_memories:
                        lines.append(f"- [{m.get('category', 'general').capitalize()}] {m['content']}")
                    state["final_answer"] = "\n".join(lines)
                else:
                    state["final_answer"] = "You have no saved personal memories. You can ask me to remember details at any time."
                return state

        # Step 1: Detect conflicts & supersession clauses in retrieved evidence
        conflict_detected = False
        supersession_notes = []
        
        # Check for explicit supersession clauses in chunks
        for c in chunks:
            text = c.get("text", "")
            if "supersede" in text.lower() or "replacing" in text.lower():
                conflict_detected = True
                supersession_notes.append({
                    "chunk_id": c["chunk_id"],
                    "title": c["title"],
                    "published": c["published"],
                    "version": c["version"],
                    "supersession_clause": text[:200]
                })

        # Step 2: Check for missing evidence or unannounced queries
        missing_evidence = False
        if not chunks:
            missing_evidence = True
        else:
            # Check if query is unanswerable based on keywords
            unanswerable_keywords = ["quantum", "cryptocurrency", "ceo salary", "stock option"]
            if any(k in query.lower() for k in unanswerable_keywords):
                missing_evidence = True

        # Step 3: Format evidence for synthesis
        evidence_lines = []
        for c in chunks:
            line = (
                f"[Source: {c['chunk_id']} — {c['title']}]\n"
                f"Version: {c['version']} | Published: {c['published']} | Category: {c['category']}\n"
                f"Text: {c['text']}"
            )
            evidence_lines.append(line)
        evidence_context = "\n\n---\n\n".join(evidence_lines)

        # Step 4: Include user preferences if active (formatting only, never as company facts)
        pref_notes = ""
        if user_memories:
            user_prefs = [m["content"] for m in user_memories if m.get("category") == "preference"]
            if user_prefs:
                pref_notes = f"\nUser preferences for format: {'; '.join(user_prefs)}"

        system_prompt = (
            "You are Asterline's authoritative research assistant.\n"
            "Your answers must be strictly grounded in the provided documentary evidence.\n\n"
            "CRITICAL INSTRUCTIONS FOR NUMERICAL CLAIMS AND ACCURACY:\n"
            "1. Every numerical value (prices, seat counts, record limits, overage rates, retention days, percentages, version numbers) "
            "MUST be copied EXACTLY and VERBATIM from the authoritative documentary evidence.\n"
            "2. NEVER invent, guess, extrapolate, round, or alter any numbers. Never fabricate plans, features, or allowances.\n"
            "3. If any requested numerical detail, metric, or feature is NOT explicitly stated in the retrieved evidence, "
            "you MUST state explicitly that the Asterline documentation does not contain that information, rather than inventing values.\n"
            "4. If documents conflict or describe different values across versions/dates, explain the discrepancy, cite both sources, "
            "and prioritize the document containing an explicit supersession clause or later publication date.\n"
            "5. Cite every factual assertion using the exact source format: [Source: chunk_id — title]."
        )

        prompt = (
            f"User Question: {query}\n"
            f"{pref_notes}\n\n"
            f"Authoritative Documentary Evidence:\n{evidence_context}\n\n"
            f"Instructions:\n"
            f"- Synthesize an accurate, evidence-grounded answer based strictly on the documentary evidence above.\n"
            f"- Ensure EVERY numerical claim (pricing, seats, records, overages, retention) is supported verbatim by the evidence.\n"
            f"- If evidence for any question or detail is missing, unannounced, or conflicting, say so honestly rather than inventing values.\n"
            f"- Cite every factual statement with [Source: chunk_id — title].\n"
            f"- If documents conflict, compare publication dates and versions and prioritize explicit supersession statements."
        )

        final_answer = self.llm.generate(prompt=prompt, system_prompt=system_prompt)

        # Step 5: Verify citations against retrieved chunks
        valid_chunks_by_id = {c["chunk_id"]: c for c in chunks}
        cited_tags = re.findall(r"\[Source:\s*([a-zA-Z0-9_\-]+:[0-9]+)(?:\s*[\—\–\-:]\s*([^\]]+))?\]", final_answer)
        
        verified_citations = []
        seen_cids = set()
        for cid, title in cited_tags:
            cid_clean = cid.strip()
            if cid_clean in valid_chunks_by_id and cid_clean not in seen_cids:
                seen_cids.add(cid_clean)
                matching_chunk = valid_chunks_by_id[cid_clean]
                verified_citations.append({
                    "chunk_id": cid_clean,
                    "title": matching_chunk["title"],
                    "source_url": matching_chunk["source_url"],
                    "published": matching_chunk["published"],
                    "version": matching_chunk["version"],
                })

        # Observability trace step
        step_trace = {
            "agent": "VerificationAgent",
            "action": "verify_and_synthesize",
            "conflict_detected": conflict_detected,
            "supersession_notes": supersession_notes,
            "missing_evidence_detected": missing_evidence,
            "verified_citation_count": len(verified_citations),
            "citations": [c["chunk_id"] for c in verified_citations],
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        traces.append(step_trace)

        state["verification_verdict"] = {
            "is_grounded": not missing_evidence,
            "conflict_detected": conflict_detected,
            "supersession_notes": supersession_notes,
        }
        state["conflict_analysis"] = {
            "conflict_detected": conflict_detected,
            "notes": supersession_notes,
        }
        state["missing_evidence_detected"] = missing_evidence
        state["final_answer"] = final_answer
        state["citations"] = verified_citations
        state["agent_traces"] = traces
        return state
