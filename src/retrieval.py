"""
Document retrieval service for Asterline Research Assistant.
Provides semantic search over ChromaDB with metadata preservation,
score calculation, evidence formatting, and multi-query fusion.
"""

import math
import logging
from typing import List, Dict, Any, Optional, Set
from pathlib import Path

from src.config import settings
from src.models import CorpusChunk, RetrievalResult
from src.ingestion import LocalVectorStore, load_corpus_chunks

logger = logging.getLogger("asterline.retrieval")


class DocumentRetriever:
    """
    Retrieves authoritative Asterline documentary evidence using semantic search.
    Preserves exact chunk IDs, titles, dates, versions, and source URLs.
    """

    def __init__(self, vector_store: Optional[LocalVectorStore] = None):
        self.vector_store = vector_store or LocalVectorStore()
        # Ensure chunks are indexed
        if self.vector_store.count() < 75:
            try:
                chunks = load_corpus_chunks()
                self.vector_store.ingest_corpus(chunks)
            except Exception as e:
                logger.warning(f"Auto-ingestion during retriever init encountered: {e}")

    def search(
        self,
        query: str,
        top_k: int = 5,
        category: Optional[str] = None
    ) -> List[RetrievalResult]:
        """
        Executes semantic search against the vector store.
        Returns top_k matching CorpusChunk objects with similarity scores.
        """
        cleaned_query = query.strip()
        if not cleaned_query:
            return []

        # If ChromaDB collection is active
        if self.vector_store._collection:
            where_filter = {"category": category} if category else None
            try:
                results = self.vector_store._collection.query(
                    query_texts=[cleaned_query],
                    n_results=min(top_k, 75),
                    where=where_filter,
                    include=["metadatas", "documents", "distances"]
                )
                
                retrieval_results: List[RetrievalResult] = []
                if results and results.get("ids") and results["ids"][0]:
                    ids = results["ids"][0]
                    docs = results["documents"][0]
                    metas = results["metadatas"][0]
                    distances = results["distances"][0] if "distances" in results else [0.0] * len(ids)

                    for cid, doc, meta, dist in zip(ids, docs, metas, distances):
                        # Convert cosine distance to similarity score in [0, 1]
                        # For cosine space: dist in [0, 2], score = max(0.0, 1.0 - dist)
                        similarity = max(0.0, min(1.0, 1.0 - dist))
                        
                        chunk = CorpusChunk(
                            chunk_id=cid,
                            doc_id=meta.get("doc_id", ""),
                            title=meta.get("title", ""),
                            category=meta.get("category", ""),
                            owner=meta.get("owner", ""),
                            source_url=meta.get("source_url", ""),
                            published=str(meta.get("published", "")),
                            version=meta.get("version", ""),
                            text=doc
                        )
                        retrieval_results.append(RetrievalResult(chunk=chunk, score=similarity, distance=dist))

                return retrieval_results
            except Exception as e:
                logger.error(f"ChromaDB query error: {e}. Falling back to lexical/dense ranking.")

        # Resilient fallback ranking
        return self._fallback_rank(cleaned_query, top_k, category)

    def _fallback_rank(self, query: str, top_k: int = 5, category: Optional[str] = None) -> List[RetrievalResult]:
        """
        Fallback keyword and lexical ranking over corpus chunks.
        Ensures search works reliably even when external vector servers are unavailable.
        """
        chunks = load_corpus_chunks()
        if category:
            chunks = [c for c in chunks if c.category == category]

        terms = [t.lower() for t in query.split() if len(t) > 2]
        scored: List[RetrievalResult] = []

        for c in chunks:
            text_lower = c.text.lower()
            title_lower = c.title.lower()
            score = 0.0

            # Match exact phrase
            if query.lower() in text_lower:
                score += 0.5
            if query.lower() in title_lower:
                score += 0.8

            # Match individual terms with frequency weighting
            for term in terms:
                count_text = text_lower.count(term)
                count_title = title_lower.count(term)
                if count_text > 0:
                    score += 0.1 * min(count_text, 5)
                if count_title > 0:
                    score += 0.3 * min(count_title, 3)

            # Boost exact chunk_id matches
            if c.chunk_id.lower() in query.lower() or c.doc_id.lower() in query.lower():
                score += 1.0

            if score > 0:
                normalized_score = min(1.0, score / 2.0)
                scored.append(RetrievalResult(chunk=c, score=normalized_score, distance=1.0 - normalized_score))

        scored.sort(key=lambda r: r.score, reverse=True)
        return scored[:top_k]

    def multi_search(self, queries: List[str], top_k: int = 6) -> List[RetrievalResult]:
        """
        Execute multiple search queries and fuse results using Reciprocal Rank Fusion (RRF).
        Aggregates reciprocal rank scores across queries so chunks matching multiple intents rank highest.
        """
        rrf_scores: Dict[str, float] = {}
        chunk_map: Dict[str, CorpusChunk] = {}
        best_similarity: Dict[str, float] = {}

        k_constant = 60.0
        for q in queries:
            results = self.search(q, top_k=min(top_k * 2, 20))
            for rank, res in enumerate(results, start=1):
                cid = res.chunk.chunk_id
                chunk_map[cid] = res.chunk
                rrf_scores[cid] = rrf_scores.get(cid, 0.0) + (1.0 / (k_constant + rank))
                best_similarity[cid] = max(best_similarity.get(cid, 0.0), res.score)

        sorted_cids = sorted(rrf_scores.keys(), key=lambda cid: rrf_scores[cid], reverse=True)
        return [
            RetrievalResult(chunk=chunk_map[cid], score=best_similarity[cid])
            for cid in sorted_cids[:top_k]
        ]

    @staticmethod
    def format_evidence_for_prompt(results: List[RetrievalResult]) -> str:
        """
        Formats retrieved documentary chunks into structured context for agent verification.
        Includes exact source tags: [Source: chunk_id — title]
        """
        if not results:
            return "No documentary evidence was found in the Asterline knowledge base."

        formatted_blocks = []
        for r in results:
            c = r.chunk
            header = f"[Source: {c.chunk_id} — {c.title}]"
            meta = f"Doc ID: {c.doc_id} | Version: {c.version} | Published: {c.published} | Category: {c.category}"
            block = f"{header}\n{meta}\nContent: {c.text}"
            formatted_blocks.append(block)

        return "\n\n---\n\n".join(formatted_blocks)
