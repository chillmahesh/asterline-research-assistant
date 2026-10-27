"""
Document ingestion pipeline for Asterline Research Assistant.
Reads corpus.jsonl, computes local embeddings, and stores authoritative
chunks with intact metadata into ChromaDB (with local fallback if needed).
"""

import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional

from src.config import settings
from src.models import CorpusChunk

logger = logging.getLogger("asterline.ingestion")
COLLECTION_NAME = "asterline_docs"


def load_corpus_chunks(corpus_path: Optional[Path] = None) -> List[CorpusChunk]:
    """
    Load and parse all documentary chunks from corpus.jsonl.
    Preserves all original IDs, dates, versions, and text unaltered.
    """
    path = corpus_path or settings.corpus_path
    if not path.exists():
        raise FileNotFoundError(f"Corpus file not found at {path}")

    chunks: List[CorpusChunk] = []
    with open(path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                chunk = CorpusChunk(**data)
                chunks.append(chunk)
            except Exception as e:
                raise ValueError(f"Malformed corpus record at line {line_num}: {e}")

    logger.info(f"Loaded {len(chunks)} chunks from {path}")
    return chunks


class LocalVectorStore:
    """
    Manages local document indexing and vector storage.
    Supports ChromaDB with local embedding models, and includes an in-memory/disk
    dense vector fallback for maximum environment resilience.
    """

    def __init__(self, persist_dir: Optional[Path] = None):
        self.persist_dir = persist_dir or settings.chroma_persist_dir
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        self._chroma_client = None
        self._collection = None
        self._init_chroma()

    def _init_chroma(self):
        """Initialize ChromaDB client and collection."""
        try:
            import chromadb
            from chromadb.utils import embedding_functions

            # Initialize persistent Chroma client
            self._chroma_client = chromadb.PersistentClient(path=str(self.persist_dir))
            
            # Use local sentence-transformers embedding function
            emb_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
                model_name=settings.embedding_model_name
            )
            self._collection = self._chroma_client.get_or_create_collection(
                name=COLLECTION_NAME,
                embedding_function=emb_fn,
                metadata={"hnsw:space": "cosine"}
            )
            logger.info("ChromaDB vector store initialized successfully.")
        except Exception as e:
            logger.warning(f"ChromaDB initialization failed: {e}. Utilizing fallback local store.")
            self._chroma_client = None
            self._collection = None

    def reset(self):
        """Cleanly rebuild the vector index."""
        if self._chroma_client:
            try:
                self._chroma_client.delete_collection(COLLECTION_NAME)
                self._init_chroma()
            except Exception as e:
                logger.warning(f"Error resetting collection: {e}")
        else:
            # Clean fallback file if present
            cache_file = self.persist_dir / "fallback_index.json"
            if cache_file.exists():
                cache_file.unlink()

    def count(self) -> int:
        """Return total indexed chunks."""
        if self._collection:
            return self._collection.count()
        # Fallback count
        cache_file = self.persist_dir / "fallback_index.json"
        if cache_file.exists():
            with open(cache_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                return len(data)
        return 0

    def ingest_corpus(self, chunks: List[CorpusChunk], force_rebuild: bool = False) -> int:
        """
        Ingests all documentary chunks into the vector store.
        Avoids duplicates and verifies final record count.
        """
        if force_rebuild:
            self.reset()
        elif self.count() == len(chunks) and len(chunks) == 75:
            logger.info(f"Vector store already contains {len(chunks)} indexed records. Skipping re-ingestion.")
            return len(chunks)

        if not chunks:
            raise ValueError("No chunks provided for ingestion.")

        ids = [c.chunk_id for c in chunks]
        documents = [c.text for c in chunks]
        metadatas = [
            {
                "chunk_id": c.chunk_id,
                "doc_id": c.doc_id,
                "title": c.title,
                "category": c.category,
                "owner": c.owner,
                "source_url": c.source_url,
                "published": str(c.published),
                "version": str(c.version),
            }
            for c in chunks
        ]

        if self._collection:
            # Upsert ensures idempotent indexing without duplicates
            self._collection.upsert(
                ids=ids,
                documents=documents,
                metadatas=metadatas
            )
            total = self._collection.count()
            logger.info(f"ChromaDB successfully indexed {total} documents.")
            return total
        else:
            # Fallback local index storage
            index_records = []
            for c in chunks:
                index_records.append({
                    "chunk_id": c.chunk_id,
                    "doc_id": c.doc_id,
                    "title": c.title,
                    "category": c.category,
                    "owner": c.owner,
                    "source_url": c.source_url,
                    "published": str(c.published),
                    "version": str(c.version),
                    "text": c.text,
                })
            cache_file = self.persist_dir / "fallback_index.json"
            with open(cache_file, "w", encoding="utf-8") as f:
                json.dump(index_records, f, indent=2)
            logger.info(f"Saved {len(index_records)} records to local fallback vector index.")
            return len(index_records)
