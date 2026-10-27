"""
Tests for corpus.jsonl integrity and schema validation.
"""

import json
import hashlib
from pathlib import Path
import pytest

CORPUS_PATH = Path(__file__).resolve().parent.parent / "corpus.jsonl"
EXPECTED_SHA256 = "f63fc05cf43204485c98bd8fb6d89860a32c16a7e906bc90975e7c3954fe9fb1"
REQUIRED_FIELDS = {
    "chunk_id",
    "doc_id",
    "title",
    "category",
    "owner",
    "source_url",
    "published",
    "version",
    "text",
}

def test_corpus_exists_and_unaltered():
    assert CORPUS_PATH.exists(), f"corpus.jsonl must exist at {CORPUS_PATH}"
    
    # Calculate SHA-256
    hasher = hashlib.sha256()
    with open(CORPUS_PATH, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            hasher.update(chunk)
    file_hash = hasher.hexdigest()
    
    assert file_hash == EXPECTED_SHA256, (
        f"corpus.jsonl SHA-256 hash mismatch! "
        f"Expected {EXPECTED_SHA256}, got {file_hash}. "
        f"The original corpus must remain completely unchanged."
    )

def test_corpus_chunk_and_doc_counts():
    with open(CORPUS_PATH, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]
        
    assert len(lines) == 75, f"Expected exactly 75 records, got {len(lines)}"
    
    chunks = [json.loads(line) for line in lines]
    chunk_ids = [c["chunk_id"] for c in chunks]
    doc_ids = set(c["doc_id"] for c in chunks)
    
    assert len(chunk_ids) == 75
    assert len(set(chunk_ids)) == 75, "Chunk IDs must all be unique"
    assert len(doc_ids) == 25, f"Expected exactly 25 documents, got {len(doc_ids)}"

def test_corpus_required_fields_and_content():
    with open(CORPUS_PATH, "r", encoding="utf-8") as f:
        chunks = [json.loads(line) for line in f if line.strip()]
        
    for chunk in chunks:
        missing = REQUIRED_FIELDS - set(chunk.keys())
        assert not missing, f"Chunk {chunk.get('chunk_id')} missing fields: {missing}"
        assert chunk["chunk_id"].strip(), f"Chunk has empty chunk_id: {chunk}"
        assert chunk["doc_id"].strip(), f"Chunk has empty doc_id: {chunk}"
        assert chunk["title"].strip(), f"Chunk has empty title: {chunk}"
        assert chunk["text"].strip(), f"Chunk has empty text: {chunk}"
        assert str(chunk["version"]).strip(), f"Chunk has empty version: {chunk}"
        assert str(chunk["published"]).strip(), f"Chunk has empty published date: {chunk}"
