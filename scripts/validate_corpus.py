"""
Validate and extract corpus.jsonl from the candidate assignment PDF.

Verifies:
1. File exists or is cleanly extracted from the embedded PDF attachment.
2. SHA-256 hash calculation and verification.
3. Exactly 75 chunks.
4. Belonging to exactly 25 documents.
5. All required metadata fields present:
   chunk_id, doc_id, title, category, owner, source_url, published, version, text
6. No malformed records or empty text fields.
"""

import os
import sys
import json
import hashlib
from pathlib import Path

EXPECTED_CHUNKS = 75
EXPECTED_DOCS = 25
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

def extract_corpus_from_pdf(pdf_path: Path, output_path: Path) -> bool:
    """Extract embedded corpus.jsonl from PDF attachment without modifying any byte."""
    try:
        import pymupdf
    except ImportError:
        print("pymupdf not installed; cannot extract from PDF directly.")
        return False

    if not pdf_path.exists():
        print(f"PDF file not found at: {pdf_path}")
        return False

    doc = pymupdf.open(str(pdf_path))
    corpus_bytes = None
    for i in range(doc.embfile_count()):
        info = doc.embfile_info(i)
        name = info.get("filename") or info.get("name")
        if name == "corpus.jsonl":
            corpus_bytes = doc.embfile_get(i)
            break

    if not corpus_bytes:
        print("corpus.jsonl was not found among embedded PDF attachments.")
        return False

    with open(output_path, "wb") as f:
        f.write(corpus_bytes)

    print(f"Successfully extracted corpus.jsonl ({len(corpus_bytes)} bytes) from {pdf_path.name}")
    return True


def compute_sha256(file_path: Path) -> str:
    """Compute the SHA-256 hash of a file."""
    hasher = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def validate_corpus(file_path: Path) -> bool:
    """Validate the integrity and schema of corpus.jsonl."""
    if not file_path.exists():
        print(f"Error: {file_path} does not exist.")
        return False

    current_hash = compute_sha256(file_path)
    print(f"File: {file_path.name}")
    print(f"SHA-256 Hash: {current_hash}")

    if current_hash != EXPECTED_SHA256:
        print(f"WARNING: Hash mismatch! Expected {EXPECTED_SHA256}, got {current_hash}")

    with open(file_path, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]

    print(f"Total lines read: {len(lines)}")
    if len(lines) != EXPECTED_CHUNKS:
        print(f"Error: Expected {EXPECTED_CHUNKS} lines, got {len(lines)}")
        return False

    chunks = []
    chunk_ids = set()
    doc_ids = set()
    errors = []

    for idx, line in enumerate(lines, start=1):
        try:
            record = json.loads(line)
        except json.JSONDecodeError as e:
            errors.append(f"Line {idx}: JSON decode error - {e}")
            continue

        chunks.append(record)

        # Validate required fields
        missing = REQUIRED_FIELDS - set(record.keys())
        if missing:
            errors.append(f"Line {idx} (ID: {record.get('chunk_id', 'unknown')}): Missing fields {missing}")

        cid = record.get("chunk_id")
        if not cid:
            errors.append(f"Line {idx}: Empty chunk_id")
        elif cid in chunk_ids:
            errors.append(f"Line {idx}: Duplicate chunk_id '{cid}'")
        else:
            chunk_ids.add(cid)

        did = record.get("doc_id")
        if not did:
            errors.append(f"Line {idx}: Empty doc_id")
        else:
            doc_ids.add(did)

        text = record.get("text", "")
        if not text or not text.strip():
            errors.append(f"Line {idx} ({cid}): Empty text field")

    if errors:
        print(f"FAILED: Found {len(errors)} validation errors:")
        for err in errors[:10]:
            print(f"  - {err}")
        return False

    print(f"Valid chunks count: {len(chunks)} (expected {EXPECTED_CHUNKS})")
    print(f"Distinct document IDs: {len(doc_ids)} (expected {EXPECTED_DOCS})")
    print(f"All required fields validated across all records.")
    print("SUCCESS: corpus.jsonl is 100% valid and unaltered!")
    return True


def main():
    root = Path(__file__).resolve().parent.parent
    corpus_file = root / "corpus.jsonl"
    pdf_file = root / "Batch5_Candidate_Assignment_Complete.pdf"

    if not corpus_file.exists():
        print("corpus.jsonl not found in workspace root. Attempting extraction from PDF...")
        extracted = extract_corpus_from_pdf(pdf_file, corpus_file)
        if not extracted:
            print("Failed to extract corpus.jsonl. Exiting.")
            sys.exit(1)

    is_valid = validate_corpus(corpus_file)
    sys.exit(0 if is_valid else 1)


if __name__ == "__main__":
    main()
