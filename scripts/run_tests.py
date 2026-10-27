"""
Standalone test runner for Asterline Research Assistant test suite.
Discovers and executes all test functions in tests/ with fixture resolution and detailed reporting.
"""

import sys
import os
import time
import inspect
import tempfile
import traceback
from pathlib import Path

# Add project root to path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.retrieval import DocumentRetriever
from src.graph import AsterlineGraph
from src.memory.repository import SQLiteMemoryRepository
from src.memory.service import MemoryService
from src.ingestion import load_corpus_chunks


def build_fixtures():
    """Build shared fixtures for tests."""
    fixtures = {}
    
    # Corpus data fixture
    chunks = load_corpus_chunks()
    fixtures["corpus_data"] = {c.chunk_id: c.model_dump() for c in chunks}
    
    # Retriever fixture
    fixtures["retriever"] = DocumentRetriever()
    
    # Graph fixture
    fixtures["graph"] = AsterlineGraph(retriever=fixtures["retriever"])
    
    return fixtures


def run_all_tests():
    print("================================================================")
    print(" Asterline Research Assistant — Automated Test Suite")
    print("================================================================\n")

    test_dir = ROOT / "tests"
    test_files = sorted(test_dir.glob("test_*.py"))
    
    total_passed = 0
    total_failed = 0
    total_duration = 0.0
    failures = []

    shared_fixtures = build_fixtures()

    for tfile in test_files:
        module_name = f"tests.{tfile.stem}"
        print(f"--- Running {tfile.name} ---")
        try:
            # Import test module dynamically
            mod = __import__(module_name, fromlist=["*"])
        except Exception as e:
            print(f"  [ERROR] Failed to import {tfile.name}: {e}")
            total_failed += 1
            failures.append((tfile.name, "import_error", traceback.format_exc()))
            continue

        # Find test functions
        test_funcs = [
            (name, func) for name, func in inspect.getmembers(mod, inspect.isfunction)
            if name.startswith("test_")
        ]

        for fname, func in test_funcs:
            sig = inspect.signature(func)
            kwargs = {}
            temp_ctx = None

            # Resolve fixtures
            try:
                for param in sig.parameters:
                    if param == "temp_service":
                        # Generate temporary SQLite database and service
                        tmp_dir = tempfile.mkdtemp()
                        db_path = Path(tmp_dir) / "test.db"
                        repo = SQLiteMemoryRepository(db_path=db_path)
                        svc = MemoryService(repository=repo)
                        kwargs[param] = (svc, db_path, repo)
                    elif param in shared_fixtures:
                        kwargs[param] = shared_fixtures[param]
                    else:
                        print(f"  [WARN] Unknown fixture parameter: {param} in {fname}")

                t0 = time.time()
                func(**kwargs)
                dt = time.time() - t0
                total_duration += dt
                print(f"  [PASS] {fname} ({dt:.3f}s)")
                total_passed += 1
            except Exception as e:
                dt = time.time() - t0
                total_duration += dt
                total_failed += 1
                tb = traceback.format_exc()
                print(f"  [FAIL] {fname} ({dt:.3f}s) - {e}")
                failures.append((f"{tfile.name}::{fname}", str(e), tb))

        print()

    print("================================================================")
    print(f" TEST RUN COMPLETE: {total_passed} Passed, {total_failed} Failed in {total_duration:.2f}s")
    print("================================================================")

    if failures:
        print("\nFailures Detail:")
        for name, err, tb in failures:
            print(f"\n[{name}]:\n{tb}")
        return False
    else:
        print("\nAll automated tests passed successfully! 100% GREEN.\n")
        return True


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
