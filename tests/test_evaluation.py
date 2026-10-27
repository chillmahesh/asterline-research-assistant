"""
Tests for automated evaluation framework, metrics, and measured improvement.
"""

import json
from pathlib import Path
import pytest

from src.evaluation.metrics import (
    calculate_retrieval_recall,
    calculate_citation_precision,
    calculate_groundedness,
    calculate_conflict_resolution,
    calculate_unanswerable_honesty,
)
from src.evaluation.runner import EvaluationRunner, run_comparative_evaluation

CASES_FILE = Path(__file__).resolve().parent.parent / "evaluation" / "cases.json"
RESULTS_DIR = Path(__file__).resolve().parent.parent / "evaluation" / "results"


def test_cases_json_schema_and_completeness():
    assert CASES_FILE.exists(), f"Cases file missing at {CASES_FILE}"
    with open(CASES_FILE, "r", encoding="utf-8") as f:
        cases = json.load(f)

    assert len(cases) >= 10, f"Expected at least 10 evaluation cases, got {len(cases)}"
    
    categories = {c["category"] for c in cases}
    assert "research_factual" in categories
    assert "conflict_resolution" in categories
    assert "incident_investigation" in categories
    assert "missing_or_unannounced" in categories
    assert "unanswerable" in categories
    assert "memory_behavior" in categories
    assert "privacy_security" in categories
    assert "security_isolation" in categories


def test_metric_calculation_accuracy():
    # Recall
    assert calculate_retrieval_recall(["a", "b"], ["a", "b", "c"]) == 1.0
    assert calculate_retrieval_recall(["a", "b"], ["a"]) == 0.5
    assert calculate_retrieval_recall([], ["a"]) == 1.0

    # Precision
    assert calculate_citation_precision(["a", "b"], ["a", "b"], {"a", "b", "c"}) == 1.0
    assert calculate_citation_precision(["fake:1"], ["a"], {"a"}) == 0.0

    # Groundedness
    assert calculate_groundedness(["$49", "Launch"], "The Launch plan is $49") == 1.0
    assert calculate_groundedness(["$49", "Enterprise"], "The Launch plan is $49") == 0.5

    # Conflict resolution
    ans = "The current stipend is $175, which supersedes the old $140 figure."
    assert calculate_conflict_resolution(ans, "$175", "$140") == 1.0

    # Unanswerable honesty
    honest_ans = "This question cannot be answered from the supplied Asterline documentation."
    assert calculate_unanswerable_honesty(honest_ans) == 1.0


def test_comparative_evaluation_shows_measured_improvement():
    results = run_comparative_evaluation()
    
    assert results["improved_pass_rate"] >= results["baseline_pass_rate"], "Improved pass rate must equal or exceed baseline"
    assert results["delta_pass_rate"] >= 0.0
    
    # Check that result files were saved
    baseline_file = RESULTS_DIR / "baseline_results.json"
    improved_file = RESULTS_DIR / "improved_results.json"
    summary_file = RESULTS_DIR / "comparison_summary.json"
    
    assert baseline_file.exists()
    assert improved_file.exists()
    assert summary_file.exists()
