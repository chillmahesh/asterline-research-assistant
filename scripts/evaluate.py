"""
CLI script to execute the reproducible evaluation suite and generate comparison reports.
"""

import sys
import json
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.evaluation.runner import EvaluationRunner, run_comparative_evaluation


def main():
    print("================================================================")
    print(" Asterline Research Assistant — Automated Evaluation Suite")
    print("================================================================\n")

    comparison = run_comparative_evaluation()

    print("\n----------------------------------------------------------------")
    print(" EVALUATION RESULTS SUMMARY")
    print("----------------------------------------------------------------")
    print(f" Baseline Pass Rate:         {comparison['baseline_pass_rate']}%")
    print(f" Improved Pass Rate:         {comparison['improved_pass_rate']}%")
    print(f" Measured Improvement:       +{comparison['delta_pass_rate']}%")
    print(f" Mean Retrieval Recall:      {comparison['improved_recall']}")
    print(f" Mean Citation Precision:    {comparison['improved_precision']}")
    print("----------------------------------------------------------------")
    print(" Detailed reports saved in:  evaluation/results/")
    print("================================================================\n")


if __name__ == "__main__":
    main()
