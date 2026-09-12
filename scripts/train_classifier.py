"""
Trains the TF-IDF + Logistic Regression intent classifier used by the live
agent (src/agent.py), using ONLY the development corpus (never the reserved
golden pool). See src/classifier.py module docstring for the design
rationale (weak supervision, deliberately distinct from the proposer used
for golden-set annotation assistance).

Usage:
    python scripts/train_classifier.py

Outputs:
    data/processed/intent_classifier.pkl
    reports/classifier_training_report.json
"""

import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.classifier import train_from_dev_corpus, DEFAULT_MODEL_PATH


def main():
    print("[*] Training intent classifier on data/processed/dev_corpus.parquet ...")
    clf, report = train_from_dev_corpus()

    clf.save(DEFAULT_MODEL_PATH)
    print(f"[+] Saved trained classifier to: {DEFAULT_MODEL_PATH}")

    os.makedirs("reports", exist_ok=True)
    report_path = "reports/classifier_training_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"[+] Saved training report to: {report_path}")

    print(f"\n[+] Training examples: {report['training_examples']:,}")
    print(f"[+] Vocabulary size: {report['vocabulary_size']:,}")
    print(f"[+] Classes ({len(report['classes'])}): {report['classes']}")
    print("\n[+] Label distribution (weak labels used for training):")
    for label, count in sorted(report["label_distribution"].items(), key=lambda x: -x[1]):
        print(f"    {label:<35} {count:>6}")

    print(
        "\n[NOTE] These weak labels come from a single-tier keyword heuristic "
        "(src/heuristic_intent.py), documented to be wrong ~15-25% of the "
        "time. This classifier is trained via weak supervision, not on clean "
        "human labels. See docs/decision_log.md."
    )


if __name__ == "__main__":
    main()
