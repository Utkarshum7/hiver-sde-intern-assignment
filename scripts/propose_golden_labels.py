"""
Phase 5A — Generate proposed labels for the Delta Golden Evaluation Set.

Reads:  data/golden_eval/golden_annotation.csv  (frozen — NOT modified)
Writes: data/golden_eval/golden_proposals.csv   (proposals only)

IMPORTANT:
  - This script does NOT write to golden_annotation.csv.
  - All output is PROPOSED labels requiring human review.
  - Use scripts/review_golden_proposals.py to review and approve.

Usage:
    python scripts/propose_golden_labels.py
    python scripts/propose_golden_labels.py --force   # overwrite existing proposals
"""

import os
import sys
import argparse
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.proposer import propose_labels, PROPOSAL_METHOD

ANNOTATION_CSV = "data/golden_eval/golden_annotation.csv"
PROPOSALS_CSV  = "data/golden_eval/golden_proposals.csv"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true",
                        help="Overwrite existing golden_proposals.csv")
    args = parser.parse_args()

    if os.path.exists(PROPOSALS_CSV) and not args.force:
        print(f"[ERROR] {PROPOSALS_CSV} already exists.")
        print("        Use --force to regenerate (this will overwrite existing proposals).")
        sys.exit(1)

    if not os.path.exists(ANNOTATION_CSV):
        print(f"[ERROR] {ANNOTATION_CSV} not found.")
        sys.exit(1)

    print(f"[*] Loading golden annotation scaffold: {ANNOTATION_CSV}")
    ann = pd.read_csv(ANNOTATION_CSV, dtype=str).fillna("")
    n = len(ann)
    print(f"[+] Loaded {n} rows.")

    # Verify the scaffold is intact before touching anything
    assert n == 200, f"Expected 200 rows, got {n}"
    assert ann["conversation_id"].nunique() == 200, "Duplicate conversation_ids in scaffold!"

    print(f"[*] Generating proposals using: {PROPOSAL_METHOD}")
    print("    (Proposals are NOT ground truth — all require human review.)")
    print()

    records = []
    for _, row in ann.iterrows():
        result = propose_labels(row["customer_message"], row["delta_response"])
        records.append({
            "sample_id":                   row["sample_id"],
            "conversation_id":             row["conversation_id"],
            "customer_message":            row["customer_message"],
            "heuristic_intent":            row["heuristic_intent"],
            "proposed_gold_intent":        result["proposed_gold_intent"],
            "proposed_escalate":           result["proposed_escalate"],
            "proposed_escalation_reason":  result["proposed_escalation_reason"],
            "proposed_difficulty":         result["proposed_difficulty"],
            "proposed_annotator_notes":    result["proposed_annotator_notes"],
            "proposal_method":             result["proposal_method"],
        })

    proposals = pd.DataFrame(records)
    proposals.to_csv(PROPOSALS_CSV, index=False, encoding="utf-8")
    print(f"[+] Wrote {len(proposals)} proposals to: {PROPOSALS_CSV}")

    # ---- Statistics ----
    print()
    print("=" * 60)
    print("  PROPOSAL STATISTICS (NOT gold results)")
    print("=" * 60)

    print("\n  Proposed intent distribution:")
    intent_dist = proposals["proposed_gold_intent"].value_counts()
    for intent in proposals["proposed_gold_intent"].cat.categories if hasattr(
            proposals["proposed_gold_intent"], "cat") else sorted(
            proposals["proposed_gold_intent"].unique()):
        print(f"    {intent:<35} {int(intent_dist.get(intent, 0)):>4}")

    print("\n  Proposed escalation distribution:")
    esc_dist = proposals["proposed_escalate"].value_counts()
    for val in ["yes", "no"]:
        print(f"    escalate={val:<3}  {int(esc_dist.get(val, 0)):>4}")

    print("\n  Proposed difficulty distribution:")
    diff_dist = proposals["proposed_difficulty"].value_counts()
    for val in ["easy", "ambiguous", "hard"]:
        print(f"    {val:<10}  {int(diff_dist.get(val, 0)):>4}")

    # Disagreement with heuristic
    disagree = (proposals["proposed_gold_intent"] != proposals["heuristic_intent"]).sum()
    print(f"\n  Proposed intent differs from heuristic: {disagree}/200 "
          f"({disagree/200:.1%})")

    ambig_hard = proposals["proposed_difficulty"].isin(["ambiguous", "hard"]).sum()
    print(f"  Flagged ambiguous or hard:              {ambig_hard}/200 "
          f"({ambig_hard/200:.1%})")

    print()
    print("=" * 60)
    print("  NEXT STEP:")
    print("  python scripts/review_golden_proposals.py --batch")
    print("  (to batch-review 10 examples at a time)")
    print("  python scripts/review_golden_proposals.py")
    print("  (to review one example at a time)")
    print("=" * 60)


if __name__ == "__main__":
    main()
