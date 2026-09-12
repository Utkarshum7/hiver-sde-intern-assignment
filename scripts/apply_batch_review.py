"""
Phase 5A — Non-interactive batch-approval applier for chat-based review.

This is a thin variant of review_golden_proposals.py's batch-accept logic,
built for sessions where the human reviewer approves batches via a chat
message (e.g. "A A R A A A A R A A") rather than an interactive terminal.

It NEVER invents or edits label content. For each sample_id in --accept it
writes the exact proposed_* fields from golden_proposals.csv into
golden_annotation.csv with annotation_source='human_approved_ai_proposal'.
Sample_ids passed via --reject are left untouched (blank) for individual
edit-review later.

SAFETY:
  - Refuses to overwrite a row that already has gold_intent filled.
  - Only ever copies fields for sample_ids explicitly listed in --accept.
  - Exits non-zero and writes nothing if any --accept id is unknown.

Usage:
    python scripts/apply_batch_review.py --accept S027,S028,S030 --reject S029
"""

import argparse
import os
import sys

import pandas as pd

ANNOTATION_CSV = "data/golden_eval/golden_annotation.csv"
PROPOSALS_CSV = "data/golden_eval/golden_proposals.csv"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--accept", default="", help="Comma-separated sample_ids to accept as-is.")
    parser.add_argument("--reject", default="", help="Comma-separated sample_ids sent to manual review (no-op here, logged only).")
    args = parser.parse_args()

    accept_ids = [s.strip() for s in args.accept.split(",") if s.strip()]
    reject_ids = [s.strip() for s in args.reject.split(",") if s.strip()]

    for path in (ANNOTATION_CSV, PROPOSALS_CSV):
        if not os.path.exists(path):
            print(f"[ERROR] {path} not found.")
            sys.exit(1)

    ann = pd.read_csv(ANNOTATION_CSV, dtype=str).fillna("")
    prop = pd.read_csv(PROPOSALS_CSV, dtype=str).fillna("")

    unknown = [sid for sid in accept_ids if sid not in set(ann["sample_id"])]
    if unknown:
        print(f"[ERROR] Unknown sample_id(s) in --accept: {unknown}")
        sys.exit(1)

    already_done = [
        sid for sid in accept_ids
        if ann.loc[ann["sample_id"] == sid, "gold_intent"].iloc[0] != ""
    ]
    if already_done:
        print(f"[ERROR] Already-approved sample_id(s), refusing to overwrite: {already_done}")
        sys.exit(1)

    applied = []
    for sid in accept_ids:
        prop_rows = prop[prop["sample_id"] == sid]
        if prop_rows.empty:
            print(f"[ERROR] No proposal found for {sid}, skipping.")
            continue
        p = prop_rows.iloc[0]
        idx = ann.index[ann["sample_id"] == sid][0]
        ann.at[idx, "gold_intent"] = p["proposed_gold_intent"]
        ann.at[idx, "escalate"] = p["proposed_escalate"]
        ann.at[idx, "escalation_reason"] = p["proposed_escalation_reason"]
        ann.at[idx, "annotator_notes"] = p["proposed_annotator_notes"]
        ann.at[idx, "difficulty"] = p["proposed_difficulty"]
        ann.at[idx, "annotation_source"] = "human_approved_ai_proposal"
        applied.append(sid)

    ann.to_csv(ANNOTATION_CSV, index=False, encoding="utf-8")

    done = int((ann["gold_intent"] != "").sum())
    total = len(ann)
    print(f"[OK] Applied {len(applied)} approvals: {applied}")
    if reject_ids:
        print(f"[INFO] Flagged for manual edit (left blank): {reject_ids}")
    print(f"[STATUS] {done}/{total} golden examples complete ({done/total:.1%})")


if __name__ == "__main__":
    main()
