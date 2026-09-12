"""
Phase 5A — Accelerated Human Review Tool for Golden Evaluation Set.

Reads proposals from:  data/golden_eval/golden_proposals.csv
Writes approved labels to: data/golden_eval/golden_annotation.csv

IMPORTANT:
  - Only HUMAN-APPROVED examples are written to golden_annotation.csv.
  - Every approved example records annotation_source:
      'human_approved_ai_proposal'  — proposal accepted without changes
      'human_edited_ai_proposal'    — proposal modified before approval
  - Ctrl+C safely saves progress and exits.

Modes:
  python scripts/review_golden_proposals.py          # one example at a time
  python scripts/review_golden_proposals.py --batch  # 10 examples at a time

Options:
  --from <sample_id>  Start review from a specific sample_id (e.g. S050)
"""

import os
import sys
import argparse
import textwrap
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

ANNOTATION_CSV = "data/golden_eval/golden_annotation.csv"
PROPOSALS_CSV  = "data/golden_eval/golden_proposals.csv"
BATCH_SIZE     = 10

VALID_INTENTS = [
    "flight_status_inquiry",
    "flight_delay_rebooking",
    "baggage_allowance_policy",
    "lost_damaged_baggage",
    "seat_assignment_upgrade",
    "skymiles_loyalty_program",
    "refund_credit_voucher",
    "checkin_boarding_pass",
    "inflight_amenities_service",
    "general_complaint_feedback",
]

INTENT_NUM = {
    "1": "flight_status_inquiry",
    "2": "flight_delay_rebooking",
    "3": "baggage_allowance_policy",
    "4": "lost_damaged_baggage",
    "5": "seat_assignment_upgrade",
    "6": "skymiles_loyalty_program",
    "7": "refund_credit_voucher",
    "8": "checkin_boarding_pass",
    "9": "inflight_amenities_service",
    "10": "general_complaint_feedback",
}

VALID_DIFFICULTIES = {"easy", "ambiguous", "hard"}
DIFF_NUM = {"1": "easy", "2": "ambiguous", "3": "hard"}

SEP   = "=" * 68
THIN  = "-" * 68

# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------

def wrap(text: str, width: int = 66, indent: str = "  ") -> str:
    lines = str(text).splitlines()
    out = []
    for line in lines:
        if line.strip() == "":
            out.append("")
        else:
            out.extend(textwrap.wrap(line, width=width,
                                     initial_indent=indent,
                                     subsequent_indent=indent))
    return "\n".join(out)


def _trunc(text: str, n: int = 90) -> str:
    t = str(text).replace("\n", " ").strip()
    return t[:n] + "…" if len(t) > n else t


def _intent_line(intent: str) -> str:
    return f"{intent:<35}"


# ---------------------------------------------------------------------------
# Persistence helpers
# ---------------------------------------------------------------------------

def load_both():
    """Load annotation CSV and proposals CSV. Returns (ann_df, prop_df)."""
    ann = pd.read_csv(ANNOTATION_CSV, dtype=str).fillna("")
    prop = pd.read_csv(PROPOSALS_CSV, dtype=str).fillna("")
    # Ensure annotation_source column exists
    if "annotation_source" not in ann.columns:
        ann["annotation_source"] = ""
    return ann, prop


def save_ann(ann: pd.DataFrame):
    ann.to_csv(ANNOTATION_CSV, index=False, encoding="utf-8")


def _get_proposal(prop_df: pd.DataFrame, sample_id: str) -> dict:
    rows = prop_df[prop_df["sample_id"] == sample_id]
    if rows.empty:
        return {}
    return rows.iloc[0].to_dict()


def _write_approval(
    ann: pd.DataFrame,
    sample_id: str,
    gold_intent: str,
    escalate: str,
    reason: str,
    notes: str,
    difficulty: str,
    source: str,
):
    """Write approved fields to the annotation dataframe (in-place)."""
    idx = ann.index[ann["sample_id"] == sample_id]
    if idx.empty:
        return
    i = idx[0]
    # SAFETY: never overwrite an already-approved row
    if ann.at[i, "gold_intent"] != "":
        return
    ann.at[i, "gold_intent"]         = gold_intent
    ann.at[i, "escalate"]            = escalate
    ann.at[i, "escalation_reason"]   = reason
    ann.at[i, "annotator_notes"]     = notes
    ann.at[i, "difficulty"]          = difficulty
    ann.at[i, "annotation_source"]   = source


# ---------------------------------------------------------------------------
# Input helpers (validated)
# ---------------------------------------------------------------------------

def _ask_intent(current: str) -> str:
    print()
    print("  Intents:")
    for num, intent in INTENT_NUM.items():
        marker = " <-- current" if intent == current else ""
        print(f"    {num:>2}  {intent}{marker}")
    while True:
        raw = input("  New intent [1-10 or full name]: ").strip()
        if raw in INTENT_NUM:
            return INTENT_NUM[raw]
        if raw in VALID_INTENTS:
            return raw
        print("  [!] Invalid. Enter 1-10 or full intent name.")


def _ask_escalate(current: str) -> str:
    while True:
        raw = input(f"  Escalate? [yes/no] (current: {current}): ").strip().lower()
        if raw in ("yes", "y"):
            return "yes"
        if raw in ("no", "n"):
            return "no"
        print("  [!] Enter yes or no.")


def _ask_reason(current: str, escalate: str) -> str:
    if escalate == "yes":
        while True:
            raw = input(f"  Escalation reason (required, Enter to keep current):\n"
                        f"  [{current[:60]}]\n  > ").strip()
            if raw:
                return raw
            if current:
                return current
            print("  [!] Reason required when escalate=yes.")
    else:
        raw = input(f"  Escalation reason (optional, Enter to keep):\n"
                    f"  [{current[:60]}]\n  > ").strip()
        return raw if raw else current


def _ask_difficulty(current: str) -> str:
    while True:
        raw = input(
            f"  Difficulty [1=easy/2=ambiguous/3=hard] (current: {current}): "
        ).strip().lower()
        if raw in DIFF_NUM:
            return DIFF_NUM[raw]
        if raw in VALID_DIFFICULTIES:
            return raw
        print("  [!] Enter 1, 2, 3, or easy/ambiguous/hard.")


def _ask_notes(current: str) -> str:
    raw = input(
        f"  Notes (Enter to keep current):\n  [{current[:60]}]\n  > "
    ).strip()
    return raw if raw else current


# ---------------------------------------------------------------------------
# Single-example review
# ---------------------------------------------------------------------------

def _show_example(ann_row: pd.Series, prop: dict, done: int, total: int):
    sample_id = ann_row["sample_id"]
    print()
    print(SEP)
    print(f"  {done+1}/{total}  |  {sample_id}  |  conv: {ann_row['conversation_id']}")
    print(f"  {ann_row['timestamp']}")
    print(SEP)
    print()
    print("  CUSTOMER MESSAGE:")
    print(wrap(ann_row["customer_message"]))
    print()
    print("  DELTA RESPONSE (context only):")
    print(wrap(ann_row["delta_response"]))
    print()
    print(THIN)
    print(f"  HEURISTIC (sampling ref only): {ann_row['heuristic_intent']}")
    print(THIN)
    pi = prop.get("proposed_gold_intent", "—")
    pe = prop.get("proposed_escalate", "—")
    pr = prop.get("proposed_escalation_reason", "—")
    pd_ = prop.get("proposed_difficulty", "—")
    pn = prop.get("proposed_annotator_notes", "—")
    print(f"  PROPOSED INTENT    : {pi}")
    print(f"  PROPOSED ESCALATE  : {pe}  |  DIFFICULTY: {pd_}")
    print(f"  PROPOSED REASON    : {_trunc(pr, 70)}")
    print(f"  PROPOSED NOTES     : {_trunc(pn, 70)}")
    print()


def _single_review(ann_row: pd.Series, prop: dict) -> dict:
    """
    Interactive single-example review.
    Returns final field values or None if skipped.
    """
    gold_intent = prop.get("proposed_gold_intent", "general_complaint_feedback")
    escalate    = prop.get("proposed_escalate", "no")
    reason      = prop.get("proposed_escalation_reason", "")
    difficulty  = prop.get("proposed_difficulty", "ambiguous")
    notes       = prop.get("proposed_annotator_notes", "")
    edited      = False

    while True:
        print()
        print("  OPTIONS:  A=accept all   I=intent   E=escalate   D=difficulty")
        print("            R=reason       N=notes    S=skip")
        choice = input("  > ").strip().upper()

        if choice == "A":
            return {
                "gold_intent": gold_intent,
                "escalate": escalate,
                "reason": reason,
                "notes": notes,
                "difficulty": difficulty,
                "edited": edited,
            }
        elif choice == "S":
            return None
        elif choice == "I":
            new = _ask_intent(gold_intent)
            if new != gold_intent:
                edited = True
                gold_intent = new
            print(f"  Intent set to: {gold_intent}")
        elif choice == "E":
            new = _ask_escalate(escalate)
            if new != escalate:
                edited = True
                escalate = new
            if escalate == "yes" and not reason:
                reason = _ask_reason(reason, escalate)
            print(f"  Escalate set to: {escalate}")
        elif choice == "D":
            new = _ask_difficulty(difficulty)
            if new != difficulty:
                edited = True
                difficulty = new
            print(f"  Difficulty set to: {difficulty}")
        elif choice == "R":
            new = _ask_reason(reason, escalate)
            if new != reason:
                edited = True
                reason = new
        elif choice == "N":
            new = _ask_notes(notes)
            if new != notes:
                edited = True
                notes = new
        else:
            print("  [!] Invalid choice.")


# ---------------------------------------------------------------------------
# Batch review
# ---------------------------------------------------------------------------

def _parse_batch_choices(raw: str, n: int):
    """
    Parse user input for batch choices. Returns list of 'A' or 'R' of length n.
    Accepts: "A R A A A A R A A A", "ARAAAARAAA", "ALL"
    """
    raw = raw.strip().upper()
    if raw == "ALL":
        return ["A"] * n
    # Remove spaces and split
    raw_clean = raw.replace(" ", "")
    if len(raw_clean) == n and all(c in "AR" for c in raw_clean):
        return list(raw_clean)
    parts = raw.split()
    if len(parts) == n and all(p in ("A", "R") for p in parts):
        return parts
    return None


def _show_batch(batch_rows, batch_props, done, total):
    """Display a compact summary of 10 proposals."""
    print()
    print(SEP)
    start_n = done + 1
    end_n   = done + len(batch_rows)
    approved_count = done
    print(f"  BATCH REVIEW  —  Examples {start_n}–{end_n} of {total}"
          f"  |  Approved so far: {approved_count}")
    print(SEP)
    print()
    for local_i, (_, ann_row) in enumerate(batch_rows, start=1):
        sid  = ann_row["sample_id"]
        prop = batch_props[local_i - 1]
        hi   = ann_row["heuristic_intent"]
        pi   = prop.get("proposed_gold_intent", "—")
        pe   = prop.get("proposed_escalate", "—")
        pd_  = prop.get("proposed_difficulty", "—")
        changed = "[DIFF]" if pi != hi else "      "
        msg_preview = _trunc(ann_row["customer_message"], 80)
        print(f"  {local_i:>2}. {sid}  {changed}")
        print(f"      heuristic: {hi:<35}  proposed: {pi}")
        print(f"      escalate={pe}  difficulty={pd_}")
        print(f"      \"{msg_preview}\"")
        print()
    print(THIN)
    print("  Mark each: A=accept  R=review-one-by-one")
    print("  Enter choices (e.g.  A A R A A A A A A A   or   ARAAAAAAAA   or   ALL)")


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Review proposed golden-set labels and approve to annotation CSV."
    )
    parser.add_argument("--batch", action="store_true",
                        help="Batch-review mode: 10 examples at a time.")
    parser.add_argument("--from", dest="from_id", default=None,
                        help="Start from a specific sample_id (e.g. S050).")
    args = parser.parse_args()

    for path in (ANNOTATION_CSV, PROPOSALS_CSV):
        if not os.path.exists(path):
            print(f"[ERROR] {path} not found.")
            sys.exit(1)

    ann, prop_df = load_both()

    # Ensure annotation_source column exists (defensive)
    if "annotation_source" not in ann.columns:
        ann["annotation_source"] = ""

    total = len(ann)

    # Find unapproved rows
    unapproved_mask = ann["gold_intent"] == ""
    unapproved_indices = ann[unapproved_mask].index.tolist()

    # Optionally start from a specific sample_id
    if args.from_id:
        target_indices = ann.index[ann["sample_id"] == args.from_id].tolist()
        if not target_indices:
            print(f"[ERROR] sample_id '{args.from_id}' not found.")
            sys.exit(1)
        target_pos = unapproved_indices.index(target_indices[0]) \
            if target_indices[0] in unapproved_indices else None
        if target_pos is None:
            print(f"[INFO] {args.from_id} is already approved. Starting from next unapproved.")
        else:
            unapproved_indices = unapproved_indices[target_pos:]

    done   = int((ann["gold_intent"] != "").sum())
    remaining = len(unapproved_indices)

    print(SEP)
    print("  Delta Golden Set — Proposal Review Tool")
    print(SEP)
    print(f"  Total:     {total}")
    print(f"  Approved:  {done}")
    print(f"  Remaining: {remaining}")
    print(f"  Mode:      {'BATCH (10 at a time)' if args.batch else 'SINGLE'}")
    print(SEP)

    if remaining == 0:
        print("  All 200 examples have been reviewed and approved!")
        print("  Run: python scripts/validate_golden_set.py")
        return

    print()
    print("  Press Ctrl+C at any time to save progress and exit.")
    print()

    review_queue = []  # For batch mode: samples marked 'R' go here

    try:
        if args.batch:
            # ---- BATCH MODE ----
            i = 0
            while i < len(unapproved_indices):
                batch_idx = unapproved_indices[i: i + BATCH_SIZE]
                batch_rows = [(idx, ann.loc[idx]) for idx in batch_idx]
                batch_props = [
                    _get_proposal(prop_df, ann.loc[idx]["sample_id"])
                    for idx in batch_idx
                ]

                _show_batch(batch_rows, batch_props, done, total)

                # Get choices
                while True:
                    raw = input("  > ").strip()
                    choices = _parse_batch_choices(raw, len(batch_idx))
                    if choices is not None:
                        break
                    print(f"  [!] Enter {len(batch_idx)} choices (A or R), "
                          f"space-separated, or ALL.")

                # Process choices
                for local_i, (choice, (idx, ann_row)) in enumerate(
                        zip(choices, batch_rows)):
                    sid = ann_row["sample_id"]
                    prop = batch_props[local_i]

                    if choice == "A":
                        _write_approval(
                            ann, sid,
                            gold_intent=prop.get("proposed_gold_intent", "general_complaint_feedback"),
                            escalate=prop.get("proposed_escalate", "no"),
                            reason=prop.get("proposed_escalation_reason", ""),
                            notes=prop.get("proposed_annotator_notes", ""),
                            difficulty=prop.get("proposed_difficulty", "ambiguous"),
                            source="human_approved_ai_proposal",
                        )
                        done += 1
                    else:
                        review_queue.append(idx)

                save_ann(ann)
                print(f"\n  [Saved] Approved so far: {done}/{total}")

                # Process single-review queue after each batch
                while review_queue:
                    idx = review_queue.pop(0)
                    ann_row = ann.loc[idx]
                    # Skip if already approved (shouldn't happen, but defensive)
                    if ann.at[idx, "gold_intent"] != "":
                        continue
                    sid = ann_row["sample_id"]
                    prop = _get_proposal(prop_df, sid)
                    _show_example(ann_row, prop, done, total)
                    result = _single_review(ann_row, prop)
                    if result is None:
                        print(f"  [Skipped] {sid} — will not appear again in this session.")
                        continue
                    source = ("human_edited_ai_proposal" if result["edited"]
                              else "human_approved_ai_proposal")
                    _write_approval(
                        ann, sid,
                        gold_intent=result["gold_intent"],
                        escalate=result["escalate"],
                        reason=result["reason"],
                        notes=result["notes"],
                        difficulty=result["difficulty"],
                        source=source,
                    )
                    done += 1
                    save_ann(ann)
                    pct = done / total * 100
                    print(f"\n  [Saved] {done}/{total} ({pct:.1f}%) approved.")

                i += BATCH_SIZE

        else:
            # ---- SINGLE MODE ----
            for i, idx in enumerate(unapproved_indices):
                ann_row = ann.loc[idx]
                sid = ann_row["sample_id"]
                prop = _get_proposal(prop_df, sid)

                _show_example(ann_row, prop, done, total)
                result = _single_review(ann_row, prop)

                if result is None:
                    print(f"  [Skipped] {sid}")
                    continue

                source = ("human_edited_ai_proposal" if result["edited"]
                          else "human_approved_ai_proposal")
                _write_approval(
                    ann, sid,
                    gold_intent=result["gold_intent"],
                    escalate=result["escalate"],
                    reason=result["reason"],
                    notes=result["notes"],
                    difficulty=result["difficulty"],
                    source=source,
                )
                done += 1
                save_ann(ann)
                pct = done / total * 100
                print(f"\n  [Saved] {done}/{total} ({pct:.1f}%) | "
                      f"source={source}")

    except KeyboardInterrupt:
        save_ann(ann)
        approved_now = int((ann["gold_intent"] != "").sum())
        print(f"\n\n  [Interrupted] Progress saved. {approved_now}/{total} approved.")
        print("  Resume: python scripts/review_golden_proposals.py"
              + (" --batch" if args.batch else ""))
        sys.exit(0)

    # Completion summary
    save_ann(ann)
    final_approved = int((ann["gold_intent"] != "").sum())
    print()
    print(SEP)
    print(f"  Review session complete. Approved: {final_approved}/{total}")
    if final_approved == total:
        print("  Run: python scripts/validate_golden_set.py")
    print(SEP)


if __name__ == "__main__":
    main()
