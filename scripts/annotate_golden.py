"""
Phase 5A — Interactive Golden Evaluation Set Annotation Tool.

Displays each conversation to the human annotator and collects:
  - gold_intent         : one of the 10 approved intents
  - escalate            : yes | no
  - escalation_reason   : required if escalate=yes
  - annotator_notes     : optional free-text
  - difficulty          : easy | ambiguous | hard

Usage:
    python scripts/annotate_golden.py

The tool saves progress after every completed annotation. It is safe to quit
(Ctrl+C) and resume at any time. Only rows where gold_intent is blank are shown.

SAFETY: the tool NEVER overwrites a row that already has gold_intent filled.

Output: data/golden_eval/golden_annotation.csv (updated in-place)
"""

import os
import sys
import textwrap
import pandas as pd

ANNOTATION_CSV = "data/golden_eval/golden_annotation.csv"

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

INTENT_SHORT = {
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

DIFFICULTY_SHORT = {
    "1": "easy",
    "2": "ambiguous",
    "3": "hard",
}
VALID_DIFFICULTIES = {"easy", "ambiguous", "hard"}

# -----------------------------------------------------------------
# Display strings
# -----------------------------------------------------------------

HEURISTIC_WARNING = """\
  ╔═══════════════════════════════════════════════════════════════╗
  ║  SAMPLING HEURISTIC — FOR REFERENCE ONLY. NOT A GOLD LABEL.  ║
  ║  Read the conversation above FIRST.  Do NOT copy this value  ║
  ║  without independently deciding the customer's intent.        ║
  ╚═══════════════════════════════════════════════════════════════╝
  Heuristic: {heuristic}
"""

INTENT_MENU = """
  Intent Options
  ─────────────────────────────────────────────────────────────
   1  flight_status_inquiry       Customer asks about gate / status / arrival
                                  WITHOUT requesting a different flight
   2  flight_delay_rebooking      Cancelled flight / missed connection /
                                  needs a new flight booked
   3  baggage_allowance_policy    Asks how much / size / rules for bags
                                  (not a claim for a missing/damaged bag)
   4  lost_damaged_baggage        Reports bag missing, delayed, or damaged
   5  seat_assignment_upgrade     Wants to pick, change, or upgrade a seat
   6  skymiles_loyalty_program    SkyMiles balance / Medallion / SkyClub /
                                  partner miles / missing miles
   7  refund_credit_voucher       Wants a monetary refund, eCredit, or voucher
   8  checkin_boarding_pass       Online/app check-in, boarding pass, TSA issues
   9  inflight_amenities_service  Wi-Fi / entertainment / meals / outlets / crew
  10  general_complaint_feedback  Vague praise or rant without actionable request
  ─────────────────────────────────────────────────────────────
"""

ESCALATION_POLICY = """
  Escalation Quick Reference
  ─────────────────────────────────────────────────────────────
  escalate=YES  when the agent would need account/booking access or
  human judgement:
    - Rebooking a cancelled or delayed flight
    - Lost / damaged baggage claim filing
    - Refund or eCredit application
    - Account-specific SkyMiles dispute
    - Safety, medical, or disability concerns
    - Complex multi-leg itinerary changes

  escalate=NO  when the agent can handle with policy information alone:
    - General fee / size / policy questions
    - Flight status (no account access needed)
    - Generic check-in troubleshooting steps
    - SkyMiles general benefit questions
    - Compliments / general feedback
  ─────────────────────────────────────────────────────────────
"""

DIFFICULTY_MENU = """
  Difficulty
  ─────────────────────────────────────────────────────────────
   1  easy       Intent is unambiguous; most annotators would agree
   2  ambiguous  Could fit 2+ intents; correct label needs interpretation
   3  hard       Genuinely borderline; high inter-annotator disagreement likely
  ─────────────────────────────────────────────────────────────
"""

SEP = "=" * 70
THIN = "-" * 70


def wrap(text: str, width: int = 68, indent: str = "  ") -> str:
    """Word-wraps multi-line text for terminal display."""
    lines = text.splitlines()
    wrapped = []
    for line in lines:
        if line.strip() == "":
            wrapped.append("")
        else:
            wrapped.extend(
                textwrap.wrap(
                    line,
                    width=width,
                    initial_indent=indent,
                    subsequent_indent=indent,
                )
            )
    return "\n".join(wrapped)


# -----------------------------------------------------------------
# Input prompts with validation
# -----------------------------------------------------------------

def prompt_intent() -> str:
    """Prompt for gold_intent. Heuristic is shown BEFORE this, not in the prompt."""
    while True:
        print(INTENT_MENU)
        raw = input("  Gold intent [1-10]: ").strip()
        if raw in INTENT_SHORT:
            return INTENT_SHORT[raw]
        if raw in VALID_INTENTS:
            return raw
        print("  [!] Invalid. Enter a number 1-10 or the full intent name.")


def prompt_escalate() -> str:
    while True:
        raw = input("  Escalate to human? [yes/no]: ").strip().lower()
        if raw in ("yes", "y"):
            return "yes"
        if raw in ("no", "n"):
            return "no"
        print("  [!] Please type yes or no.")


def prompt_reason(escalate: str) -> str:
    if escalate == "yes":
        while True:
            reason = input("  Escalation reason (required): ").strip()
            if reason:
                return reason
            print("  [!] Reason is required when escalate=yes.")
    else:
        return input("  Escalation reason (optional, Enter to skip): ").strip()


def prompt_notes() -> str:
    return input("  Annotator notes (optional, Enter to skip): ").strip()


def prompt_difficulty() -> str:
    while True:
        print(DIFFICULTY_MENU)
        raw = input("  Difficulty [1-3 or easy/ambiguous/hard]: ").strip().lower()
        if raw in DIFFICULTY_SHORT:
            return DIFFICULTY_SHORT[raw]
        if raw in VALID_DIFFICULTIES:
            return raw
        print("  [!] Invalid. Enter 1 (easy), 2 (ambiguous), or 3 (hard).")


# -----------------------------------------------------------------
# Persistence
# -----------------------------------------------------------------

def save(df: pd.DataFrame):
    df.to_csv(ANNOTATION_CSV, index=False, encoding="utf-8")


# -----------------------------------------------------------------
# Main
# -----------------------------------------------------------------

def main():
    if not os.path.exists(ANNOTATION_CSV):
        print(
            f"[ERROR] {ANNOTATION_CSV} not found. "
            "Run scripts/sample_golden_set.py first."
        )
        sys.exit(1)

    df = pd.read_csv(ANNOTATION_CSV, dtype=str)
    df = df.fillna("")

    # Ensure difficulty column exists (defensive — in case tool is run on
    # an older CSV that pre-dates the column addition)
    if "difficulty" not in df.columns:
        df["difficulty"] = ""

    total = len(df)
    # Only rows where gold_intent is blank are candidates for annotation.
    # This also implicitly protects already-annotated rows from overwrite.
    todo_mask = df["gold_intent"] == ""
    done = int((~todo_mask).sum())
    remaining = int(todo_mask.sum())

    print(SEP)
    print("  Delta Golden Evaluation Set — Annotation Tool")
    print(SEP)
    print(f"  Total examples : {total}")
    print(f"  Annotated      : {done}")
    print(f"  Remaining      : {remaining}")
    print(SEP)

    if remaining == 0:
        print("  All 200 examples have been annotated. Annotation complete!")
        print("  Run: python scripts/validate_golden_set.py to verify.")
        return

    print("  Instructions:")
    print("    - Read the CUSTOMER MESSAGE carefully before doing anything else.")
    print("    - The heuristic reference shown is a SAMPLING SIGNAL only.")
    print("      It is NOT ground truth. Do not copy it without independent judgment.")
    print("    - The historical Delta response is context/evidence — not automatically")
    print("      the correct answer. Judge intent from the customer message.")
    print("    - Press Ctrl+C at any time to save progress and exit safely.")
    print()

    todo_indices = df[todo_mask].index.tolist()

    try:
        for i, idx in enumerate(todo_indices, start=1):
            row = df.loc[idx]

            # SAFETY: double-check this row hasn't been annotated since load
            if df.at[idx, "gold_intent"] != "":
                print(f"  [SKIP] {row['sample_id']} already annotated in memory — skipping.")
                continue

            print()
            print(SEP)
            print(
                f"  Example {done + i} of {total}"
                f"  |  ID: {row['sample_id']}"
                f"  |  conv: {row['conversation_id']}"
            )
            print(f"  Date: {row['timestamp']}")
            print(SEP)

            # ---- Show the conversation ----
            print()
            print("  CUSTOMER MESSAGE:")
            print(wrap(row["customer_message"]))
            print()
            print("  DELTA RESPONSE (historical evidence — context only):")
            print(wrap(row["delta_response"]))
            print()
            print(THIN)

            # ---- Show heuristic BEFORE the menu, clearly framed ----
            print(HEURISTIC_WARNING.format(heuristic=row["heuristic_intent"]))

            # ---- Collect: gold_intent ----
            gold_intent = prompt_intent()

            # ---- Collect: escalation ----
            print()
            print(ESCALATION_POLICY)
            escalate = prompt_escalate()
            reason = prompt_reason(escalate)

            # ---- Collect: notes ----
            notes = prompt_notes()

            # ---- Collect: difficulty ----
            print()
            difficulty = prompt_difficulty()

            # Persist all fields atomically
            df.at[idx, "gold_intent"] = gold_intent
            df.at[idx, "escalate"] = escalate
            df.at[idx, "escalation_reason"] = reason
            df.at[idx, "annotator_notes"] = notes
            df.at[idx, "difficulty"] = difficulty
            # Direct annotation (not via proposal review) = human_annotated_direct
            if "annotation_source" in df.columns:
                df.at[idx, "annotation_source"] = "human_annotated_direct"
            save(df)

            annotated_so_far = done + i
            pct = annotated_so_far / total * 100
            print(
                f"\n  [Saved] {annotated_so_far}/{total} ({pct:.1f}%)"
                f" | intent={gold_intent} | escalate={escalate}"
                f" | difficulty={difficulty}"
            )

    except KeyboardInterrupt:
        save(df)
        annotated_now = int((df["gold_intent"] != "").sum())
        print(
            f"\n\n  [Interrupted] Progress saved. "
            f"{annotated_now}/{total} annotated."
        )
        print("  Resume anytime: python scripts/annotate_golden.py")
        sys.exit(0)

    print()
    print(SEP)
    print("  Annotation session complete!")
    annotated_now = int((df["gold_intent"] != "").sum())
    print(f"  Annotated {annotated_now}/{total} examples.")
    if annotated_now == total:
        print("  Run: python scripts/validate_golden_set.py")
    print(SEP)


if __name__ == "__main__":
    main()
