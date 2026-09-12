"""
Phase 5A — Golden Evaluation Set Completeness Validator.

Checks that data/golden_eval/golden_annotation.csv meets all quality gates
before it can be used for evaluation.

Required quality gates (FAIL = set is not ready):
  G1.  Exactly 200 rows.
  G2.  All sample_ids unique.
  G3.  All conversation_ids unique.
  G4.  All conversation_ids are from the reserved_golden_pool (leakage check).
  G5.  No conversation_ids appear in the dev_corpus (leakage check).
  G6a. gold_intent: no blanks.
  G6b. gold_intent: all values are valid intent codes.
  G7a. escalate: no blanks.
  G7b. escalate: all values are 'yes' or 'no'.
  G8.  escalation_reason: non-empty for all rows where escalate == 'yes'.
  G9.  All 10 intents represented at least once in gold_intent.
  G10. Non-trivial annotator disagreement with heuristic (>5%).

Advisory checks (reported but do NOT fail the set):
  A1. difficulty column present.
  A2. No blank difficulty values (advisory — partial annotation permitted).
  A3. All difficulty values are valid (easy | ambiguous | hard).

Usage:
    python scripts/validate_golden_set.py
"""

import os
import sys
import pandas as pd

ANNOTATION_CSV = "data/golden_eval/golden_annotation.csv"
RESERVED_PARQUET = "data/processed/reserved_golden_pool.parquet"
DEV_PARQUET = "data/processed/dev_corpus.parquet"

VALID_INTENTS = {
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
}

VALID_DIFFICULTIES = {"easy", "ambiguous", "hard"}


def gate(name: str, condition: bool, detail: str = "") -> bool:
    status = "PASS" if condition else "FAIL"
    print(f"  [{status}] {name}")
    if detail:
        print(f"         {detail}")
    return condition


def advisory(name: str, condition: bool, detail: str = ""):
    status = "OK  " if condition else "WARN"
    print(f"  [{status}] {name}")
    if detail:
        print(f"         {detail}")


def main():
    required_pass = True

    print("=" * 60)
    print("  Golden Evaluation Set — Validation Report")
    print("=" * 60)

    if not os.path.exists(ANNOTATION_CSV):
        print(f"[ERROR] {ANNOTATION_CSV} not found.")
        sys.exit(1)

    df = pd.read_csv(ANNOTATION_CSV, dtype=str)
    df = df.fillna("")

    print(f"\n  File: {ANNOTATION_CSV}")
    print(f"  Rows loaded: {len(df)}\n")
    print("  --- Required Gates ---\n")

    # G1
    required_pass &= gate("G1 : Exactly 200 rows", len(df) == 200,
                          f"actual={len(df)}")

    # G2
    dup_ids = df["sample_id"].duplicated().sum()
    required_pass &= gate("G2 : All sample_ids unique", dup_ids == 0,
                          f"duplicates={dup_ids}")

    # G3
    dup_convs = df["conversation_id"].duplicated().sum()
    required_pass &= gate("G3 : All conversation_ids unique", dup_convs == 0,
                          f"duplicates={dup_convs}")

    annotation_ids = set(df["conversation_id"].astype(str))

    # G4
    if os.path.exists(RESERVED_PARQUET):
        reserved_df = pd.read_parquet(RESERVED_PARQUET)
        reserved_ids = set(reserved_df["conversation_id"].astype(str))
        not_in_reserved = annotation_ids - reserved_ids
        required_pass &= gate(
            "G4 : All conv_ids from reserved pool",
            len(not_in_reserved) == 0,
            f"ids not in reserved pool: {len(not_in_reserved)}"
        )
    else:
        print(f"  [SKIP] G4: {RESERVED_PARQUET} not found")

    # G5
    if os.path.exists(DEV_PARQUET):
        dev_df = pd.read_parquet(DEV_PARQUET)
        dev_ids = set(dev_df["conversation_id"].astype(str))
        leaked = annotation_ids & dev_ids
        required_pass &= gate(
            "G5 : No overlap with dev corpus",
            len(leaked) == 0,
            f"leaked conversation_ids: {len(leaked)}"
        )
    else:
        print(f"  [SKIP] G5: {DEV_PARQUET} not found")

    # G6a
    blank_intent = (df["gold_intent"] == "").sum()
    required_pass &= gate("G6a: No blank gold_intent", blank_intent == 0,
                          f"blank rows: {blank_intent}")

    # G6b
    invalid_intent = [
        v for v in df["gold_intent"].unique()
        if v != "" and v not in VALID_INTENTS
    ]
    required_pass &= gate("G6b: All gold_intent values valid",
                          len(invalid_intent) == 0,
                          f"invalid values: {invalid_intent}")

    # G7a
    blank_esc = (df["escalate"] == "").sum()
    required_pass &= gate("G7a: No blank escalate", blank_esc == 0,
                          f"blank rows: {blank_esc}")

    # G7b
    invalid_esc = [
        v for v in df["escalate"].unique()
        if v not in {"yes", "no", ""}
    ]
    required_pass &= gate("G7b: All escalate values are yes|no",
                          len(invalid_esc) == 0,
                          f"invalid values: {invalid_esc}")

    # G8
    esc_yes = df[df["escalate"] == "yes"]
    missing_reason = (esc_yes["escalation_reason"] == "").sum()
    required_pass &= gate(
        "G8 : escalation_reason filled for all escalate=yes",
        missing_reason == 0,
        f"rows missing reason: {missing_reason}"
    )

    # G9
    represented = set(df["gold_intent"].unique()) - {""}
    missing_intents = VALID_INTENTS - represented
    required_pass &= gate(
        "G9 : All 10 intents represented in gold_intent",
        len(missing_intents) == 0,
        f"missing intents: {sorted(missing_intents)}"
    )

    # G10
    labelled_mask = df["gold_intent"] != ""
    total_labelled = labelled_mask.sum()
    if total_labelled > 0:
        matches = (df.loc[labelled_mask, "gold_intent"]
                   == df.loc[labelled_mask, "heuristic_intent"]).sum()
        disagreement_rate = 1 - matches / total_labelled
    else:
        disagreement_rate = 0.0
    required_pass &= gate(
        "G10: Non-trivial annotator disagreement with heuristic (>5%)",
        disagreement_rate > 0.05,
        f"annotator disagreement rate: {disagreement_rate:.1%}"
    )

    # ---- Advisory checks ----
    print()
    print("  --- Advisory Checks (informational, do not block) ---\n")

    has_difficulty = "difficulty" in df.columns
    advisory("A1 : difficulty column present", has_difficulty,
             "" if has_difficulty else "Run scripts/_add_difficulty_col.py to add it.")

    if has_difficulty:
        blank_diff = (df["difficulty"] == "").sum()
        advisory("A2 : No blank difficulty values", blank_diff == 0,
                 f"blank rows: {blank_diff} (annotation in progress or not yet started)")

        invalid_diff = [
            v for v in df["difficulty"].unique()
            if v != "" and v not in VALID_DIFFICULTIES
        ]
        advisory("A3 : All difficulty values valid (easy|ambiguous|hard)",
                 len(invalid_diff) == 0,
                 f"invalid values: {invalid_diff}")
    else:
        print("  [SKIP] A2, A3: difficulty column not present")

    # ---- New Required Gates (G11-G14) for Annotation Source ----
    VALID_SOURCES = {"human_approved_ai_proposal", "human_edited_ai_proposal", "human_annotated_direct"}
    has_src = "annotation_source" in df.columns
    required_pass &= gate(
        "G11: annotation_source column present", has_src,
        "Run review tool to add annotation_source."
    )

    if has_src:
        labelled_src_df = df[df["gold_intent"] != ""]
        missing_src = (labelled_src_df["annotation_source"] == "").sum()
        required_pass &= gate(
            "G12: All completed rows have annotation_source",
            missing_src == 0,
            f"rows missing annotation_source: {missing_src}"
        )
        invalid_src = [
            v for v in df["annotation_source"].unique()
            if v != "" and v not in VALID_SOURCES
        ]
        required_pass &= gate(
            "G13: All annotation_source values are valid",
            len(invalid_src) == 0,
            f"invalid values: {invalid_src}"
        )
        unlabelled_with_src = (
            (df["gold_intent"] == "") & (df["annotation_source"] != "")
        ).sum()
        required_pass &= gate(
            "G14: No blank gold_intent rows have annotation_source set",
            unlabelled_with_src == 0,
            f"rows with annotation_source but no gold_intent: {unlabelled_with_src}"
        )
    else:
        print("  [SKIP] G12, G13, G14: annotation_source column not present")

    # ---- Distribution summary ----
    if total_labelled > 0:
        print()
        print("  --- Label Distributions ---\n")
        gold_dist = df[labelled_mask]["gold_intent"].value_counts()
        print("  Gold Intent:")
        for intent in sorted(VALID_INTENTS):
            count = int(gold_dist.get(intent, 0))
            print(f"    {intent:<35} {count:>4}")

        print()
        esc_yes_count = (df["escalate"] == "yes").sum()
        esc_no_count = (df["escalate"] == "no").sum()
        print(f"  Escalate=yes : {esc_yes_count}")
        print(f"  Escalate=no  : {esc_no_count}")

        if has_difficulty:
            diff_dist = df[df["difficulty"] != ""]["difficulty"].value_counts()
            print()
            print("  Difficulty:")
            for d in ["easy", "ambiguous", "hard"]:
                print(f"    {d:<12} {int(diff_dist.get(d, 0)):>4}")

    # ---- Final verdict ----
    print()
    print("=" * 60)
    if required_pass:
        print("  ALL REQUIRED GATES PASSED.")
        print("  Golden set is ready for evaluation.")
    else:
        print("  ONE OR MORE REQUIRED GATES FAILED.")
        print("  Fix issues before using for evaluation.")
    print("=" * 60)

    sys.exit(0 if required_pass else 1)


if __name__ == "__main__":
    main()
