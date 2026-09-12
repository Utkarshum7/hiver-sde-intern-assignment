"""
Phase 5A — Golden Evaluation Set Sampling Script.

Samples exactly 200 conversations from the reserved_golden_pool.parquet using
stratified sampling across the 10-intent taxonomy.

Rules:
  - Uses the SAME heuristic classifier as Phase 3 for stratification guidance ONLY.
  - The heuristic label is stored as 'heuristic_intent' — for reference during annotation.
  - 'gold_intent' and all annotation fields are left BLANK for human labelling.
  - Over-samples rare intents up to their pool ceiling.
  - Deterministic: seed=42.
  - Output: data/golden_eval/golden_annotation.csv

Usage:
    python scripts/sample_golden_set.py
"""

import os
import re
import sys
import pandas as pd
import numpy as np

TARGET_TOTAL = 200
SEED = 42

# Target per-intent allocation (will be adjusted for rare intents).
# Sum = 200. Weighted toward intents underrepresented in heuristic counts.
TARGET_PER_INTENT = {
    "flight_status_inquiry":       32,
    "flight_delay_rebooking":      25,
    "baggage_allowance_policy":    18,
    "lost_damaged_baggage":        10,   # Will be capped at pool size (7)
    "seat_assignment_upgrade":     25,
    "skymiles_loyalty_program":    20,
    "refund_credit_voucher":       18,   # Will be capped at pool size (38)
    "checkin_boarding_pass":       20,
    "inflight_amenities_service":  18,
    "general_complaint_feedback":  14,
}

INTENT_PATTERNS = {
    "flight_status_inquiry": [
        r"\bflight\b", r"\bgate\b", r"\barrive\b", r"\bdepart\b",
        r"\bschedule\b", r"\bstatus\b", r"\btime\b", r"\blanded\b", r"\bdelayed\b"
    ],
    "flight_delay_rebooking": [
        r"\bcancel\b", r"\bcancelled\b", r"\bmissed connection\b", r"\brebook\b",
        r"\bstandby\b", r"\bweather delay\b", r"\bstuck\b", r"\bconnection\b"
    ],
    "baggage_allowance_policy": [
        r"\bbaggage fee\b", r"\bbag fee\b", r"\bcarry[- ]on\b", r"\bchecked bag\b",
        r"\bweight limit\b", r"\bdimension\b", r"\bmilitary bag\b", r"\bgolf\b", r"\bski\b"
    ],
    "lost_damaged_baggage": [
        r"\blost bag\b", r"\bmissing bag\b", r"\bdamaged bag\b", r"\bbaggage claim\b",
        r"\bdelayed bag\b", r"\bbroken suitcase\b"
    ],
    "seat_assignment_upgrade": [
        r"\bseat\b", r"\bseating\b", r"\bupgrade\b", r"\bfirst class\b",
        r"\bcomfort\+\b", r"\baisle\b", r"\bwindow\b", r"\bsit together\b", r"\bmain cabin\b"
    ],
    "skymiles_loyalty_program": [
        r"\bskymiles\b", r"\bmedallion\b", r"\bmiles\b", r"\bredeem miles\b",
        r"\baccount number\b", r"\bpartner miles\b", r"\bskyclub\b", r"\bbonus miles\b"
    ],
    "refund_credit_voucher": [
        r"\brefund\b", r"\becredit\b", r"\bvoucher\b", r"\b24[- ]hour\b",
        r"\bcancel ticket\b", r"\bcredit card refund\b", r"\breimbursed\b"
    ],
    "checkin_boarding_pass": [
        r"\bcheck[- ]in\b", r"\bboarding pass\b", r"\bapp\b", r"\bpassport scan\b",
        r"\btsa\b", r"\bprecheck\b", r"\bbarcode\b", r"\berror code\b"
    ],
    "inflight_amenities_service": [
        r"\bwi[- ]?fi\b", r"\bentertainment\b", r"\bmeal\b", r"\bfood\b", r"\bdrink\b",
        r"\bpower outlet\b", r"\bscreen\b", r"\bflight attendant\b", r"\bonboard\b"
    ],
    "general_complaint_feedback": [
        r"\bshout out\b", r"\bkudos\b", r"\bterrible service\b", r"\bawful\b",
        r"\bworst airline\b", r"\bthank you\b", r"\bcompliment\b", r"\bgreat job\b"
    ],
}

VALID_INTENTS = list(INTENT_PATTERNS.keys())
ESCALATE_OPTIONS = ["yes", "no"]


def classify(text: str) -> str:
    """Keyword-heuristic classifier (Phase 3 taxonomy)."""
    t = text.lower()
    scores = {intent: sum(1 for p in pats if re.search(p, t))
              for intent, pats in INTENT_PATTERNS.items()}
    best, best_score = max(scores.items(), key=lambda x: x[1])
    return "general_complaint_feedback" if best_score == 0 else best


def compute_adjusted_targets(df_by_intent: dict) -> dict:
    """
    Adjusts per-intent targets where the pool is smaller than the target.
    Redistributes surplus budget to intents with ample supply.
    """
    targets = dict(TARGET_PER_INTENT)
    total_budget = TARGET_TOTAL
    surplus = 0

    # First pass: cap intents that don't have enough examples
    capped = {}
    for intent, tgt in targets.items():
        pool_size = len(df_by_intent.get(intent, []))
        if pool_size < tgt:
            capped[intent] = pool_size
            surplus += tgt - pool_size
        else:
            capped[intent] = tgt

    # Redistribute surplus to intents with ample supply, proportionally
    ample_intents = [i for i, t in targets.items()
                     if len(df_by_intent.get(i, [])) >= capped[i]
                     and capped[i] == targets[i]]

    if ample_intents and surplus > 0:
        # distribute surplus round-robin by pool size descending
        ample_sorted = sorted(ample_intents,
                              key=lambda i: len(df_by_intent.get(i, [])),
                              reverse=True)
        for i, intent in enumerate(ample_sorted):
            extra = surplus // (len(ample_sorted) - i)
            surplus -= extra
            capped[intent] += extra

    assert sum(capped.values()) == total_budget, (
        f"Budget mismatch: {sum(capped.values())} != {total_budget}"
    )
    return capped


def main():
    reserved_path = "data/processed/reserved_golden_pool.parquet"
    output_dir = "data/golden_eval"
    output_csv = os.path.join(output_dir, "golden_annotation.csv")

    if not os.path.exists(reserved_path):
        print(f"[ERROR] {reserved_path} not found.")
        sys.exit(1)

    if os.path.exists(output_csv):
        print(f"[ERROR] {output_csv} already exists. Delete it first if you want to re-sample.")
        sys.exit(1)

    print(f"[*] Loading reserved golden pool from: {reserved_path}")
    df = pd.read_parquet(reserved_path)
    df = df[df["retrieval_eligible"]].reset_index(drop=True)
    print(f"[+] Retrieval-eligible reserved conversations: {len(df):,}")

    # Apply heuristic classifier (for stratification guidance ONLY)
    print("[*] Applying heuristic classifier for stratified sampling...")
    df["heuristic_intent"] = df["customer_message"].apply(classify)

    dist = df["heuristic_intent"].value_counts()
    print("[+] Heuristic distribution in reserved pool:")
    for intent in VALID_INTENTS:
        print(f"    {intent:<35} {dist.get(intent, 0):>5}")

    # Build per-intent sub-dataframes
    df_by_intent = {intent: df[df["heuristic_intent"] == intent]
                    for intent in VALID_INTENTS}

    # Compute adjusted targets
    adjusted = compute_adjusted_targets(df_by_intent)
    print("\n[+] Adjusted sampling targets per intent:")
    for intent in VALID_INTENTS:
        pool_n = len(df_by_intent[intent])
        tgt = adjusted[intent]
        flag = " *** CAPPED at pool size ***" if pool_n < TARGET_PER_INTENT[intent] else ""
        print(f"    {intent:<35} target={tgt:>3}  pool={pool_n:>5}{flag}")
    print(f"    {'TOTAL':<35} {sum(adjusted.values()):>3}")

    # Sample deterministically
    np.random.seed(SEED)
    sampled_frames = []
    for intent in VALID_INTENTS:
        sub = df_by_intent[intent]
        n = adjusted[intent]
        if n == 0:
            continue
        sampled = sub.sample(n=n, random_state=SEED, replace=False)
        sampled_frames.append(sampled)

    sampled_df = pd.concat(sampled_frames).reset_index(drop=True)
    assert len(sampled_df) == TARGET_TOTAL, (
        f"Sampling produced {len(sampled_df)} rows, expected {TARGET_TOTAL}"
    )

    # Verify no duplicate conversation IDs
    dups = sampled_df["conversation_id"].duplicated().sum()
    assert dups == 0, f"Duplicate conversation_ids in sample: {dups}"

    # Verify no exact-duplicate customer messages within the sample
    dup_msgs = sampled_df["customer_message"].duplicated().sum()
    if dup_msgs > 0:
        print(f"[WARN] {dup_msgs} exact-duplicate customer_message texts found in sample.")
    else:
        print("[+] No exact-duplicate customer_message texts within the 200 (checked).")
    print("[NOTE] Fuzzy/semantic near-duplicate detection was NOT performed.")

    # Build annotation CSV
    os.makedirs(output_dir, exist_ok=True)
    annotation_df = pd.DataFrame({
        "sample_id": [f"S{i+1:03d}" for i in range(TARGET_TOTAL)],
        "conversation_id": sampled_df["conversation_id"].values,
        "customer_message": sampled_df["customer_message"].values,
        "delta_response": sampled_df["delta_response"].values,
        "timestamp": sampled_df["timestamp"].values,
        "heuristic_intent": sampled_df["heuristic_intent"].values,
        # ---- Human annotation fields (all blank) ----
        "gold_intent": [""] * TARGET_TOTAL,
        "escalate": [""] * TARGET_TOTAL,           # yes | no
        "escalation_reason": [""] * TARGET_TOTAL,  # free text; required if escalate=yes
        "annotator_notes": [""] * TARGET_TOTAL,    # optional free-text
        "difficulty": [""] * TARGET_TOTAL,         # easy | ambiguous | hard
        "annotation_source": [""] * TARGET_TOTAL,  # human_approved_ai_proposal | human_edited_ai_proposal | human_annotated_direct
    })

    # Shuffle so annotator doesn't see all intent clusters in a row
    annotation_df = annotation_df.sample(frac=1.0, random_state=SEED).reset_index(drop=True)
    # Re-assign sample_id after shuffle so IDs are contiguous
    annotation_df["sample_id"] = [f"S{i+1:03d}" for i in range(TARGET_TOTAL)]

    annotation_df.to_csv(output_csv, index=False, encoding="utf-8")
    print(f"\n[+] Saved annotation scaffold to: {output_csv}")
    print(f"[+] Total rows: {len(annotation_df)}")
    print(f"[+] Columns: {annotation_df.columns.tolist()}")
    print("\n[+] gold_intent blank rows (expected 200):", (annotation_df["gold_intent"] == "").sum())
    print("[+] Intent distribution in sample (by heuristic):")
    print(annotation_df["heuristic_intent"].value_counts().to_string())
    print("\n[OK] Annotation scaffold ready. Run: python scripts/annotate_golden.py")


if __name__ == "__main__":
    main()
