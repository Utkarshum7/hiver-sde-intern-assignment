"""
Shared single-tier keyword heuristic classifier for the 10-intent taxonomy.

This is the SAME classification logic originally written independently in
scripts/analyze_delta_intents.py (Phase 3 taxonomy validation) and
scripts/sample_golden_set.py (golden-set stratification). It is centralized
here so:

  1. src/retrieval.py can compute a real `intent_heuristic` value when
     building evidence units (previously this silently defaulted to a
     constant value — see docs/decision_log.md for the bug writeup).
  2. Future callers don't have to keep 2-3 copies of the same pattern table
     in sync by hand.

IMPORTANT: This is a coarse, single-tier keyword heuristic used ONLY for
weak-signal purposes (dataset stratification, retrieval evidence metadata,
sanity displays). It is NOT the proposer used for golden-set annotation
proposals (that is the more sophisticated weighted/tie-broken classifier in
src/proposer.py) and it is NOT ground truth. See docs/golden_set_methodology.md.
"""

import re
from typing import Dict, List

INTENT_PATTERNS: Dict[str, List[str]] = {
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


def classify(text: str) -> str:
    """Single-tier keyword-heuristic classifier (Phase 3 taxonomy)."""
    t = str(text or "").lower()
    scores = {intent: sum(1 for p in pats if re.search(p, t))
              for intent, pats in INTENT_PATTERNS.items()}
    best, best_score = max(scores.items(), key=lambda x: x[1])
    return "general_complaint_feedback" if best_score == 0 else best
