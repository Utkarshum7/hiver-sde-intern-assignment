"""
Rule-based label proposer for the Delta Golden Evaluation Set.

Proposal method identifier: rule_based_v1

IMPORTANT: All outputs from this module are PROPOSED labels only.
They are NOT ground truth. Every proposal must be individually reviewed
and approved (or edited) by a human annotator before being written
to the final golden_annotation.csv.

Design:
  - Intent scoring via weighted pattern matching (3 tiers of specificity).
  - Priority tiebreaking when top scores are equal.
  - Escalation via intent baseline + customer/response signals.
  - Difficulty via normalised score gap and message properties.
  - Deterministic: no randomness, same input → same output.
"""

import re
from typing import Dict, List, Tuple

from src import escalation as _escalation

PROPOSAL_METHOD = "rule_based_v1"

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

# Priority order used when top intent scores are equal.
# Lower index = wins the tie. Represents urgency/specificity hierarchy.
_PRIORITY: Dict[str, int] = {intent: i for i, intent in enumerate([
    "lost_damaged_baggage",
    "refund_credit_voucher",
    "flight_delay_rebooking",
    "checkin_boarding_pass",
    "seat_assignment_upgrade",
    "skymiles_loyalty_program",
    "baggage_allowance_policy",
    "inflight_amenities_service",
    "flight_status_inquiry",
    "general_complaint_feedback",
])}

# ---------------------------------------------------------------------------
# Intent signals: List of (regex_pattern, weight) tuples.
# Weight 3 = highly specific to this intent.
# Weight 2 = moderately specific.
# Weight 1 = weak/general — needs other signals to confirm.
# Scoring is done on customer_message (lowercased).
# ---------------------------------------------------------------------------
_INTENT_SIGNALS: Dict[str, List[Tuple[str, int]]] = {

    "flight_status_inquiry": [
        (r"\bflight\s+status\b", 3),
        (r"\bon\s+time\b", 3),
        (r"\bstill\s+on\s+time\b", 3),
        (r"\btrack(?:ing)?\s+(?:my\s+)?flight\b", 3),
        (r"\barrival\s+time\b", 2),
        (r"\bdeparture\s+time\b", 2),
        (r"\bwhat\s+(?:gate|time)\b", 2),
        (r"\bgates?\s+(?:number|change|assignment)\b", 2),
        (r"\bis\s+(?:my|the)\s+flight\b", 2),
        (r"\blanded\b", 2),
        (r"\bflight\b", 1),
        (r"\bdepart(?:ure|ing|ed)?\b", 1),
        (r"\barrive?(?:al|d|s)?\b", 1),
        (r"\bschedule(?:d)?\b", 1),
    ],

    "flight_delay_rebooking": [
        (r"\bmissed?\s+(?:my\s+)?connection\b", 3),
        (r"\brebook(?:ing|ed)?\b", 3),
        (r"\balternate\s+flight\b", 3),
        (r"\bnext\s+available\s+flight\b", 3),
        (r"\bneed\s+(?:a\s+)?new\s+flight\b", 3),
        (r"\bmiss(?:ed|ing)?\s+(?:my\s+)?flight\b", 3),
        (r"\bcan(?:\'t|not)\s+make\s+(?:my|the)\s+flight\b", 3),
        (r"\bstrandedd?\b", 3),
        (r"\bstandby\b", 3),
        (r"\bcancel(?:led|ation)?\b", 2),
        (r"\bweather\s+delay\b", 2),
        (r"\bstuck\s+(?:at|in|on)\b", 2),
        (r"\bchange\s+(?:my\s+)?flight\b", 2),
        (r"\bnext\s+flight\b", 2),
        (r"\bconnection\b", 2),
        (r"\bcan\s+(?:you|someone)\s+(?:rebook|help\s+me\s+get)\b", 2),
    ],

    "baggage_allowance_policy": [
        (r"\bbaggage\s+fee\b", 3),
        (r"\bbag\s+fee\b", 3),
        (r"\bfee\s+for\s+(?:a\s+)?(?:checked|bag)\b", 3),
        (r"\bhow\s+much\s+(?:is|does|for)\s+(?:a\s+)?(?:checked|bag|baggage)\b", 3),
        (r"\bcarry[- ]on\s+(?:size|limit|rule|allowance|dimension|policy)\b", 3),
        (r"\bweight\s+limit\b", 3),
        (r"\bbag(?:gage)?\s+(?:size|dimension|limit|weight|allowance|policy|rule)\b", 3),
        (r"\bfirst\s+(?:checked\s+)?bag\s+free\b", 3),
        (r"\bmilitary\s+(?:bag|allowance)\b", 3),
        (r"\bgolf\s+(?:clubs?|bag)\b", 2),
        (r"\bski(?:s|ing|equipment)?\s+bag\b", 2),
        (r"\bspecial\s+(?:item|equipment)\s+(?:fee|policy)\b", 2),
        (r"\bhow\s+many\s+bags?\b", 2),
        (r"\bcarry[- ]on\b", 1),
        (r"\bchecked\s+bag\b", 1),
    ],

    "lost_damaged_baggage": [
        (r"\b(?:lost|missing|damaged|broken|destroyed|delayed)\s+(?:my\s+)?(?:bag|luggage|suitcase)\b", 3),
        (r"\b(?:bag|luggage|suitcase).*\b(?:lost|missing|damaged|broken|destroyed|delayed)\b", 3),
        (r"\b(?:bag|luggage)\s+(?:didn.?t|never|not)\s+(?:arrive[d]?|come|show)\b", 3),
        (r"\bcarousel\b", 3),
        (r"\bbaggage\s+(?:claim|office|service)\b", 2),
        (r"\bfile\s+(?:a\s+)?(?:baggage\s+)?claim\b", 3),
        (r"\blost\s+and\s+found\b", 2),
        (r"\bmissing\s+(?:item|content)\b", 2),
    ],

    "seat_assignment_upgrade": [
        (r"\bupgrade\s+(?:to|me|my|list|request|waitlist)\b", 3),
        (r"\bfirst\s+class\s+upgrade\b", 3),
        (r"\bchange\s+(?:my\s+)?seat\b", 3),
        (r"\bselect\s+(?:a\s+|my\s+)?seat\b", 3),
        (r"\bseat\s+(?:assignment|selection|change|request)\b", 3),
        (r"\bsit\s+(?:next\s+to|together|with)\b", 3),
        (r"\baisle\s+seat\b", 2),
        (r"\bwindow\s+seat\b", 2),
        (r"\bmiddle\s+seat\b", 2),
        (r"\bcomfort\s*\+\b", 2),
        (r"\bseating\b", 2),
        (r"\bfamily\s+seating\b", 3),
        (r"\bgroup\s+seat(?:ing)?\b", 2),
        (r"\bseat(?:ed)?\b", 1),
        (r"\bupgrade\b", 1),
        (r"\bmain\s+cabin\s+extra\b", 2),
        (r"\bdelta\s+one\b", 2),
    ],

    "skymiles_loyalty_program": [
        (r"\bsky\s*miles?\b", 3),
        (r"\bmedallion\b", 3),
        (r"\bsky\s*club\b", 3),
        (r"\bbonus\s+miles\b", 3),
        (r"\bpartner\s+miles\b", 3),
        (r"\bredeem\s+miles\b", 3),
        (r"\bmiles\s+(?:not\s+)?(?:posted|credited|earned|appeared)\b", 3),
        (r"\bmissing\s+miles\b", 3),
        (r"\bmedallion\s+status\b", 3),
        (r"\b(?:silver|gold|platinum|diamond)\s+medallion\b", 3),
        (r"\bMQ[MSD]\b", 3),
        (r"\bstatus\s+(?:match|challenge|qualification)\b", 3),
        (r"\bamex.*miles\b", 2),
        (r"\blounge\s+access\b", 2),
        (r"\bsky\s*miles?\s+(?:number|account|member)\b", 3),
        (r"\bmiles\b", 1),
        (r"\bpoints\b", 1),
    ],

    "refund_credit_voucher": [
        (r"\brefund\b", 3),
        (r"\becredit\b", 3),
        (r"\be[- ]credit\b", 3),
        (r"\bvoucher\b", 3),
        (r"\b24[- ]hour\s+(?:rule|cancel|cancellation)\b", 3),
        (r"\bcancel\s+(?:my\s+)?ticket\b", 3),
        (r"\bcredit\s+card\s+refund\b", 3),
        (r"\breimburse(?:ment|d)?\b", 3),
        (r"\bwant\s+(?:my\s+)?money\s+back\b", 3),
        (r"\bget\s+(?:a\s+)?refund\b", 3),
        (r"\bmoney\s+back\b", 3),
        (r"\bcompensation\b", 2),
        (r"\bcredit\s+(?:me|back|to)\b", 2),
        (r"\bowed?\s+(?:a\s+)?refund\b", 3),
        (r"\bprocessed?\s+(?:a\s+)?refund\b", 2),
    ],

    "checkin_boarding_pass": [
        (r"\bcheck[- ]in\b", 3),
        (r"\bchecking\s+in\b", 2),
        (r"\bboarding\s+pass\b", 3),
        (r"\bpassport\s+scan\b", 3),
        (r"\btsa\s+pre[- ]?check\b", 3),
        (r"\bprecheck\b", 3),
        (r"\bknown\s+traveler\s+(?:number|#)\b", 3),
        (r"\bbarcode\b", 3),
        (r"\berror\s+code\b", 2),
        (r"\b(?:can.?t|cannot|unable)\s+(?:check|log)\s+in\b", 3),
        (r"\bonline\s+check[- ]in\b", 3),
        (r"\bmobile\s+(?:boarding|check[- ]in)\b", 3),
        (r"\bfly\s+delta\s+app\b", 2),
        (r"\bapp\s+(?:won.t|won\'t|can.?t|error|not\s+working|down)\b", 2),
        (r"\bdigital\s+(?:boarding|check)\b", 2),
        (r"\bself[- ]serve\s+(?:kiosk|check)\b", 2),
    ],

    "inflight_amenities_service": [
        (r"\bwi[- ]?fi\b", 3),
        (r"\bwifi\b", 3),
        (r"\bgogo\b", 3),
        (r"\bin[- ]flight\s+(?:wifi|internet|entertainment|food|meal|service)\b", 3),
        (r"\binflight\b", 2),
        (r"\bentertainment\s+(?:screen|system|option)\b", 3),
        (r"\bseatback\s+screen\b", 3),
        (r"\bspecial\s+meal\b", 3),
        (r"\bvegan\s+(?:meal|option|food)\b", 2),
        (r"\bkosher\b", 2),
        (r"\bgluten[- ]free\b", 2),
        (r"\bpower\s+outlet\b", 3),
        (r"\busb\s+port\b", 2),
        (r"\bflight\s+attendant\b", 3),
        (r"\bcrew\s+member\b", 2),
        (r"\bonboard\b", 2),
        (r"\bon\s+(?:the\s+)?(?:plane|board|flight)\b", 1),
        (r"\bmeal\b", 1),
        (r"\bfood\b", 1),
        (r"\bdrink\b", 1),
        (r"\bscreen\b", 1),
    ],

    "general_complaint_feedback": [
        (r"\bshout[- ]?out\b", 3),
        (r"\bkudos\b", 3),
        (r"\bworst\s+airline\b", 3),
        (r"\bcompliment\b", 3),
        (r"\bgreat\s+job\b", 2),
        (r"\bthank\s+you\b", 2),
        (r"\bgreat\s+(?:experience|service|flight|crew)\b", 2),
        (r"\bawful\b", 2),
        (r"\bterrible\s+(?:experience|service|airline)\b", 3),
        (r"\bunacceptable\b", 2),
        (r"\bdisappointed\b", 2),
        (r"\blove\s+(?:flying|delta|you)\b", 2),
        (r"\bwonderful\b", 1),
        (r"\bfabulous\b", 1),
        (r"\bamazing\b", 1),
        (r"\bawesome\b", 1),
        (r"\bthanks?\b", 1),
        (r"\bhate\b", 1),
    ],
}

# ---------------------------------------------------------------------------
# Escalation logic
#
# Escalation policy lives in src/escalation.py and is shared with the live
# agent (src/agent.py) so annotation-assist proposals and production
# decisions never drift apart. See src/escalation.py module docstring for
# why sharing this specific piece of logic is not evaluation leakage.
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Core scoring
# ---------------------------------------------------------------------------

def _score_intents(text: str) -> Dict[str, int]:
    """Compute weighted pattern score for each intent against the given text."""
    t = text.lower()
    scores: Dict[str, int] = {}
    for intent, signals in _INTENT_SIGNALS.items():
        total = 0
        for pattern, weight in signals:
            if re.search(pattern, t):
                total += weight
        scores[intent] = total
    return scores


def _resolve_intent(scores: Dict[str, int]) -> Tuple[str, int, int]:
    """
    Pick the winning intent using scores + priority tiebreaking.
    Returns (winner, winner_score, runner_up_score).
    """
    if not scores or max(scores.values()) == 0:
        return "general_complaint_feedback", 0, 0

    max_score = max(scores.values())
    candidates = [i for i, s in scores.items() if s == max_score]

    if len(candidates) == 1:
        winner = candidates[0]
    else:
        # Break ties by priority order (lower index wins)
        winner = min(candidates, key=lambda i: _PRIORITY.get(i, 99))

    # Runner-up: highest score excluding winner
    runner_up_score = max(
        (s for i, s in scores.items() if i != winner),
        default=0
    )
    return winner, max_score, runner_up_score


def _compute_difficulty(
    winner_score: int,
    runner_up_score: int,
    message: str,
) -> str:
    """
    Estimate labelling difficulty based on score separation and message properties.

    easy     : winner_score >= 4 AND gap >= 3  (clear, multiple strong signals)
    hard     : winner_score <= 1 OR gap == 0   (no meaningful evidence or dead tie)
    ambiguous: everything else
    Also hard: message is very short (< 20 chars after stripping handles/punctuation)
    """
    # Strip @handles and URLs to judge actual content length
    clean = re.sub(r"@\w+|https?://\S+|#\w+", "", message).strip()
    content_len = len(clean)

    gap = winner_score - runner_up_score

    if content_len < 20:
        return "hard"
    if winner_score == 0:
        return "hard"
    if winner_score >= 4 and gap >= 3:
        return "easy"
    if winner_score <= 1 or gap == 0:
        return "hard"
    return "ambiguous"


def _should_escalate(
    intent: str,
    customer_message: str,
    delta_response: str,
) -> bool:
    """Delegates to the shared escalation policy (src/escalation.py)."""
    return _escalation.should_escalate(intent, customer_message, delta_response)


def _build_reason(intent: str, escalate: bool, customer_message: str) -> str:
    """Delegates to the shared escalation policy (src/escalation.py)."""
    return _escalation.build_reason(intent, escalate)


def _build_notes(
    intent: str,
    winner_score: int,
    runner_up_score: int,
    scores: Dict[str, int],
    difficulty: str,
    customer_message: str,
) -> str:
    """Generate brief annotator notes explaining the proposal rationale."""
    parts = []
    gap = winner_score - runner_up_score

    if winner_score == 0:
        parts.append("No strong intent signals detected; defaulted to general_complaint_feedback.")
    elif difficulty == "hard":
        clean = re.sub(r"@\w+|https?://\S+|#\w+", "", customer_message).strip()
        if len(clean) < 20:
            parts.append("Very short message with limited signal.")
        elif gap == 0:
            # Find the tied runner-up
            max_s = winner_score
            tied = [i for i, s in scores.items() if s == max_s]
            parts.append(f"Tied signals: {', '.join(tied)}. Resolved by priority order.")
    elif difficulty == "ambiguous":
        # Find second-place intent
        sorted_intents = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        if len(sorted_intents) >= 2 and sorted_intents[1][1] > 0:
            runner_up_intent = sorted_intents[1][0]
            parts.append(
                f"Primary signal: {intent} (score={winner_score}). "
                f"Competing signal: {runner_up_intent} (score={runner_up_score}). "
                "Verify independently."
            )

    if not parts:
        parts.append(f"Intent score={winner_score}, gap={gap}.")

    return " ".join(parts)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def propose_labels(customer_message: str, delta_response: str = "") -> dict:
    """
    Generate proposed labels for a single golden-set example.

    Parameters
    ----------
    customer_message : str
        The customer's tweet text (combined if multi-turn).
    delta_response : str
        The historical Delta reply text (used for escalation signals only).

    Returns
    -------
    dict with keys:
        proposed_gold_intent, proposed_escalate, proposed_escalation_reason,
        proposed_difficulty, proposed_annotator_notes, proposal_method,
        intent_scores (internal, for diagnostics)
    """
    customer_message = str(customer_message or "").strip()
    delta_response = str(delta_response or "").strip()

    scores = _score_intents(customer_message)
    intent, winner_score, runner_up_score = _resolve_intent(scores)
    difficulty = _compute_difficulty(winner_score, runner_up_score, customer_message)
    escalate = _should_escalate(intent, customer_message, delta_response)
    reason = _build_reason(intent, escalate, customer_message)
    notes = _build_notes(
        intent, winner_score, runner_up_score, scores, difficulty, customer_message
    )

    return {
        "proposed_gold_intent": intent,
        "proposed_escalate": "yes" if escalate else "no",
        "proposed_escalation_reason": reason,
        "proposed_difficulty": difficulty,
        "proposed_annotator_notes": notes,
        "proposal_method": PROPOSAL_METHOD,
        "_intent_scores": scores,
        "_winner_score": winner_score,
        "_runner_up_score": runner_up_score,
    }
