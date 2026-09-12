"""
Escalation Policy Module — Delta Customer Support AI Agent.

This module is the single source of truth for the escalation decision policy
described in docs/escalation_policy.md. It is deterministic (no ML, no
randomness) and is used by BOTH:
  - src/proposer.py (annotation-assist proposals for the golden set)
  - src/agent.py (the live production agent's escalation decisions)

Keeping one implementation avoids policy drift between what annotators saw
as a "proposed" escalation call and what the live agent actually does.

Design note (see docs/decision_log.md): escalation here is a POLICY decision,
not a learned classifier, so sharing this logic between the annotation-assist
tool and the production agent is not evaluation leakage — the golden labels
are still the human's independent judgement call, not copied from this
module's output. See docs/golden_set_methodology.md Section 5 for the human
review protocol and disclosure of this design choice.
"""

import re
from typing import Dict, List

# Intents that almost always require human/account escalation, regardless
# of other signals.
ALWAYS_ESCALATE = {
    "flight_delay_rebooking",
    "lost_damaged_baggage",
    "refund_credit_voucher",
}

# Signals in the customer message that indicate account-specific action is
# needed, escalating even for intents that are usually auto-handleable.
ACCOUNT_SIGNALS: List[str] = [
    r"\bconfirmation\s+(?:number|code|#)\b",
    r"\bticket\s+(?:number|#)\b",
    r"\bmy\s+booking\b",
    r"\bmy\s+reservation\b",
    r"\bpnr\b",
    r"\bmy\s+account\b",
    r"\bmy\s+skymiles\s+(?:number|account)\b",
    r"\bmerge\s+(?:my\s+)?accounts?\b",
    r"\bmy\s+miles\s+(?:didn.t|never|not)\b",
    r"\bfile\s+(?:a\s+)?claim\b",
]

# Intent-specific conditional escalation signals.
CONDITIONAL_ESCALATE: Dict[str, List[str]] = {
    "seat_assignment_upgrade": [
        r"\bmy\s+(?:booking|reservation|flight|confirmation)\b",
        r"\bspecific(?:ally)?\s+(?:seat|row)\b",
        r"\bassign\s+(?:me|us)\b",
        r"\badd\s+(?:me\s+to|to)\s+(?:upgrade|waitlist)\b",
        r"\bmy\s+seat\s+was\s+(?:moved|changed|taken)\b",
    ],
    "skymiles_loyalty_program": [
        r"\bmissing\s+miles\b",
        r"\bmiles\s+(?:not|didn.t|never)\s+(?:post|credit|appear|show)\b",
        r"\bmy\s+(?:skymiles|miles|account)\b",
        r"\bskymiles\s+number\b",
        r"\bmerge\s+(?:my\s+)?accounts?\b",
        r"\bstatus\s+challenge\b",
    ],
    "checkin_boarding_pass": [
        r"\bmy\s+(?:booking|reservation|confirmation|tsa|precheck|ktn)\b",
        r"\bknown\s+traveler\b",
        r"\bprecheck\s+(?:missing|not\s+(?:showing|on))\b",
        r"\b(?:can.?t|cannot|unable)\s+(?:access|retrieve|find)\b",
    ],
    "flight_status_inquiry": [
        r"\bstuck\b",
        r"\bneed\s+help\b",
        r"\bstranded\b",
        r"\bmissing\s+connection\b",
    ],
    "inflight_amenities_service": [
        r"\brefund\b",
        r"\bcompensation\b",
        r"\breimburse\b",
        r"\bcharge\s+(?:me|my)\b",
    ],
    "baggage_allowance_policy": [
        r"\bmy\s+(?:flight|booking|reservation|ticket)\b.*\bbag\b",
    ],
}

# Signals from a historical Delta response suggesting escalation was needed.
# Used only as a weak assist signal for the proposer; the live agent never
# has a "historical response" for an incoming message, so this is not part
# of the live agent's decision path (see decide() below).
RESPONSE_ESCALATION_SIGNALS: List[str] = [
    r"\bplease\s+(?:dm|direct\s+message|message)\b",
    r"\bfollow\s+(?:and\s+)?dm\b",
    r"\bprovide\s+(?:your\s+)?(?:confirmation|ticket|record)\b",
    r"\bcall\s+(?:us|our)\b",
    r"\bspeak\s+with\s+(?:a|an|our)\s+(?:agent|supervisor|representative|specialist)\b",
    r"\bcontact\s+(?:our|the)\s+baggage\b",
    r"\bfile\s+(?:a\s+)?claim\b",
]

ESCALATION_REASONS: Dict[str, str] = {
    "flight_delay_rebooking": (
        "Customer needs flight rebooking due to cancellation, delay, or missed "
        "connection; requires access to booking system and live flight availability."
    ),
    "lost_damaged_baggage": (
        "Customer reporting missing, delayed, or damaged baggage; requires baggage "
        "claim lookup and coordination with airport baggage services."
    ),
    "refund_credit_voucher": (
        "Customer requesting a monetary refund, eCredit, or voucher; requires "
        "financial transaction processing with account and ticket verification."
    ),
    "seat_assignment_upgrade": (
        "Customer requesting a specific seat change or upgrade that requires "
        "access to their booking record and seat inventory."
    ),
    "skymiles_loyalty_program": (
        "Customer reporting a SkyMiles discrepancy or requesting account-specific "
        "action that requires SkyMiles account lookup."
    ),
    "checkin_boarding_pass": (
        "Customer unable to complete check-in or access boarding pass; may require "
        "account-level troubleshooting or reservation access."
    ),
    "flight_status_inquiry": (
        "Customer appears stranded or in urgent need of assistance beyond "
        "publicly available flight status information."
    ),
    "inflight_amenities_service": (
        "Customer requesting compensation or reimbursement for in-flight service "
        "failure; requires account access to process credit."
    ),
    "baggage_allowance_policy": (
        "Customer referencing a specific booking in context of baggage policy; "
        "may require reservation lookup to confirm applicable rules."
    ),
    "general_complaint_feedback": (
        "Customer feedback escalated due to account-specific reference detected."
    ),
}

NO_ESCALATION_REASONS: Dict[str, str] = {
    "flight_status_inquiry": (
        "General flight status or schedule inquiry answerable with publicly "
        "available information; no account access required."
    ),
    "baggage_allowance_policy": (
        "General baggage fee or policy question answerable with published "
        "Delta policy; no account or booking access required."
    ),
    "seat_assignment_upgrade": (
        "General seat or upgrade policy question; no account-specific access needed."
    ),
    "skymiles_loyalty_program": (
        "General SkyMiles benefit or Medallion policy question; no account lookup needed."
    ),
    "checkin_boarding_pass": (
        "General check-in troubleshooting steps can be provided without account access."
    ),
    "inflight_amenities_service": (
        "In-flight amenities question or feedback; no account access required."
    ),
    "general_complaint_feedback": (
        "General compliment or feedback; no account or booking access needed."
    ),
}


def should_escalate(intent: str, customer_message: str, delta_response: str = "") -> bool:
    """
    Determine whether a case should be escalated to a human agent.

    Parameters
    ----------
    intent : str
        The classified intent for the customer message.
    customer_message : str
        The customer's message text.
    delta_response : str, optional
        Historical Delta reply text. ONLY used by the annotation-assist
        proposer (which has this available in historical data); the live
        agent calls this with delta_response="" since it has no historical
        reply for a brand-new incoming message.

    Logic (in order):
      1. Always-escalate intents -> yes
      2. Account-specific signals in customer message -> yes
      3. Intent-conditional escalation signals -> yes
      4. Response signals suggesting account access was needed (proposer only) -> yes
      5. Otherwise -> no
    """
    if intent in ALWAYS_ESCALATE:
        return True

    cust_lower = (customer_message or "").lower()
    resp_lower = (delta_response or "").lower()

    for pat in ACCOUNT_SIGNALS:
        if re.search(pat, cust_lower):
            return True

    for pat in CONDITIONAL_ESCALATE.get(intent, []):
        if re.search(pat, cust_lower):
            return True

    if resp_lower:
        for pat in RESPONSE_ESCALATION_SIGNALS:
            if re.search(pat, resp_lower):
                return True

    return False


def build_reason(intent: str, escalate: bool) -> str:
    """Return the escalation reason (or '' if not escalating)."""
    if not escalate:
        return ""
    return ESCALATION_REASONS.get(
        intent,
        "Case requires human agent review based on customer request and escalation policy."
    )


def build_no_escalation_reason(intent: str) -> str:
    """Return the reason auto-handling is safe for this intent (for transparency in output)."""
    return NO_ESCALATION_REASONS.get(
        intent, "No account-specific or high-risk signals detected."
    )


def decide(intent: str, customer_message: str, delta_response: str = "") -> Dict[str, str]:
    """
    Full escalation decision for one message. Returns dict with
    escalate ('yes'|'no') and escalation_reason (str, '' if not escalating).
    """
    escalate = should_escalate(intent, customer_message, delta_response)
    return {
        "escalate": "yes" if escalate else "no",
        "escalation_reason": build_reason(intent, escalate) if escalate
        else build_no_escalation_reason(intent),
    }
