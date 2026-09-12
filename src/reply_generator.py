"""
Reply Generation Module — Delta Customer Support AI Agent.

Produces a customer-facing reply GROUNDED in retrieved historical Delta
support evidence (src/retrieval.py). No LLM call is required — this module
is a deterministic, fully local/offline fallback so the pipeline works with
zero API keys. See src/llm_client.py + src/llm_judge.py for the
(currently unconfigured) optional LLM path.

Hard safety rules enforced here:
  1. NEVER claim an account-specific action was performed for the current
     customer (no live flight lookup, no seat change, no refund, no miles
     credit, no baggage trace). Historical evidence text sometimes contains
     first-person completed-action language ("I've added 500 miles...")
     because it was written for a DIFFERENT customer in the past — copying
     that verbatim into a new reply would be a fabricated claim. This module
     strips such sentences before reuse (see _strip_unsupported_claims).
  2. When escalating, the reply must say so plainly and give the reason,
     never pretend the underlying action (rebooking, refund, claim filing)
     has already happened.
  3. Every non-escalated reply must be traceable to something in `evidence`
     or to a static policy fact already documented in docs/intent_taxonomy.md
     (POLICY_FACTS below) — nothing is invented at generation time.
"""

import re
from typing import Dict, List

# Static policy facts already documented in docs/intent_taxonomy.md
# (Phase 3 research), reused here as fallback grounding when retrieved
# evidence for an intent is thin. These are NOT invented during evaluation;
# they mirror what was already written into the project's own docs.
POLICY_FACTS: Dict[str, str] = {
    "flight_status_inquiry": (
        "You can check real-time flight status at delta.com/flightstatus or "
        "in the Fly Delta app."
    ),
    "baggage_allowance_policy": (
        "Standard checked-bag fees are $30 for the first bag and $40 for the "
        "second bag on most domestic routes; carry-on and checked bag size/"
        "weight limits are listed at delta.com/baggage."
    ),
    "seat_assignment_upgrade": (
        "Seat selection and upgrade options (including Comfort+ and "
        "Complimentary Upgrades for eligible Medallion Members) can be "
        "viewed and changed under 'My Trips' at delta.com or in the app."
    ),
    "skymiles_loyalty_program": (
        "SkyMiles balance, Medallion status, and mile-earning details are "
        "available at delta.com/skymiles or in the Fly Delta app."
    ),
    "checkin_boarding_pass": (
        "Online check-in opens 24 hours before departure at delta.com or in "
        "the Fly Delta app, where you can also access your mobile boarding pass."
    ),
    "inflight_amenities_service": (
        "In-flight Wi-Fi, entertainment, and meal/beverage options vary by "
        "aircraft and route and are listed at delta.com/inflight-entertainment."
    ),
    "general_complaint_feedback": (
        "We appreciate you taking the time to share this feedback with us."
    ),
}

# Sentence-level patterns that claim a specific account action was already
# taken. These are stripped from reused historical evidence text because the
# action was (if genuine) performed for a DIFFERENT customer in the past,
# never for the current one.
_FILLER = r"(?:\w+\s+){0,2}"  # tolerates up to 2 filler words, e.g. "already", "just now"
_ACTION_VERBS = r"(?:added|credited|applied|processed|refunded|rebooked|upgraded|changed|updated|fixed|removed|cancelled|reset)"

_UNSUPPORTED_ACTION_PATTERNS = [
    rf"\bi(?:'ve| have)\s+{_FILLER}{_ACTION_VERBS}\b",
    rf"\bwe(?:'ve| have)\s+{_FILLER}{_ACTION_VERBS}\b",
    r"\byour\s+(?:refund|miles|upgrade|seat|booking|reservation|claim)\s+(?:has|have)\s+been\b",
    r"\bi(?:'ve| have)\s+(?:checked|verified|confirmed|looked into)\s+(?:your|the)\b",
    r"\byour\s+account\s+(?:has been|is now)\b",
]

_AGENT_SIGNATURE_RE = re.compile(r"\*[A-Z]{2,4}\b")           # e.g. "*HJH"
_MENTION_RE = re.compile(r"@\d+|@\w+")                          # @115850 or @Delta
_URL_RE = re.compile(r"https?://\S+")
_WHITESPACE_RE = re.compile(r"\s+")


def _clean_evidence_text(text: str) -> str:
    """Strip @mentions, agent-initial signatures, and tracking shortlinks
    from a historical Delta response, leaving the substantive content."""
    text = _AGENT_SIGNATURE_RE.sub("", text or "")
    text = _MENTION_RE.sub("", text)
    text = _URL_RE.sub("", text)
    text = _WHITESPACE_RE.sub(" ", text).strip()
    return text


def _strip_unsupported_claims(text: str) -> str:
    """Remove sentences that claim a specific account action was already
    performed (see module docstring, safety rule 1)."""
    sentences = re.split(r"(?<=[.!?])\s+", text)
    kept = []
    for s in sentences:
        s_lower = s.lower()
        if any(re.search(pat, s_lower) for pat in _UNSUPPORTED_ACTION_PATTERNS):
            continue
        kept.append(s)
    return " ".join(kept).strip()


def contains_unsupported_action_claim(text: str) -> bool:
    """
    Automated safety check: does `text` claim an account-specific action was
    performed? Used both as an internal self-check in generate_reply() and
    as an evaluation-time automated metric (see src/metrics.py).
    """
    t = (text or "").lower()
    return any(re.search(pat, t) for pat in _UNSUPPORTED_ACTION_PATTERNS)


def _best_evidence_snippet(evidence: List[Dict], intent: str) -> str:
    """Pick the most useful cleaned, claim-stripped snippet from evidence."""
    for item in evidence:
        raw = item.get("delta_response", "")
        cleaned = _strip_unsupported_claims(_clean_evidence_text(raw))
        if len(cleaned) >= 20:
            return cleaned
    return ""


def generate_reply(
    customer_message: str,
    intent: str,
    evidence: List[Dict],
    escalate: bool,
    escalation_reason: str = "",
) -> Dict[str, str]:
    """
    Generate a grounded, escalation-safe customer reply.

    Returns a dict:
      {
        "reply": str,
        "grounding_source": "evidence" | "policy_fact" | "escalation_template",
        "used_evidence_conversation_id": str | None,
      }
    """
    intent = intent if intent in POLICY_FACTS else "general_complaint_feedback"

    if escalate:
        reason_clause = f" {escalation_reason}" if escalation_reason else ""
        reply = (
            "Thanks for reaching out, and sorry for the trouble. This one needs "
            "a specialist who can pull up your booking/account details, so I'm "
            f"flagging it for our support team to take over.{reason_clause} "
            "We are not able to make account or booking changes in this "
            "channel, but a human agent will pick this up from here."
        )
        return {
            "reply": reply.strip(),
            "grounding_source": "escalation_template",
            "used_evidence_conversation_id": None,
        }

    snippet = _best_evidence_snippet(evidence, intent)
    used_conv_id = None
    if snippet:
        for item in evidence:
            if _strip_unsupported_claims(_clean_evidence_text(item.get("delta_response", ""))) == snippet:
                used_conv_id = item.get("conversation_id")
                break
        grounding_source = "evidence"
        body = snippet
    else:
        grounding_source = "policy_fact"
        body = POLICY_FACTS.get(intent, POLICY_FACTS["general_complaint_feedback"])

    reply = f"Thanks for reaching out to Delta! {body}".strip()

    # Final self-check: never ship a reply that slipped through with an
    # unsupported action claim (defense in depth).
    if contains_unsupported_action_claim(reply):
        reply = f"Thanks for reaching out to Delta! {POLICY_FACTS.get(intent, POLICY_FACTS['general_complaint_feedback'])}"
        grounding_source = "policy_fact"
        used_conv_id = None

    return {
        "reply": reply,
        "grounding_source": grounding_source,
        "used_evidence_conversation_id": used_conv_id,
    }
