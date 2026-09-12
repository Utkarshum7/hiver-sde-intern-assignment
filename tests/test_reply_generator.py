"""Tests for src/reply_generator.py — grounding and safety contracts."""

from src.reply_generator import (
    generate_reply,
    contains_unsupported_action_claim,
    _clean_evidence_text,
    _strip_unsupported_claims,
)


def test_escalation_reply_never_claims_action_performed():
    result = generate_reply(
        customer_message="My bag never arrived!",
        intent="lost_damaged_baggage",
        evidence=[],
        escalate=True,
        escalation_reason="Requires baggage claim lookup.",
    )
    assert result["grounding_source"] == "escalation_template"
    assert not contains_unsupported_action_claim(result["reply"])
    assert "specialist" in result["reply"].lower() or "human agent" in result["reply"].lower()


def test_auto_handle_reply_uses_evidence_when_available():
    evidence = [{
        "conversation_id": "999",
        "delta_response": "@12345 Checked bag fees are $30 for the first bag and $40 for the second. *ABC https://t.co/xyz",
        "similarity_score": 0.7,
        "timestamp": "t",
        "intent_heuristic": "baggage_allowance_policy",
    }]
    result = generate_reply(
        customer_message="How much for a second bag?",
        intent="baggage_allowance_policy",
        evidence=evidence,
        escalate=False,
    )
    assert result["grounding_source"] == "evidence"
    assert result["used_evidence_conversation_id"] == "999"
    assert "$30" in result["reply"] or "$40" in result["reply"]
    assert "@12345" not in result["reply"]
    assert "*ABC" not in result["reply"]


def test_auto_handle_falls_back_to_policy_fact_when_no_evidence():
    result = generate_reply(
        customer_message="How much for a bag?",
        intent="baggage_allowance_policy",
        evidence=[],
        escalate=False,
    )
    assert result["grounding_source"] == "policy_fact"
    assert result["used_evidence_conversation_id"] is None
    assert len(result["reply"]) > 0


def test_unsupported_action_claims_are_stripped_from_reused_evidence():
    evidence = [{
        "conversation_id": "1",
        "delta_response": "I've credited 500 miles to your account. Thanks for your patience!",
        "similarity_score": 0.5,
        "timestamp": "t",
        "intent_heuristic": "skymiles_loyalty_program",
    }]
    result = generate_reply(
        customer_message="Where are my miles?",
        intent="skymiles_loyalty_program",
        evidence=evidence,
        escalate=False,
    )
    # The claim sentence must never appear; generator should fall back to
    # a policy fact instead of shipping a fabricated action claim.
    assert not contains_unsupported_action_claim(result["reply"])


def test_contains_unsupported_action_claim_detector():
    assert contains_unsupported_action_claim("I've processed your refund already.")
    assert contains_unsupported_action_claim("We have rebooked you on the next flight.")
    assert not contains_unsupported_action_claim(
        "You can check flight status at delta.com/flightstatus."
    )


def test_clean_evidence_text_strips_mentions_and_signatures():
    raw = "@115850 We understand your concern. *TBW https://t.co/6iDGBJAc2m"
    cleaned = _clean_evidence_text(raw)
    assert "@115850" not in cleaned
    assert "*TBW" not in cleaned
    assert "https://" not in cleaned
    assert "understand your concern" in cleaned


def test_strip_unsupported_claims_keeps_safe_sentences():
    text = "I've added the miles to your account. You can view SkyMiles balance at delta.com."
    stripped = _strip_unsupported_claims(text)
    assert "added the miles" not in stripped
    assert "SkyMiles balance" in stripped
