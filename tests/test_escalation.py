"""Tests for the shared escalation policy (src/escalation.py)."""

from src import escalation


def test_always_escalate_intents():
    for intent in ["flight_delay_rebooking", "lost_damaged_baggage", "refund_credit_voucher"]:
        assert escalation.should_escalate(intent, "any message here") is True


def test_no_escalate_for_clean_policy_question():
    result = escalation.decide(
        "baggage_allowance_policy", "How much is a second checked bag?"
    )
    assert result["escalate"] == "no"
    assert result["escalation_reason"] != ""  # still explains WHY it's safe


def test_account_signal_forces_escalation():
    result = escalation.decide(
        "skymiles_loyalty_program", "My SkyMiles number is 123, please check my account"
    )
    assert result["escalate"] == "yes"
    assert result["escalation_reason"] != ""


def test_escalation_reason_required_when_escalating():
    result = escalation.decide("flight_delay_rebooking", "My flight was cancelled")
    assert result["escalate"] == "yes"
    assert len(result["escalation_reason"]) > 0


def test_no_escalation_reason_still_present_when_not_escalating():
    result = escalation.decide("flight_status_inquiry", "What is the standard check-in time?")
    assert result["escalate"] == "no"
    # We always return a (non-escalation) reason for transparency, but it
    # must be distinguishable from a "must escalate" reason.
    assert result["escalation_reason"] != ""


def test_delta_response_signal_only_used_when_provided():
    # Live agent never passes delta_response (see src/agent.py); proposer does.
    no_resp = escalation.should_escalate(
        "checkin_boarding_pass", "The app won't let me check in", delta_response=""
    )
    with_resp = escalation.should_escalate(
        "checkin_boarding_pass",
        "The app won't let me check in",
        delta_response="Please follow and DM your confirmation number.",
    )
    assert with_resp is True
    # Without response text, this particular message alone should not trigger.
    assert no_resp is False
