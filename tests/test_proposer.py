"""
Unit tests for src/proposer.py — rule-based label proposer.

Tests verify:
  - Intent proposals for clear, unambiguous cases (should match expected intent)
  - Escalation logic (always-escalate intents, conditional signals)
  - Difficulty scoring (easy/ambiguous/hard thresholds)
  - Edge cases (empty message, very short, all-whitespace)
  - Proposer output schema (required keys present)
"""

import sys
import os
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.proposer import propose_labels, PROPOSAL_METHOD, VALID_INTENTS


# ---------------------------------------------------------------------------
# Schema tests
# ---------------------------------------------------------------------------

class TestOutputSchema:
    def test_required_keys_present(self):
        result = propose_labels("my flight was delayed")
        required = {
            "proposed_gold_intent",
            "proposed_escalate",
            "proposed_escalation_reason",
            "proposed_difficulty",
            "proposed_annotator_notes",
            "proposal_method",
        }
        assert required.issubset(result.keys()), (
            f"Missing keys: {required - result.keys()}"
        )

    def test_intent_is_valid(self):
        for msg in [
            "is my flight on time",
            "my bag is missing",
            "please rebook my flight",
            "how much for a checked bag",
            "skymiles points missing",
        ]:
            r = propose_labels(msg)
            assert r["proposed_gold_intent"] in VALID_INTENTS, (
                f"Invalid intent '{r['proposed_gold_intent']}' for: {msg}"
            )

    def test_escalate_is_yes_or_no(self):
        for msg in ["my flight was cancelled", "how do I check in"]:
            r = propose_labels(msg)
            assert r["proposed_escalate"] in ("yes", "no")

    def test_difficulty_is_valid(self):
        for msg in ["my bag is missing", "love flying delta", "need help"]:
            r = propose_labels(msg)
            assert r["proposed_difficulty"] in ("easy", "ambiguous", "hard")

    def test_proposal_method_correct(self):
        r = propose_labels("help")
        assert r["proposal_method"] == PROPOSAL_METHOD == "rule_based_v1"

    def test_always_returns_dict(self):
        for msg in ["", " ", "@Delta", "   \n   "]:
            r = propose_labels(msg)
            assert isinstance(r, dict)
            assert r["proposed_gold_intent"] in VALID_INTENTS


# ---------------------------------------------------------------------------
# Intent classification tests
# ---------------------------------------------------------------------------

class TestIntentClassification:
    """Test that clear messages get the correct proposed intent."""

    def _intent(self, msg: str, delta_resp: str = "") -> str:
        return propose_labels(msg, delta_resp)["proposed_gold_intent"]

    # flight_status_inquiry
    def test_flight_status_inquiry_on_time(self):
        assert self._intent("Is my flight still on time?") == "flight_status_inquiry"

    def test_flight_status_inquiry_gate(self):
        assert self._intent("What gate is my flight departing from?") == "flight_status_inquiry"

    def test_flight_status_inquiry_tracking(self):
        assert self._intent("Can you track my flight arrival time?") == "flight_status_inquiry"

    # flight_delay_rebooking
    def test_rebooking_missed_connection(self):
        assert self._intent(
            "I missed my connection in Atlanta, I need to be rebooked on next flight"
        ) == "flight_delay_rebooking"

    def test_rebooking_cancelled_flight(self):
        assert self._intent(
            "My flight was cancelled and I need a new flight tonight please"
        ) == "flight_delay_rebooking"

    def test_rebooking_stranded(self):
        assert self._intent("I am stranded at JFK, need standby on next flight") == "flight_delay_rebooking"

    # baggage_allowance_policy
    def test_baggage_policy_fee(self):
        assert self._intent("How much is the baggage fee for a checked bag?") == "baggage_allowance_policy"

    def test_baggage_policy_carry_on(self):
        assert self._intent("What are the carry-on size limits for Delta?") == "baggage_allowance_policy"

    def test_baggage_policy_weight_limit(self):
        assert self._intent("Is there a weight limit for checked luggage?") == "baggage_allowance_policy"

    # lost_damaged_baggage
    def test_lost_bag(self):
        assert self._intent("My bag never arrived at baggage claim, it's lost!") == "lost_damaged_baggage"

    def test_damaged_bag(self):
        assert self._intent("My suitcase was damaged and the wheel is broken") == "lost_damaged_baggage"

    def test_missing_bag(self):
        assert self._intent("My luggage is missing after the flight") == "lost_damaged_baggage"

    # seat_assignment_upgrade
    def test_seat_selection(self):
        assert self._intent("Can I select an aisle seat for my flight?") == "seat_assignment_upgrade"

    def test_upgrade_request(self):
        assert self._intent("I would like to upgrade to first class if possible") == "seat_assignment_upgrade"

    def test_sit_together(self):
        assert self._intent("Can my family sit together on this flight?") == "seat_assignment_upgrade"

    # skymiles_loyalty_program
    def test_skymiles_balance(self):
        assert self._intent("How many SkyMiles do I have in my account?") == "skymiles_loyalty_program"

    def test_missing_miles(self):
        assert self._intent("My miles from last week's flight never posted to my account") == "skymiles_loyalty_program"

    def test_medallion_status(self):
        assert self._intent("How do I qualify for Gold Medallion status?") == "skymiles_loyalty_program"

    # refund_credit_voucher
    def test_refund_request(self):
        assert self._intent("I want a refund for my cancelled flight please") == "refund_credit_voucher"

    def test_ecredit(self):
        assert self._intent("Can I get an eCredit for my unused ticket?") == "refund_credit_voucher"

    def test_money_back(self):
        assert self._intent("I need my money back for the ticket you cancelled") == "refund_credit_voucher"

    # checkin_boarding_pass
    def test_checkin_online(self):
        assert self._intent("I cannot check in online, the website gives an error") == "checkin_boarding_pass"

    def test_boarding_pass(self):
        assert self._intent("My boarding pass is not showing up in the Fly Delta app") == "checkin_boarding_pass"

    def test_tsa_precheck(self):
        assert self._intent("TSA PreCheck is not on my boarding pass") == "checkin_boarding_pass"

    # inflight_amenities_service
    def test_wifi(self):
        assert self._intent("The WiFi on my flight is not working") == "inflight_amenities_service"

    def test_entertainment(self):
        assert self._intent("The entertainment screen at my seat is broken") == "inflight_amenities_service"

    def test_special_meal(self):
        assert self._intent("Did I request a vegan meal option for my flight?") == "inflight_amenities_service"

    # general_complaint_feedback
    def test_compliment(self):
        assert self._intent("Shout out to the crew at ATL for being amazing!") == "general_complaint_feedback"

    def test_general_complaint(self):
        assert self._intent("Worst airline experience I have ever had, terrible service") == "general_complaint_feedback"

    def test_thank_you(self):
        assert self._intent("Thank you Delta, great experience as always!") == "general_complaint_feedback"


# ---------------------------------------------------------------------------
# Priority / tiebreaking tests
# ---------------------------------------------------------------------------

class TestPriorityTiebreaking:
    def test_refund_wins_over_rebooking(self):
        """When customer asks for refund on cancelled flight, refund is primary intent."""
        result = propose_labels(
            "My flight was cancelled and I want a full refund please"
        )
        assert result["proposed_gold_intent"] == "refund_credit_voucher", (
            f"Expected refund_credit_voucher, got {result['proposed_gold_intent']}"
        )

    def test_lost_baggage_wins_over_status(self):
        """Lost baggage should win over flight status when both signal."""
        result = propose_labels(
            "My flight landed but my bag never arrived at baggage claim"
        )
        assert result["proposed_gold_intent"] == "lost_damaged_baggage"

    def test_rebooking_wins_over_status(self):
        """Rebooking intent wins over status inquiry when rebooking is needed."""
        result = propose_labels(
            "Flight delayed 6 hours, I missed my connection, need next available flight"
        )
        assert result["proposed_gold_intent"] == "flight_delay_rebooking"


# ---------------------------------------------------------------------------
# Escalation tests
# ---------------------------------------------------------------------------

class TestEscalation:
    def _esc(self, msg: str, delta: str = "") -> str:
        return propose_labels(msg, delta)["proposed_escalate"]

    # Always-escalate intents
    def test_rebooking_always_escalates(self):
        assert self._esc("I missed my connection and need to be rebooked") == "yes"

    def test_lost_baggage_always_escalates(self):
        assert self._esc("My bag is lost after my flight") == "yes"

    def test_refund_always_escalates(self):
        assert self._esc("I want a refund for my cancelled flight") == "yes"

    # Policy questions — no escalation
    def test_baggage_policy_no_escalate(self):
        assert self._esc("How much does a checked bag cost?") == "no"

    def test_feedback_no_escalate(self):
        assert self._esc("Great experience on my Delta flight today!") == "no"

    def test_general_skymiles_no_escalate(self):
        assert self._esc("How do I earn SkyMiles on partner airlines?") == "no"

    # Conditional escalation via account signals
    def test_account_signal_triggers_escalation(self):
        assert self._esc("Please check my booking, confirmation number is ABC123") == "yes"

    def test_my_skymiles_account_triggers_escalation(self):
        assert self._esc(
            "My SkyMiles miles didn't post to my account after last flight"
        ) == "yes"

    # Delta response signals
    def test_dm_response_triggers_escalation(self):
        assert self._esc(
            "How do I change my seat?",
            "Please DM us your booking details so we can look into this."
        ) == "yes"


# ---------------------------------------------------------------------------
# Difficulty tests
# ---------------------------------------------------------------------------

class TestDifficulty:
    def _diff(self, msg: str) -> str:
        return propose_labels(msg)["proposed_difficulty"]

    def test_clear_message_is_easy(self):
        # Multiple strong signals, single clear intent
        d = self._diff(
            "My checked bag was lost at baggage claim, the wheel is broken and missing"
        )
        assert d == "easy"

    def test_very_short_message_is_hard(self):
        d = self._diff("help")
        assert d == "hard"

    def test_empty_message_is_hard(self):
        d = self._diff("")
        assert d == "hard"

    def test_ambiguous_multi_intent_detected(self):
        # Mentions both refund and rebooking — should not be easy
        d = self._diff(
            "My flight was cancelled. Can I get a refund or be rebooked?"
        )
        assert d in ("ambiguous", "hard")


# ---------------------------------------------------------------------------
# Determinism test
# ---------------------------------------------------------------------------

class TestDeterminism:
    def test_same_input_same_output(self):
        msg = "My SkyMiles from last month's trip to LAX never posted"
        r1 = propose_labels(msg)
        r2 = propose_labels(msg)
        assert r1["proposed_gold_intent"]  == r2["proposed_gold_intent"]
        assert r1["proposed_escalate"]     == r2["proposed_escalate"]
        assert r1["proposed_difficulty"]   == r2["proposed_difficulty"]

    def test_whitespace_variations_consistent(self):
        """Minor whitespace differences should not change the intent."""
        msg1 = "How much is the baggage fee?"
        msg2 = "  How much is the baggage fee?  "
        r1 = propose_labels(msg1)["proposed_gold_intent"]
        r2 = propose_labels(msg2)["proposed_gold_intent"]
        assert r1 == r2
