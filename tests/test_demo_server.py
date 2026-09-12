"""
Tests for scripts/demo_server.py — the local demo wrapper around the
existing agent pipeline.

These tests never load the real DeltaSupportAgent (which takes ~17s) —
every test mocks the model load, so this file stays fast and does not
depend on data/processed/*.pkl existing on disk. Nothing here touches
src/agent.py, evaluation logic, the golden set, or reports/.
"""

import importlib

import pydantic
import pytest

from scripts import demo_server


class _FakeAgent:
    """Minimal stand-in for DeltaSupportAgent, matching its .handle() contract."""

    def __init__(self):
        self.handle_calls = []

    def handle(self, message: str) -> dict:
        self.handle_calls.append(message)
        return {
            "intent": "baggage_allowance_policy",
            "reply": "Thanks for reaching out to Delta! Checked bag fees are $30/$40.",
            "escalate": "no",
            "escalation_reason": "General baggage fee policy question; no account access required.",
            "evidence": [
                {
                    "conversation_id": "12345",
                    "customer_message": "How much for a bag?",
                    "delta_response": "Checked bag fees are $30/$40.",
                    "similarity_score": 0.62,
                    "timestamp": "2017-01-01",
                    "intent_heuristic": "baggage_allowance_policy",
                }
            ],
            # extra diagnostic fields the demo must NOT leak through untouched
            "confidence": 0.81,
            "grounding_source": "evidence",
            "used_evidence_conversation_id": "12345",
        }


@pytest.fixture(autouse=True)
def reset_singleton():
    """Ensure each test starts with a clean module-level agent singleton."""
    demo_server._agent = None
    yield
    demo_server._agent = None


# ---------------------------------------------------------------------------
# 1. Server import / startup behavior (mocking the model load)
# ---------------------------------------------------------------------------

def test_import_does_not_eagerly_load_the_model():
    # Re-import fresh to be sure module-level code never calls .load().
    module = importlib.reload(demo_server)
    assert module._agent is None


def test_get_agent_loads_exactly_once(monkeypatch):
    fake = _FakeAgent()
    load_calls = {"n": 0}

    def fake_load():
        load_calls["n"] += 1
        return fake

    monkeypatch.setattr(demo_server.DeltaSupportAgent, "load", staticmethod(fake_load))

    a1 = demo_server.get_agent()
    a2 = demo_server.get_agent()
    a3 = demo_server.get_agent()

    assert a1 is a2 is a3 is fake
    assert load_calls["n"] == 1  # loaded once, cached after that


def test_startup_event_populates_singleton(monkeypatch):
    fake = _FakeAgent()
    monkeypatch.setattr(demo_server.DeltaSupportAgent, "load", staticmethod(lambda: fake))

    assert demo_server._agent is None
    demo_server._load_agent_on_startup()
    assert demo_server._agent is fake


# ---------------------------------------------------------------------------
# 2. Valid request
# ---------------------------------------------------------------------------

def test_handle_valid_request_returns_agent_result(monkeypatch):
    fake = _FakeAgent()
    monkeypatch.setattr(demo_server.DeltaSupportAgent, "load", staticmethod(lambda: fake))

    req = demo_server.HandleRequest(message="How much for a second checked bag?")
    resp = demo_server.handle(req)

    assert resp.intent == "baggage_allowance_policy"
    assert resp.escalate == "no"
    assert "checked bag" in resp.reply.lower() or "$30" in resp.reply
    assert len(resp.evidence) == 1
    assert fake.handle_calls == ["How much for a second checked bag?"]


def test_handle_passes_message_through_unmodified(monkeypatch):
    fake = _FakeAgent()
    monkeypatch.setattr(demo_server.DeltaSupportAgent, "load", staticmethod(lambda: fake))

    req = demo_server.HandleRequest(message="  My flight was cancelled!  ")
    demo_server.handle(req)

    # pydantic does not strip whitespace by default; the agent should see
    # exactly what the user typed (agent.py handles its own normalization).
    assert fake.handle_calls == ["  My flight was cancelled!  "]


# ---------------------------------------------------------------------------
# 3. Empty message rejection
# ---------------------------------------------------------------------------

def test_empty_message_is_rejected():
    with pytest.raises(pydantic.ValidationError):
        demo_server.HandleRequest(message="")


def test_whitespace_only_message_is_rejected():
    with pytest.raises(pydantic.ValidationError):
        demo_server.HandleRequest(message="   ")


# ---------------------------------------------------------------------------
# 4. Response schema
# ---------------------------------------------------------------------------

def test_response_schema_has_exactly_the_required_fields(monkeypatch):
    fake = _FakeAgent()
    monkeypatch.setattr(demo_server.DeltaSupportAgent, "load", staticmethod(lambda: fake))

    req = demo_server.HandleRequest(message="test")
    resp = demo_server.handle(req)

    assert set(resp.model_dump().keys()) == {
        "intent", "reply", "escalate", "escalation_reason", "evidence",
    }


def test_response_schema_rejects_missing_required_field():
    with pytest.raises(pydantic.ValidationError):
        demo_server.HandleResponse(
            intent="baggage_allowance_policy",
            reply="...",
            escalate="no",
            # escalation_reason missing
            evidence=[],
        )


# ---------------------------------------------------------------------------
# Index page
# ---------------------------------------------------------------------------

def test_index_page_contains_expected_elements():
    html = demo_server.index()
    assert "<textarea" in html
    assert "Send" in html
    assert "loading" in html.lower()
    assert "/handle" in html


def test_index_page_header_and_offline_labeling():
    html = demo_server.index()
    assert "Delta Support Intelligence" in html
    assert "Offline demo" in html
    assert "AI-powered customer-support responses from historical @Delta support data" in html
    # Must clearly disclose this is not a live system, in more than one place.
    assert "not sent to a live Delta system" in html
    assert (
        "This demo does not access live Delta systems and cannot perform "
        "flight-status lookups, bookings, refunds, account actions, or "
        "baggage-system operations."
    ) in html


def test_index_page_contains_no_assignment_wording():
    html = demo_server.index()
    lowered = html.lower()
    for banned in ("hiver", "sde", "internship", "take-home", "take home"):
        assert banned not in lowered
    # "intern" specifically, but not as a substring of "international" etc.
    import re
    assert not re.search(r"\bintern\b", lowered)


def test_index_page_has_composer_and_result_elements():
    html = demo_server.index()
    # Composer
    assert 'id="message"' in html
    assert "char-count" in html
    assert "Clear" in html
    assert "Ctrl+Enter" in html
    # Three required example categories
    assert "Baggage fee question" in html
    assert "Cancelled flight" in html
    assert "Missing baggage" in html
    # Result panel structure
    assert 'id="empty-state"' in html
    assert 'id="r-intent"' in html
    assert 'id="r-decision-banner"' in html
    assert 'id="r-reason-wrap"' in html
    assert 'id="r-reply"' in html
    assert 'id="r-evidence"' in html
    assert 'id="alert-box"' in html
    assert 'role="alert"' in html


def test_index_page_has_accessibility_hooks():
    html = demo_server.index()
    assert 'lang="en"' in html
    assert "aria-live" in html
    assert "aria-busy" in html
    assert 'for="message"' in html  # label associated with the textarea
    assert "aria-labelledby" in html


def test_index_page_escalation_styling_hooks_present():
    html = demo_server.index()
    # Both decision states must have distinct, named styling hooks.
    assert "decision-banner.safe" in html
    assert "decision-banner.escalate" in html
    assert "Auto-handled" in html
    assert "Escalated to a human agent" in html


def test_index_page_has_sidebar_and_functional_nav_items():
    html = demo_server.index()
    assert 'class="sidebar"' in html
    assert "New Conversation" in html
    assert "Example Queries" in html
    assert "How It Works" in html
    assert "About the Project" in html
    # Each nav item must call a real local function, not link to a nonexistent page.
    assert "onclick=\"clearMessage()\"" in html
    assert "onclick=\"focusExamples()\"" in html
    assert "onclick=\"openPanel('how-it-works')\"" in html
    assert "onclick=\"openPanel('about')\"" in html
    assert "<a href=" not in html  # no fake navigation links anywhere on the page


def test_index_page_has_original_inline_svg_logo_and_illustration():
    html = demo_server.index()
    assert "<svg" in html
    assert "brand-mark" in html
    # No external image/network assets of any kind.
    assert "<img" not in html
    assert "http://" not in html.replace("http://127.0.0.1", "")
    assert "https://" not in html
    assert "fonts.googleapis" not in html
    assert "cdn." not in html.lower()


def test_index_page_info_panels_are_factual_and_honest():
    html = demo_server.index()
    assert "PANEL_CONTENT" in html
    assert "Kaggle" in html and "Customer Support on Twitter dataset" in html
    assert "cannot perform flight-status lookups" in html
    assert "modal-overlay" in html


# ---------------------------------------------------------------------------
# DEMO-ONLY safety override
#
# This is the fix for a real reported bug: "My suitcase came out with a
# broken wheel, how do I file a claim?" was classified flight_status_inquiry
# and auto-handled with an unrelated reply. These tests exercise
# demo_server.py's own override logic in isolation — they never touch
# src/agent.py, src/escalation.py, src/classifier.py, or
# scripts/evaluate.py, so the measured golden-set evaluation numbers in
# reports/evaluation_results.json are provably unaffected by this feature.
# ---------------------------------------------------------------------------

class _FakeNonEscalatingAgent:
    """Always returns escalate='no', regardless of the message — isolates
    the override's own text-matching logic from whatever the real
    classifier would have decided."""

    def handle(self, message: str) -> dict:
        return {
            "intent": "flight_status_inquiry",  # the real misclassification observed
            "reply": "Thanks for reaching out to Delta! Here is some unrelated flight status info.",
            "escalate": "no",
            "escalation_reason": "General flight status or schedule inquiry answerable with publicly available information; no account access required.",
            "evidence": [],
        }


class _FakeAlreadyEscalatingAgent:
    """Always returns escalate='yes' for a real (non-override) reason —
    used to confirm the override never overwrites a genuine escalation."""

    def handle(self, message: str) -> dict:
        return {
            "intent": "lost_damaged_baggage",
            "reply": "Thanks for reaching out, and sorry for the trouble. This one needs a specialist...",
            "escalate": "yes",
            "escalation_reason": "Customer reporting missing, delayed, or damaged baggage; requires baggage claim lookup.",
            "evidence": [],
        }


@pytest.mark.parametrize("message", [
    "My suitcase came out with a broken wheel, how do I file a claim?",
    "I lost my bag on the flight and want to report it, please help",
    "Please check my booking reference and reservation number for my refund request",
])
def test_safety_override_forces_escalation(monkeypatch, message):
    fake = _FakeNonEscalatingAgent()
    monkeypatch.setattr(demo_server.DeltaSupportAgent, "load", staticmethod(lambda: fake))

    resp = demo_server.handle(demo_server.HandleRequest(message=message))

    assert resp.escalate == "yes"
    assert resp.escalation_reason.startswith(demo_server.SAFETY_OVERRIDE_REASON_PREFIX)
    # Must NOT ship the original agent's unrelated/confident policy answer.
    assert "unrelated flight status info" not in resp.reply
    # Must be a genuine escalation notice, not a fabricated action claim.
    from src.reply_generator import contains_unsupported_action_claim
    assert not contains_unsupported_action_claim(resp.reply)


def test_ordinary_baggage_fee_question_still_auto_handled(monkeypatch):
    fake = _FakeNonEscalatingAgent()
    monkeypatch.setattr(demo_server.DeltaSupportAgent, "load", staticmethod(lambda: fake))

    resp = demo_server.handle(demo_server.HandleRequest(
        message="How much does a second checked bag cost on an international flight?"
    ))

    assert resp.escalate == "no"
    assert resp.reply == "Thanks for reaching out to Delta! Here is some unrelated flight status info."
    assert not resp.escalation_reason.startswith(demo_server.SAFETY_OVERRIDE_REASON_PREFIX)


def test_safety_override_never_overwrites_a_real_escalation(monkeypatch):
    fake = _FakeAlreadyEscalatingAgent()
    monkeypatch.setattr(demo_server.DeltaSupportAgent, "load", staticmethod(lambda: fake))

    resp = demo_server.handle(demo_server.HandleRequest(
        message="My bag was lost, please file a claim"
    ))

    assert resp.escalate == "yes"
    # The agent's own real reason is preserved, not replaced by the override.
    assert "baggage claim lookup" in resp.escalation_reason
    assert not resp.escalation_reason.startswith(demo_server.SAFETY_OVERRIDE_REASON_PREFIX)


@pytest.mark.parametrize("message,expected_terms", [
    ("My suitcase came out with a broken wheel, how do I file a claim?", {"broken wheel", "file a claim"}),
    ("I lost my bag on the flight, can you help me find it?", {"lost baggage"}),
    ("My luggage is missing, please help", {"missing luggage"}),
    ("My bag was damaged during the flight and I need reimbursement", {"damaged suitcase", "reimbursement"}),
    ("Please check my booking reference and reservation number for my refund request",
     {"booking reference", "reservation number", "refund request"}),
    ("Can you refund my booking please", {"refund my booking"}),
    ("I filed a baggage claim yesterday, any update?", {"baggage claim"}),
    # Reverse phrasing order ("noun ... adjective") must also be caught.
    ("My bag was lost somewhere between gates", {"lost baggage"}),
    ("The wheel on my suitcase was broken when it arrived", {"broken wheel"}),
])
def test_match_safety_override_terms_detects_expected_terms(message, expected_terms):
    matched = set(demo_server._match_safety_override_terms(message))
    assert expected_terms.issubset(matched)


@pytest.mark.parametrize("message", [
    "How much does a second checked bag cost on an international flight?",
    # These would have false-triggered on the old bare-word list
    # ("report", "account", "missing", "damaged") — must NOT trigger now.
    "I want to report how great my flight attendant was today",
    "My account has great SkyMiles rewards, thanks Delta!",
    "Can I bring my golf bag as checked baggage?",
    "The crew did an amazing job today",
    "I'm missing my connecting flight because of the delay",
])
def test_match_safety_override_terms_no_false_positives(message):
    assert demo_server._match_safety_override_terms(message) == []


def test_match_safety_override_terms_no_match_on_ordinary_policy_question():
    matched = demo_server._match_safety_override_terms(
        "How much does a second checked bag cost on an international flight?"
    )
    assert matched == []


def test_apply_demo_safety_override_is_pure_and_does_not_mutate_input():
    original = {
        "intent": "flight_status_inquiry",
        "reply": "some unrelated reply",
        "escalate": "no",
        "escalation_reason": "some reason",
        "evidence": [],
    }
    original_copy = dict(original)

    result = demo_server.apply_demo_safety_override("I lost my bag, please file a claim", original)

    assert original == original_copy  # input dict untouched
    assert result is not original
    assert result["escalate"] == "yes"


def test_safety_override_badge_markup_present_and_hidden_by_default():
    html = demo_server.index()
    assert 'id="r-override-badge"' in html
    assert "display:none;" in html.split('id="r-override-badge"')[1][:60]
    assert "Safety override applied" in html
