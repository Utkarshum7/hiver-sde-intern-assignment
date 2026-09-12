"""
Tests for src/agent.py — end-to-end pipeline contract.

Uses small locally-trained fixtures (not the real 20k-row dev corpus / real
retrieval index) so tests run fast and don't depend on generated artifacts
existing on disk.
"""

import pandas as pd
import pytest

from src.agent import DeltaSupportAgent
from src.classifier import TfidfIntentClassifier
from src.retrieval import TFIDFRetriever, build_evidence_corpus

REQUIRED_OUTPUT_FIELDS = {"intent", "reply", "escalate", "escalation_reason", "evidence"}


@pytest.fixture
def toy_agent():
    texts = [
        "How much is a checked bag fee?",
        "What is the carry-on weight limit for bags?",
        "My flight was cancelled, need to rebook now",
        "Missed my connection due to delay, need a new flight",
        "Great crew today, thank you so much!",
        "Loved the flight attendants, amazing service",
    ] * 4
    labels = [
        "baggage_allowance_policy", "baggage_allowance_policy",
        "flight_delay_rebooking", "flight_delay_rebooking",
        "general_complaint_feedback", "general_complaint_feedback",
    ] * 4
    clf = TfidfIntentClassifier()
    clf.fit(texts, labels)

    conv_df = pd.DataFrame([
        {"conversation_id": str(i), "first_customer_text": t,
         "delta_responses_text": f"Response about: {t}", "start_time": "t",
         "is_two_way": True}
        for i, t in enumerate(texts)
    ])
    evidence_df = build_evidence_corpus(conv_df)
    retriever = TFIDFRetriever()
    retriever.fit(evidence_df)

    return DeltaSupportAgent(classifier=clf, retriever=retriever, top_k=2)


def test_handle_returns_required_output_contract(toy_agent):
    result = toy_agent.handle("How much for a checked bag?")
    assert REQUIRED_OUTPUT_FIELDS.issubset(result.keys())
    assert isinstance(result["evidence"], list)
    assert result["escalate"] in ("yes", "no")


def test_handle_escalates_flight_delay_rebooking(toy_agent):
    result = toy_agent.handle("My flight was cancelled, need to rebook now")
    assert result["intent"] == "flight_delay_rebooking"
    assert result["escalate"] == "yes"
    assert result["escalation_reason"] != ""


def test_handle_auto_handles_baggage_policy(toy_agent):
    result = toy_agent.handle("What is the carry-on weight limit?")
    assert result["intent"] == "baggage_allowance_policy"
    assert result["escalate"] == "no"


def test_handle_empty_message_does_not_crash(toy_agent):
    result = toy_agent.handle("")
    assert REQUIRED_OUTPUT_FIELDS.issubset(result.keys())


def test_handle_reply_never_claims_unsupported_action(toy_agent):
    from src.reply_generator import contains_unsupported_action_claim
    for msg in [
        "How much for a checked bag?",
        "My flight was cancelled, need to rebook now",
        "Great crew today!",
    ]:
        result = toy_agent.handle(msg)
        assert not contains_unsupported_action_claim(result["reply"])


def test_load_raises_clear_error_when_artifacts_missing(tmp_path):
    with pytest.raises(FileNotFoundError):
        DeltaSupportAgent.load(
            classifier_path=str(tmp_path / "nope.pkl"),
            index_path=str(tmp_path / "nope2.pkl"),
        )
