"""Tests for src/baselines.py."""

import pandas as pd
import pytest

from src.baselines import TrivialBaseline, SimpleRetrievalBaseline
from src.retrieval import TFIDFRetriever, build_evidence_corpus


@pytest.fixture
def small_dev_corpus_df(tmp_path):
    df = pd.DataFrame([
        {"conversation_id": "1", "customer_message": "How much for a checked bag?",
         "delta_response": "Checked bag fee is $30 for the first bag.",
         "timestamp": "t", "is_two_way": True, "retrieval_eligible": True,
         "intent_heuristic": "baggage_allowance_policy"},
        {"conversation_id": "2", "customer_message": "Flight cancelled and I missed my connection, need rebooking",
         "delta_response": "Please DM your confirmation number for rebooking help.",
         "timestamp": "t", "is_two_way": True, "retrieval_eligible": True,
         "intent_heuristic": "flight_delay_rebooking"},
        {"conversation_id": "3", "customer_message": "Great flight crew today, thanks!",
         "delta_response": "So glad to hear it! Thanks for flying Delta.",
         "timestamp": "t", "is_two_way": True, "retrieval_eligible": True,
         "intent_heuristic": "general_complaint_feedback"},
    ])
    path = tmp_path / "dev_corpus.parquet"
    df.to_parquet(path, index=False)
    return df, str(path)


def test_trivial_baseline_ignores_content_for_intent_and_reply():
    baseline = TrivialBaseline(majority_intent="general_complaint_feedback")
    r1 = baseline.handle("How much is a checked bag?")
    r2 = baseline.handle("My seat was changed without asking!")
    assert r1["intent"] == r2["intent"] == "general_complaint_feedback"
    assert r1["reply"] == r2["reply"]
    assert r1["evidence"] == []


def test_trivial_baseline_from_dev_corpus(small_dev_corpus_df):
    df, path = small_dev_corpus_df
    baseline = TrivialBaseline.from_dev_corpus(path)
    assert baseline.majority_intent in df["intent_heuristic"].values


def test_trivial_baseline_flat_keyword_escalation():
    baseline = TrivialBaseline()
    assert baseline.handle("I want a refund please")["escalate"] == "yes"
    assert baseline.handle("What is the weather like")["escalate"] == "no"


def test_simple_retrieval_baseline_predicts_nearest_neighbor_intent(small_dev_corpus_df):
    df, _ = small_dev_corpus_df
    evidence_df = build_evidence_corpus(pd.DataFrame([
        {"conversation_id": r["conversation_id"], "first_customer_text": r["customer_message"],
         "delta_responses_text": r["delta_response"], "start_time": r["timestamp"],
         "is_two_way": True}
        for _, r in df.iterrows()
    ]))
    retriever = TFIDFRetriever()
    retriever.fit(evidence_df)
    baseline = SimpleRetrievalBaseline(retriever=retriever, top_k=2)

    result = baseline.handle("How much does a second checked bag cost?")
    assert result["intent"] == "baggage_allowance_policy"
    assert result["reply"] == result["evidence"][0]["delta_response"]  # verbatim reuse
    assert len(result["evidence"]) > 0


def test_simple_retrieval_baseline_fixed_escalation_table(small_dev_corpus_df):
    df, _ = small_dev_corpus_df
    evidence_df = build_evidence_corpus(pd.DataFrame([
        {"conversation_id": r["conversation_id"], "first_customer_text": r["customer_message"],
         "delta_responses_text": r["delta_response"], "start_time": r["timestamp"],
         "is_two_way": True}
        for _, r in df.iterrows()
    ]))
    retriever = TFIDFRetriever()
    retriever.fit(evidence_df)
    baseline = SimpleRetrievalBaseline(retriever=retriever, top_k=1)

    result = baseline.handle("Flight cancelled, missed my connection, need to rebook")
    assert result["intent"] == "flight_delay_rebooking"
    assert result["escalate"] == "yes"  # always-escalate intent, per fixed table
