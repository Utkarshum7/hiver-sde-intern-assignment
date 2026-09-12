"""
Unit and safety tests for Phase 4 retrieval index and data leakage prevention.
"""

import os
import sys
import pandas as pd
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.retrieval import (
    build_evidence_corpus,
    create_leakage_safe_splits,
    filter_retrieval_eligibility,
    TFIDFRetriever
)


@pytest.fixture
def sample_conv_df():
    data = [
        {
            "conversation_id": "c1",
            "first_customer_text": "How much for a second checked bag to London?",
            "delta_responses_text": "Checked bag fee for second bag is $40. Details at delta.com/baggage.",
            "start_time": "2017-10-10 10:00:00",
            "is_two_way": True,
        },
        {
            "conversation_id": "c2",
            "first_customer_text": "DL123 cancelled and I missed my connection, need rebooking to Boston!",
            "delta_responses_text": "Please DM us your confirmation number so we can assist.",
            "start_time": "2017-10-10 11:00:00",
            "is_two_way": True,
        },
        {
            "conversation_id": "c3",
            "first_customer_text": "Can I upgrade my seat to First Class with SkyMiles?",
            "delta_responses_text": "Yes, Medallion upgrades are available in app or at delta.com.",
            "start_time": "2017-10-10 12:00:00",
            "is_two_way": True,
        },
        {
            "conversation_id": "c4",
            "first_customer_text": "My missing bag never came out on the carousel in Miami!",
            "delta_responses_text": "Please file a claim at the baggage desk or DM us reference number.",
            "start_time": "2017-10-10 13:00:00",
            "is_two_way": True,
        }
    ]
    return pd.DataFrame(data)


def test_split_group_isolation(sample_conv_df):
    """Test 2: Verify all messages from one conversation remain in the same split (0 intersection)."""
    evidence_df = build_evidence_corpus(sample_conv_df)
    dev_df, reserved_df = create_leakage_safe_splits(evidence_df, dev_ratio=0.50, seed=42)

    dev_ids = set(dev_df["conversation_id"])
    reserved_ids = set(reserved_df["conversation_id"])

    # Intersection must be empty (group isolation)
    assert len(dev_ids.intersection(reserved_ids)) == 0
    assert len(dev_ids) + len(reserved_ids) == len(sample_conv_df)


def test_reserved_pool_leakage_prevention(sample_conv_df):
    """Test 1: A conversation in the reserved pool CANNOT exist in the fitted retrieval index."""
    evidence_df = build_evidence_corpus(sample_conv_df)
    dev_df, reserved_df = create_leakage_safe_splits(evidence_df, dev_ratio=0.50, seed=42)

    retriever = TFIDFRetriever()
    retriever.fit(dev_df)

    indexed_ids = set(retriever.corpus_df["conversation_id"])
    reserved_ids = set(reserved_df["conversation_id"])

    # Reserved pool IDs must NEVER be in the index corpus
    assert len(indexed_ids.intersection(reserved_ids)) == 0


def test_empty_message_handling(sample_conv_df):
    """Test 4: Empty or whitespace-only queries do not crash retrieval."""
    evidence_df = build_evidence_corpus(sample_conv_df)
    retriever = TFIDFRetriever()
    retriever.fit(evidence_df)

    res1 = retriever.query("", top_k=2)
    res2 = retriever.query("   ", top_k=2)

    assert isinstance(res1, list)
    assert isinstance(res2, list)


def test_deterministic_retrieval_results(sample_conv_df):
    """Test 5: Retrieval returns identical deterministic results for the same query."""
    evidence_df = build_evidence_corpus(sample_conv_df)
    retriever = TFIDFRetriever()
    retriever.fit(evidence_df)

    query = "checked bag fee"
    res_a = retriever.query(query, top_k=2)
    res_b = retriever.query(query, top_k=2)

    assert res_a == res_b


def test_intent_heuristic_computed_from_message_not_constant(sample_conv_df):
    """
    Regression test for a bug where build_evidence_corpus() read a
    'predicted_intent' column that is never present on the real
    delta_conversations.parquet data, silently defaulting every row's
    intent_heuristic to the same constant value ('general_complaint_feedback').

    The 4 fixture conversations have clearly distinct topics (baggage fee,
    flight cancellation/rebooking, seat upgrade, lost baggage), so a working
    heuristic must NOT collapse them all to the same intent.
    """
    evidence_df = build_evidence_corpus(sample_conv_df)
    heuristics = evidence_df.set_index("conversation_id")["intent_heuristic"]

    assert heuristics["c1"] == "baggage_allowance_policy"
    assert heuristics["c2"] == "flight_delay_rebooking"
    assert heuristics["c3"] == "seat_assignment_upgrade"
    assert heuristics["c4"] == "lost_damaged_baggage"
    # The core regression guard: not every row collapsed to one value.
    assert evidence_df["intent_heuristic"].nunique() > 1


def test_retrieved_payload_permitted_fields(sample_conv_df):
    """Test 6: Retrieved evidence payload contains only permitted schema fields."""
    evidence_df = build_evidence_corpus(sample_conv_df)
    retriever = TFIDFRetriever()
    retriever.fit(evidence_df)

    results = retriever.query("upgrading my seat", top_k=1)
    assert len(results) > 0

    item = results[0]
    expected_fields = {"conversation_id", "customer_message", "delta_response", "similarity_score", "timestamp", "intent_heuristic"}
    assert set(item.keys()) == expected_fields
