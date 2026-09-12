"""Tests for src/classifier.py (TF-IDF + LogisticRegression, weak supervision)."""

import pandas as pd
import pytest

from src.classifier import TfidfIntentClassifier, train_from_dev_corpus


@pytest.fixture
def toy_training_data():
    texts = [
        "How much is a checked bag fee?",
        "What is the carry-on weight limit?",
        "My flight was cancelled, need to rebook now",
        "Missed my connection due to delay, need a new flight",
        "Great crew today, thank you so much!",
        "Loved the flight attendants, amazing service",
    ] * 5  # repeat so TF-IDF/LogReg has enough signal
    labels = [
        "baggage_allowance_policy",
        "baggage_allowance_policy",
        "flight_delay_rebooking",
        "flight_delay_rebooking",
        "general_complaint_feedback",
        "general_complaint_feedback",
    ] * 5
    return texts, labels


def test_fit_and_predict_returns_valid_structure(toy_training_data):
    texts, labels = toy_training_data
    clf = TfidfIntentClassifier()
    clf.fit(texts, labels)

    intent, confidence = clf.predict("How much for a second checked bag?")
    assert intent in set(labels)
    assert 0.0 <= confidence <= 1.0


def test_predict_before_fit_raises():
    clf = TfidfIntentClassifier()
    with pytest.raises(RuntimeError):
        clf.predict("hello")


def test_empty_message_does_not_crash(toy_training_data):
    texts, labels = toy_training_data
    clf = TfidfIntentClassifier()
    clf.fit(texts, labels)
    intent, confidence = clf.predict("")
    assert intent == "general_complaint_feedback"
    assert confidence == 0.0


def test_save_and_load_roundtrip(toy_training_data, tmp_path):
    texts, labels = toy_training_data
    clf = TfidfIntentClassifier()
    clf.fit(texts, labels)

    path = str(tmp_path / "clf.pkl")
    clf.save(path)
    loaded = TfidfIntentClassifier.load(path)

    intent_a, _ = clf.predict("How much is a checked bag fee?")
    intent_b, _ = loaded.predict("How much is a checked bag fee?")
    assert intent_a == intent_b


def test_train_from_dev_corpus_refuses_reserved_or_golden_path():
    with pytest.raises(ValueError):
        train_from_dev_corpus("data/processed/reserved_golden_pool.parquet")
    with pytest.raises(ValueError):
        train_from_dev_corpus("data/golden_eval/golden_annotation.csv")


def test_train_from_dev_corpus_uses_intent_heuristic_labels(tmp_path):
    df = pd.DataFrame([
        {"conversation_id": "1", "customer_message": "checked bag fee please", "delta_response": "x",
         "timestamp": "t", "is_two_way": True, "retrieval_eligible": True,
         "intent_heuristic": "baggage_allowance_policy"},
        {"conversation_id": "2", "customer_message": "flight cancelled rebook me", "delta_response": "x",
         "timestamp": "t", "is_two_way": True, "retrieval_eligible": True,
         "intent_heuristic": "flight_delay_rebooking"},
    ] * 5)
    path = tmp_path / "dev_corpus.parquet"
    df.to_parquet(path, index=False)

    clf, report = train_from_dev_corpus(str(path))
    assert clf.is_fitted
    assert report["training_examples"] == 10
    assert "intent_heuristic" in report["label_source"]
