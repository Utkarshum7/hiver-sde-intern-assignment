"""Tests for src/metrics.py — evaluation metric correctness on synthetic data.

IMPORTANT: these fixtures are synthetic / made up for testing purposes only.
They are NOT real golden-set labels and must never be reported as results.
"""

from src.metrics import (
    intent_metrics,
    escalation_metrics,
    retrieval_intent_match_at_k,
    unsupported_claim_rate,
)


def test_intent_metrics_perfect_predictions():
    labels = ["a", "b", "c"]
    y_true = ["a", "b", "c", "a"]
    y_pred = ["a", "b", "c", "a"]
    result = intent_metrics(y_true, y_pred, labels)
    assert result["accuracy"] == 1.0
    assert result["macro_f1"] == 1.0
    assert result["n"] == 4
    for label in labels:
        assert result["per_intent"][label]["f1"] in (0.0, 1.0)


def test_intent_metrics_confusion_matrix_shape_and_values():
    labels = ["a", "b"]
    y_true = ["a", "a", "b", "b"]
    y_pred = ["a", "b", "b", "b"]
    result = intent_metrics(y_true, y_pred, labels)
    cm = result["confusion_matrix"]
    assert cm["a"]["a"] == 1
    assert cm["a"]["b"] == 1
    assert cm["b"]["b"] == 2
    assert result["accuracy"] == 0.75


def test_escalation_metrics_basic():
    y_true = ["yes", "yes", "no", "no"]
    y_pred = ["yes", "no", "no", "no"]
    result = escalation_metrics(y_true, y_pred)
    assert result["accuracy"] == 0.75
    assert result["precision"] == 1.0  # 1 predicted yes, correct
    assert result["recall"] == 0.5     # 2 true yes, only 1 caught


def test_escalation_metrics_flags_unsafe_false_negatives():
    # index 1: gold=yes, pred=no -> unsafe false negative
    y_true = ["no", "yes", "yes"]
    y_pred = ["no", "no", "yes"]
    result = escalation_metrics(y_true, y_pred)
    assert result["unsafe_false_negative_count"] == 1
    assert result["unsafe_false_negative_indices"] == [1]
    assert result["false_negative_count"] == 1
    assert result["false_negative_indices"] == [1]


def test_escalation_metrics_confusion_matrix_and_false_positives():
    # idx0: TN, idx1: FN(unsafe), idx2: TP, idx3: FP
    y_true = ["no", "yes", "yes", "no"]
    y_pred = ["no", "no", "yes", "yes"]
    result = escalation_metrics(y_true, y_pred)
    cm = result["confusion_matrix"]
    assert cm == {
        "true_positive": 1,
        "true_negative": 1,
        "false_positive": 1,
        "false_negative": 1,
    }
    assert result["false_positive_count"] == 1
    assert result["false_positive_indices"] == [3]
    assert result["false_negative_count"] == 1
    assert result["false_negative_indices"] == [1]


def test_retrieval_intent_match_at_k():
    gold_intents = ["a", "b", "c"]
    evidence_intents = [["a", "x"], ["y", "z"], ["c"]]
    result = retrieval_intent_match_at_k(gold_intents, evidence_intents)
    assert result["n"] == 3
    assert result["match_at_k_rate"] == round(2 / 3, 4)
    assert result["is_proxy_metric"] is True


def test_retrieval_intent_match_at_k_empty():
    result = retrieval_intent_match_at_k([], [])
    assert result["n"] == 0
    assert result["match_at_k_rate"] is None


def test_unsupported_claim_rate():
    replies = [
        "You can check status at delta.com.",
        "I've already credited your account with 500 miles.",
        "We have rebooked you on the next available flight.",
    ]
    result = unsupported_claim_rate(replies)
    assert result["n"] == 3
    assert result["unsupported_claim_count"] == 2
    assert result["flagged_indices"] == [1, 2]
