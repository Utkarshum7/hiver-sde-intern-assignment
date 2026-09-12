"""
Evaluation metrics for the Delta Customer Support AI Agent.

All functions here are pure (input arrays/lists -> metrics dict); they do
not read any files themselves, so they can be unit-tested with small
synthetic fixtures without touching the golden set or any held-out data.

See scripts/evaluate.py for how these are wired to real data (gated on
golden-set annotation completeness).
"""

from typing import Dict, List, Sequence

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    precision_score,
    recall_score,
)

from src.reply_generator import contains_unsupported_action_claim


def intent_metrics(y_true: Sequence[str], y_pred: Sequence[str], labels: Sequence[str]) -> Dict:
    """
    Intent classification metrics: accuracy, macro F1, per-intent P/R/F1,
    and a confusion matrix (as a nested dict keyed by [true][pred]).
    """
    labels = list(labels)
    accuracy = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)

    precisions, recalls, f1s, supports = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average=None, zero_division=0
    )
    per_intent = {
        label: {
            "precision": round(float(p), 4),
            "recall": round(float(r), 4),
            "f1": round(float(f), 4),
            "support": int(s),
        }
        for label, p, r, f, s in zip(labels, precisions, recalls, f1s, supports)
    }

    cm = confusion_matrix(y_true, y_pred, labels=labels)
    cm_dict = {
        true_label: {
            pred_label: int(cm[i][j])
            for j, pred_label in enumerate(labels)
        }
        for i, true_label in enumerate(labels)
    }

    return {
        "accuracy": round(float(accuracy), 4),
        "macro_f1": round(float(macro_f1), 4),
        "per_intent": per_intent,
        "confusion_matrix": cm_dict,
        "n": len(y_true),
    }


def escalation_metrics(y_true: Sequence[str], y_pred: Sequence[str]) -> Dict:
    """
    Escalation metrics, treating 'yes' as the positive class.

    Reports a full 2x2 confusion matrix plus explicit false-positive and
    false-negative counts/indices. The false-negative case (gold=yes,
    predicted=no) is additionally labelled "unsafe" throughout this project
    because it means the agent tried to auto-handle something that actually
    needed a human — the most dangerous error type for this system.
    """
    y_true_bin = [1 if v == "yes" else 0 for v in y_true]
    y_pred_bin = [1 if v == "yes" else 0 for v in y_pred]

    accuracy = accuracy_score(y_true_bin, y_pred_bin)
    precision = precision_score(y_true_bin, y_pred_bin, zero_division=0)
    recall = recall_score(y_true_bin, y_pred_bin, zero_division=0)
    f1 = f1_score(y_true_bin, y_pred_bin, zero_division=0)

    tp_indices = [i for i, (t, p) in enumerate(zip(y_true_bin, y_pred_bin)) if t == 1 and p == 1]
    tn_indices = [i for i, (t, p) in enumerate(zip(y_true_bin, y_pred_bin)) if t == 0 and p == 0]
    fp_indices = [i for i, (t, p) in enumerate(zip(y_true_bin, y_pred_bin)) if t == 0 and p == 1]
    fn_indices = [i for i, (t, p) in enumerate(zip(y_true_bin, y_pred_bin)) if t == 1 and p == 0]

    return {
        "accuracy": round(float(accuracy), 4),
        "precision": round(float(precision), 4),
        "recall": round(float(recall), 4),
        "f1": round(float(f1), 4),
        "confusion_matrix": {
            "true_positive": len(tp_indices),
            "true_negative": len(tn_indices),
            "false_positive": len(fp_indices),
            "false_negative": len(fn_indices),
        },
        "false_positive_count": len(fp_indices),
        "false_positive_indices": fp_indices,
        "false_negative_count": len(fn_indices),
        "false_negative_indices": fn_indices,
        # Alias: every false negative here IS an "unsafe" one (gold=escalate,
        # predicted=auto-handle), kept under both names for continuity with
        # earlier reports that only used this term.
        "unsafe_false_negative_count": len(fn_indices),
        "unsafe_false_negative_indices": fn_indices,
        "n": len(y_true),
    }


def retrieval_intent_match_at_k(
    query_gold_intents: Sequence[str],
    evidence_intent_lists: Sequence[List[str]],
) -> Dict:
    """
    PROXY retrieval metric: for each query, does ANY of the top-k retrieved
    evidence items' intent_heuristic match the query's gold_intent?

    This is a proxy, not a true relevance judgment, because the golden set
    has no per-example "correct historical conversation_id" label (building
    that would require a second, separate human-labelling pass that this
    project's timeline does not include). It is honestly reported as such.
    A qualitative top-1 relevance check on non-golden dev queries already
    exists separately (see reports/retrieval_sanity_check.md).
    """
    if not query_gold_intents:
        return {"n": 0, "match_at_k_rate": None, "note": "no queries provided"}

    hits = 0
    for gold_intent, evidence_intents in zip(query_gold_intents, evidence_intent_lists):
        if gold_intent in evidence_intents:
            hits += 1

    return {
        "n": len(query_gold_intents),
        "match_at_k_rate": round(hits / len(query_gold_intents), 4),
        "is_proxy_metric": True,
        "caveat": (
            "Proxy metric: checks whether retrieved evidence's weak "
            "intent_heuristic label (not a human relevance judgment) "
            "matches the query's gold_intent. See docs for why a true "
            "relevance-labelled retrieval metric was not built."
        ),
    }


def unsupported_claim_rate(replies: Sequence[str]) -> Dict:
    """
    Automated safety metric: fraction of generated replies that contain an
    unsupported account-action claim (see src/reply_generator.py). This is
    a fully local, non-LLM proxy for the "safety / unsupported claims"
    dimension of the LLM-judge rubric (src/llm_judge.py), usable even when
    no LLM API key is configured.
    """
    if not replies:
        return {"n": 0, "unsupported_claim_rate": None}

    flags = [contains_unsupported_action_claim(r) for r in replies]
    return {
        "n": len(replies),
        "unsupported_claim_count": sum(flags),
        "unsupported_claim_rate": round(sum(flags) / len(replies), 4),
        "flagged_indices": [i for i, f in enumerate(flags) if f],
    }
