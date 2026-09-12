"""
Baselines for the Delta Customer Support AI Agent evaluation.

Two baselines, both non-LLM, both meaningfully weaker than the main system
(src/agent.py) but neither a strawman:

BASELINE 1 — TrivialBaseline
    Intent:      always predicts the single majority intent class,
                 computed from the development corpus's intent_heuristic
                 distribution (currently 'general_complaint_feedback').
    Reply:       one fixed generic canned message, ignores message content
                 and retrieved evidence entirely.
    Escalation:  a single flat keyword rule — escalate only if the message
                 contains one of 4 high-risk words ("refund", "cancel",
                 "lost", "damaged"); otherwise never escalate. No per-intent
                 policy, no account-signal detection.
    Evidence:    none retrieved (this baseline does not use the retrieval
                 index at all).

BASELINE 2 — SimpleRetrievalBaseline
    Intent:      1-nearest-neighbour via the SAME TF-IDF retrieval index
                 used by the main system (src/retrieval.py) — predicts the
                 intent_heuristic label of the single most similar
                 historical dev-corpus conversation. No classifier is
                 trained; this is a pure lookup.
    Reply:       the top-1 retrieved historical Delta response, reused
                 VERBATIM (no @mention/signature cleanup, no unsupported-
                 claim stripping) — this is intentionally naive so its
                 grounding-safety failures are visible in evaluation.
    Escalation:  a fixed intent -> escalate lookup table (only the 3
                 "always escalate" intents from src/escalation.py trigger
                 escalation; no message-level account-signal detection at all).
    Evidence:    top-k retrieved historical conversations (same retriever,
                 same k, as the main system — so evidence quality is held
                 constant and only classification/reply/escalation logic
                 differs between baseline 2 and the main system).

Both baselines expose the same `.handle(customer_message)` contract as
DeltaSupportAgent so they can be evaluated identically.
"""

from typing import Dict, List, Optional

import pandas as pd

from src.retrieval import TFIDFRetriever
from src.escalation import ALWAYS_ESCALATE, build_reason, build_no_escalation_reason

_TRIVIAL_ESCALATE_KEYWORDS = ["refund", "cancel", "lost", "damaged"]


class TrivialBaseline:
    """Baseline 1: majority-class intent + generic reply + one flat keyword rule."""

    GENERIC_REPLY = (
        "Thanks for reaching out to Delta! A member of our team will follow "
        "up with more information."
    )

    def __init__(self, majority_intent: str = "general_complaint_feedback"):
        self.majority_intent = majority_intent

    @classmethod
    def from_dev_corpus(cls, dev_corpus_path: str = "data/processed/dev_corpus.parquet") -> "TrivialBaseline":
        df = pd.read_parquet(dev_corpus_path)
        majority_intent = df["intent_heuristic"].value_counts().idxmax()
        return cls(majority_intent=majority_intent)

    def handle(self, customer_message: str) -> Dict:
        customer_message = str(customer_message or "")
        msg_lower = customer_message.lower()
        escalate = any(kw in msg_lower for kw in _TRIVIAL_ESCALATE_KEYWORDS)
        return {
            "intent": self.majority_intent,
            "reply": self.GENERIC_REPLY,
            "escalate": "yes" if escalate else "no",
            "escalation_reason": (
                "Message contains a high-risk keyword; routed to a human agent."
                if escalate else ""
            ),
            "evidence": [],
        }


class SimpleRetrievalBaseline:
    """Baseline 2: 1-NN TF-IDF intent transfer + verbatim reply reuse + fixed escalation table."""

    def __init__(self, retriever: TFIDFRetriever, top_k: int = 3):
        self.retriever = retriever
        self.top_k = top_k

    @classmethod
    def load(cls, index_path: str = "data/processed/retrieval_index.pkl", top_k: int = 3) -> "SimpleRetrievalBaseline":
        return cls(retriever=TFIDFRetriever.load(index_path), top_k=top_k)

    def handle(self, customer_message: str) -> Dict:
        evidence = self.retriever.query(customer_message, top_k=self.top_k)

        if not evidence:
            # No similar historical case found at all (e.g. empty message).
            return {
                "intent": "general_complaint_feedback",
                "reply": "Thanks for reaching out to Delta! A member of our team will follow up.",
                "escalate": "no",
                "escalation_reason": "",
                "evidence": [],
            }

        top1 = evidence[0]
        intent = top1["intent_heuristic"]
        escalate = intent in ALWAYS_ESCALATE
        reason = build_reason(intent, escalate) if escalate else build_no_escalation_reason(intent)

        return {
            "intent": intent,
            "reply": top1["delta_response"],  # verbatim reuse, deliberately naive
            "escalate": "yes" if escalate else "no",
            "escalation_reason": reason,
            "evidence": evidence,
        }
