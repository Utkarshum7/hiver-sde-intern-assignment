"""
Delta Customer Support AI Agent — end-to-end pipeline.

    CUSTOMER MESSAGE
          v
    INTENT CLASSIFICATION   (src/classifier.py — TF-IDF + LogisticRegression,
          |                   weak-supervised on dev corpus)
          v
    HISTORICAL RETRIEVAL    (src/retrieval.py — TF-IDF cosine similarity,
          |                   dev corpus only, reserved pool never indexed)
          v
    EVIDENCE                (top-k historical Delta resolutions)
          v
    REPLY GENERATION        (src/reply_generator.py — grounded, escalation-safe,
          |                   local template; LLM path optional via src/llm_client.py)
          v
    ESCALATION DECISION     (src/escalation.py — deterministic policy, shared
          |                   with the annotation-assist proposer)
          v
    FINAL RESPONSE

Each stage is independently testable (see tests/test_agent.py) and this
class only orchestrates calls between them — no business logic lives here.
"""

import os
from typing import Dict, List, Optional

from src.classifier import TfidfIntentClassifier, DEFAULT_MODEL_PATH
from src.retrieval import TFIDFRetriever
from src import escalation as escalation_policy
from src.reply_generator import generate_reply

DEFAULT_RETRIEVAL_INDEX_PATH = "data/processed/retrieval_index.pkl"


class DeltaSupportAgent:
    """The main (non-baseline) support agent for @Delta."""

    def __init__(
        self,
        classifier: Optional[TfidfIntentClassifier] = None,
        retriever: Optional[TFIDFRetriever] = None,
        top_k: int = 3,
    ):
        self.classifier = classifier
        self.retriever = retriever
        self.top_k = top_k

    @classmethod
    def load(
        cls,
        classifier_path: str = DEFAULT_MODEL_PATH,
        index_path: str = DEFAULT_RETRIEVAL_INDEX_PATH,
        top_k: int = 3,
    ) -> "DeltaSupportAgent":
        if not os.path.exists(classifier_path):
            raise FileNotFoundError(
                f"Classifier not found at {classifier_path}. "
                "Run: python scripts/train_classifier.py"
            )
        if not os.path.exists(index_path):
            raise FileNotFoundError(
                f"Retrieval index not found at {index_path}. "
                "Run: python scripts/build_retrieval_index.py"
            )
        classifier = TfidfIntentClassifier.load(classifier_path)
        retriever = TFIDFRetriever.load(index_path)
        return cls(classifier=classifier, retriever=retriever, top_k=top_k)

    def classify(self, customer_message: str) -> Dict:
        intent, confidence = self.classifier.predict(customer_message)
        return {"intent": intent, "confidence": confidence}

    def retrieve(self, customer_message: str) -> List[Dict]:
        return self.retriever.query(customer_message, top_k=self.top_k)

    def decide_escalation(self, intent: str, customer_message: str) -> Dict:
        # NOTE: delta_response="" always — the live agent, unlike the
        # golden-set proposer, has no historical Delta reply for a brand
        # new incoming message. See src/escalation.py docstring.
        return escalation_policy.decide(intent, customer_message, delta_response="")

    def handle(self, customer_message: str) -> Dict:
        """
        Run the full pipeline for one customer message.

        Returns a dict with exactly the fields required by the assignment:
          intent, reply, escalate, escalation_reason, evidence
        plus a few extra diagnostic fields (confidence, grounding_source)
        that are useful for evaluation/debugging but not part of the core
        contract.
        """
        if self.classifier is None or self.retriever is None:
            raise RuntimeError(
                "Agent has no loaded classifier/retriever. Use DeltaSupportAgent.load()."
            )

        customer_message = str(customer_message or "").strip()

        cls_result = self.classify(customer_message)
        intent = cls_result["intent"]

        evidence = self.retrieve(customer_message)

        esc_result = self.decide_escalation(intent, customer_message)
        escalate = esc_result["escalate"] == "yes"

        gen = generate_reply(
            customer_message=customer_message,
            intent=intent,
            evidence=evidence,
            escalate=escalate,
            escalation_reason=esc_result["escalation_reason"] if escalate else "",
        )

        return {
            "intent": intent,
            "reply": gen["reply"],
            "escalate": esc_result["escalate"],
            "escalation_reason": esc_result["escalation_reason"],
            "evidence": evidence,
            # Diagnostics (not required by the assignment output contract,
            # but retained for evaluation/debugging/interview explanation):
            "confidence": cls_result["confidence"],
            "grounding_source": gen["grounding_source"],
            "used_evidence_conversation_id": gen["used_evidence_conversation_id"],
        }
