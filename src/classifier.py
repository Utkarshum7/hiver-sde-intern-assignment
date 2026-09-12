"""
Intent Classifier for the live Delta Customer Support AI Agent.

DESIGN NOTE (see docs/decision_log.md): this is DELIBERATELY a different
algorithm from src/proposer.py's rule_based_v1. The proposer's output was
shown to human annotators as a "proposed" label during golden-set
construction, so reusing that exact same rule engine as the production
classifier would raise a fair concern about circularity (the thing being
graded and the thing that helped write the answer key would be identical).

Instead, the live classifier here is a TF-IDF + Logistic Regression model
trained via WEAK SUPERVISION on `intent_heuristic` labels from the
DEVELOPMENT CORPUS ONLY (data/processed/dev_corpus.parquet, ~20,913
retrieval-eligible conversations). Those weak labels come from
src/heuristic_intent.py's single-tier keyword matcher (used for Phase 3
taxonomy validation and dataset stratification) — a DIFFERENT, simpler
heuristic than the proposer's weighted/tie-broken scorer. The classifier
never sees the reserved golden pool or any human gold label during training.

This is honestly a weak-supervision classifier, not a classifier trained on
clean human labels — that limitation is disclosed in the report. It gives a
genuinely distinct "main system" classifier to compare against the
rule-based baseline, without touching held-out data.
"""

import os
from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from src.heuristic_intent import VALID_INTENTS

DEFAULT_MODEL_PATH = "data/processed/intent_classifier.pkl"


class TfidfIntentClassifier:
    """TF-IDF + Logistic Regression intent classifier trained via weak supervision."""

    def __init__(self, max_features: int = 20000, C: float = 4.0):
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=(1, 2),
            stop_words="english",
            lowercase=True,
            min_df=2,
        )
        self.model = LogisticRegression(
            C=C,
            max_iter=1000,
            class_weight="balanced",
        )
        self.classes_: List[str] = []
        self.is_fitted = False

    def fit(self, texts: List[str], labels: List[str]) -> "TfidfIntentClassifier":
        """Fit on (text, weak_label) pairs. Caller is responsible for ensuring
        these come only from non-held-out data (see module docstring)."""
        X = self.vectorizer.fit_transform(texts)
        self.model.fit(X, labels)
        self.classes_ = list(self.model.classes_)
        self.is_fitted = True
        return self

    def predict(self, text: str) -> Tuple[str, float]:
        """Predict (intent, confidence) for a single message.

        confidence = the model's predicted probability for the winning class.
        Falls back to 'general_complaint_feedback' with confidence 0.0 for
        empty/whitespace-only input (nothing to classify) rather than raising.
        """
        if not self.is_fitted:
            raise RuntimeError("Classifier is not fitted. Call fit() or load a saved model.")
        text = str(text or "").strip()
        if not text:
            return "general_complaint_feedback", 0.0

        X = self.vectorizer.transform([text])
        proba = self.model.predict_proba(X)[0]
        best_idx = int(np.argmax(proba))
        intent = self.classes_[best_idx]
        confidence = float(proba[best_idx])
        return intent, confidence

    def predict_batch(self, texts: List[str]) -> List[Tuple[str, float]]:
        return [self.predict(t) for t in texts]

    def save(self, filepath: str = DEFAULT_MODEL_PATH):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        joblib.dump(self, filepath)

    @classmethod
    def load(cls, filepath: str = DEFAULT_MODEL_PATH) -> "TfidfIntentClassifier":
        return joblib.load(filepath)


def train_from_dev_corpus(
    dev_corpus_path: str = "data/processed/dev_corpus.parquet",
) -> Tuple[TfidfIntentClassifier, Dict]:
    """
    Trains the classifier on the development corpus ONLY.

    Returns (classifier, training_report_dict). Raises if the dev corpus
    path looks like it might be a golden/reserved file, as a defensive
    leakage guard.
    """
    if "reserved" in dev_corpus_path.lower() or "golden" in dev_corpus_path.lower():
        raise ValueError(
            f"Refusing to train on '{dev_corpus_path}' — filename suggests "
            "held-out/reserved data. Training must only use the dev corpus."
        )

    df = pd.read_parquet(dev_corpus_path)
    df = df[df["customer_message"].astype(str).str.strip() != ""].reset_index(drop=True)

    texts = df["customer_message"].tolist()
    labels = df["intent_heuristic"].tolist()

    clf = TfidfIntentClassifier()
    clf.fit(texts, labels)

    label_counts = pd.Series(labels).value_counts().to_dict()
    report = {
        "training_examples": len(texts),
        "source_file": dev_corpus_path,
        "label_source": "intent_heuristic (weak, single-tier keyword heuristic — see src/heuristic_intent.py)",
        "vocabulary_size": len(clf.vectorizer.vocabulary_),
        "classes": clf.classes_,
        "label_distribution": label_counts,
    }
    return clf, report
