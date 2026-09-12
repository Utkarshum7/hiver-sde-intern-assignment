"""
Retrieval & Evidence Corpus Module for Delta Customer Support AI Agent.

Handles:
1. Evidence unit extraction from raw/processed conversation threads.
2. Group-based train/dev vs reserved golden candidate pool splitting (leakage-safe).
3. Retrieval eligibility filtering.
4. Lightweight TF-IDF + Cosine Similarity local vector search index.
"""

import os
import re
import json
import joblib
from typing import List, Dict, Any, Tuple
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.heuristic_intent import classify as _classify_heuristic_intent


def filter_retrieval_eligibility(first_cust_text: str, delta_resp_text: str) -> bool:
    """
    Determines if a historical Delta support interaction is eligible for retrieval.
    Criteria:
    - Customer message exists and is non-empty (>5 chars)
    - Delta response exists and contains actionable content (>15 chars)
    - Excludes single-word/empty replies
    """
    if not first_cust_text or len(first_cust_text.strip()) < 5:
        return False
    if not delta_resp_text or len(delta_resp_text.strip()) < 15:
        return False
    return True


def build_evidence_corpus(df_conversations: pd.DataFrame) -> pd.DataFrame:
    """Extracts historical support evidence units from conversation dataframe."""
    records = []
    
    for _, row in df_conversations.iterrows():
        conv_id = str(row["conversation_id"])
        cust_text = str(row.get("first_customer_text", "")).strip()
        delta_text = str(row.get("delta_responses_text", "")).strip()
        timestamp = str(row.get("start_time", "")).strip()
        is_two_way = bool(row.get("is_two_way", True))
        # BUGFIX: this previously read row.get("predicted_intent", ...), but
        # delta_conversations.parquet never contains a "predicted_intent"
        # column (that field only ever existed in an in-memory dataframe
        # inside scripts/analyze_delta_intents.py and was never persisted),
        # so every row silently fell back to the same default value. Compute
        # the heuristic directly from the customer message instead.
        intent_heuristic = _classify_heuristic_intent(cust_text)

        eligible = is_two_way and filter_retrieval_eligibility(cust_text, delta_text)

        records.append({
            "conversation_id": conv_id,
            "customer_message": cust_text,
            "delta_response": delta_text,
            "timestamp": timestamp,
            "is_two_way": is_two_way,
            "retrieval_eligible": eligible,
            "intent_heuristic": intent_heuristic
        })

    return pd.DataFrame(records)


def create_leakage_safe_splits(
    df_conversations: pd.DataFrame, 
    dev_ratio: float = 0.80, 
    seed: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Splits conversation dataset into Development Corpus (80%) and Reserved Golden Pool (20%)
    at the CONVERSATION ID level using a deterministic random seed.
    """
    unique_conv_ids = df_conversations["conversation_id"].unique()
    np.random.seed(seed)
    shuffled_ids = np.random.permutation(unique_conv_ids)
    
    num_dev = int(len(shuffled_ids) * dev_ratio)
    dev_ids = set(shuffled_ids[:num_dev])
    reserved_ids = set(shuffled_ids[num_dev:])

    dev_df = df_conversations[df_conversations["conversation_id"].isin(dev_ids)].copy()
    reserved_df = df_conversations[df_conversations["conversation_id"].isin(reserved_ids)].copy()

    return dev_df, reserved_df


class TFIDFRetriever:
    """Local, explainable TF-IDF + Cosine Similarity index for historical support retrieval."""

    def __init__(self, max_features: int = 25000):
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=(1, 2),
            stop_words="english",
            lowercase=True
        )
        self.tfidf_matrix = None
        self.corpus_df = None

    def fit(self, dev_corpus_df: pd.DataFrame):
        """Fits TF-IDF vectorizer on eligible development customer messages only."""
        # Filter strictly for retrieval eligible cases in dev corpus
        self.corpus_df = dev_corpus_df[dev_corpus_df["retrieval_eligible"] == True].reset_index(drop=True)
        
        texts = self.corpus_df["customer_message"].tolist()
        self.tfidf_matrix = self.vectorizer.fit_transform(texts)

    def query(self, query_text: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """Queries index with customer text and returns top-k historically similar Delta cases."""
        if not query_text or self.tfidf_matrix is None:
            return []

        query_vec = self.vectorizer.transform([query_text])
        sim_scores = cosine_similarity(query_vec, self.tfidf_matrix).flatten()

        top_indices = np.argsort(sim_scores)[::-1][:top_k]

        results = []
        for idx in top_indices:
            score = float(sim_scores[idx])
            row = self.corpus_df.iloc[idx]
            results.append({
                "conversation_id": row["conversation_id"],
                "customer_message": row["customer_message"],
                "delta_response": row["delta_response"],
                "similarity_score": round(score, 4),
                "timestamp": row["timestamp"],
                "intent_heuristic": row["intent_heuristic"]
            })

        return results

    def save(self, filepath: str):
        """Saves retriever instance to disk using joblib."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        joblib.dump(self, filepath)

    @classmethod
    def load(cls, filepath: str):
        """Loads retriever instance from disk."""
        return joblib.load(filepath)
