"""
Script to build the leakage-safe development split, reserved golden pool,
and TF-IDF retrieval index for Delta historical support evidence.

Usage:
    python scripts/build_retrieval_index.py
"""

import os
import sys
import json
import pandas as pd

# Ensure src directory is in Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.retrieval import (
    build_evidence_corpus,
    create_leakage_safe_splits,
    TFIDFRetriever
)


def main():
    parquet_path = "data/processed/delta_conversations.parquet"
    if not os.path.exists(parquet_path):
        print(f"[ERROR] {parquet_path} not found. Please run scripts/build_delta_dataset.py first.")
        sys.exit(1)

    print(f"[*] Loading processed Delta conversations from: {parquet_path}")
    df_convs = pd.read_parquet(parquet_path)
    total_convs = len(df_convs)
    print(f"[+] Total Delta Conversations: {total_convs:,}")

    # Build evidence units
    print("[*] Building evidence units and evaluating retrieval eligibility...")
    evidence_df = build_evidence_corpus(df_convs)

    # Perform Leakage-Safe Group Split (Seed=42, Dev=80%, Reserved Golden Pool=20%)
    print("[*] Performing group-based conversation split (80% Dev / 20% Reserved Golden Pool)...")
    dev_df, reserved_pool_df = create_leakage_safe_splits(evidence_df, dev_ratio=0.80, seed=42)

    dev_count = len(dev_df)
    reserved_count = len(reserved_pool_df)
    dev_eligible_count = int(dev_df["retrieval_eligible"].sum())
    reserved_eligible_count = int(reserved_pool_df["retrieval_eligible"].sum())

    print(f"[+] Development Corpus Conversations: {dev_count:,} ({dev_count / total_convs:.1%})")
    print(f"[+] Development Corpus Retrieval-Eligible: {dev_eligible_count:,}")
    print(f"[+] Reserved Golden Pool Conversations (HELD OUT): {reserved_count:,} ({reserved_count / total_convs:.1%})")
    print(f"[+] Reserved Golden Pool Retrieval-Eligible: {reserved_eligible_count:,}")

    # Save split parquet files
    dev_parquet_path = "data/processed/dev_corpus.parquet"
    reserved_parquet_path = "data/processed/reserved_golden_pool.parquet"
    dev_df.to_parquet(dev_parquet_path, index=False)
    reserved_pool_df.to_parquet(reserved_parquet_path, index=False)
    print(f"[+] Saved Development Corpus to: {dev_parquet_path}")
    print(f"[+] Saved Reserved Golden Candidate Pool to: {reserved_parquet_path}")

    # Fit TF-IDF Retrieval Index on Development Corpus only!
    print("[*] Fitting TF-IDF retriever index on eligible Development Corpus cases...")
    retriever = TFIDFRetriever(max_features=25000)
    retriever.fit(dev_df)

    index_path = "data/processed/retrieval_index.pkl"
    retriever.save(index_path)
    index_bytes = os.path.getsize(index_path)
    index_mb = index_bytes / (1024 * 1024)
    print(f"[+] Saved Index File to: {index_path} ({index_mb:.2f} MB)")

    # Save Index Metadata Report
    reports_dir = os.path.abspath("reports")
    os.makedirs(reports_dir, exist_ok=True)
    summary = {
        "total_conversations": total_convs,
        "dev_split_count": dev_count,
        "dev_split_pct": round(dev_count / total_convs * 100, 2),
        "dev_retrieval_eligible_count": dev_eligible_count,
        "reserved_golden_pool_count": reserved_count,
        "reserved_golden_pool_pct": round(reserved_count / total_convs * 100, 2),
        "reserved_retrieval_eligible_count": reserved_eligible_count,
        "retrieval_method": "TF-IDF Vectorizer + Cosine Similarity",
        "vectorizer_max_features": 25000,
        "ngram_range": [1, 2],
        "index_file_size_mb": round(index_mb, 2),
        "random_seed": 42
    }
    
    summary_json_path = os.path.join(reports_dir, "retrieval_index_summary.json")
    with open(summary_json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"[+] Saved index summary to: {summary_json_path}")


if __name__ == "__main__":
    main()
