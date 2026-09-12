"""
Script to extract and process all Delta customer support conversations from raw twcs.csv.

Outputs:
- data/processed/delta_conversations.parquet
- data/processed/delta_messages.parquet
- reports/delta_dataset_summary.json
"""

import os
import sys
import json
from collections import defaultdict
import pandas as pd
import numpy as np

# Target brand handle
TARGET_BRAND = "Delta"


def extract_delta_dataset(raw_csv_path: str):
    print(f"[*] Reading raw CSV: {raw_csv_path} to identify Delta messages...")
    
    tweets_by_id = {}
    parent_map = {}
    delta_tweet_ids = set()

    # Step 1: Stream chunks to collect all tweets & map parents
    chunk_reader = pd.read_csv(raw_csv_path, chunksize=100000, dtype=str)
    total_scanned = 0

    for chunk in chunk_reader:
        chunk["tweet_id"] = chunk["tweet_id"].astype(str).str.strip()
        chunk["inbound"] = chunk["inbound"].astype(str).str.lower().isin(["true", "1", "t"])
        chunk["author_id"] = chunk["author_id"].astype(str).str.strip()
        chunk["text"] = chunk["text"].fillna("").astype(str)

        total_scanned += len(chunk)

        for _, row in chunk.iterrows():
            tid = row["tweet_id"]
            author = row["author_id"]
            inbound = row["inbound"]
            parent_id = str(row.get("in_response_to_tweet_id", "")).strip()
            created_at = str(row.get("created_at", "")).strip()
            text = row["text"]

            tweets_by_id[tid] = {
                "tweet_id": tid,
                "author_id": author,
                "inbound": inbound,
                "created_at": created_at,
                "text": text,
                "parent_id": parent_id
            }

            if parent_id and parent_id.lower() not in ["nan", "none", "", "<na>"]:
                parent_map[tid] = parent_id

            if author == TARGET_BRAND:
                delta_tweet_ids.add(tid)

    print(f"[+] Total raw tweets scanned: {total_scanned:,}")
    print(f"[+] Total Delta outbound tweets identified: {len(delta_tweet_ids):,}")

    # Step 2: Build children map
    children_map = defaultdict(list)
    for child_id, parent_id in parent_map.items():
        children_map[parent_id].append(child_id)

    # Step 3: Reconstruct conversation trees containing Delta
    print("[*] Reconstructing conversation threads for Delta...")
    seen_roots = set()
    conversations = []
    messages = []

    for b_tid in delta_tweet_ids:
        # Trace root
        curr = b_tid
        while curr in parent_map and parent_map[curr] in tweets_by_id:
            curr = parent_map[curr]

        if curr in seen_roots:
            continue
        seen_roots.add(curr)

        # BFS/DFS from root to collect full thread
        stack = [curr]
        conv_tweets = []
        visited = set()

        while stack:
            node = stack.pop()
            if node in visited:
                continue
            visited.add(node)
            if node in tweets_by_id:
                conv_tweets.append(tweets_by_id[node])
            for child in children_map.get(node, []):
                if child not in visited:
                    stack.append(child)

        if not conv_tweets:
            continue

        # Check conversation attributes
        has_cust = any(t["inbound"] for t in conv_tweets)
        has_delta = any(t["author_id"] == TARGET_BRAND and not t["inbound"] for t in conv_tweets)

        # Sort conversation tweets by timestamp or ID
        conv_id = curr
        cust_texts = [t["text"] for t in conv_tweets if t["inbound"]]
        delta_texts = [t["text"] for t in conv_tweets if t["author_id"] == TARGET_BRAND and not t["inbound"]]
        
        dates = [t["created_at"] for t in conv_tweets if t["created_at"]]
        start_time = dates[0] if dates else "N/A"
        end_time = dates[-1] if dates else "N/A"

        for t in conv_tweets:
            msg_dict = dict(t)
            msg_dict["conversation_id"] = conv_id
            messages.append(msg_dict)

        conversations.append({
            "conversation_id": conv_id,
            "total_messages": len(conv_tweets),
            "customer_messages_count": len(cust_texts),
            "delta_messages_count": len(delta_texts),
            "is_two_way": bool(has_cust and has_delta),
            "start_time": start_time,
            "end_time": end_time,
            "first_customer_text": cust_texts[0] if cust_texts else "",
            "all_customer_text": " | ".join(cust_texts),
            "delta_responses_text": " | ".join(delta_texts),
            "full_messages_json": json.dumps(conv_tweets)
        })

    conv_df = pd.DataFrame(conversations)
    msg_df = pd.DataFrame(messages)

    print(f"[+] Total Delta conversations reconstructed: {len(conv_df):,}")
    print(f"[+] Total 2-way Delta conversations: {conv_df['is_two_way'].sum():,}")
    print(f"[+] Total Delta thread messages: {len(msg_df):,}")

    # Save to processed Parquet files
    os.makedirs("data/processed", exist_ok=True)
    conv_parquet_path = "data/processed/delta_conversations.parquet"
    msg_parquet_path = "data/processed/delta_messages.parquet"

    conv_df.to_parquet(conv_parquet_path, index=False)
    msg_df.to_parquet(msg_parquet_path, index=False)

    print(f"[+] Saved processed conversations to: {conv_parquet_path}")
    print(f"[+] Saved processed messages to: {msg_parquet_path}")

    # Save summary report JSON
    os.makedirs("reports", exist_ok=True)
    summary = {
        "brand": TARGET_BRAND,
        "total_raw_tweets_scanned": total_scanned,
        "total_reconstructed_conversations": len(conv_df),
        "two_way_conversations_count": int(conv_df["is_two_way"].sum()),
        "one_way_conversations_count": int((~conv_df["is_two_way"]).sum()),
        "total_thread_messages": len(msg_df),
        "avg_thread_length": round(float(conv_df["total_messages"].mean()), 2),
        "median_thread_length": float(conv_df["total_messages"].median()),
        "date_range": {
            "start": str(conv_df["start_time"].min()),
            "end": str(conv_df["end_time"].max())
        }
    }
    
    summary_path = "reports/delta_dataset_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"[+] Saved dataset summary to: {summary_path}")

    return conv_df, msg_df, summary


def main():
    raw_csv = "data/raw/twcs.csv"
    if not os.path.exists(raw_csv):
        print(f"[ERROR] {raw_csv} not found.")
        sys.exit(1)
        
    extract_delta_dataset(raw_csv)


if __name__ == "__main__":
    main()
