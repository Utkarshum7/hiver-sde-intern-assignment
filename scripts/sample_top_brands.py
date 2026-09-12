"""
Script to extract and analyze a reproducible stratified sample of conversations
for the top 5 candidate brands in the Customer Support on Twitter dataset.

Target Brands:
1. AmazonHelp
2. AppleSupport
3. Uber_Support
4. SpotifyCares
5. Delta
"""

import os
import sys
import json
import re
from collections import Counter, defaultdict
import pandas as pd
import numpy as np

# Ensure src directory is in Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

TARGET_BRANDS = [
    "AmazonHelp",
    "AppleSupport",
    "Uber_Support",
    "SpotifyCares",
    "Delta"
]

# Patterns for DM / Private Redirects
DM_PATTERNS = [
    r"\bdm\b", r"\bprivate message\b", r"\bdirect message\b",
    r"\blink in bio\b", r"\bmessage us\b", r"\bsend us a message\b",
    r"\bpm us\b", r"\bhead over to dms\b", r"\bhop into dms\b"
]

# Patterns for Actionable Content (URLs, steps, specific instructions)
ACTIONABLE_PATTERNS = [
    r"http[s]?://", r"\bsettings\b", r"\brestart\b", r"\bupdate\b",
    r"\bclick\b", r"\bgo to\b", r"\bstep\b", r"\bcheck out\b",
    r"\bapp store\b", r"\bbrowser\b", r"\bclear cache\b", r"\bpolicy\b"
]

# Patterns for Customer Resolution / Sentiment
RESOLVED_PATTERNS = [
    r"\bthanks\b", r"\bthank you\b", r"\bworked\b", r"\bfixed\b",
    r"\bsolved\b", r"\bgreat\b", r"\bawesome\b", r"\bperfect\b", r"\bhelped\b"
]


def is_dm_redirect(text: str) -> bool:
    text_lower = text.lower()
    return any(re.search(pat, text_lower) for pat in DM_PATTERNS)


def is_actionable(text: str) -> bool:
    text_lower = text.lower()
    return any(re.search(pat, text_lower) for pat in ACTIONABLE_PATTERNS)


def is_customer_satisfied(text: str) -> bool:
    text_lower = text.lower()
    return any(re.search(pat, text_lower) for pat in RESOLVED_PATTERNS)


def sample_and_analyze_brands(csv_path: str, samples_per_brand: int = 500):
    print(f"[*] Reading dataset in chunks to extract brand threads for: {TARGET_BRANDS}")
    
    # Track tweets per brand
    # brand -> list of conversations (each conversation is list of tweet dicts)
    parent_map = {}
    tweets_by_id = {}
    brand_tweet_ids = defaultdict(set)

    chunk_reader = pd.read_csv(csv_path, chunksize=100000, dtype=str)
    for chunk in chunk_reader:
        chunk["tweet_id"] = chunk["tweet_id"].astype(str).str.strip()
        chunk["inbound"] = chunk["inbound"].astype(str).str.lower().isin(["true", "1", "t"])
        chunk["author_id"] = chunk["author_id"].astype(str).str.strip()
        chunk["text"] = chunk["text"].fillna("").astype(str)

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

            if not inbound and author in TARGET_BRANDS:
                brand_tweet_ids[author].add(tid)

    print("[*] Reconstructing conversation threads for target brands...")
    
    # Children linkage
    children_map = defaultdict(list)
    for child_id, parent_id in parent_map.items():
        children_map[parent_id].append(child_id)

    brand_conversations = defaultdict(list)

    for brand in TARGET_BRANDS:
        seen_conv_roots = set()
        for b_tid in brand_tweet_ids[brand]:
            # Trace root of this thread
            curr = b_tid
            while curr in parent_map and parent_map[curr] in tweets_by_id:
                curr = parent_map[curr]
            
            if curr in seen_conv_roots:
                continue
            seen_conv_roots.add(curr)

            # Traverse full conversation tree from root
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

            # Keep 2-way conversations (must have customer inbound + brand outbound)
            has_cust = any(t["inbound"] for t in conv_tweets)
            has_brand = any(t["author_id"] == brand and not t["inbound"] for t in conv_tweets)

            if has_cust and has_brand:
                # Sort by tweet_id or timestamp
                brand_conversations[brand].append(conv_tweets)

    print("[*] Performing empirical content sampling and resolution quality profiling...")

    brand_results = {}
    np.random.seed(42)  # Reproducible seed

    for brand in TARGET_BRANDS:
        convs = brand_conversations[brand]
        total_available = len(convs)
        
        # Sample deterministically
        sample_size = min(samples_per_brand, total_available)
        sampled_indices = np.random.choice(total_available, size=sample_size, replace=False)
        sampled_convs = [convs[i] for i in sampled_indices]

        # Resolution Quality Metrics
        dm_redirect_count = 0
        actionable_count = 0
        customer_satisfied_count = 0
        unresolved_discontinued_count = 0
        account_specific_info_count = 0
        sensitive_situation_count = 0

        # Quality categories
        cat_resolved = 0
        cat_partially_resolved = 0
        cat_generic_dm = 0
        cat_unresolved = 0
        cat_unclear = 0

        # Topic/Keyword counter for customer messages
        customer_keywords = Counter()

        sensitive_keywords = [
            "refund", "money", "charged", "billing", "stolen", "hacked",
            "cancel", "fraud", "scam", "flight", "delay", "passport",
            "safety", "accident", "crash", "driver", "locked out"
        ]

        account_info_keywords = [
            "account", "email", "phone", "order number", "order #",
            "tracking", "confirmation", "receipt", "credit card", "passport", "policy #"
        ]

        for conv in sampled_convs:
            cust_texts = [t["text"] for t in conv if t["inbound"]]
            brand_texts = [t["text"] for t in conv if t["author_id"] == brand and not t["inbound"]]

            full_cust_str = " ".join(cust_texts).lower()
            full_brand_str = " ".join(brand_texts).lower()

            # DM redirect check
            has_dm = any(is_dm_redirect(t) for t in brand_texts)
            if has_dm:
                dm_redirect_count += 1

            # Actionable info check
            has_actionable = any(is_actionable(t) for t in brand_texts)
            if has_actionable:
                actionable_count += 1

            # Customer satisfaction check
            last_cust_msg = cust_texts[-1] if cust_texts else ""
            satisfied = is_customer_satisfied(last_cust_msg)
            if satisfied:
                customer_satisfied_count += 1

            # Sensitive situation check
            if any(k in full_cust_str for k in sensitive_keywords):
                sensitive_situation_count += 1

            # Account info check
            if any(k in full_cust_str for k in account_info_keywords):
                account_specific_info_count += 1

            # Categorize Resolution Quality
            if satisfied and (has_actionable or not has_dm):
                cat_resolved += 1
            elif has_dm:
                cat_generic_dm += 1
            elif has_actionable:
                cat_partially_resolved += 1
            elif len(conv) <= 2 and not satisfied:
                cat_unresolved += 1
            else:
                cat_unclear += 1

            # Words for intent topics
            words = [w.strip("#@,.!?").lower() for w in full_cust_str.split() if len(w) > 3]
            customer_keywords.update(words)

        brand_results[brand] = {
            "total_available_2way_conversations": total_available,
            "sampled_conversations_analyzed": sample_size,
            "resolution_breakdown": {
                "clearly_resolved_actionable_pct": round(cat_resolved / sample_size * 100, 1),
                "partially_resolved_pct": round(cat_partially_resolved / sample_size * 100, 1),
                "generic_low_info_dm_redirect_pct": round(cat_generic_dm / sample_size * 100, 1),
                "unresolved_discontinued_pct": round(cat_unresolved / sample_size * 100, 1),
                "unclear_escalated_pct": round(cat_unclear / sample_size * 100, 1)
            },
            "key_pattern_stats": {
                "dm_redirect_pct": round(dm_redirect_count / sample_size * 100, 1),
                "actionable_info_pct": round(actionable_count / sample_size * 100, 1),
                "customer_satisfied_signal_pct": round(customer_satisfied_count / sample_size * 100, 1),
                "account_specific_info_pct": round(account_specific_info_count / sample_size * 100, 1),
                "sensitive_high_risk_pct": round(sensitive_situation_count / sample_size * 100, 1)
            },
            "top_customer_topic_keywords": customer_keywords.most_common(20),
            "sample_conversations_preview": [
                {
                    "conv_id": idx,
                    "thread_len": len(c),
                    "messages": [{"author": t["author_id"], "inbound": t["inbound"], "text": t["text"]} for t in c]
                }
                for idx, c in enumerate(sampled_convs[:5])  # First 5 full sample previews
            ]
        }

    # Save output report JSON
    os.makedirs("reports", exist_ok=True)
    out_json = "reports/top_brands_deep_sample.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(brand_results, f, indent=2)
    print(f"[+] Saved deep sample empirical analysis to: {out_json}")

    return brand_results


def main():
    csv_path = "data/raw/twcs.csv"
    if not os.path.exists(csv_path):
        print(f"[ERROR] {csv_path} not found.")
        sys.exit(1)
        
    sample_and_analyze_brands(csv_path, samples_per_brand=500)


if __name__ == "__main__":
    main()
