"""
Dataset Profiling Module for Customer Support on Twitter Dataset.
Designed to process large CSV files (>500MB) efficiently using chunked streaming
without loading the entire dataset into memory at once.
"""

import json
from collections import Counter, defaultdict
from typing import Dict, Any, Tuple, Optional, Set
import pandas as pd
import numpy as np


def inspect_schema(csv_path: str) -> Dict[str, Any]:
    """Inspects the CSV header and sample row to determine exact schema."""
    sample = pd.read_csv(csv_path, nrows=5, dtype=str)
    return {
        "columns": list(sample.columns),
        "num_columns": len(sample.columns),
        "sample_row": sample.iloc[0].to_dict() if len(sample) > 0 else {}
    }


class DatasetProfiler:
    """Accumulates dataset statistics in chunks to minimize memory footprint."""
    
    def __init__(self, chunk_size: int = 100000):
        self.chunk_size = chunk_size
        
        # Overall metrics
        self.total_messages = 0
        self.inbound_count = 0
        self.outbound_count = 0
        self.missing_counts = defaultdict(int)
        
        # Unique tracking
        self.seen_tweet_ids: Set[str] = set()
        self.duplicate_tweet_ids = 0
        self.unique_authors: Set[str] = set()
        
        # Text statistics
        self.char_lengths = []
        self.word_lengths = []
        
        # Date range tracking
        self.min_date: Optional[str] = None
        self.max_date: Optional[str] = None
        
        # Brand candidate tracking
        # brand_name -> {inbound: int, outbound: int, authors: set, conversations: set}
        self.brand_outbound_counts = Counter()
        self.brand_inbound_counts = Counter()
        
        # Thread & Conversation tracking
        # parent_map: child_tweet_id -> parent_tweet_id
        # tweet_author: tweet_id -> author_id
        # tweet_inbound: tweet_id -> inbound (bool)
        self.parent_map: Dict[str, str] = {}
        self.tweet_author: Dict[str, str] = {}
        self.tweet_inbound: Dict[str, bool] = {}

    def process_chunk(self, chunk: pd.DataFrame):
        """Processes a single DataFrame chunk."""
        self.total_messages += len(chunk)
        
        # Missing values
        for col in chunk.columns:
            self.missing_counts[col] += int(chunk[col].isna().sum())
            
        # Standardize types
        chunk["tweet_id"] = chunk["tweet_id"].astype(str).str.strip()
        chunk["inbound"] = chunk["inbound"].astype(str).str.lower().isin(["true", "1", "t"])
        chunk["author_id"] = chunk["author_id"].astype(str).str.strip()
        chunk["text"] = chunk["text"].fillna("").astype(str)
        
        # Duplicate tweet_ids
        for tid in chunk["tweet_id"]:
            if tid in self.seen_tweet_ids:
                self.duplicate_tweet_ids += 1
            else:
                self.seen_tweet_ids.add(tid)
                
        # Unique authors
        self.unique_authors.update(chunk["author_id"].unique())
        
        # Inbound vs Outbound
        inbound_mask = chunk["inbound"]
        self.inbound_count += int(inbound_mask.sum())
        self.outbound_count += int((~inbound_mask).sum())
        
        # Track brands (outbound authors are support handles in twcs dataset)
        outbound_df = chunk[~inbound_mask]
        for brand in outbound_df["author_id"]:
            self.brand_outbound_counts[brand] += 1
            
        # Message lengths
        char_lens = chunk["text"].str.len().values
        word_lens = chunk["text"].str.split().str.len().values
        self.char_lengths.extend(char_lens)
        self.word_lengths.extend(word_lens)
        
        # Dates (if created_at present)
        if "created_at" in chunk.columns:
            dates = pd.to_datetime(chunk["created_at"], errors="coerce", format="mixed")
            valid_dates = dates.dropna()
            if not valid_dates.empty:
                c_min = str(valid_dates.min())
                c_max = str(valid_dates.max())
                if self.min_date is None or c_min < self.min_date:
                    self.min_date = c_min
                if self.max_date is None or c_max > self.max_date:
                    self.max_date = c_max

        # Populate thread linkages for conversation reconstruction
        for _, row in chunk.iterrows():
            tid = row["tweet_id"]
            author = row["author_id"]
            inbound = row["inbound"]
            parent_id = str(row.get("in_response_to_tweet_id", "")).strip()
            
            self.tweet_author[tid] = author
            self.tweet_inbound[tid] = inbound
            
            if parent_id and parent_id.lower() not in ["nan", "none", "", "<na>"]:
                self.parent_map[tid] = parent_id

    def finalize_profile(self) -> Dict[str, Any]:
        """Calculates final aggregated metrics and thread metrics."""
        # Message length statistics
        char_arr = np.array(self.char_lengths) if self.char_lengths else np.array([0])
        word_arr = np.array(self.word_lengths) if self.word_lengths else np.array([0])
        
        char_stats = {
            "min": int(np.min(char_arr)),
            "max": int(np.max(char_arr)),
            "mean": round(float(np.mean(char_arr)), 2),
            "median": float(np.median(char_arr)),
            "p95": float(np.percentile(char_arr, 95))
        }
        
        word_stats = {
            "min": int(np.min(word_arr)),
            "max": int(np.max(word_arr)),
            "mean": round(float(np.mean(word_arr)), 2),
            "median": float(np.median(word_arr)),
            "p95": float(np.percentile(word_arr, 95))
        }
        
        # Thread / Conversation reconstruction
        threads, thread_authors = self._reconstruct_threads()
        thread_lengths = [len(t) for t in threads]
        
        thread_stats = {
            "total_conversations": len(threads),
            "min_length": int(np.min(thread_lengths)) if thread_lengths else 0,
            "max_length": int(np.max(thread_lengths)) if thread_lengths else 0,
            "mean_length": round(float(np.mean(thread_lengths)), 2) if thread_lengths else 0,
            "median_length": float(np.median(thread_lengths)) if thread_lengths else 0
        }
        
        # Brand Candidate Analysis (Top 15 support accounts)
        top_brands_info = self._analyze_brand_candidates(threads, thread_authors)
        
        profile = {
            "dataset_summary": {
                "total_messages": self.total_messages,
                "inbound_messages": self.inbound_count,
                "outbound_messages": self.outbound_count,
                "unique_authors": len(self.unique_authors),
                "duplicate_tweet_ids": self.duplicate_tweet_ids,
                "date_range": {
                    "start": self.min_date,
                    "end": self.max_date
                }
            },
            "missing_values_by_column": dict(self.missing_counts),
            "message_length_stats": {
                "characters": char_stats,
                "words": word_stats
            },
            "conversation_stats": thread_stats,
            "top_brand_candidates": top_brands_info
        }
        return profile

    def _reconstruct_threads(self) -> Tuple[list, list]:
        """Reconstructs threads from parent linkage graph."""
        # Children map: parent_id -> list of child_ids
        children_map = defaultdict(list)
        for child_id, parent_id in self.parent_map.items():
            children_map[parent_id].append(child_id)
            
        # Find root tweets (tweets that have no parent or parent is missing from dataset)
        all_tweets = set(self.tweet_author.keys())
        parent_tweet_ids = set(self.parent_map.values())
        root_candidates = (all_tweets - set(self.parent_map.keys())) | (parent_tweet_ids - all_tweets)
        
        threads = []
        thread_authors = []
        visited = set()
        
        for root in root_candidates:
            if root not in children_map and root not in self.tweet_author:
                continue
            
            # Simple BFS/DFS traversal from root to collect thread
            stack = [root]
            thread_tweets = []
            authors = set()
            
            while stack:
                curr = stack.pop()
                if curr in visited:
                    continue
                visited.add(curr)
                
                if curr in self.tweet_author:
                    thread_tweets.append(curr)
                    authors.add(self.tweet_author[curr])
                    
                for child in children_map.get(curr, []):
                    if child not in visited:
                        stack.append(child)
                        
            if thread_tweets:
                threads.append(thread_tweets)
                thread_authors.append(authors)
                
        return threads, thread_authors

    def _analyze_brand_candidates(self, threads: list, thread_authors: list) -> list:
        """Analyzes candidate support accounts / brands."""
        top_brands = [brand for brand, _ in self.brand_outbound_counts.most_common(15)]
        
        brand_stats = []
        for brand in top_brands:
            outbound_vol = self.brand_outbound_counts[brand]
            
            # Count conversations involving brand
            brand_conv_count = 0
            two_way_conv_count = 0
            total_conv_len = 0
            
            for t_idx, authors in enumerate(thread_authors):
                if brand in authors:
                    brand_conv_count += 1
                    thread = threads[t_idx]
                    total_conv_len += len(thread)
                    
                    # Check if thread has both customer (inbound) and support (outbound brand)
                    has_customer = any(self.tweet_inbound.get(tid, False) for tid in thread)
                    has_brand = any(self.tweet_author.get(tid) == brand and not self.tweet_inbound.get(tid, True) for tid in thread)
                    
                    if has_customer and has_brand:
                        two_way_conv_count += 1
                        
            brand_stats.append({
                "brand_id": brand,
                "outbound_support_messages": outbound_vol,
                "total_conversations_involved": brand_conv_count,
                "two_way_conversations": two_way_conv_count,
                "avg_conversation_length": round(total_conv_len / brand_conv_count, 2) if brand_conv_count > 0 else 0
            })
            
        return brand_stats
