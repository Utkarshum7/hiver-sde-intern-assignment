import os
import sys
import tempfile
import pandas as pd
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.profiler import inspect_schema, DatasetProfiler


@pytest.fixture
def sample_csv():
    """Creates a temporary CSV file mimicking twcs.csv format."""
    data = """tweet_id,author_id,inbound,created_at,text,response_tweet_id,in_response_to_tweet_id
1,AppleSupport,False,Tue Oct 10 21:54:19 +0000 2017,@115712 Hi there!,2,
2,115712,True,Tue Oct 10 21:55:00 +0000 2017,My phone screen is black and won't turn on,3,1
3,AppleSupport,False,Tue Oct 10 21:56:00 +0000 2017,@115712 Please send us a DM with your iOS version.,,2
4,AmazonHelp,False,Tue Oct 10 21:50:00 +0000 2017,@115713 We can help with your order!,5,
5,115713,True,Tue Oct 10 21:51:00 +0000 2017,Where is my package?,6,4
6,AmazonHelp,False,Tue Oct 10 21:52:00 +0000 2017,@115713 Please check your email for tracking info.,,5
"""
    with tempfile.NamedTemporaryFile("w", delete=False, suffix=".csv") as f:
        f.write(data)
        temp_path = f.name
    yield temp_path
    if os.path.exists(temp_path):
        os.remove(temp_path)


def test_inspect_schema(sample_csv):
    schema = inspect_schema(sample_csv)
    assert schema["num_columns"] == 7
    assert "tweet_id" in schema["columns"]
    assert "author_id" in schema["columns"]
    assert "inbound" in schema["columns"]
    assert "text" in schema["columns"]


def test_dataset_profiler(sample_csv):
    profiler = DatasetProfiler(chunk_size=2)
    for chunk in pd.read_csv(sample_csv, chunksize=2, dtype=str):
        profiler.process_chunk(chunk)
        
    profile = profiler.finalize_profile()
    
    summary = profile["dataset_summary"]
    assert summary["total_messages"] == 6
    assert summary["inbound_messages"] == 2
    assert summary["outbound_messages"] == 4
    assert summary["unique_authors"] == 4  # AppleSupport, 115712, AmazonHelp, 115713
    assert summary["duplicate_tweet_ids"] == 0
    
    # Check top brands
    candidates = profile["top_brand_candidates"]
    brand_ids = [c["brand_id"] for c in candidates]
    assert "AppleSupport" in brand_ids
    assert "AmazonHelp" in brand_ids
    
    # Check conversation stats
    assert profile["conversation_stats"]["total_conversations"] > 0
