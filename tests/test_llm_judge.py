"""Tests for src/llm_judge.py — must never fabricate scores when unconfigured."""

import json
import os

import pytest

from src import llm_judge


def test_judge_reply_returns_pending_without_api_key(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    result = llm_judge.judge_reply(
        customer_message="How much for a bag?",
        evidence=[],
        reply="Bags are $30.",
    )
    assert result["status"] == "pending_no_api_key"
    assert result["scores"] is None
    assert result["aggregate"] is None
    assert result["rubric_version"] == llm_judge.RUBRIC_VERSION


def test_parse_json_scores_valid():
    raw = '{"relevance": 5, "correctness": 4, "groundedness": 5, "helpfulness": 4, "safety": 5, "rationale": "good"}'
    parsed = llm_judge._parse_json_scores(raw)
    assert parsed["relevance"] == 5
    assert parsed["safety"] == 5


def test_parse_json_scores_rejects_out_of_range():
    raw = '{"relevance": 9, "correctness": 4, "groundedness": 5, "helpfulness": 4, "safety": 5}'
    assert llm_judge._parse_json_scores(raw) is None


def test_parse_json_scores_rejects_missing_key():
    raw = '{"relevance": 5, "correctness": 4}'
    assert llm_judge._parse_json_scores(raw) is None


def test_parse_json_scores_handles_none_or_garbage():
    assert llm_judge._parse_json_scores(None) is None
    assert llm_judge._parse_json_scores("not json at all") is None


def test_judge_reply_uses_cache_on_second_call(monkeypatch, tmp_path):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "fake-key-for-test")

    call_count = {"n": 0}

    def fake_complete(prompt, system=None, max_tokens=300):
        call_count["n"] += 1
        return '{"relevance": 5, "correctness": 5, "groundedness": 5, "helpfulness": 5, "safety": 5, "rationale": "ok"}'

    monkeypatch.setattr(llm_judge.llm_client, "complete", fake_complete)
    cache_path = str(tmp_path / "cache.json")

    r1 = llm_judge.judge_reply("msg", [], "reply text", cache_path=cache_path)
    r2 = llm_judge.judge_reply("msg", [], "reply text", cache_path=cache_path)

    assert r1["status"] == "ok"
    assert r2["status"] == "cached"
    assert call_count["n"] == 1  # second call served from cache, no re-call
    assert r2["aggregate"] == 5.0
