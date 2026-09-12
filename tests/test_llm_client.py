"""Tests for src/llm_client.py — must never fabricate a completion when unconfigured."""

import os

import pytest

from src import llm_client


def test_is_available_false_without_env_var(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    assert llm_client.is_available() is False


def test_is_available_true_with_env_var(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "fake-key-for-test")
    assert llm_client.is_available() is True


def test_complete_returns_none_when_not_configured(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    result = llm_client.complete("Hello")
    assert result is None  # must not fabricate a fake completion
