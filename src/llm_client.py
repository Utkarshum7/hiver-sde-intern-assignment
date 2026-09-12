"""
Thin optional LLM client wrapper.

Used by src/llm_judge.py (reply-quality judging) and, optionally, as a
higher-fluency alternative to src/reply_generator.py's deterministic
template generator. Never required for the pipeline to run — every caller
must handle `is_available() == False` gracefully.

Reads the API key from the ANTHROPIC_API_KEY environment variable ONLY.
NEVER hardcode a key here, never log the key value, never commit one.

This module makes no network calls unless a caller explicitly invokes
complete(). Importing it costs nothing and requires no key.
"""

import os
from typing import Optional

MODEL_ID = "claude-sonnet-5"


def is_available() -> bool:
    """True if an API key is present in the environment."""
    return bool(os.environ.get("ANTHROPIC_API_KEY", "").strip())


def complete(prompt: str, system: Optional[str] = None, max_tokens: int = 512) -> Optional[str]:
    """
    Calls the Anthropic API if configured. Returns the text completion, or
    None if no API key is configured (caller must fall back to a
    deterministic path — never fabricate a substitute for a real LLM call).

    Raises RuntimeError only for genuine API-level failures (network error,
    invalid key, rate limit) so callers can distinguish "not configured"
    (expected, silent fallback) from "configured but failed" (surface it).
    """
    if not is_available():
        return None

    try:
        import anthropic  # imported lazily so the dependency is optional
    except ImportError as e:
        raise RuntimeError(
            "ANTHROPIC_API_KEY is set but the 'anthropic' package is not "
            "installed. Run: pip install anthropic"
        ) from e

    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    kwargs = {}
    if system:
        kwargs["system"] = system
    response = client.messages.create(
        model=MODEL_ID,
        max_tokens=max_tokens,
        messages=[{"role": "user", "content": prompt}],
        **kwargs,
    )
    return "".join(
        block.text for block in response.content if getattr(block, "type", None) == "text"
    )
