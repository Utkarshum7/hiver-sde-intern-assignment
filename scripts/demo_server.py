"""
Minimal local demo server for the Delta Customer Support AI Agent.

This is a thin, read-only demonstration wrapper around the existing agent
pipeline (`src/agent.py`) — it does NOT change any agent, retrieval,
escalation, or evaluation logic. It exists only so the already-built
pipeline can be exercised from a browser instead of the Python REPL.

`DeltaSupportAgent` is loaded exactly ONCE, at server startup (not per
request) — model/index deserialization takes ~17 seconds; every request
after that is a normal ~15ms `agent.handle(message)` call.

Usage:
    python -m uvicorn scripts.demo_server:app --reload

Then open http://127.0.0.1:8000 in a browser.

OFFLINE DEMONSTRATION ONLY: this demo (like the underlying agent) never
performs any live Delta action — no flight-status lookup, no booking/PNR
change, no refund, no baggage trace, no account access. Every reply is
either grounded in retrieved historical evidence / static policy facts, or
an escalation notice. See docs/escalation_policy.md for the full capability
boundary, which this demo inherits unchanged.
"""

import os
import re
import sys
from contextlib import asynccontextmanager
from typing import Dict, List, Optional, Tuple

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field, field_validator

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.agent import DeltaSupportAgent  # noqa: E402  (after sys.path fix-up)
from src.reply_generator import generate_reply as _generate_reply  # noqa: E402

# Module-level singleton. Populated once by the lifespan startup hook below
# (or lazily by get_agent() if something calls it first, e.g. tests).
_agent: Optional[DeltaSupportAgent] = None


# ---------------------------------------------------------------------------
# DEMO-ONLY safety override
#
# IMPORTANT — SCOPE: everything in this section runs only inside this demo
# server. It never touches src/agent.py, src/classifier.py,
# src/escalation.py, or anything scripts/evaluate.py depends on, so it has
# ZERO effect on the golden-set evaluation results already recorded in
# reports/evaluation_results.json / reports/evaluation_raw_predictions.json.
# Those numbers reflect the real, unmodified agent and stay exactly as
# measured. This override exists purely as an extra, disclosed safety net
# in the interactive demo, layered on top of (not instead of) the real
# agent's own escalation decision.
#
# It exists because a real, already-documented weakness (REPORT.md §16
# failure mode 3: the weak-supervised classifier over-predicts
# flight_status_inquiry) can cause a baggage-claim / account-specific
# message to be auto-handled with an unrelated policy reply — exactly what
# happened for "My suitcase came out with a broken wheel, how do I file a
# claim?". Rather than silently patching the evaluated classifier (which
# would invalidate the measured numbers without a re-run), the demo adds a
# narrow, deterministic, fully-transparent second check.
# ---------------------------------------------------------------------------

# Each entry is (regex pattern, human-readable label for the reason text).
# Word-boundary matched, case-insensitive — deterministic substring
# detection only, no fuzzy/semantic matching, no ML.
#
# DESIGN NOTE (narrowed after review): an earlier version of this list used
# bare single words like "report", "missing", "account", "damaged", "lost",
# "claim", "broken" — each of those is common enough in ordinary,
# non-risky messages ("I want to report how great my flight was", "my
# account has great SkyMiles rewards") that it would over-trigger. Every
# entry below is now a context-specific phrase instead: either a fixed
# multi-word phrase, or a baggage noun (bag/baggage/luggage/suitcase)
# paired with a risk adjective — matched in BOTH natural word orders
# ("damaged bag" and "bag was damaged"), since real customer messages use
# both about equally often, and only the paired form counts as a match.
_BAGGAGE_NOUN = r"(?:bag|baggage|luggage|suitcase)"
_OPT_POSSESSIVE = r"(?:my\s+|the\s+|a\s+|an\s+|our\s+)?"
_LINKING_VERB = r"(?:is|was|got|came|arrived)\s+(?:badly\s+)?"


def _adjective_noun_pair(adjective: str, label: str) -> List[Tuple[str, str]]:
    """Matches '<adjective> [possessive] <baggage noun>' and
    '<baggage noun> [is/was/got] <adjective>' — both real phrasing orders."""
    return [
        (rf"\b{adjective}\s+{_OPT_POSSESSIVE}{_BAGGAGE_NOUN}\b", label),
        (rf"\b{_BAGGAGE_NOUN}\s+{_LINKING_VERB}{adjective}\b", label),
    ]


_SAFETY_OVERRIDE_TERMS: List[Tuple[str, str]] = [
    (r"\bfile (?:a |an )?claim\b", "file a claim"),
    (r"\bbaggage claim\b", "baggage claim"),
    *_adjective_noun_pair("lost", "lost baggage"),
    *_adjective_noun_pair("missing", "missing luggage"),
    *_adjective_noun_pair("damaged", "damaged suitcase"),
    (rf"\bbroken\s+{_OPT_POSSESSIVE}(?:wheel|handle|zipper|{_BAGGAGE_NOUN})\b", "broken wheel"),
    (rf"\b(?:wheel|handle|zipper|{_BAGGAGE_NOUN})\s+{_LINKING_VERB}broken\b", "broken wheel"),
    (r"\brefund my (?:booking|ticket|flight|reservation)\b", "refund my booking"),
    (r"\brefund request\b", "refund request"),
    (r"\bbooking reference\b", "booking reference"),
    (r"\breservation number\b", "reservation number"),
    (r"\breimbursement\b", "reimbursement"),
]

SAFETY_OVERRIDE_REASON_PREFIX = "Safety override:"


def _match_safety_override_terms(message: str) -> List[str]:
    """Returns the (deduplicated, in definition order) list of matched term
    labels for a message, or an empty list if none matched."""
    lowered = (message or "").lower()
    matched = []
    for pattern, label in _SAFETY_OVERRIDE_TERMS:
        if re.search(pattern, lowered) and label not in matched:
            matched.append(label)
    return matched


def apply_demo_safety_override(customer_message: str, result: Dict) -> Dict:
    """
    If the real agent did NOT escalate but the message contains
    claim/account-specific language, force escalation and replace the reply
    with a safe escalation notice — never a confident policy answer for a
    claim or account-specific action.

    `result` is exactly what DeltaSupportAgent.handle() returned; this
    function does not mutate it, it returns a (possibly) new dict.
    """
    if result.get("escalate") == "yes":
        return result  # already escalating for a real reason; nothing to override

    matched = _match_safety_override_terms(customer_message)
    if not matched:
        return result

    reason = (
        f"{SAFETY_OVERRIDE_REASON_PREFIX} message contains claim/account-specific "
        f"language ({', '.join(matched)}) that the automatic classifier may "
        "misroute; escalating to a human rather than risking an unrelated "
        "policy answer."
    )
    gen = _generate_reply(
        customer_message=customer_message,
        intent=result.get("intent", "general_complaint_feedback"),
        evidence=[],  # do not ground an overridden reply in possibly-mismatched evidence
        escalate=True,
        escalation_reason=reason,
    )

    overridden = dict(result)
    overridden["escalate"] = "yes"
    overridden["escalation_reason"] = reason
    overridden["reply"] = gen["reply"]
    return overridden


def get_agent() -> DeltaSupportAgent:
    """Returns the process-wide DeltaSupportAgent, loading it once if needed."""
    global _agent
    if _agent is None:
        _agent = DeltaSupportAgent.load()
    return _agent


def _load_agent_on_startup() -> None:
    # Pay the ~17s classifier/retrieval-index load cost once, before the
    # server starts accepting traffic — not on the first incoming request.
    get_agent()


@asynccontextmanager
async def _lifespan(_app: FastAPI):
    _load_agent_on_startup()
    yield


app = FastAPI(title="Delta Support Agent Demo", lifespan=_lifespan)


# ---------------------------------------------------------------------------
# Request/response contracts
# ---------------------------------------------------------------------------

class HandleRequest(BaseModel):
    message: str = Field(..., min_length=1, description="Customer message text.")

    @field_validator("message")
    @classmethod
    def not_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("message must not be blank")
        return v


class HandleResponse(BaseModel):
    intent: str
    reply: str
    escalate: str
    escalation_reason: str
    evidence: List[dict]


@app.post("/handle", response_model=HandleResponse)
def handle(req: HandleRequest) -> HandleResponse:
    agent = get_agent()
    result = agent.handle(req.message)
    # DEMO-ONLY safety net layered on top of the real agent's decision — see
    # the "DEMO-ONLY safety override" section above for exactly why and what
    # this does and does not affect. Never applied to scripts/evaluate.py.
    result = apply_demo_safety_override(req.message, result)
    # Expose exactly the agent's existing output fields — no relabeling,
    # no added/invented fields, no logic here beyond picking these 5 keys.
    return HandleResponse(
        intent=result["intent"],
        reply=result["reply"],
        escalate=result["escalate"],
        escalation_reason=result["escalation_reason"],
        evidence=result["evidence"],
    )


# ---------------------------------------------------------------------------
# Inline HTML page (no external assets, no CDN — works fully offline)
# ---------------------------------------------------------------------------

_HTML_PAGE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Delta Support Intelligence — Local Demo</title>
<style>
  :root {
    --bg: #f2f3f5;
    --surface: #ffffff;
    --sidebar-bg: #fbfbfc;
    --border: #e3e5ea;
    --border-strong: #d3d6dd;
    --text: #1a1d24;
    --text-muted: #5b6270;
    --text-faint: #8992a3;
    --accent: #c8102e;
    --accent-dark: #a10d26;
    --accent-tint: #fdeaec;
    --focus: #2563eb;
    --safe-bg: #e7f6ec;
    --safe-border: #a9dfba;
    --safe-text: #146c34;
    --warn-bg: #fdecec;
    --warn-border: #f3b4b4;
    --warn-text: #a1121f;
    --radius: 12px;
    --radius-sm: 8px;
    --shadow: 0 1px 2px rgba(16,24,40,.04), 0 2px 6px rgba(16,24,40,.05);
    --shadow-lg: 0 8px 24px rgba(16,24,40,.12);
  }
  * { box-sizing: border-box; }
  html, body { margin: 0; padding: 0; height: 100%; }
  body {
    background: var(--bg);
    color: var(--text);
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    -webkit-font-smoothing: antialiased;
  }
  :focus-visible { outline: 2px solid var(--focus); outline-offset: 2px; }
  button { font-family: inherit; }

  /* ---------- Header ---------- */
  header.topbar {
    background: var(--surface);
    border-bottom: 1px solid var(--border);
    padding: 0.75rem 1.5rem;
    display: flex;
    align-items: center;
    gap: 0.75rem;
    position: sticky;
    top: 0;
    z-index: 20;
  }
  .brand-mark {
    width: 38px; height: 38px; border-radius: 10px; flex-shrink: 0;
    background: linear-gradient(155deg, var(--accent), var(--accent-dark));
    display: flex; align-items: center; justify-content: center;
    box-shadow: var(--shadow);
  }
  .brand-text h1 { margin: 0; font-size: 1.08rem; font-weight: 800; line-height: 1.2; letter-spacing: -0.01em; }
  .brand-text p { margin: 0.15rem 0 0; font-size: 0.78rem; color: var(--text-muted); }
  .status-badge {
    margin-left: auto; flex-shrink: 0;
    display: inline-flex; align-items: center; gap: 0.35rem;
    background: #f1f2f5; border: 1px solid var(--border); color: var(--text-muted);
    font-size: 0.72rem; font-weight: 700; padding: 0.32rem 0.65rem; border-radius: 999px;
    white-space: nowrap;
  }
  .status-dot { width: 7px; height: 7px; border-radius: 50%; background: #16a34a; }

  /* ---------- App shell: sidebar + main ---------- */
  .app-body { display: flex; align-items: flex-start; }

  .sidebar {
    width: 216px; flex-shrink: 0;
    background: var(--sidebar-bg); border-right: 1px solid var(--border);
    padding: 1.1rem 0.8rem; position: sticky; top: 57px;
    height: calc(100vh - 57px);
    display: flex; flex-direction: column;
  }
  .nav-group-label {
    font-size: 0.68rem; font-weight: 700; text-transform: uppercase; letter-spacing: .05em;
    color: var(--text-faint); margin: 1.1rem 0.6rem 0.4rem;
  }
  .nav-group-label:first-child { margin-top: 0; }
  .nav-item {
    width: 100%; display: flex; align-items: center; gap: 0.55rem;
    background: transparent; border: 1px solid transparent; border-radius: var(--radius-sm);
    padding: 0.55rem 0.6rem; margin-bottom: 0.2rem;
    font-size: 0.85rem; font-weight: 600; color: var(--text-muted);
    cursor: pointer; text-align: left;
  }
  .nav-item svg { flex-shrink: 0; }
  .nav-item:hover { background: #eef0f4; color: var(--text); }
  .nav-item.active {
    background: var(--accent-tint); color: var(--accent-dark); border-color: #f7c9d0;
  }
  .sidebar-footer {
    margin-top: auto; padding-top: 0.9rem; border-top: 1px solid var(--border);
  }
  .sidebar-footer-badge {
    font-size: 0.7rem; color: var(--text-faint); font-weight: 600;
    display: flex; align-items: center; gap: 0.35rem;
  }

  main {
    flex: 1; min-width: 0; max-width: 980px; margin: 0 auto; padding: 1.5rem;
    display: grid; grid-template-columns: 1fr 1fr; gap: 1.25rem; align-items: start;
  }

  @media (max-width: 860px) {
    .app-body { flex-direction: column; }
    .sidebar {
      width: 100%; height: auto; position: static;
      flex-direction: row; overflow-x: auto; gap: 0.4rem;
      padding: 0.7rem 0.8rem;
    }
    .nav-group-label { display: none; }
    .nav-item { width: auto; white-space: nowrap; margin-bottom: 0; }
    .sidebar-footer { display: none; }
    main { grid-template-columns: 1fr; padding: 1rem; max-width: 100%; }
  }

  /* ---------- Panels ---------- */
  .panel {
    background: var(--surface); border: 1px solid var(--border); border-radius: var(--radius);
    box-shadow: var(--shadow); padding: 1.3rem;
  }
  .panel h2 { margin: 0 0 0.9rem; font-size: 0.95rem; font-weight: 700; }

  label.field-label {
    display: block; font-size: 0.8rem; font-weight: 600; color: var(--text-muted);
    margin-bottom: 0.4rem;
  }
  textarea#message {
    width: 100%; min-height: 140px; font-size: 0.95rem; font-family: inherit;
    padding: 0.7rem 0.8rem; border: 1px solid var(--border); border-radius: var(--radius-sm);
    resize: vertical; line-height: 1.4;
  }
  .composer-meta {
    display: flex; justify-content: space-between; align-items: center;
    margin-top: 0.4rem; font-size: 0.75rem; color: var(--text-faint);
  }

  .examples { margin: 1rem 0; }
  .examples-label { font-size: 0.75rem; font-weight: 600; color: var(--text-muted); margin-bottom: 0.5rem; }
  .examples-grid { display: grid; grid-template-columns: 1fr; gap: 0.5rem; }
  .example-card {
    font-family: inherit; text-align: left;
    display: flex; align-items: center; gap: 0.6rem;
    background: var(--surface); border: 1px solid var(--border); color: var(--text);
    font-size: 0.83rem; font-weight: 600; padding: 0.55rem 0.7rem; border-radius: var(--radius-sm);
    cursor: pointer; transition: box-shadow .12s, border-color .12s;
  }
  .example-card:hover { border-color: var(--border-strong); box-shadow: var(--shadow); }
  .example-card .dot { width: 7px; height: 7px; border-radius: 50%; background: var(--accent); flex-shrink: 0; }

  .composer-actions { display: flex; gap: 0.6rem; margin-top: 1.1rem; }
  button.btn {
    font-size: 0.9rem; font-weight: 700;
    padding: 0.6rem 1.1rem; border-radius: var(--radius-sm); border: 1px solid transparent;
    cursor: pointer; display: inline-flex; align-items: center; gap: 0.45rem;
    min-height: 40px;
  }
  button.btn-primary { background: var(--accent); color: white; }
  button.btn-primary:hover:not(:disabled) { background: var(--accent-dark); }
  button.btn-primary:disabled { background: #d7a3ac; cursor: default; }
  button.btn-secondary { background: var(--surface); color: var(--text); border-color: var(--border); }
  button.btn-secondary:hover { background: #f1f2f5; }

  .spinner {
    width: 14px; height: 14px; border-radius: 50%;
    border: 2px solid rgba(255,255,255,.5); border-top-color: white;
    animation: spin .7s linear infinite; flex-shrink: 0;
  }
  @keyframes spin { to { transform: rotate(360deg); } }

  #alert-box {
    display: none; margin-top: 1rem;
    background: var(--warn-bg); border: 1px solid var(--warn-border); color: var(--warn-text);
    padding: 0.7rem 0.9rem; border-radius: var(--radius-sm); font-size: 0.85rem;
  }

  /* ---------- Empty state ---------- */
  #empty-state {
    color: var(--text-faint); font-size: 0.88rem; text-align: center; padding: 2rem 1rem;
    border: 1px dashed var(--border); border-radius: var(--radius-sm);
  }
  #empty-state svg { margin-bottom: 0.9rem; }
  #empty-state p { margin: 0; max-width: 28ch; margin-left: auto; margin-right: auto; }
  #result { display: none; }

  .badge {
    display: inline-flex; align-items: center; font-size: 0.78rem; font-weight: 700;
    padding: 0.25rem 0.6rem; border-radius: 999px;
    background: #eef0f4; color: var(--text); border: 1px solid var(--border);
    text-transform: capitalize;
  }
  .decision-banner {
    display: flex; align-items: center; gap: 0.5rem;
    padding: 0.7rem 0.9rem; border-radius: var(--radius-sm);
    font-weight: 700; font-size: 0.9rem; margin: 0.5rem 0; border: 1px solid;
  }
  .decision-banner.safe { background: var(--safe-bg); border-color: var(--safe-border); color: var(--safe-text); }
  .decision-banner.escalate { background: var(--warn-bg); border-color: var(--warn-border); color: var(--warn-text); }

  .override-badge {
    display: inline-flex; align-items: center; gap: 0.35rem;
    background: #fff4e0; border: 1px solid #f0c987; color: #8a5a00;
    font-size: 0.75rem; font-weight: 700; padding: 0.3rem 0.6rem; border-radius: 999px;
    margin: 0.3rem 0 0.6rem;
  }

  .reply-card {
    background: #fafbfc; border: 1px solid var(--border); border-radius: var(--radius-sm);
    padding: 0.9rem; font-size: 0.92rem; line-height: 1.5; white-space: pre-wrap;
  }
  .result-section { margin-bottom: 1.1rem; }
  .result-section:last-child { margin-bottom: 0; }
  .result-label {
    font-size: 0.75rem; font-weight: 700; text-transform: uppercase; letter-spacing: .03em;
    color: var(--text-muted); margin-bottom: 0.35rem;
  }
  .offline-note { font-size: 0.72rem; color: var(--text-faint); margin-top: 0.4rem; }

  details.evidence-item {
    border: 1px solid var(--border); border-radius: var(--radius-sm);
    margin-bottom: 0.5rem; overflow: hidden;
  }
  details.evidence-item summary {
    cursor: pointer; padding: 0.55rem 0.75rem; font-size: 0.83rem; font-weight: 600;
    background: #fafbfc; list-style: none; display: flex; justify-content: space-between; gap: 0.5rem;
  }
  details.evidence-item summary::-webkit-details-marker { display: none; }
  details.evidence-item .evidence-body { padding: 0.7rem 0.85rem; font-size: 0.83rem; border-top: 1px solid var(--border); }
  .evidence-score { color: var(--text-faint); font-weight: 500; white-space: nowrap; }
  .evidence-meta { color: var(--text-faint); font-size: 0.75rem; margin-bottom: 0.4rem; }

  footer {
    max-width: 980px; margin: 0 auto; padding: 1.2rem 1.5rem 2rem; font-size: 0.75rem;
    color: var(--text-faint); text-align: center; border-top: 1px solid var(--border); margin-top: 1rem;
  }

  /* ---------- Info modal (How It Works / About) ---------- */
  .modal-overlay {
    position: fixed; inset: 0; background: rgba(20,22,28,.45);
    display: flex; align-items: center; justify-content: center; padding: 1rem; z-index: 50;
  }
  .modal-overlay[hidden] { display: none; }
  .modal-card {
    background: var(--surface); border-radius: var(--radius); box-shadow: var(--shadow-lg);
    max-width: 460px; width: 100%; padding: 1.4rem; position: relative;
    max-height: 82vh; overflow-y: auto;
  }
  .modal-card h3 { margin: 0 0 0.7rem; font-size: 1.02rem; }
  .modal-card p, .modal-card li { font-size: 0.87rem; line-height: 1.55; color: var(--text-muted); }
  .modal-card ol { padding-left: 1.1rem; margin: 0.6rem 0; }
  .modal-close {
    position: absolute; top: 0.8rem; right: 0.8rem;
    width: 28px; height: 28px; border-radius: 50%; border: 1px solid var(--border);
    background: var(--surface); color: var(--text-muted); font-size: 1rem; line-height: 1;
    cursor: pointer; display: flex; align-items: center; justify-content: center;
  }
  .modal-close:hover { background: #f1f2f5; }
</style>
</head>
<body>
  <header class="topbar">
    <div class="brand-mark" aria-hidden="true">
      <svg width="20" height="20" viewBox="0 0 24 24" fill="none">
        <rect x="3" y="4" width="18" height="12" rx="3.5" fill="#ffffff"/>
        <path d="M8 16 L8 20 L12.2 16 Z" fill="#ffffff"/>
        <circle cx="8.5" cy="10" r="1.35" fill="#c8102e"/>
        <circle cx="12.5" cy="10" r="1.35" fill="#c8102e"/>
        <circle cx="16.5" cy="10" r="1.35" fill="#c8102e" opacity="0.55"/>
      </svg>
    </div>
    <div class="brand-text">
      <h1>Delta Support Intelligence</h1>
      <p>AI-powered customer-support responses from historical @Delta support data</p>
    </div>
    <span class="status-badge"><span class="status-dot" aria-hidden="true"></span>Offline demo</span>
  </header>

  <div class="app-body">
    <aside class="sidebar" aria-label="Demo navigation">
      <div class="nav-group-label">Workspace</div>
      <button type="button" class="nav-item active" onclick="clearMessage()">
        <svg width="16" height="16" viewBox="0 0 16 16" fill="none" aria-hidden="true"><path d="M8 3v10M3 8h10" stroke="currentColor" stroke-width="1.6" stroke-linecap="round"/></svg>
        New Conversation
      </button>
      <button type="button" class="nav-item" onclick="focusExamples()">
        <svg width="16" height="16" viewBox="0 0 16 16" fill="none" aria-hidden="true"><circle cx="4" cy="4" r="1.3" fill="currentColor"/><circle cx="4" cy="8" r="1.3" fill="currentColor"/><circle cx="4" cy="12" r="1.3" fill="currentColor"/><path d="M7.5 4h5M7.5 8h5M7.5 12h5" stroke="currentColor" stroke-width="1.4" stroke-linecap="round"/></svg>
        Example Queries
      </button>

      <div class="nav-group-label">Resources</div>
      <button type="button" class="nav-item" onclick="openPanel('how-it-works')">
        <svg width="16" height="16" viewBox="0 0 16 16" fill="none" aria-hidden="true"><circle cx="8" cy="8" r="6.2" stroke="currentColor" stroke-width="1.4"/><path d="M8 7.2v4" stroke="currentColor" stroke-width="1.4" stroke-linecap="round"/><circle cx="8" cy="5" r="0.9" fill="currentColor"/></svg>
        How It Works
      </button>
      <button type="button" class="nav-item" onclick="openPanel('about')">
        <svg width="16" height="16" viewBox="0 0 16 16" fill="none" aria-hidden="true"><rect x="3" y="2.2" width="10" height="11.6" rx="1.5" stroke="currentColor" stroke-width="1.4"/><path d="M5.3 5.2h5.4M5.3 8h5.4M5.3 10.8h3.4" stroke="currentColor" stroke-width="1.3" stroke-linecap="round"/></svg>
        About the Project
      </button>

      <div class="sidebar-footer">
        <div class="sidebar-footer-badge"><span class="status-dot" aria-hidden="true"></span>Local · No live actions</div>
      </div>
    </aside>

    <main>
      <section class="panel" aria-labelledby="composer-heading">
        <h2 id="composer-heading">Customer Message</h2>

        <label class="field-label" for="message">Message</label>
        <textarea id="message" maxlength="2000" placeholder="Type a customer support message…"
                  aria-describedby="char-count composer-hint"></textarea>
        <div class="composer-meta">
          <span id="char-count">0 / 2000 characters</span>
          <span id="composer-hint">Ctrl+Enter to send</span>
        </div>

        <div class="examples" id="examples-section">
          <div class="examples-label">Try an example</div>
          <div class="examples-grid">
            <button type="button" class="example-card" onclick="fillSample(0)"><span class="dot" aria-hidden="true"></span>Baggage fee question</button>
            <button type="button" class="example-card" onclick="fillSample(1)"><span class="dot" aria-hidden="true"></span>Cancelled flight / rebooking</button>
            <button type="button" class="example-card" onclick="fillSample(2)"><span class="dot" aria-hidden="true"></span>Missing baggage</button>
          </div>
        </div>

        <div class="composer-actions">
          <button type="button" id="submitBtn" class="btn btn-primary" onclick="submitMessage()" aria-busy="false">
            <span id="submitLabel">Send Message</span>
          </button>
          <button type="button" class="btn btn-secondary" onclick="clearMessage()">Clear</button>
        </div>

        <div id="alert-box" role="alert"></div>
      </section>

      <section class="panel" aria-labelledby="result-heading" aria-live="polite">
        <h2 id="result-heading">Agent Analysis</h2>

        <div id="empty-state">
          <svg width="140" height="88" viewBox="0 0 160 100" aria-hidden="true">
            <rect x="12" y="14" width="82" height="50" rx="11" fill="#eef0f4" stroke="#d7dae1"/>
            <path d="M30 64 L30 80 L48 64 Z" fill="#eef0f4" stroke="#d7dae1"/>
            <rect x="58" y="40" width="82" height="50" rx="11" fill="#fdecec" stroke="#f3b4b4"/>
            <path d="M132 90 L132 74 L114 90 Z" fill="#fdecec" stroke="#f3b4b4"/>
            <circle cx="36" cy="39" r="3.2" fill="#b7bcc7"/>
            <circle cx="47" cy="39" r="3.2" fill="#b7bcc7"/>
            <circle cx="58" cy="39" r="3.2" fill="#b7bcc7"/>
            <circle cx="86" cy="65" r="3.2" fill="#e39a9a"/>
            <circle cx="97" cy="65" r="3.2" fill="#e39a9a"/>
            <circle cx="108" cy="65" r="3.2" fill="#e39a9a"/>
          </svg>
          <p>Send a message to see the agent's classification, reply, and escalation decision.</p>
        </div>

        <div id="result">
          <div class="result-section">
            <div class="result-label">Intent</div>
            <span class="badge" id="r-intent"></span>
          </div>

          <div class="result-section">
            <div class="result-label">Escalation decision</div>
            <div class="decision-banner" id="r-decision-banner">
              <span id="r-decision-icon" aria-hidden="true"></span>
              <span id="r-decision-text"></span>
            </div>
            <span id="r-override-badge" class="override-badge" style="display:none;" role="status">
              &#9888; Safety override applied — a claim/account-specific term forced escalation
            </span>
            <div id="r-reason-wrap">
              <div class="result-label">Reason</div>
              <div id="r-reason" class="reply-card" style="font-size:0.85rem;"></div>
            </div>
          </div>

          <div class="result-section">
            <div class="result-label">Reply</div>
            <div class="reply-card" id="r-reply"></div>
            <div class="offline-note">Offline model response — grounded in retrieved historical evidence and/or static policy; not sent to a live Delta system.</div>
          </div>

          <div class="result-section">
            <div class="result-label">Evidence</div>
            <div id="r-evidence"></div>
          </div>
        </div>
      </section>
    </main>
  </div>

  <footer>
    This demo does not access live Delta systems and cannot perform flight-status lookups, bookings, refunds, account actions, or baggage-system operations.
  </footer>

  <div class="modal-overlay" id="modal-overlay" hidden>
    <div class="modal-card" role="dialog" aria-modal="true" aria-labelledby="modal-heading">
      <button type="button" class="modal-close" onclick="closePanel()" aria-label="Close" id="modal-close-btn">&times;</button>
      <div id="modal-content"></div>
    </div>
  </div>

<script>
const SAMPLES = [
  "How much does a second checked bag cost on an international flight?",
  "My flight was cancelled, I need to get to Boston tonight!",
  "My suitcase came out with a broken wheel, how do I file a claim?"
];

const PANEL_CONTENT = {
  'how-it-works': '<h3 id="modal-heading">How it works</h3>'
    + '<ol>'
    + '<li><strong>Classify</strong> — a TF-IDF + logistic regression model assigns one of 10 intents.</li>'
    + '<li><strong>Retrieve</strong> — TF-IDF/cosine search finds similar historical Delta support exchanges.</li>'
    + '<li><strong>Generate a reply</strong> — grounded in that retrieved evidence or documented policy facts, offline and deterministic.</li>'
    + '<li><strong>Decide escalation</strong> — a rule-based policy flags anything needing account or booking access for a human.</li>'
    + '</ol>'
    + '<p>Every reply you see here came from this exact pipeline — nothing is hand-picked for the demo.</p>',
  'about': '<h3 id="modal-heading">About this project</h3>'
    + '<p>Delta Support Intelligence is an offline AI customer-support prototype built using the Kaggle '
    + 'Customer Support on Twitter dataset. It classifies customer messages, retrieves relevant historical '
    + 'support evidence, drafts a grounded response, and determines whether the request should be escalated.</p>'
    + '<p>This demo does not access live Delta systems and cannot perform flight-status lookups, bookings, '
    + 'refunds, account actions, or baggage-system operations.</p>'
};

const messageEl = document.getElementById('message');
const charCountEl = document.getElementById('char-count');
const submitBtn = document.getElementById('submitBtn');
const submitLabel = document.getElementById('submitLabel');
const alertBox = document.getElementById('alert-box');
const emptyState = document.getElementById('empty-state');
const resultBox = document.getElementById('result');
const modalOverlay = document.getElementById('modal-overlay');
const modalContent = document.getElementById('modal-content');

function updateCharCount() {
  charCountEl.textContent = messageEl.value.length + ' / 2000 characters';
}
messageEl.addEventListener('input', updateCharCount);

// Plain Enter stays a normal newline (textarea usability); Ctrl/Cmd+Enter submits.
messageEl.addEventListener('keydown', function (e) {
  if (e.key === 'Enter' && (e.ctrlKey || e.metaKey)) {
    e.preventDefault();
    submitMessage();
  }
});

function fillSample(i) {
  messageEl.value = SAMPLES[i];
  updateCharCount();
  hideAlert();
  messageEl.focus();
}

function clearMessage() {
  messageEl.value = '';
  updateCharCount();
  hideAlert();
  resultBox.style.display = 'none';
  emptyState.style.display = 'block';
  messageEl.focus();
}

function focusExamples() {
  const section = document.getElementById('examples-section');
  section.scrollIntoView({ behavior: 'smooth', block: 'center' });
  const first = section.querySelector('.example-card');
  if (first) first.focus();
}

function openPanel(name) {
  modalContent.innerHTML = PANEL_CONTENT[name] || '';
  modalOverlay.hidden = false;
  document.getElementById('modal-close-btn').focus();
}

function closePanel() {
  modalOverlay.hidden = true;
}

modalOverlay.addEventListener('click', function (e) {
  if (e.target === modalOverlay) closePanel();
});
document.addEventListener('keydown', function (e) {
  if (e.key === 'Escape' && !modalOverlay.hidden) closePanel();
});

function hideAlert() {
  alertBox.style.display = 'none';
  alertBox.textContent = '';
}

function showAlert(msg) {
  alertBox.textContent = msg;
  alertBox.style.display = 'block';
}

function setLoading(isLoading) {
  submitBtn.disabled = isLoading;
  submitBtn.setAttribute('aria-busy', isLoading ? 'true' : 'false');
  submitLabel.innerHTML = isLoading
    ? '<span class="spinner" aria-hidden="true"></span> Analyzing…'
    : 'Send Message';
}

function escapeHtml(str) {
  const div = document.createElement('div');
  div.textContent = str;
  return div.innerHTML;
}

function formatIntent(intent) {
  return (intent || '').split('_').join(' ');
}

async function submitMessage() {
  const message = messageEl.value;
  hideAlert();

  if (!message || !message.trim()) {
    showAlert('Please enter a message before sending.');
    return;
  }

  setLoading(true);

  try {
    const resp = await fetch('/handle', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ message: message })
    });

    if (!resp.ok) {
      let detail = resp.statusText;
      try {
        const errJson = await resp.json();
        if (Array.isArray(errJson.detail)) {
          detail = errJson.detail.map(function (d) { return d.msg; }).join('; ');
        } else if (errJson.detail) {
          detail = errJson.detail;
        }
      } catch (e) { /* ignore parse failure, keep statusText */ }
      throw new Error('Request failed (' + resp.status + '): ' + detail);
    }

    renderResult(await resp.json());
  } catch (err) {
    showAlert(err.message || String(err));
  } finally {
    setLoading(false);
  }
}

function renderResult(data) {
  document.getElementById('r-intent').textContent = formatIntent(data.intent);

  const banner = document.getElementById('r-decision-banner');
  const icon = document.getElementById('r-decision-icon');
  const text = document.getElementById('r-decision-text');
  const isEscalate = data.escalate === 'yes';
  banner.className = 'decision-banner ' + (isEscalate ? 'escalate' : 'safe');
  icon.textContent = isEscalate ? '\\u26A0' : '\\u2713';
  text.textContent = isEscalate ? 'Escalated to a human agent' : 'Auto-handled';

  const reasonWrap = document.getElementById('r-reason-wrap');
  const reasonEl = document.getElementById('r-reason');
  if (data.escalation_reason && data.escalation_reason.trim()) {
    reasonEl.textContent = data.escalation_reason;
    reasonWrap.style.display = 'block';
  } else {
    reasonWrap.style.display = 'none';
  }

  // The demo-only safety override (scripts/demo_server.py) always prefixes
  // its reason with this exact string — never generated by the real agent.
  const isSafetyOverride = !!(data.escalation_reason && data.escalation_reason.indexOf('Safety override:') === 0);
  document.getElementById('r-override-badge').style.display = isSafetyOverride ? 'inline-flex' : 'none';

  document.getElementById('r-reply').textContent = data.reply;

  const evEl = document.getElementById('r-evidence');
  evEl.innerHTML = '';
  if (!data.evidence || data.evidence.length === 0) {
    const p = document.createElement('div');
    p.className = 'offline-note';
    p.textContent = 'No historical evidence retrieved for this message.';
    evEl.appendChild(p);
  } else {
    data.evidence.forEach(function (ev, idx) {
      const details = document.createElement('details');
      details.className = 'evidence-item';
      if (idx === 0) details.open = true;

      const summary = document.createElement('summary');
      const label = document.createElement('span');
      label.textContent = 'Evidence ' + (idx + 1) + (ev.conversation_id ? ' \\u2014 conv ' + ev.conversation_id : '');
      const score = document.createElement('span');
      score.className = 'evidence-score';
      score.textContent = (ev.similarity_score !== undefined) ? 'similarity ' + ev.similarity_score : '';
      summary.appendChild(label);
      summary.appendChild(score);
      details.appendChild(summary);

      const body = document.createElement('div');
      body.className = 'evidence-body';

      const meta = document.createElement('div');
      meta.className = 'evidence-meta';
      const metaParts = [];
      if (ev.intent_heuristic) metaParts.push('heuristic intent: ' + ev.intent_heuristic);
      if (ev.timestamp) metaParts.push(ev.timestamp);
      meta.textContent = metaParts.join(' \\u00B7 ');
      body.appendChild(meta);

      if (ev.customer_message) {
        const cust = document.createElement('div');
        cust.style.marginBottom = '0.5rem';
        cust.innerHTML = '<strong>Customer:</strong> ' + escapeHtml(ev.customer_message);
        body.appendChild(cust);
      }
      if (ev.delta_response) {
        const resp = document.createElement('div');
        resp.innerHTML = '<strong>Delta:</strong> ' + escapeHtml(ev.delta_response);
        body.appendChild(resp);
      }

      details.appendChild(body);
      evEl.appendChild(details);
    });
  }

  emptyState.style.display = 'none';
  resultBox.style.display = 'block';
}

updateCharCount();
</script>
</body>
</html>
"""


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return _HTML_PAGE
