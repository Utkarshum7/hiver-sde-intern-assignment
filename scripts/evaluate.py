"""
Headline evaluation entry point for the Delta Customer Support AI Agent.

    python scripts/evaluate.py

This script assumes DATA PREPARATION has already been run once:
    python scripts/build_delta_dataset.py     (raw twcs.csv -> conversations)
    python scripts/build_retrieval_index.py   (leakage-safe split + TF-IDF index)
    python scripts/train_classifier.py        (weak-supervised intent classifier)
All three write to data/processed/ and only need to be re-run if the raw
dataset or splitting logic changes. This script itself does none of that —
it only loads already-built artifacts, so it runs in well under a minute.

METHODOLOGY GUARDRAIL (non-negotiable per project rules): this script
REFUSES to compute intent/escalation accuracy metrics against the golden
set until data/golden_eval/golden_annotation.csv is FULLY human-annotated
and passes every required validation gate (scripts/validate_golden_set.py).
This prevents evaluating on a partially-labelled held-out set and reporting
numbers that would need to be recomputed (and could look like they were
being "optimized towards") once the rest of the labels come in.

While the golden set is incomplete, this script:
  1. Reports exact completion status (N/200, which gates fail/pass).
  2. Runs a non-golden smoke test (a handful of example messages) to prove
     the pipeline itself works end-to-end.
  3. Exits 0 (this is an expected state, not an error) without writing any
     "results" file that could be mistaken for real evaluation numbers.

Once the golden set is complete, it computes and writes:
    reports/evaluation_results.json
    reports/evaluation_report.md
covering intent metrics, escalation metrics (incl. unsafe false negatives),
a retrieval proxy metric, and an automated reply-safety metric — for the
main agent AND both baselines, side by side.
"""

import json
import os
import subprocess
import sys

import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.agent import DeltaSupportAgent
from src.baselines import TrivialBaseline, SimpleRetrievalBaseline
from src.metrics import (
    intent_metrics,
    escalation_metrics,
    retrieval_intent_match_at_k,
    unsupported_claim_rate,
)
from src.heuristic_intent import VALID_INTENTS
from src import llm_client

GOLDEN_CSV = "data/golden_eval/golden_annotation.csv"
RESERVED_PARQUET = "data/processed/reserved_golden_pool.parquet"
DEV_PARQUET = "data/processed/dev_corpus.parquet"
RETRIEVAL_INDEX_PATH = "data/processed/retrieval_index.pkl"
HUMAN_RATING_SHEET = "data/eval/human_rating_sheet.csv"
JUDGE_RESULTS_PATH = "data/eval/llm_judge_results.json"

SMOKE_TEST_MESSAGES = [
    "How much for a second checked bag on an international flight?",
    "My flight was cancelled, I need to get to Boston tonight!",
    "My suitcase came out with a broken wheel, how do I file a claim?",
    "Great flight attendant on my flight today, please recognize her!",
]


def _golden_set_is_ready() -> bool:
    """Re-runs the same required gates as scripts/validate_golden_set.py
    (subprocess call, so there is exactly one source of truth for the gates)."""
    result = subprocess.run(
        [sys.executable, "scripts/validate_golden_set.py"],
        capture_output=True, text=True,
    )
    print(result.stdout)
    if result.returncode != 0:
        return False
    return True


def _run_smoke_test():
    print("=" * 70)
    print("  PIPELINE SMOKE TEST (non-golden example messages)")
    print("=" * 70)
    agent = DeltaSupportAgent.load()
    for msg in SMOKE_TEST_MESSAGES:
        r = agent.handle(msg)
        print(f"\n  MSG: {msg}")
        print(f"  -> intent={r['intent']} (conf={r['confidence']:.2f})  escalate={r['escalate']}")
        if r["escalate"] == "yes":
            print(f"     reason: {r['escalation_reason']}")
        print(f"     reply: {r['reply'][:140]}")


def _recheck_leakage(golden_conv_ids):
    """
    Re-verifies, immediately before evaluation, that the golden set's
    conversation_ids: (a) do not appear in the dev corpus, (b) do not appear
    among the conversations actually indexed by the fitted retriever. This
    is a fresh check, not a cached assumption, and its result is included
    in the evaluation artifact.
    """
    result = {}
    if os.path.exists(DEV_PARQUET):
        dev_ids = set(pd.read_parquet(DEV_PARQUET)["conversation_id"].astype(str))
        leaked_dev = sorted(golden_conv_ids & dev_ids)
        result["golden_conv_ids_in_dev_corpus"] = len(leaked_dev)
        result["golden_conv_ids_in_dev_corpus_sample"] = leaked_dev[:10]
    else:
        result["golden_conv_ids_in_dev_corpus"] = "SKIPPED (dev_corpus.parquet not found)"

    if os.path.exists(RETRIEVAL_INDEX_PATH):
        retriever = SimpleRetrievalBaseline.load(RETRIEVAL_INDEX_PATH).retriever
        indexed_ids = set(retriever.corpus_df["conversation_id"].astype(str))
        leaked_index = sorted(golden_conv_ids & indexed_ids)
        result["golden_conv_ids_in_retrieval_index"] = len(leaked_index)
        result["golden_conv_ids_in_retrieval_index_sample"] = leaked_index[:10]
    else:
        result["golden_conv_ids_in_retrieval_index"] = "SKIPPED (retrieval_index.pkl not found)"

    result["leakage_free"] = (
        result.get("golden_conv_ids_in_dev_corpus") == 0
        and result.get("golden_conv_ids_in_retrieval_index") == 0
    )
    return result


def _check_llm_judge_status():
    if llm_client.is_available():
        status = "api_key_configured"
    else:
        status = "pending_no_api_key"

    judge_results = {}
    if os.path.exists(JUDGE_RESULTS_PATH):
        with open(JUDGE_RESULTS_PATH, "r", encoding="utf-8") as f:
            judge_results = json.load(f)
    statuses = [v.get("status") for v in judge_results.values()]
    ok_count = sum(1 for s in statuses if s in ("ok", "cached"))

    return {
        "api_key_status": status,
        "existing_judge_results_file": JUDGE_RESULTS_PATH if judge_results else None,
        "existing_judge_results_count": len(judge_results),
        "existing_judge_results_ok_count": ok_count,
        "note": (
            "Real judge scores available." if ok_count > 0 else
            "No API key configured in this environment; no judge scores computed. "
            "Run scripts/build_human_rating_sheet.py with ANTHROPIC_API_KEY set to "
            "get real scores. See src/llm_judge.py for the rubric."
        ),
    }


def _check_human_agreement_status():
    if not os.path.exists(HUMAN_RATING_SHEET):
        return {
            "status": "pending_no_rating_sheet",
            "note": f"{HUMAN_RATING_SHEET} not found. Run scripts/build_human_rating_sheet.py first.",
        }

    sheet = pd.read_csv(HUMAN_RATING_SHEET, dtype=str).fillna("")
    human_cols = [c for c in sheet.columns if c.startswith("human_") and c != "human_notes"]
    rated_mask = sheet[human_cols].apply(lambda row: all(v.strip() != "" for v in row), axis=1)
    n_rated = int(rated_mask.sum())
    n_total = len(sheet)

    return {
        "status": "complete" if n_rated == n_total else "pending_incomplete_ratings",
        "rows_total": n_total,
        "rows_rated": n_rated,
        "rows_remaining": n_total - n_rated,
        "note": (
            f"{n_rated}/{n_total} rows rated. Fill in columns {human_cols} in "
            f"{HUMAN_RATING_SHEET}, then run scripts/compute_judge_human_agreement.py."
            if n_rated < n_total else
            "All rows rated; run scripts/compute_judge_human_agreement.py for agreement stats."
        ),
    }


def _run_full_evaluation():
    print("=" * 70)
    print("  HEADLINE EVALUATION — golden set is complete and validated")
    print("=" * 70)

    df = pd.read_csv(GOLDEN_CSV, dtype=str).fillna("")
    n = len(df)
    print(f"[+] Evaluating on {n} golden examples.")

    golden_conv_ids = set(df["conversation_id"].astype(str))
    print("\n[*] Re-checking leakage immediately before evaluation...")
    leakage = _recheck_leakage(golden_conv_ids)
    print(f"    golden conv_ids in dev corpus: {leakage['golden_conv_ids_in_dev_corpus']}")
    print(f"    golden conv_ids in retrieval index: {leakage['golden_conv_ids_in_retrieval_index']}")
    print(f"    leakage_free: {leakage['leakage_free']}")
    if not leakage["leakage_free"]:
        print("\n[FATAL] Leakage detected. Aborting evaluation without writing results.")
        sys.exit(1)

    agent = DeltaSupportAgent.load()
    baseline1 = TrivialBaseline.from_dev_corpus()
    baseline2 = SimpleRetrievalBaseline.load()

    systems = {
        "main_agent": agent,
        "baseline_1_trivial": baseline1,
        "baseline_2_simple_retrieval": baseline2,
    }

    all_results = {}
    all_raw_predictions = {}
    for name, system in systems.items():
        print(f"\n[*] Running {name} on {n} golden examples...")
        predictions = [system.handle(msg) for msg in df["customer_message"]]

        pred_intents = [p["intent"] for p in predictions]
        pred_escalate = [p["escalate"] for p in predictions]
        pred_replies = [p["reply"] for p in predictions]
        evidence_intents = [
            [e.get("intent_heuristic", "") for e in p["evidence"]] for p in predictions
        ]

        results = {
            "intent": intent_metrics(df["gold_intent"].tolist(), pred_intents, VALID_INTENTS),
            "escalation": escalation_metrics(df["escalate"].tolist(), pred_escalate),
            "retrieval_proxy_supplementary_not_required": retrieval_intent_match_at_k(
                df["gold_intent"].tolist(), evidence_intents
            ),
            "reply_safety": unsupported_claim_rate(pred_replies),
        }
        all_results[name] = results

        # Raw per-example predictions, preserved for reproducibility and
        # for real failure-mode inspection (never aggregated-only).
        raw = []
        for idx, row in df.iterrows():
            p = predictions[idx]
            raw.append({
                "sample_id": row["sample_id"],
                "conversation_id": row["conversation_id"],
                "customer_message": row["customer_message"],
                "gold_intent": row["gold_intent"],
                "pred_intent": p["intent"],
                "intent_correct": row["gold_intent"] == p["intent"],
                "gold_escalate": row["escalate"],
                "pred_escalate": p["escalate"],
                "escalate_correct": row["escalate"] == p["escalate"],
                "difficulty": row["difficulty"],
                "pred_reply": p["reply"],
                "evidence_conversation_ids": [e.get("conversation_id") for e in p["evidence"]],
            })
        all_raw_predictions[name] = raw

        print(f"    intent accuracy={results['intent']['accuracy']}  macro_f1={results['intent']['macro_f1']}")
        em = results["escalation"]
        print(f"    escalation f1={em['f1']}  FP={em['false_positive_count']}  "
              f"FN(unsafe)={em['false_negative_count']}")

    print("\n[*] Checking LLM-judge and human-agreement status (not fabricating either)...")
    judge_status = _check_llm_judge_status()
    human_status = _check_human_agreement_status()
    print(f"    LLM judge: {judge_status['api_key_status']}")
    print(f"    Human agreement: {human_status['status']}")

    artifact = {
        "n_golden_examples": n,
        "leakage_check": leakage,
        "results": all_results,
        "llm_judge_status": judge_status,
        "human_agreement_status": human_status,
    }

    os.makedirs("reports", exist_ok=True)
    with open("reports/evaluation_results.json", "w", encoding="utf-8") as f:
        json.dump(artifact, f, indent=2, default=str)
    print("\n[+] Saved reports/evaluation_results.json (summary metrics + status)")

    with open("reports/evaluation_raw_predictions.json", "w", encoding="utf-8") as f:
        json.dump(all_raw_predictions, f, indent=2, default=str)
    print("[+] Saved reports/evaluation_raw_predictions.json (per-example predictions, all systems)")

    _write_markdown_report(all_results, n, leakage, judge_status, human_status)


def _write_markdown_report(all_results: dict, n: int, leakage: dict, judge_status: dict, human_status: dict):
    lines = [f"# Evaluation Results (n={n} golden examples)\n"]
    lines.append(f"**Leakage check (re-verified at evaluation time):** leakage_free={leakage['leakage_free']}\n")
    for name, results in all_results.items():
        lines.append(f"## {name}\n")
        im = results["intent"]
        em = results["escalation"]
        lines.append(f"- Intent accuracy: **{im['accuracy']}**, macro F1: **{im['macro_f1']}** (n={im['n']})")
        lines.append(
            f"- Escalation: accuracy={em['accuracy']}, precision={em['precision']}, "
            f"recall={em['recall']}, F1={em['f1']}"
        )
        lines.append(
            f"- Escalation confusion matrix: TP={em['confusion_matrix']['true_positive']}, "
            f"TN={em['confusion_matrix']['true_negative']}, "
            f"**FP={em['false_positive_count']}**, **FN(unsafe)={em['false_negative_count']}**"
        )
        rp = results["retrieval_proxy_supplementary_not_required"]
        lines.append(
            f"- (Supplementary, not a required metric) Retrieval proxy (intent-match@k): "
            f"{rp['match_at_k_rate']} — {rp['caveat']}"
        )
        rs = results["reply_safety"]
        lines.append(f"- Unsupported-claim rate in replies (automated, non-LLM): {rs['unsupported_claim_rate']}\n")

    lines.append("## LLM-as-judge status\n")
    lines.append(f"- {judge_status['api_key_status']}: {judge_status['note']}\n")
    lines.append("## Human-agreement status\n")
    lines.append(f"- {human_status['status']}: {human_status['note']}\n")

    with open("reports/evaluation_report.md", "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print("[+] Saved reports/evaluation_report.md")


def main():
    if not os.path.exists(GOLDEN_CSV):
        print(f"[ERROR] {GOLDEN_CSV} not found.")
        sys.exit(1)

    ready = _golden_set_is_ready()

    if not ready:
        print("\n" + "=" * 70)
        print("  RESULT: PENDING")
        print("=" * 70)
        print(
            "  The golden set is not yet fully human-annotated (see gate "
            "failures above). Per project methodology, this harness refuses "
            "to compute intent/escalation accuracy against a partially "
            "labelled held-out set. Finish annotation, then re-run:\n"
            "      python scripts/validate_golden_set.py\n"
            "      python scripts/evaluate.py\n"
        )
        _run_smoke_test()
        sys.exit(0)

    _run_full_evaluation()


if __name__ == "__main__":
    main()
