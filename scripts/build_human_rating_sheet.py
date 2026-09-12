"""
Builds a human-rating sheet for reply-quality evaluation (Step 11:
LLM-judge / human agreement).

IMPORTANT — deliberately does NOT sample from the golden set:
  The golden set (data/golden_eval/golden_annotation.csv) is reserved for
  intent/escalation accuracy evaluation once fully human-annotated, per
  project rules ("do not evaluate on the golden set before final
  annotation"). Reply-QUALITY rating doesn't need gold_intent labels at
  all, so this script instead samples from the DEVELOPMENT corpus
  (data/processed/dev_corpus.parquet) — never held out, available now —
  which fully decouples the reply-quality/judge/human-agreement workflow
  from golden-set annotation progress.

Also runs the LLM judge (src/llm_judge.py) on each generated reply. With no
ANTHROPIC_API_KEY configured this will show status="pending_no_api_key" for
every row — that is expected and correctly reported, not an error.

Usage:
    python scripts/build_human_rating_sheet.py [--n 20]

Outputs:
    data/eval/human_rating_sheet.csv   <- what a human fills in
    data/eval/llm_judge_results.json   <- judge output per sample_id (or PENDING)
"""

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.agent import DeltaSupportAgent
from src.llm_judge import judge_reply

SEED = 42
OUTPUT_DIR = "data/eval"
SHEET_PATH = os.path.join(OUTPUT_DIR, "human_rating_sheet.csv")
JUDGE_PATH = os.path.join(OUTPUT_DIR, "llm_judge_results.json")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=20, help="Number of examples to sample.")
    args = parser.parse_args()

    dev = pd.read_parquet("data/processed/dev_corpus.parquet")
    dev = dev[
        (dev["retrieval_eligible"]) & (dev["customer_message"].astype(str).str.strip() != "")
    ].reset_index(drop=True)

    # Stratify roughly evenly across intent_heuristic buckets so the rating
    # sample isn't dominated by the 2 majority classes.
    np.random.seed(SEED)
    per_intent = max(1, args.n // dev["intent_heuristic"].nunique())
    sampled_frames = []
    for intent, group in dev.groupby("intent_heuristic"):
        n = min(per_intent, len(group))
        sampled_frames.append(group.sample(n=n, random_state=SEED))
    sample = pd.concat(sampled_frames).sample(frac=1.0, random_state=SEED).head(args.n).reset_index(drop=True)

    print(f"[*] Sampled {len(sample)} dev-corpus examples for rating (seed={SEED}).")

    agent = DeltaSupportAgent.load()

    rows = []
    judge_results = {}
    for i, row in sample.iterrows():
        rating_id = f"R{i+1:03d}"
        result = agent.handle(row["customer_message"])
        evidence_preview = " | ".join(
            e["delta_response"][:120] for e in result["evidence"][:2]
        )

        rows.append({
            "rating_id": rating_id,
            "conversation_id": row["conversation_id"],
            "customer_message": row["customer_message"],
            "generated_intent": result["intent"],
            "generated_escalate": result["escalate"],
            "generated_reply": result["reply"],
            "evidence_used_preview": evidence_preview,
            # ---- Blank fields for a REAL human rater to fill in (1-5) ----
            "human_relevance": "",
            "human_correctness": "",
            "human_groundedness": "",
            "human_helpfulness": "",
            "human_safety": "",
            "human_notes": "",
        })

        judge_results[rating_id] = judge_reply(
            customer_message=row["customer_message"],
            evidence=result["evidence"],
            reply=result["reply"],
        )

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    pd.DataFrame(rows).to_csv(SHEET_PATH, index=False, encoding="utf-8")
    with open(JUDGE_PATH, "w", encoding="utf-8") as f:
        json.dump(judge_results, f, indent=2)

    print(f"[+] Wrote human rating sheet: {SHEET_PATH} ({len(rows)} rows)")
    print(f"[+] Wrote LLM judge results: {JUDGE_PATH}")

    statuses = pd.Series([v["status"] for v in judge_results.values()]).value_counts()
    print("\n[+] LLM judge status breakdown:")
    print(statuses.to_string())
    if (statuses.index == "pending_no_api_key").any():
        print(
            "\n[NOTE] No ANTHROPIC_API_KEY configured — all judge scores are "
            "PENDING, not fabricated. Set the env var and re-run to get real "
            "judge scores."
        )
    print(
        f"\n[NEXT STEP] Open {SHEET_PATH} and fill in the human_* columns "
        "(1-5) for each row with your own judgment, then run:\n"
        "    python scripts/compute_judge_human_agreement.py"
    )


if __name__ == "__main__":
    main()
