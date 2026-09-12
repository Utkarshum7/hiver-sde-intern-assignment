"""
Computes LLM-judge / human-rater agreement statistics (Step 11).

Requires BOTH:
  1. data/eval/human_rating_sheet.csv with the human_* columns filled in
     (see scripts/build_human_rating_sheet.py) — REAL human ratings only.
  2. data/eval/llm_judge_results.json with status="ok" entries (requires
     ANTHROPIC_API_KEY to have been set when build_human_rating_sheet.py /
     a judge re-run was executed).

If either precondition isn't met, this script reports exactly what is
missing and computes NOTHING — it never fabricates agreement statistics.

Usage:
    python scripts/compute_judge_human_agreement.py
"""

import json
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import cohen_kappa_score

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

SHEET_PATH = "data/eval/human_rating_sheet.csv"
JUDGE_PATH = "data/eval/llm_judge_results.json"

DIMENSIONS = ["relevance", "correctness", "groundedness", "helpfulness", "safety"]


def main():
    if not os.path.exists(SHEET_PATH):
        print(f"[PENDING] {SHEET_PATH} not found. Run scripts/build_human_rating_sheet.py first.")
        sys.exit(0)
    if not os.path.exists(JUDGE_PATH):
        print(f"[PENDING] {JUDGE_PATH} not found. Run scripts/build_human_rating_sheet.py first.")
        sys.exit(0)

    sheet = pd.read_csv(SHEET_PATH, dtype=str).fillna("")
    with open(JUDGE_PATH, "r", encoding="utf-8") as f:
        judge_results = json.load(f)

    human_cols = [f"human_{d}" for d in DIMENSIONS]
    rated_mask = sheet[human_cols].apply(lambda row: all(v.strip() != "" for v in row), axis=1)
    n_rated = int(rated_mask.sum())

    if n_rated == 0:
        print(
            f"[PENDING] No rows in {SHEET_PATH} have human ratings filled in yet.\n"
            f"  Fill in columns: {human_cols}\n"
            "  This script computes ZERO fabricated agreement numbers until real "
            "ratings exist."
        )
        sys.exit(0)

    judged_ok_ids = {
        rid for rid, v in judge_results.items() if v.get("status") in ("ok", "cached")
    }
    if not judged_ok_ids:
        statuses = pd.Series([v["status"] for v in judge_results.values()]).value_counts()
        print(
            f"[PENDING] {n_rated} human ratings exist, but zero LLM judge scores "
            f"are available (status breakdown below). Set ANTHROPIC_API_KEY and "
            "re-run scripts/build_human_rating_sheet.py to get real judge scores.\n"
            f"{statuses.to_string()}"
        )
        sys.exit(0)

    both_mask = rated_mask & sheet["rating_id"].isin(judged_ok_ids)
    n_both = int(both_mask.sum())
    print(f"[+] {n_rated} human-rated rows, {len(judged_ok_ids)} judge-scored rows, "
          f"{n_both} rows have BOTH (usable for agreement).")

    if n_both == 0:
        print("[PENDING] No overlapping rows have both a human rating and a judge score.")
        sys.exit(0)

    both = sheet[both_mask].copy()

    results = {}
    for dim in DIMENSIONS:
        human_scores = both[f"human_{dim}"].astype(int).tolist()
        judge_scores = [judge_results[rid]["scores"][dim] for rid in both["rating_id"]]

        exact_agreement = np.mean([h == j for h, j in zip(human_scores, judge_scores)])
        within_one = np.mean([abs(h - j) <= 1 for h, j in zip(human_scores, judge_scores)])

        pearson_r, pearson_p = (None, None)
        spearman_r, spearman_p = (None, None)
        kappa = None
        if len(set(human_scores)) > 1 and len(set(judge_scores)) > 1:
            pearson_r, pearson_p = pearsonr(human_scores, judge_scores)
            spearman_r, spearman_p = spearmanr(human_scores, judge_scores)
            kappa = cohen_kappa_score(human_scores, judge_scores, weights="linear")

        results[dim] = {
            "n": n_both,
            "exact_agreement": round(float(exact_agreement), 4),
            "within_one_agreement": round(float(within_one), 4),
            "pearson_r": round(float(pearson_r), 4) if pearson_r is not None else None,
            "spearman_r": round(float(spearman_r), 4) if spearman_r is not None else None,
            "weighted_kappa": round(float(kappa), 4) if kappa is not None else None,
        }

    out_path = "reports/judge_human_agreement.json"
    os.makedirs("reports", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"\n[+] Saved agreement results to: {out_path}\n")
    for dim, r in results.items():
        print(f"  {dim:<15} exact={r['exact_agreement']:.2f}  within_1={r['within_one_agreement']:.2f}  "
              f"pearson_r={r['pearson_r']}  kappa={r['weighted_kappa']}")


if __name__ == "__main__":
    main()
