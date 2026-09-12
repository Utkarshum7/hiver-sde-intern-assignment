# Evaluation Results (n=200 golden examples)

**Leakage check (re-verified at evaluation time):** leakage_free=True

## main_agent

- Intent accuracy: **0.565**, macro F1: **0.583** (n=200)
- Escalation: accuracy=0.74, precision=0.8364, recall=0.5169, F1=0.6389
- Escalation confusion matrix: TP=46, TN=102, **FP=9**, **FN(unsafe)=43**
- (Supplementary, not a required metric) Retrieval proxy (intent-match@k): 0.575 — Proxy metric: checks whether retrieved evidence's weak intent_heuristic label (not a human relevance judgment) matches the query's gold_intent. See docs for why a true relevance-labelled retrieval metric was not built.
- Unsupported-claim rate in replies (automated, non-LLM): 0.0

## baseline_1_trivial

- Intent accuracy: **0.255**, macro F1: **0.0406** (n=200)
- Escalation: accuracy=0.645, precision=0.875, recall=0.236, F1=0.3717
- Escalation confusion matrix: TP=21, TN=108, **FP=3**, **FN(unsafe)=68**
- (Supplementary, not a required metric) Retrieval proxy (intent-match@k): 0.0 — Proxy metric: checks whether retrieved evidence's weak intent_heuristic label (not a human relevance judgment) matches the query's gold_intent. See docs for why a true relevance-labelled retrieval metric was not built.
- Unsupported-claim rate in replies (automated, non-LLM): 0.0

## baseline_2_simple_retrieval

- Intent accuracy: **0.36**, macro F1: **0.358** (n=200)
- Escalation: accuracy=0.61, precision=1.0, recall=0.1236, F1=0.22
- Escalation confusion matrix: TP=11, TN=111, **FP=0**, **FN(unsafe)=78**
- (Supplementary, not a required metric) Retrieval proxy (intent-match@k): 0.575 — Proxy metric: checks whether retrieved evidence's weak intent_heuristic label (not a human relevance judgment) matches the query's gold_intent. See docs for why a true relevance-labelled retrieval metric was not built.
- Unsupported-claim rate in replies (automated, non-LLM): 0.0

## LLM-as-judge status

- pending_no_api_key: No API key configured in this environment; no judge scores computed. Run scripts/build_human_rating_sheet.py with ANTHROPIC_API_KEY set to get real scores. See src/llm_judge.py for the rubric.

## Human-agreement status

- pending_incomplete_ratings: 0/20 rows rated. Fill in columns ['human_relevance', 'human_correctness', 'human_groundedness', 'human_helpfulness', 'human_safety'] in data/eval/human_rating_sheet.csv, then run scripts/compute_judge_human_agreement.py.
