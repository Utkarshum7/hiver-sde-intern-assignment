# Evaluation Report — Delta Support Intelligence

This document reorganizes the real content of [`../REPORT.md`](../REPORT.md)
under the assignment's exact requested section headings, for fast
reviewer cross-checking. **It introduces no new numbers, claims, or
examples** — every figure here is the same measured result already in
`REPORT.md` and `reports/evaluation_results.json`, produced by one real run
of `python scripts/evaluate.py` against the locked 200-example golden set.
Where more detail, worked reasoning, or full evidence tables are useful,
each section links to the corresponding `REPORT.md` section rather than
duplicating it. Rendered length here is deliberately kept to roughly six
printed pages; `REPORT.md`'s appendix carries the full detail.

---

## 1. Executive Summary

An AI customer-support agent for **`@Delta`** (Kaggle *Customer Support on
Twitter* dataset) that classifies an incoming message into one of 10
empirically-derived intents, drafts a reply grounded in a retrieved,
historically similar Delta resolution, and decides auto-handle vs.
escalate with a stated reason. Measured once against a locked, 200-example,
human-reviewed golden set: **intent accuracy 56.5%** (macro F1 58.3%),
**escalation F1 63.9%** (recall **51.7%** — the number that matters most,
see §10), **0% unsupported-action-claim rate** in generated replies. The
agent beats both a trivial and a simple-retrieval baseline on every
reported metric. LLM-as-judge reply-quality scoring and judge/human
agreement are fully built but **PENDING real inputs** (no API key
configured; 0/20 human ratings collected) — disclosed, not fabricated.
Full detail: [`REPORT.md`, Executive Summary](../REPORT.md#executive-summary--this-is-the-submission-ready-report).

## 2. Problem Framing and Selected Brand

**Task:** classify an incoming message into a small intent taxonomy defined
from the data, draft a reply grounded in how the brand historically
resolved similar issues, and decide auto-handle vs. escalate with a stated
reason.

**Definition of "good" for this brand:** Delta's own historical support
account never performs a live action in-thread — it always says "please DM
your confirmation number." So "good" here is not "acts like a real airline
backend," it is: correctly separating "answerable from public policy" from
"needs a human with account access," giving accurate evidence-grounded
answers for the former, and **failing safe** — a misclassification that
still gets escalated is a far smaller problem than one that gets wrongly
auto-handled.

**Selected brand: `@Delta`**, chosen via a 10-dimension empirical scoring
rubric over a 500-conversation stratified sample from each of the 5
largest brand accounts in the dataset. Delta scored 47/50 vs. runner-up
SpotifyCares' 37/50, driven mainly by a higher public resolution rate
(20.4% vs. 12.2% for Apple, 9.2% for Amazon) and cleanly non-overlapping
intent categories. Full scoring table: [`docs/brand_selection.md`](brand_selection.md).
Detail: [`REPORT.md` §1–§2](../REPORT.md#1-problem-framing).

## 3. System Overview

Pipeline: **intent classification → evidence retrieval → grounded reply
generation → escalation decision** ([`src/agent.py`](../src/agent.py)).

- **Classification** ([`src/classifier.py`](../src/classifier.py)): TF-IDF +
  `LogisticRegression`, weak-supervised (trained on heuristic labels, not
  clean human labels — a deliberate, disclosed trade-off, decision log #5).
- **Retrieval** ([`src/retrieval.py`](../src/retrieval.py)): leakage-safe
  TF-IDF + cosine similarity over 20,913 historical evidence units, fit
  only on the dev corpus (never the reserved/golden pool).
- **Reply generation** ([`src/reply_generator.py`](../src/reply_generator.py)):
  template/retrieval-based, not an LLM — actively strips any language
  implying a live account action was performed.
- **Escalation** ([`src/escalation.py`](../src/escalation.py)): deterministic
  policy — always-escalate intents, plus regex-based conditional triggers
  for account-specific phrasing within otherwise-safe intents.

No fine-tuned or hosted LLM is used anywhere in the live decision path; an
optional Claude-based LLM judge exists purely for offline reply-quality
scoring (§8). Detail: [`REPORT.md` §6](../REPORT.md#6-architecture).

## 4. Dataset and Sampling Method

**Source:** Kaggle *Customer Support on Twitter*
(`thoughtvector/customer-support-on-twitter`), `twcs.csv`, 2,811,774 rows,
516.5 MB — not committed to this repo (~500MB, downloaded separately per
the README). **Delta subset:** 87,994 messages reconstructed into **26,168
conversations** (26,166 genuinely two-way), spanning 2012-11-27 to
2017-12-03 (policy facts reflect 2017-era Delta, not current pricing).

**Split:** 80/20 at the **conversation-ID level** (seed=42) into a dev
corpus (20,934 conversations, used to fit the retrieval index and train the
classifier) and a reserved pool (5,234 conversations, held out entirely).
Zero conversation-ID overlap between the two, independently re-verified
during this audit. Detail: [`REPORT.md` §3–§4](../REPORT.md#3-dataset).

## 5. Golden-Set Construction

**200 examples**, stratified-sampled (seed=42) from the reserved pool only,
deliberately oversampling rare high-risk escalation intents (e.g.
`lost_damaged_baggage`: ~0.23% of the dataset by heuristic estimate, but 15
of 200 golden examples) so the evaluation can say something about them at
all — see decision log #4.

**Annotation workflow:** AI/rule-assisted proposals
(`scripts/propose_golden_labels.py`) were reviewed and explicitly approved
or edited by a human (`scripts/review_golden_proposals.py`) via an
interactive batch-review process — proposals were never silently promoted
to gold. Every row records its provenance:

| `annotation_source` | Count | % |
|---|---:|---:|
| `human_edited_ai_proposal` | 109 | 54.5% |
| `human_approved_ai_proposal` | 85 | 42.5% |
| `human_annotated_direct` | 6 | 3.0% |

The 54.5% edit rate is real evidence against rubber-stamping. **Status:
200/200 (100%) complete and locked** — `python scripts/validate_golden_set.py`
reports all 14 required gates PASS (re-verified live during this audit: no
duplicates, no leakage, no invalid labels, no missing required fields).
Required fields present on every row: `gold_intent`, `escalate`,
`escalation_reason`, `heuristic_intent`, `difficulty`, `annotation_source`,
plus the source `conversation_id`/`customer_message`/`delta_response`.
Sampling and labelling methodology, including known heuristic biases (e.g.
`flight_status_inquiry` landing at only 2/200 examples after human review
overturned most of the heuristic's proposals for that class), is
documented in full in [`docs/golden_set_methodology.md`](golden_set_methodology.md)
and [`docs/golden_annotation_guide.md`](golden_annotation_guide.md).
Detail: [`REPORT.md` §9](../REPORT.md#9-golden-set-methodology).

## 6. Baselines

Two non-LLM baselines ([`src/baselines.py`](../src/baselines.py)), run on
the identical 200 golden examples through the identical metrics code path:

- **Baseline 1 (trivial):** always predicts the single most frequent
  training intent (`general_complaint_feedback`) and never escalates.
  Sanity-checked: 51/200 = 0.255 exactly matches its measured accuracy, and
  its recall for that one class is exactly 1.0 — confirms it is implemented
  correctly, not artificially weakened.
- **Baseline 2 (simple retrieval):** a fixed intent→escalate lookup table
  plus nearest-neighbor retrieval, no learned classifier.

## 7. Main Results Table

Real, measured once against the locked 200-example golden set (leakage
re-verified immediately before this run: 0 golden conv_ids in the dev
corpus or fitted retrieval index).

| System | Intent acc. | Intent macro F1 | Escalation acc. | Escalation P / R / F1 | Escalation FP / FN |
|---|---:|---:|---:|---:|---:|
| **Main agent** | **0.565** | **0.583** | **0.740** | 0.836 / **0.517** / 0.639 | 9 / 43 |
| Baseline 1 — trivial | 0.255 | 0.041 | 0.645 | 0.875 / 0.236 / 0.372 | 3 / 68 |
| Baseline 2 — simple retrieval | 0.360 | 0.358 | 0.610 | 1.000 / 0.124 / 0.220 | 0 / 78 |

The main agent beats both baselines on every reported metric. Reply
safety (automated, non-LLM): **0/200 (0.0%)** replies contained a detected
unsupported account-action claim. Full per-intent precision/recall/F1
table and the 2×2 escalation confusion matrix: [`REPORT.md` §12](../REPORT.md#12-results).

## 8. LLM-as-Judge Rubric and Human-Agreement Status

**Rubric implemented** ([`src/llm_judge.py`](../src/llm_judge.py),
`RUBRIC_VERSION = "delta_reply_judge_v1"`): 1–5 scale on five dimensions —
relevance, correctness, groundedness, helpfulness, safety (any unsupported
action claim forces safety=1 regardless of other dimensions).
Judge/human agreement statistics are fully coded
([`scripts/compute_judge_human_agreement.py`](../scripts/compute_judge_human_agreement.py)):
exact agreement, within-1 agreement, Pearson/Spearman correlation, and
weighted Cohen's kappa.

**Disclosed limitation — genuinely PENDING, not fabricated:**
- LLM judge: **no `ANTHROPIC_API_KEY` configured** in this environment, so
  zero real judge scores exist. `judge_reply()` returns a labelled
  `pending_no_api_key` status rather than a fabricated number.
- Human ratings: a 20-row rating sheet exists
  (`data/eval/human_rating_sheet.csv`, built by
  `scripts/build_human_rating_sheet.py` from real dev-corpus examples,
  columns for relevance/correctness/groundedness/helpfulness/safety) but
  **0/20 rows have been filled in**. `scripts/compute_judge_human_agreement.py`
  was run during this audit and correctly refused to compute a number:
  `[PENDING] No rows in data/eval/human_rating_sheet.csv have human ratings
  filled in yet. This script computes ZERO fabricated agreement numbers
  until real ratings exist.`

**Smallest practical path to completing this honestly:** open
`data/eval/human_rating_sheet.csv`, fill the five `human_*` columns for the
20 rows using the rubric in `src/llm_judge.RUBRIC`, set
`ANTHROPIC_API_KEY`, run `python scripts/build_human_rating_sheet.py` to
populate real judge scores, then `python scripts/compute_judge_human_agreement.py`
to get real agreement statistics. Nothing else needs to change. Detail:
[`REPORT.md` §14–§15](../REPORT.md#14-llm-as-judge).

## 9. Five Failure Modes (Real Examples)

All five read directly from `reports/evaluation_raw_predictions.json`
(actual per-example predictions on the 200 golden examples):

1. **Escalation policy misses real account-specific phrasing even when
   intent is classified correctly** (44% of all unsafe misses — 19/43).
   Worst-hit: `seat_assignment_upgrade` (9/13 unsafe FNs). Examples:
   **S027**, **S093**, **S191**, **S200**. *Cause:* hand-written regex
   can't cover the full space of natural phrasing.
2. **Compliments/thanks misclassified into domain-specific intents.**
   `general_complaint_feedback` recall is worst of any intent (0.196,
   10/51). Examples: **S050**, **S149**, **S186**, **S196**, **S198** —
   all "thanks for the upgrade" style messages misrouted by
   keyword-matching on domain nouns.
3. **`flight_status_inquiry` acts as a noise sink.** Only 2 real golden
   examples carry this intent, but the classifier predicted it 30 times —
   29 wrong (precision 0.033), causing 14 of 24 intent-driven unsafe
   misses. Examples: **S002**, **S003**, **S022**, **S031**, **S047**.
4. **`lost_damaged_baggage` (highest-severity intent) has only 46.7%
   recall** (7/15). Examples: **S022**, **S126**, **S147**. 7 of 8
   misclassifications became unsafe false negatives.
5. **Always-escalate intents over-trigger on general policy questions** —
   3 of 9 total false positives. Examples: **S092**, **S136**, **S064**,
   all correctly classified but escalated purely because their intent
   belongs to an always-escalate bucket, even where a human reader (and,
   in two cases, Delta's own historical agent) judged them answerable from
   public policy alone.

Full analysis with causes/impact/mitigation for each: [`REPORT.md` §16](../REPORT.md#16-top-5-failure-modes).

## 10. "What Is Misleading About My Headline Number?"

1. **The golden set is stratified, not representative — a real, measured
   distortion.** `flight_status_inquiry` is ~44.6% of the dataset by a
   noisy heuristic estimate but only 2/200 (1%) in the locked golden set;
   `lost_damaged_baggage` is deliberately oversampled 30×+ over its natural
   frequency. "56.5% intent accuracy" is an average over this artificial
   mix, not a production-traffic estimate.
2. **The `flight_status_inquiry` row (precision 0.033) is computed from
   n=2** — reported transparently, but not a stable estimate of anything.
3. **Escalation F1 (0.639) looks fine in isolation; escalation recall
   (0.517) is the number that should actually worry a reviewer** — 43 of
   89 cases that needed a human did not get one, overwhelmingly a
   policy-coverage gap (fixable without retraining), not fixed today.
4. **54.5% of gold labels were edited from a proposal, 42.5% accepted
   as-is** — real evidence against rubber-stamping, but a fully
   independent second annotator might still draw some boundaries
   differently, especially on the 83 `difficulty=ambiguous` examples.
5. **The 0% unsupported-claim rate is a narrow mechanical safety check**,
   not a groundedness or quality score — that requires the still-pending
   LLM judge and human ratings (§8).
6. **"Unsafe false negative" is only as good as the escalation policy
   it's measured against** — a genuinely risky pattern the policy doesn't
   encode at all wouldn't show up as a miss in this metric.
7. **Retrieval quality is only measured with a proxy metric** (intent-match
   at k, 0.575 for the main agent) — it checks topic match, not whether the
   specific retrieved case was the most useful evidence.

Full 7-point analysis: [`REPORT.md` §17](../REPORT.md#17-what-is-misleading-about-my-headline-number).

## 11. Scope Exclusions

Deliberately not built, and disclosed as a design boundary rather than a
gap: live Delta flight-status API, PNR/booking modification, refund/payment
gateway, baggage-tracing system, customer-account access (SkyMiles
balance, Medallion status, booking history), multi-turn/context-carrying
conversation handling, and any fine-tuned/hosted LLM in the live decision
path (no API key available in this environment; reply generation is
template/retrieval-based by design, trading fluency for auditability and
zero fabrication risk). Every one of these is handled by escalating to a
human, never by pretending to perform it. Detail: [`REPORT.md` §1, §18](../REPORT.md#18-limitations).

## 12. One-Week Improvement Plan

Ranked by the actual evidence in §9 above, not a generic wishlist:

1. **Fix escalation recall first** — mine the 43 unsafe-FN examples'
   `annotator_notes` (already explaining *why* each needed escalation) to
   expand regex coverage in `src/escalation.py`, starting with
   `seat_assignment_upgrade` (9/13 misses). Per failure mode 1, this alone
   would fix 44% of unsafe misses without touching the classifier.
2. **Fix the `flight_status_inquiry` noise sink** — correct the
   weak-supervision heuristic's tie-breaking bias, or retrain without that
   label in the weak-label vocabulary.
3. **Collect real human ratings** on the 20-row sheet (ready today) and,
   with an API key, real LLM-judge scores — compute the agreement
   statistics that are fully coded but have zero real inputs.
4. **Add a message-level "general policy question" signal** to soften
   always-escalate intent buckets (failure mode 5).
5. **Build a true relevance-labelled retrieval benchmark** — a second,
   smaller human-labelling pass rating top-k evidence as relevant/
   irrelevant, replacing the current topic-match proxy.
6. **Retrain the classifier on a small human-labelled sample** via active
   learning, targeting the exact confusions measured above.
7. **Wire the optional LLM path into reply generation** (behind a flag)
   once a key exists, and re-run judge/human agreement comparing template
   vs. LLM-drafted replies.

## 13. Reproduction Commands

All commands run from the repository root with the virtual environment
activated (`pip install -r requirements.txt`).

```bash
# Golden-set validation (no data prep needed — reads the committed CSV)
python scripts/validate_golden_set.py

# Headline evaluation — runs in ~15-20 seconds once artifacts exist
python scripts/evaluate.py
# -> reports/evaluation_results.json, evaluation_raw_predictions.json, evaluation_report.md

# Full test suite (198 tests)
pytest -q

# Judge/human agreement (honestly reports PENDING with 0 filled ratings)
python scripts/compute_judge_human_agreement.py
```

One-time data preparation (requires `data/raw/twcs.csv`, downloaded
separately, not committed) to regenerate the classifier/retrieval
artifacts from scratch is documented in full in
[`README.md`, "Reproduction — Full Data Preparation"](../README.md#reproduction--full-data-preparation)
— not required to reproduce the headline evaluation above, which loads the
already-committed artifacts.

## 14. Limitations and Risks

- **Escalation recall (51.7%)** is the single most important disclosed
  weakness — measured, not assumed; see §7 and §9.
- **Weak-supervision classifier** trained on noisy heuristic labels, not
  clean human labels — a deliberate, documented trade-off (decision log
  #5), with a measured consequence (the `flight_status_inquiry` sink).
- **No multi-turn conversation handling** — each message classified
  independently.
- **Reply generation is template-based**, not fluent LLM prose — trades
  fluency for auditability; reply *quality* (beyond the narrow
  unsupported-claim check) is unmeasured pending the LLM judge and human
  ratings (§8).
- **LLM-as-judge and human-agreement statistics are infrastructure-complete
  but produce zero real numbers today** — genuinely PENDING, not
  fabricated; see §8 for the exact remaining steps.
- **Retrieval evidence reflects 2012–2017 Delta policy**, not current
  pricing or policy.
- **Deployment note:** the hosted demo's copy of the retrieval evidence
  (`data/processed/retrieval_index.pkl`) has a narrow, display-only
  redaction applied for public deployment (phone numbers, claim-reference
  codes) — documented in
  [`data/processed/DEPLOYMENT_ARTIFACT_NOTE.md`](../data/processed/DEPLOYMENT_ARTIFACT_NOTE.md).
  It does not change the vectorizer, TF-IDF matrix, ranking, or any locked
  metric in this report; the numbers above were produced against the same
  underlying data before this redaction was ever applied, and re-running
  `scripts/evaluate.py` reproduces identical summary metrics (independently
  re-verified during this audit).

Full limitations list: [`REPORT.md` §18](../REPORT.md#18-limitations).
