# Delta Customer Support AI Agent — Report

**Hiver SDE Intern Take-Home Assignment**

## Executive Summary — this is the submission-ready report

**This section alone is the assignment's required report** (problem
framing, both baselines, real results, top 5 failure modes, misleading-headline-number,
one-more-week, decision log, limitations) and is written to stand on its
own within the assignment's 6-page guideline. **Everything after it, under
"Appendix — Full Detail & Evidence," is optional supporting material** —
confusion matrices, worked reasoning, real `sample_id` citations for every
claim made here — for a reviewer who wants to verify a specific number, not
additional required reading, and it is not itself claimed to fit in 6 pages.

**Status, stated in full:** the golden set is **200/200 (100%) complete and
locked** (`data/golden_eval/golden_annotation.csv`), and the evaluation
below was run **once, for real**, against it (`reports/evaluation_results.json`,
`reports/evaluation_raw_predictions.json`) — the system was not tuned
against the golden set and re-run to chase a better number. No LLM API key
is configured and no human reply-quality ratings have been collected, so
those two pieces are marked **PENDING** below, not guessed at.

**Problem & brand.** Build an AI agent for one Twitter-support brand that
classifies intent, drafts a grounded reply, and decides auto-handle vs.
escalate — then prove it's trustworthy. **@Delta** was selected via a
10-dimension empirical scoring rubric over a 500-conversation sample per
candidate brand (Delta scored 47/50 vs. runner-up SpotifyCares's 37/50;
§2). "Good" here means never claiming an action the system can't perform
(§1, §18) — this system has no live flight-status, booking, refund, or
account access, by design.

**Data & split.** 26,168 reconstructed Delta conversations from the Kaggle
*Customer Support on Twitter* dataset, split 80/20 at the conversation-ID
level (seed=42) into a dev corpus (20,934, used for retrieval index +
classifier training) and a reserved pool (5,234, held out). Verified zero
conversation-ID leakage between them, re-checked fresh at evaluation time (§3, §12).

**Architecture** (§6): classify (TF-IDF + Logistic Regression, weak-supervised
on dev-corpus heuristic labels) → retrieve (TF-IDF/cosine, top-3, dev-corpus
only) → generate reply (deterministic, offline, strips unsupported action
claims from reused evidence) → decide escalation (deterministic policy) →
respond. Each stage is independently unit-tested.

**Golden set** (§9): 200 examples, stratified-sampled from the reserved
pool, **100% human-annotated and locked** (`data/golden_eval/golden_annotation.csv`).
Every label is `human_approved_ai_proposal` (42.5%), `human_edited_ai_proposal`
(54.5%), or `human_annotated_direct` (3%) — AI proposals were never silently
promoted to gold. 45.5% annotator/heuristic disagreement rate. Final
distribution is heavily skewed (`general_complaint_feedback`=51,
`flight_status_inquiry`=2) — this is the **locked human-approved ground
truth**, distinct from the Phase-3 *heuristic estimate* of real dataset
frequency (§5) and from real Delta traffic itself, which this golden set
does **not** claim to represent (§17).

**Baselines** (§10): a trivial baseline (majority-class + generic reply +
one flat keyword escalation rule) and a simple-retrieval baseline (1-NN
intent transfer + verbatim reply reuse + fixed intent→escalate table),
both using the identical output contract as the main agent for fair comparison.

**Headline results — real, measured once against the locked 200-example
set, not tuned and re-run** (§12):

| Metric | Main agent | Baseline 1 (trivial) | Baseline 2 (simple retrieval) |
|---|---:|---:|---:|
| Intent accuracy | **56.5%** | 25.5% | 36.0% |
| Intent macro F1 | **58.3%** | 4.1% | 35.8% |
| Escalation F1 | **63.9%** | 37.2% | 22.0% |
| **Escalation recall** | **51.7%** ⚠️ | 23.6% | 12.4% |
| Escalation FP / FN | 9 / 43 | 3 / 68 | 0 / 78 |
| Unsupported-claim rate in replies | **0.0%** (0/200) | 0.0% | 0.0% |

The main agent beats both baselines on every metric, but **escalation
recall of 51.7% is the number that matters most**: 43 of 89 cases that
needed a human were auto-handled instead. §16 shows 44% of those misses
happened even with the *correct* intent classified — an escalation-policy
regex-coverage gap, not primarily a model-accuracy problem.

**LLM-as-judge** (§14): fully implemented (rubric, caching, model/version
tracking) — **PENDING real scores**, no `ANTHROPIC_API_KEY` configured;
verified to output `pending_no_api_key` rather than a fabricated score.

**Human agreement** (§15): workflow fully implemented (20-row rating sheet
sampled from the dev corpus, agreement-statistics script) — **PENDING**,
0/20 rows rated. Nothing computed or invented in its absence.

**Top 5 real failure modes** (§16, from actual predictions —
`reports/evaluation_raw_predictions.json`): (1) escalation regex coverage
gaps, 44% of unsafe misses, worst on `seat_assignment_upgrade`; (2)
compliments/thanks misclassified into domain intents, `general_complaint_feedback`
recall only 19.6%; (3) `flight_status_inquiry` over-predicted 30x for only
2 real cases (precision 3.3%), a direct product of noisy weak-supervision
labels; (4) the highest-severity intent, `lost_damaged_baggage`, has only
46.7% recall; (5) always-escalate intent buckets over-trigger on general
policy questions (3 of 9 false positives).

**What's misleading about the headline number** (§17, condensed): the
golden set is stratified, not traffic-representative, so 56.5% is not a
production estimate in either direction; the `flight_status_inquiry` row is
computed from 2 examples and is not statistically stable; escalation F1
(0.639) is a rosier number to headline than the recall (0.517) that
actually matters; ~55% of gold labels were edited (not just approved) by
the human reviewer, which is reassuring against rubber-stamping but not
proof of zero anchoring; the 0% unsupported-claim rate is a narrow
mechanical safety check, not a quality or groundedness score.

**Capability boundary** (§18): no live flight-status API, no PNR
modification, no refund gateway, no baggage-tracing system, no
account-specific lookups — every one of these is handled by escalating to
a human, never by claiming to perform it (though §12/§16 show that
escalation policy currently misses 48% of cases that need it).

**Other limitations** (§18): no multi-turn/context-carrying conversation
handling (each message classified independently); the intent classifier is
trained on noisy weak-supervision labels, not clean human labels;
retrieval evidence reflects 2012-2017 Delta policy, not current pricing;
reply generation is template-based, not fluent LLM prose, given no API key
was available.

**One more week** (§19, top 3): fix escalation-policy regex coverage using
the annotator notes already collected (would resolve 44% of unsafe
misses); fix the `flight_status_inquiry` training-label bias; collect real
human ratings and LLM-judge scores (infrastructure is ready today).

**Decision log:** 14 non-obvious engineering decisions, [`docs/decision_log.md`](docs/decision_log.md).
**Reproduction:** [README.md](README.md); headline eval is `python scripts/evaluate.py` (seconds, once artifacts are built).

---

# Appendix — Full Detail & Evidence

## 1. Problem Framing

Build an AI support agent for one Twitter-support brand that (1) classifies
an incoming customer message into a small intent taxonomy, (2) drafts a
reply grounded in how that brand has historically resolved similar issues,
and (3) decides auto-handle vs. escalate with a stated reason — then prove
the system is trustworthy enough to actually use.

**What "good" means for this brand:** Delta's own historical support
account never performs live actions in-thread (it never *actually* rebooks
a flight or issues a refund inside a tweet — it always says "please DM your
confirmation number"). So "good" for this agent is **not** "acts like a
real airline backend." It is:
- Correctly separating "I can answer this from public policy" from "this
  needs a human with account access" — and never claiming the latter is the
  former.
- Giving accurate, evidence-grounded policy answers for the auto-handle
  cases.
- Failing *safe*: an intent misclassification that still gets correctly
  escalated is a much smaller problem than one that gets wrongly
  auto-handled.

**What we chose not to build:**
- No live flight status, GDS/booking, seat inventory, baggage tracking,
  account, or refund/payment integration. The dataset doesn't provide any
  of these systems, and pretending to have them would make every "auto-handled"
  reply potentially fabricated. See `docs/escalation_policy.md`.
- No multi-turn conversation state / follow-up handling — each customer
  message is classified independently, first-message-in-thread style,
  matching how the dataset itself is structured for retrieval.
- No fine-tuned language model — every reply-generation and classification
  component here is either a deterministic rule/policy engine or a small
  classical ML model (TF-IDF + Logistic Regression), because no LLM API key
  is available in this environment (see §12).

---

## 2. Why Delta

Selected via a 10-dimension empirical scoring rubric (data volume,
conversation richness, intent diversity, resolution quality, retrieval
usefulness, golden-set constructability, escalation-policy opportunities,
challenge/interest, hallucination risk, overall fit) applied to a
stratified 500-conversation sample from each of the 5 largest brand
accounts in the dataset. Delta scored 47/50, well ahead of the runner-up
(SpotifyCares, 37/50) — driven mainly by a much higher public resolution
rate (20.4% vs. 12.2% for Apple, 9.2% for Amazon) and cleanly
non-overlapping intent categories. Full methodology and scoring table:
[`docs/brand_selection.md`](docs/brand_selection.md).

---

## 3. Dataset

- **Source:** Kaggle *Customer Support on Twitter* (`thoughtvector/customer-support-on-twitter`), `twcs.csv`, 2,811,774 rows, 516.5 MB.
- **Delta subset:** 87,994 messages (42,253 outbound from @Delta + 45,296 inbound + 445 third-party thread messages), reconstructed into **26,168 conversations** (26,166 genuinely two-way).
- **Date range:** 2012-11-27 to 2017-12-03 (so policy facts referenced here — e.g. baggage fees — reflect 2017-era Delta policy, not necessarily current pricing).
- The raw 500 MB file is **not** committed to this repo (see `.gitignore` /
  reproduction instructions) — this satisfies "we will not run your code on
  the full dataset" from the assignment; everything downstream operates on
  the already-reconstructed 26,168-conversation subset.

## 4. Conversation Reconstruction

Each Delta conversation thread is reconstructed from the raw tweet graph
(`tweet_id` / `parent_id` / `conversation_id` fields) into one row containing
the full customer text, all Delta reply text concatenated, message counts,
and two-way/one-way status (`scripts/build_delta_dataset.py`). Two
one-way announcement threads are correctly counted as one-way, not
discarded silently.

## 5. Intent Taxonomy

10 intents, defined empirically from the data (not copied from an existing
label set), each with an explicit auto-handle/escalate default and worked
disambiguation rules for confusing pairs (status-vs-rebooking,
policy-vs-claim, seat-vs-refund). Full definitions, representative examples,
and disambiguation rules: [`docs/intent_taxonomy.md`](docs/intent_taxonomy.md).

| Intent | % of dataset (heuristic) | Default handling |
|---|---:|---|
| `flight_status_inquiry` | 44.6% | Auto (info only) |
| `general_complaint_feedback` | 37.5% | Auto |
| `seat_assignment_upgrade` | 5.7% | Auto (info) / Escalate (specific booking) |
| `skymiles_loyalty_program` | 3.7% | Auto (general) / Escalate (account-specific) |
| `checkin_boarding_pass` | 3.1% | Auto (general) / Escalate (account-specific) |
| `flight_delay_rebooking` | 2.3% | **Always escalate** |
| `inflight_amenities_service` | 1.6% | Auto / Escalate (compensation request) |
| `baggage_allowance_policy` | 0.7% | Auto |
| `refund_credit_voucher` | 0.65% | **Always escalate** |
| `lost_damaged_baggage` | 0.23% | **Always escalate** |

These percentages are from the Phase-3 single-tier keyword heuristic
(`src/heuristic_intent.py`) applied to all 26,168 conversations — a weak
signal used for taxonomy sanity-checking and dataset stratification, **not**
a claim about true intent distribution (it is documented to be wrong
roughly 15-25% of the time; see `docs/golden_annotation_guide.md` §2).

## 6. Architecture

```
CUSTOMER MESSAGE
      |
      v
INTENT CLASSIFICATION   src/classifier.py
      |                 TF-IDF + Logistic Regression, weak-supervised on the
      |                 dev corpus's heuristic labels (20,933 examples).
      v                 Deliberately a DIFFERENT algorithm from the proposer
      |                 used to assist golden-set annotation (see decision #5).
      v
HISTORICAL RETRIEVAL    src/retrieval.py
      |                 TF-IDF + cosine similarity over 20,913 retrieval-
      |                 eligible dev-corpus evidence units. Reserved/golden
      v                 pool is never indexed (verified, see §8).
      v
EVIDENCE                top-k (k=3) historical (customer_message, delta_response)
      |                 pairs + similarity score + weak intent label.
      v
REPLY GENERATION        src/reply_generator.py
      |                 Deterministic, offline. Strips @mentions/signatures/
      |                 tracking links and, critically, strips first-person
      v                 completed-action claims from reused evidence text
      |                 before it reaches a new customer (decision #7).
      v
ESCALATION DECISION     src/escalation.py
      |                 Deterministic policy: always-escalate intents +
      |                 account-signal regexes + intent-conditional signals.
      v                 Shared with the annotation-assist proposer (decision #6).
      v
FINAL RESPONSE          { intent, reply, escalate, escalation_reason, evidence }
```

Every stage is independently testable and unit-tested (`tests/test_classifier.py`,
`tests/test_retrieval.py`, `tests/test_reply_generator.py`, `tests/test_escalation.py`,
`tests/test_agent.py` for the wiring). `src/agent.py`'s `DeltaSupportAgent`
only orchestrates calls between stages — no business logic lives in the
orchestrator itself.

**LLM path (optional, currently unconfigured):** `src/llm_client.py` wraps
the Anthropic API behind an `is_available()` / `complete()` interface that
returns `None` (never a fabricated string) when `ANTHROPIC_API_KEY` isn't
set. Nothing in the pipeline currently calls it for reply generation — the
deterministic path above is the default and only path exercised in this
report. It exists so a reviewer can drop in a key and get LLM-drafted
replies without any code changes to the pipeline shape.

## 7. Retrieval

TF-IDF (`max_features=25000`, unigrams+bigrams, English stopwords) +
cosine similarity, fit once on the 20,913 retrieval-eligible dev-corpus
conversations, serialized to `data/processed/retrieval_index.pkl` (12.4 MB).
Evidence unit = full customer query + full concatenated Delta response
chain (chosen over single-reply-pair or LLM-summarized alternatives — see
trade-off table in `docs/retrieval_design.md` §1).

**Eligibility filter:** customer message >5 chars, Delta response >15 chars,
two-way only — excludes 21 of 20,934 dev conversations and 5 of 5,234
reserved conversations (mostly single-emoji Delta replies). Documented with
the exact excluded IDs and reasons in `docs/golden_set_methodology.md` §3.

**Retrieval quality:** a 20-example qualitative sanity check on
*development*-corpus queries (never golden/reserved) found 100% top-1
topical relevance, similarity scores 0.35-0.78 for strong matches (full
report: `reports/retrieval_sanity_check.md`). This is a small, manually-eyeballed
check, not a statistically powered benchmark — treat the 100% figure as "no
obvious retrieval failures in this small manual spot-check," not as a
precision estimate. A larger-scale automated proxy metric
(`retrieval_intent_match_at_k` in `src/metrics.py`) is implemented and unit
tested; running it against the golden set is gated on annotation completion
(§9, §11).

## 8. Escalation Policy

Deterministic policy (`src/escalation.py`), not a learned classifier:
- **Always escalate:** `flight_delay_rebooking`, `lost_damaged_baggage`, `refund_credit_voucher`.
- **Escalate on account-signal regexes** in the customer message (confirmation numbers, "my account", "my SkyMiles number", "file a claim", etc.) regardless of intent.
- **Intent-conditional escalation signals** for `seat_assignment_upgrade`, `skymiles_loyalty_program`, `checkin_boarding_pass`, `flight_status_inquiry` (e.g. "stranded", "stuck"), `inflight_amenities_service` (refund/compensation requests), `baggage_allowance_policy` (booking-specific references).
- Full intent-level decision matrix with worked examples: [`docs/escalation_policy.md`](docs/escalation_policy.md).

This same module is shared by the golden-set annotation-assist proposer —
see decision #6 for why that's not a leakage concern (it's a policy, not a
label fit to data).

## 9. Golden-Set Methodology

**200 examples**, stratified-sampled (seed=42) from the **reserved golden
pool only** (5,234 conversations, 20% split at the conversation-ID level,
never used to fit the retrieval index or any classifier). Rare high-risk
escalation intents are deliberately oversampled beyond their natural
frequency (e.g. `lost_damaged_baggage`: 0.23% of the dataset but 15 of 200
golden examples) so the evaluation can actually say something about them —
full allocation table in `docs/escalation_policy.md` §3 and
`docs/golden_set_methodology.md` §4.

**Annotation workflow:** AI/rule-assisted proposals
(`scripts/propose_golden_labels.py`, `src/proposer.py` `rule_based_v1`) are
reviewed and explicitly approved or edited by a human
(`scripts/review_golden_proposals.py`) — proposals are never silently
promoted to gold. Every completed row records `annotation_source` ∈
`{human_annotated_direct, human_approved_ai_proposal, human_edited_ai_proposal}`.
Full protocol: [`docs/golden_set_methodology.md`](docs/golden_set_methodology.md),
[`docs/golden_annotation_guide.md`](docs/golden_annotation_guide.md).

**Status: 200/200 (100%) complete and locked.** `scripts/validate_golden_set.py`
reports **all 14 required gates PASS** and all 3 advisory checks OK.
Annotator/heuristic disagreement rate on the final 200: **45.5%** — real
evidence the human reviewer was not rubber-stamping the sampling heuristic
(nearly half of proposed/heuristic labels were changed on independent
review). Provenance breakdown of the 200 final labels:

| `annotation_source` | Count | % |
|---|---:|---:|
| `human_edited_ai_proposal` (reviewer changed at least one field) | 109 | 54.5% |
| `human_approved_ai_proposal` (reviewer accepted as proposed) | 85 | 42.5% |
| `human_annotated_direct` (labelled without the proposal workflow) | 6 | 3.0% |

**Final locked golden-intent distribution** (this is the human-approved
ground truth used for evaluation — **not** the Phase-3 heuristic estimate in
§5, and **not** a claim about real Delta traffic; see §17 point 1):

| Intent | Golden count (n=200) |
|---|---:|
| `general_complaint_feedback` | 51 |
| `flight_delay_rebooking` | 25 |
| `inflight_amenities_service` | 23 |
| `refund_credit_voucher` | 21 |
| `seat_assignment_upgrade` | 21 |
| `skymiles_loyalty_program` | 17 |
| `lost_damaged_baggage` | 15 |
| `baggage_allowance_policy` | 13 |
| `checkin_boarding_pass` | 12 |
| `flight_status_inquiry` | **2** |

`flight_status_inquiry` landing at only 2 examples is a real, notable
consequence of the annotation process, not an error in it: during
human review, the large majority of messages the sampling heuristic had
proposed as `flight_status_inquiry` turned out — on independent reading —
to actually be compliments, delay reports, or amenity feedback (the
heuristic's known weakness of defaulting ambiguous "flight"-containing
messages to this bucket, documented in `docs/golden_annotation_guide.md`).
**Any per-intent metric reported for `flight_status_inquiry` in §12 is
based on n=2 and must not be read as a stable estimate of anything** — it
is reported transparently rather than hidden, but one or two examples
cannot support a meaningful precision/recall claim in either direction.
`escalate=yes`: 89, `escalate=no`: 111. Difficulty: 87 easy / 83 ambiguous /
30 hard.

**Leakage checks performed (independently re-verified during this project's
audit, not just trusted from prior documentation):**
- Zero conversation-ID overlap between dev corpus (20,934) and reserved pool (5,234).
- Zero duplicate conversation_ids within either split, and within the golden set.
- All 200 golden `conversation_id`s confirmed present in the reserved pool and absent from the dev corpus.
- One **text-level** (not conversation-level) coincidence is documented honestly: a single customer message appears verbatim across 35 dev-corpus conversations (a broadcast/retweet artifact in the raw Twitter data) — the golden example's own conversation_id is confirmed not present in the retrieval index. This is disclosed rather than hidden. See `docs/golden_set_methodology.md` §6.

## 10. Baselines

Two non-LLM baselines (`src/baselines.py`), documented in code and here:

**Baseline 1 — Trivial** (`TrivialBaseline`): always predicts the single
majority intent class from the dev corpus's heuristic distribution
(`general_complaint_feedback`); one fixed generic reply regardless of
content; a single flat keyword rule for escalation (escalate only if the
message contains "refund", "cancel", "lost", or "damaged" — no per-intent
policy at all). Uses no retrieval.

**Baseline 2 — Simple retrieval** (`SimpleRetrievalBaseline`): 1-nearest-neighbor
intent transfer using the *same* TF-IDF retrieval index as the main system
(no trained classifier — pure lookup); reuses the top-1 retrieved historical
Delta reply **verbatim**, with no cleanup (deliberately naive, so its
grounding-safety failures — e.g. reused first-person action claims — are
visible in evaluation rather than hidden); escalation via a fixed
intent→escalate lookup table (only the 3 always-escalate intents trigger
escalation; no message-level signal detection at all).

Both baselines expose the identical `.handle(message)` output contract as
the main agent so they can be evaluated identically (`scripts/evaluate.py`).
Baseline 2 deliberately reuses the main system's retriever so the
comparison isolates classification/reply/escalation quality rather than
confounding it with retrieval differences (decision #10).

## 11. Evaluation Methodology

`src/metrics.py` (pure functions, unit-tested on synthetic fixtures —
`tests/test_metrics.py`) implements:

- **Intent:** accuracy, macro F1, per-intent precision/recall/F1/support, full confusion matrix (`intent_metrics`).
- **Escalation:** accuracy, precision, recall, F1, full TP/TN/FP/FN confusion matrix, and explicit false-positive/false-negative index lists (`escalation_metrics`). Every false negative here (gold=escalate, predicted=auto-handle) is labelled "unsafe" throughout this report — the single most dangerous error type for this system.
- **Retrieval (supplementary, not a required deliverable metric):** an explicitly-labelled *proxy*, intent-match@k — whether any of the top-k retrieved evidence items' weak intent label matches the query's gold intent (`retrieval_intent_match_at_k`). Disclosed as a proxy, not a true relevance judgment, because a true relevance-labelled retrieval benchmark would need a second human-labelling pass not in scope here.
- **Reply safety (automated, non-LLM):** fraction of generated replies containing an unsupported account-action claim (`unsupported_claim_rate`), reusing the same detector that gates the reply generator itself. This is the *only* automated reply-quality metric the code implements — no grounding/relevance score exists in code, and none is claimed below.

`scripts/evaluate.py` is the single reproduction entry point. It **hard-refused
to compute any golden-set accuracy metric until `scripts/validate_golden_set.py`
reported all required gates passing** (decision #9), and re-verifies leakage
freshly at evaluation time (not from a cached assumption) before computing
anything. It has now been run once, for real, against the complete 200-example
golden set — results below (§12) are that one run's output, saved verbatim to
`reports/evaluation_results.json` and `reports/evaluation_raw_predictions.json`.

## 12. Results

**Real, measured, run once against the locked 200-example golden set.**
Leakage was re-verified immediately before this run (0 golden conv_ids in
the dev corpus, 0 in the fitted retrieval index — see `reports/evaluation_results.json["leakage_check"]`).

### Main agent — intent (n=200)

**Accuracy: 56.5%. Macro F1: 58.3%.**

| Intent | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| `flight_status_inquiry` | 0.033 | 0.500 | 0.063 | **2** ⚠️ |
| `flight_delay_rebooking` | 0.625 | 0.600 | 0.612 | 25 |
| `baggage_allowance_policy` | 0.611 | 0.846 | 0.710 | 13 |
| `lost_damaged_baggage` | 1.000 | 0.467 | 0.636 | 15 |
| `seat_assignment_upgrade` | 0.577 | 0.714 | 0.638 | 21 |
| `skymiles_loyalty_program` | 0.619 | 0.765 | 0.684 | 17 |
| `refund_credit_voucher` | 0.813 | 0.619 | 0.703 | 21 |
| `checkin_boarding_pass` | 0.600 | 1.000 | 0.750 | 12 |
| `inflight_amenities_service` | 0.800 | 0.696 | 0.744 | 23 |
| `general_complaint_feedback` | 0.556 | **0.196** | 0.290 | 51 |

⚠️ = support of 2; see §9 and §17 — this row is not a statistically stable
estimate and is reported for transparency, not as a claim. Full 10×10
confusion matrix: `reports/evaluation_results.json`.

### Main agent — escalation (n=200)

Accuracy **0.740**, precision **0.836**, recall **0.517** (⚠️ the important
number — see §16/§17), F1 **0.639**.

| | Predicted: no | Predicted: yes |
|---|---:|---:|
| **Gold: no** | TN = 102 | **FP = 9** |
| **Gold: yes** | **FN = 43 (unsafe)** | TP = 46 |

43 of 89 real escalate-needed cases (48%) were auto-handled instead of
routed to a human. That is the single most important number in this report.

### Reply safety (automated, non-LLM; n=200, main agent)

**0 of 200** generated replies contained a detected unsupported
account-action claim (`unsupported_claim_rate = 0.0`). This is the only
automated reply-quality metric the code implements (§11) — no groundedness
or relevance score is claimed.

### Baseline comparison (same 200 examples, same metrics, same code path)

| System | Intent acc. | Intent macro F1 | Escalation acc. | Escalation P / R / F1 | Escalation FP / FN |
|---|---:|---:|---:|---:|---:|
| **Main agent** | **0.565** | **0.583** | **0.740** | 0.836 / **0.517** / 0.639 | 9 / 43 |
| Baseline 1 — trivial | 0.255 | 0.041 | 0.645 | 0.875 / 0.236 / 0.372 | 3 / 68 |
| Baseline 2 — simple retrieval | 0.360 | 0.358 | 0.610 | 1.000 / 0.124 / 0.220 | 0 / 78 |

The main agent beats both baselines on every metric. Sanity check on
baseline 1 (predicts `general_complaint_feedback` for every message):
51/200 = 0.255 exactly matches its accuracy, and its recall for that one
class is exactly 1.0 — confirms the trivial baseline is implemented
correctly, not artificially weakened to flatter the main system. Baseline
2's near-zero false positives (0) alongside its worst-in-class recall
(0.124) shows its fixed intent→escalate lookup table almost never triggers
— it is *safe-looking* only because it almost never escalates at all,
which is itself informative: a system that escalates rarely will show high
precision/low FP by construction, not because it is good at the task.

### What actually drove the errors (see §16 for the full analysis)
19 of the 43 unsafe escalation misses (44%) happened even when the intent
was classified *correctly* — a gap in the escalation policy's regex
coverage, not a classification error. The remaining 24 trace mostly to one
classifier bug: `flight_status_inquiry` was predicted 30 times but correct
only once (precision 0.033), acting as a "sink" for short/ambiguous
messages — a direct consequence of the noisy weak-supervision training
labels (decision #5).

## 13. Reply Evaluation

Reply quality is meant to be judged by an LLM rubric (§14, PENDING) plus one
fully local, non-LLM automated check that **has** actually run:
`unsupported_claim_rate` (§11/§12), which flags any reply claiming a
specific account action was performed. Measured result: **0/200 (0.0%)**
generated replies from the main agent tripped this check. Unit tests
(`tests/test_reply_generator.py`) verify the generator's own claim-stripping
logic; this evaluation-time number confirms it held on the full golden set,
not just the unit-test fixtures — but it is a narrow safety check, not a
measure of reply relevance, fluency, or correctness (those need the LLM
judge / human ratings in §14–§15, both still pending).

## 14. LLM-as-Judge

Fully implemented (`src/llm_judge.py`), **PENDING actual scores** — no
`ANTHROPIC_API_KEY` is configured in this environment.

**Rubric (`RUBRIC_VERSION = "delta_reply_judge_v1"`)**, 1-5 per dimension:
relevance, correctness, groundedness, helpfulness, safety (any claim of an
unsupported action scores safety=1 regardless of other dimensions). Full
prompt text: `src/llm_judge.RUBRIC`.

Stored per judgment: rubric version, model ID, raw per-dimension scores,
aggregate (mean), and — for failed/unparseable responses — the raw response
text for debugging. Results are cached to `data/eval/llm_judge_cache.json`
keyed by a hash of (rubric version, model, message, reply), so re-running
never re-spends API calls on unchanged inputs.

Running `scripts/build_human_rating_sheet.py` today (no key configured)
correctly writes `pending_no_api_key` for all 20 sampled dev-corpus
examples — verified, not asserted. Set `ANTHROPIC_API_KEY` and re-run for
real scores.

## 15. Human Agreement

**Workflow built, PENDING real human ratings.**
`scripts/build_human_rating_sheet.py` samples 20 examples from the
**development corpus** (not the golden set — decision #8, so this doesn't
wait on annotation completion), runs the live agent, and writes
`data/eval/human_rating_sheet.csv` with blank `human_relevance`,
`human_correctness`, `human_groundedness`, `human_helpfulness`,
`human_safety`, `human_notes` columns for a real person to fill in (1-5 each).

`scripts/compute_judge_human_agreement.py` computes exact agreement,
within-one agreement, Pearson r, Spearman r, and linear-weighted Cohen's
kappa per dimension — **only once both** real human ratings and real judge
scores exist for overlapping rows; otherwise it prints exactly what's
missing and computes nothing. Verified today: with an empty rating sheet,
it correctly reports `[PENDING]` and exits without writing any output file.

**Action needed from the project owner:** open
`data/eval/human_rating_sheet.csv` and fill in the six `human_*` columns
(1-5, plus optional notes) for the 20 sampled replies.

## 16. Top 5 Failure Modes

All five are read directly from `reports/evaluation_raw_predictions.json`
(the main agent's actual per-example predictions on the 200 golden
examples) — not inferred from the taxonomy, not generic ML-failure prose.

**1. Escalation policy misses real account-specific phrasing even when
intent is classified correctly** (44% of all unsafe misses). 19 of the 43
unsafe false negatives had the *correct* gold intent — the deterministic
regex signals in `src/escalation.py` simply didn't match the actual
wording used. Worst-hit: `seat_assignment_upgrade` (9 of its 13 unsafe
FNs). Real examples: **S027** ("need seating assistance... please follow me
so I may DM"), **S093** ("did it again. Took away my exit aisle..."),
**S191** ("any way to reverse transfer of sky miles?"), **S200** ("paid for
an exit seat. This seat is not that."). *Cause:* hand-written regex
patterns, not learned from data, cannot cover the full space of natural
phrasing. *Impact:* the single largest contributor to unsafe escalation
misses — larger than any classifier error. *Mitigation:* mine the
annotator notes captured during golden-set QC (which state *why* each case
needed escalation) to expand regex coverage, or train a small supervised
escalation classifier on the now-complete 200 gold labels.

**2. Compliments/retrospective thanks misclassified into domain-specific
intents.** `general_complaint_feedback` has the worst recall of any intent
(0.196, 10/51 correct); 10 were misclassified as `seat_assignment_upgrade`,
10 as `flight_status_inquiry`, 6 as `baggage_allowance_policy`. Real
examples: **S050**, **S149**, **S186**, **S196**, **S198** — all "thanks for
the upgrade" style messages. *Cause:* the TF-IDF+LogReg classifier keys on
domain nouns ("upgrade," "seat," "miles") regardless of tense or sentiment,
a weakness inherited from its weak-supervision training labels (§6,
decision #5). *Impact:* low safety risk (this intent never escalates) but a
real customer-experience/groundedness risk — a compliment could get an
off-topic policy reply instead of a simple acknowledgment.

**3. `flight_status_inquiry` acts as a noise sink for short/ambiguous
messages.** Only 2 real golden examples are this intent, but the
classifier predicted it 30 times — **29 of those 30 were wrong**
(precision 0.033). This single pattern caused 14 of the 24
intent-driven unsafe escalation misses. Real examples: **S002**, **S003**,
**S022**, **S031**, **S047**. *Cause:* traced directly to the weak-supervision
training labels — the underlying single-tier heuristic (`src/heuristic_intent.py`)
itself over-assigns `flight_status_inquiry` on scoring ties, and the
classifier learned that same bias.

**4. The highest-severity intent, `lost_damaged_baggage`, has only 46.7%
recall** (7/15) — over half of real lost-baggage reports get misrouted,
mostly into the noise sink from failure mode 3. Real examples: **S022**
("bag was checked to Tunis but delivered to Baggage Claim 10 at JFK"),
**S126** ("HELP I LOST MY BAGS ON YOUR FLIGHT"), **S147** ("bag shipped on a
separate flight... missed an audition"). 7 of these 8 misclassifications
became unsafe false negatives (the 8th landed on `flight_delay_rebooking`,
which is also always-escalate, so it accidentally stayed safe). *Impact:*
this is the intent where a miss is most consequential — a customer with
physically lost luggage getting an auto-handled non-answer.

**5. The always-escalate intent list over-triggers on general policy
questions, causing 3 of the 9 total false positives.** **S092** ("under
what circumstances can I get a refund?"), **S136** ("why can't I use my
eCredit to buy a ticket for someone else?"), and **S064** ("how long will it
take to confirm my status challenge?") were all classified with the
*correct* intent but auto-escalated purely because that intent belongs to
an always-escalate bucket — even though a human reader (and Delta's own
historical agent, in two of the three) judged these answerable from public
policy alone. *Cause:* escalation keyed on intent alone, with no
message-level "is this a general policy question" signal. *Impact:*
efficiency/cost, not safety — unnecessary human workload on simple
questions.

## 17. What Is Misleading About My Headline Number?

If I only told you "56.5% intent accuracy, beats both baselines," here is
everything wrong with stopping there:

1. **The golden set is stratified, not representative — and the
   distortion is now visible in real numbers, not a hypothetical.**
   `flight_status_inquiry` is ~44.6% of the dataset by the *Phase-3
   heuristic estimate* (§5 — an exploratory, admittedly-noisy signal, not
   ground truth) but only **2 of 200 (1%)** in the *locked human-approved
   golden set* (§9), because most examples the heuristic guessed were that
   intent turned out on human review to be something else. `lost_damaged_baggage`
   is ~0.23% of the dataset by the same heuristic but 7.5% of the golden
   set (deliberately oversampled). A single headline "intent accuracy = 56.5%"
   is an average over this artificial mix, not over what a live Delta
   support queue would actually look like — and given the model's accuracy
   varies wildly by intent (100% on `checkin_boarding_pass`, essentially
   random on the 2-example `flight_status_inquiry` row), a production-traffic-weighted
   accuracy would almost certainly land at a **different number**, not
   simply a scaled version of this one.
2. **The `flight_status_inquiry` per-intent row (precision 0.033, recall
   0.5) is not a stable estimate of anything** — it is computed from 2
   examples. Reporting it at all (rather than hiding it) is the honest
   choice, but treating that 0.033 as "the model is bad at flight status"
   would itself be a misleading reading of a genuinely tiny sample.
3. **Escalation F1 (0.639) looks fine in isolation; escalation recall
   (0.517) is the number that should actually worry a reviewer**, and it's
   easy to headline the F1 instead because it's the bigger number. 43 of 89
   cases that needed a human did not get one. §16 shows this is
   overwhelmingly an escalation-*policy* coverage gap (regex patterns too
   narrow), not primarily a classification problem — meaning it is fixable
   without retraining anything, but it is not fixed today.
4. **45.5% of the final gold labels came from a proposal the human
   reviewer edited, and 42.5% came from a proposal accepted as-is**
   (`data/golden_eval/golden_annotation.csv` `annotation_source` column,
   §9). The high edit rate is real evidence against rubber-stamping, but a
   fully independent second annotator, working from a blank slate with no
   proposal shown at all, might still draw some boundaries differently —
   especially on the 83 examples marked `difficulty=ambiguous`.
5. **The 0% unsupported-claim rate is a narrow, mechanical safety check,
   not a groundedness or quality score.** It only proves the reply
   generator's regex-based claim-stripping held on all 200 examples; it
   says nothing about whether a reply was actually *useful*, *correct*, or
   *relevant* — that requires the LLM judge and human ratings, both
   PENDING (§14, §15).
6. **"Unsafe false negative" is only as good as the escalation policy it's
   measured against.** If the policy itself is missing a genuinely risky
   pattern entirely (rather than failing to detect a pattern it already
   encodes), this metric can't see that gap — it only measures conformance
   to the policy as written, not whether the policy is complete.
7. **Retrieval quality is only measured with a proxy metric** (§7, §11) — a
   high intent-match@k rate says the retriever found text about the same
   *topic*, not that the specific retrieved case was actually the most
   useful evidence for grounding this specific reply.

## 18. Limitations

### Known capability boundary — what this system cannot do (by design)

This is not an incomplete-feature list, it's a deliberate boundary. The
agent **never** claims to do any of the following, and the reply generator
actively strips language that would imply otherwise (decision #7):

- **No live Delta flight-status API** — no real-time gate/delay/schedule lookup.
- **No PNR/booking modification** — cannot change a seat, itinerary, or reservation.
- **No refund/payment gateway** — cannot process a refund, eCredit, or voucher.
- **No baggage-tracing system** — cannot look up or file a real baggage claim.
- **No customer-account access** — cannot look up SkyMiles balance, Medallion status, or booking history for a specific person.

**Every one of the above is handled by escalating to a human agent**, not
by pretending to perform it — that is the entire reason the escalation
policy (§8) exists. Whether that policy is *complete enough* in practice is
a separate, measured question — see the 51.7% escalation recall (§12) and
failure mode #1 (§16), which shows it currently is not.

### Other limitations

- No multi-turn/context-carrying conversation handling — each message is classified independently.
- Weak-supervision classifier trained on noisy heuristic labels, not clean
  human labels (decision #5) — now *measured*, not hypothetical: 56.5%
  intent accuracy, with `flight_status_inquiry` acting as a systematic
  error sink (§16 failure mode 3).
- Escalation policy is hand-written regex, not learned from the (now
  available) 200 gold labels — measured to miss 44% of unsafe cases purely
  on coverage gaps, independent of classifier accuracy (§16 failure mode 1).
- Retrieval evidence reflects 2012-2017 Delta policy and shortlinks; not
  current pricing/policy.
- No fuzzy/semantic near-duplicate detection was performed when sampling
  the golden set (acknowledged in `docs/golden_set_methodology.md` §4.4).
- Reply generation is template-based, not fluent LLM prose — trades
  fluency for auditability/safety given no API key was available; reply
  *quality* (as opposed to the narrow safety check in §12/§13) is unmeasured
  pending the LLM judge and human ratings (§14, §15).
- LLM-as-judge and human-agreement statistics are infrastructure-complete
  but produce zero real numbers today — genuinely PENDING, not fabricated
  (§14, §15).

## 19. What I Would Do With One More Week

Ranked by the actual evidence in §16, not a generic wishlist:

1. **Fix escalation recall first — it's the real safety gap.** Mine the 43
   unsafe-FN examples' `annotator_notes` (written during golden-set QC,
   already explaining *why* each needed escalation) to expand
   `CONDITIONAL_ESCALATE` regex coverage in `src/escalation.py`, starting
   with `seat_assignment_upgrade` (9 of 13 misses). This alone, per §16
   failure mode 1, would fix 44% of unsafe misses without touching the
   classifier at all.
2. **Fix the `flight_status_inquiry` noise-sink** (§16 failure mode 3):
   retrain the weak-supervision classifier after correcting the underlying
   single-tier heuristic's tie-breaking bias, or drop `flight_status_inquiry`
   from the weak-label vocabulary entirely and route ties to
   `general_complaint_feedback` instead, matching what human review
   actually did 96% of the time on this session's golden set.
3. **Collect real human ratings** on `data/eval/human_rating_sheet.csv`
   (20 rows, dev-corpus, ready today) and, with an `ANTHROPIC_API_KEY`, real
   LLM-judge scores; compute the agreement statistics in §15 that are fully
   coded but have zero real inputs today.
4. **Add a message-level "general policy question" signal** to soften the
   always-escalate intent buckets (§16 failure mode 5) — a cheap regex for
   "what is your policy," "under what circumstances," "why can't I," etc.,
   layered before the intent-level always-escalate check.
5. **Build the true relevance-labelled retrieval benchmark** flagged as
   out-of-scope in §7/§11 (a second, smaller human-labelling pass rating
   top-k evidence per query as relevant/irrelevant) — the current
   intent-match@k proxy (0.575 for the main agent) says nothing about
   whether the *specific* retrieved case was the most useful one.
6. **Retrain the intent classifier on a small human-labelled dev-corpus
   sample** via active learning, targeting exactly the confusions §16
   quantifies (general-complaint-vs-domain-intent, and the
   flight-status-inquiry sink), rather than continuing to train on the
   noisy weak-supervision labels.
7. **Wire the optional LLM path** (`src/llm_client.py`) into reply
   generation itself (behind a flag) once an API key exists, and re-run the
   judge/human-agreement comparison between template replies and
   LLM-drafted replies.

## 20. Decision Log

10-15+ non-obvious decisions with rationale: [`docs/decision_log.md`](docs/decision_log.md).

See also [`docs/requirements_checklist.md`](docs/requirements_checklist.md)
for a full requirement-by-requirement final adversarial review
(PASS/PENDING/FAIL with cited evidence for every line).

## 21. Reproduction Instructions

See [README.md](README.md) for the full command sequence (data prep vs.
headline evaluation, separated per the assignment's <15-minute requirement).
Quick summary:

```powershell
# One-time data preparation (requires data/raw/twcs.csv, ~500MB, not committed)
python scripts/build_delta_dataset.py
python scripts/build_retrieval_index.py
python scripts/train_classifier.py

# Headline evaluation (fast — loads pre-built artifacts only)
python scripts/evaluate.py

# Full test suite
pytest
```
