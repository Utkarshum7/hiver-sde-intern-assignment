# Golden Evaluation Set — Methodology Document

**Version:** 1.1  
**Brand:** @Delta  
**Phase:** 5A — Golden Set Construction  

---

## 1. Overview

This document describes the construction methodology for the 200-example hand-labelled
Golden Evaluation Set used to assess the Delta customer-support AI agent.

The golden set provides the **only objective human ground truth** in this project.
It is isolated from all model training, retrieval index construction, and hyperparameter
tuning. No automated labelling (LLM or heuristic) was used to produce the final gold labels.

---

## 2. Source Data

| Property | Value |
|---|---|
| Raw dataset | Customer Support on Twitter (Kaggle, `twcs.csv`) |
| Brand | @Delta (handle `delta`) |
| Total Delta conversations | 26,168 |
| Reserved golden candidate pool | 5,234 conversations (20% split, seed=42) |
| Retrieval-eligible candidates | 5,229 conversations |

The reserved pool was isolated **before** the TF-IDF retrieval index was built.
It has never been used for any model or retrieval optimization.

---

## 3. Retrieval Eligibility Filter

5 of the 5,234 reserved conversations are excluded from sampling because they fail the
retrieval eligibility criteria defined in `src/retrieval.py`:

- Customer message must be non-empty and > 5 characters
- Delta response must be > 15 characters and contain actionable content

The 5 excluded conversations and their failure reasons:

| conversation_id | Failure reason |
|---|---|
| 279537 | Delta response is a single emoji heart (`❤ *HSD`) — 8 chars, below threshold |
| 1321788 | Delta response is a single emoji heart (`❤ *HSD`) — below threshold |
| 429183 | Customer message is **empty string** — fails > 5 char check |
| 176613 | Delta response is a single emoji heart (`❤ *ACJ`) — below threshold |
| 917132 | Delta response is a single emoji heart (`❤ *HSD`) — below threshold |

These exclusions are correct. An empty customer message or a one-emoji-reply cannot
support meaningful evaluation of intent classification or reply quality.

---

## 4. Sampling Strategy

### 4.1 Why Stratified Over Uniform

Uniform random sampling from 5,229 conversations would produce a skewed set dominated
by `flight_status_inquiry` (~40%) and `general_complaint_feedback` (~43%).
Rarer but important intents such as `lost_damaged_baggage` (0.1%) and
`baggage_allowance_policy` (0.6%) would be underrepresented or absent, leaving the
evaluation unable to test those failure modes.

### 4.2 Heuristic Pre-Stratification

The Phase 3 keyword-based heuristic classifier (from `scripts/analyze_delta_intents.py`)
was applied to the 5,229 retrieval-eligible reserved conversations to estimate their
likely intent **for sampling purposes only**.

> **Important:** The heuristic label is a weak keyword-match signal used to partition
> the pool into approximate groups before sampling. It is **not** the gold label and is
> not used in any model or evaluation computation. The human annotator independently
> determines the correct `gold_intent` for each example.

| Intent | Heuristic count in pool | Target sample | Actual sample |
|---|---|---|---|
| `flight_status_inquiry` | 2,103 | 32 | 32 |
| `flight_delay_rebooking` | 113 | 25 | 25 |
| `baggage_allowance_policy` | 32 | 19 | 19 |
| `lost_damaged_baggage` | 7 | 10 → **capped at 7** | 7 |
| `seat_assignment_upgrade` | 280 | 25 | 25 |
| `skymiles_loyalty_program` | 191 | 20 | 20 |
| `refund_credit_voucher` | 38 | 19 | 19 |
| `checkin_boarding_pass` | 159 | 20 | 20 |
| `inflight_amenities_service` | 78 | 19 | 19 |
| `general_complaint_feedback` | 2,228 | 14 | 14 |
| **TOTAL** | **5,229** | **200** | **200** |

Where the heuristic pool was smaller than the target (e.g., `lost_damaged_baggage`: 7
available, 10 targeted), the entire heuristic sub-pool was taken and the surplus budget
redistributed to intents with ample supply.

### 4.3 Determinism

Sampling was performed with `numpy.random.seed(42)` and `pandas.DataFrame.sample(random_state=42)`.
The output is fully reproducible by running `python scripts/sample_golden_set.py`.

> **Note:** `sample_golden_set.py` fails if `golden_annotation.csv` already exists,
> preventing accidental overwriting of human labels.

### 4.4 Deduplication

Sampling enforces **unique conversation IDs** (verified by assertion).
After sampling, the script checks for **exact duplicate `customer_message` text**
within the final 200 examples — none were found.

**No fuzzy or semantic near-duplicate detection was performed.** Given that the 200
examples are drawn from 5,229 unique conversations, the risk of meaningful near-duplication
is low, but this is an acknowledged limitation of the sampling methodology.

---

## 5. Annotation Protocol — Accelerated Human Review Workflow

### 5.1 Overview

Annotation uses an **accelerated proposal-then-review workflow** to reduce manual
data-entry effort while preserving full human control over every final label.

**AI/rule-based labels are proposals only. Final golden labels are human-approved or human-edited.**
Proposed labels were not treated as ground truth.

The workflow has two stages:

1. **Proposal generation** (`scripts/propose_golden_labels.py`): generates
   `data/golden_eval/golden_proposals.csv` using a rule-based AI classifier
   (`src/proposer.py`, method identifier: `rule_based_v1`). This script never
   modifies `golden_annotation.csv`.

2. **Human review** (`scripts/review_golden_proposals.py`): the annotator reviews
   each proposed example. Only on explicit human approval are labels written to
   `golden_annotation.csv`. The review tool supports:
   - **Single mode**: one example at a time with per-field editing.
   - **Batch mode** (`--batch`): 10 examples displayed together; annotator marks
     each as `A` (accept) or `R` (send to single review).

### 5.2 Proposal Logic (rule_based_v1)

The proposer (`src/proposer.py`) uses:

- **Weighted keyword pattern scoring**: 3 tiers (weight 3 = highly specific;
  weight 2 = moderate; weight 1 = weak/general). Scoring applied to `customer_message`.
- **Priority tiebreaking**: when two intents score equally, the more urgent/actionable
  intent wins (order: `lost_damaged_baggage` > `refund_credit_voucher` >
  `flight_delay_rebooking` > … > `general_complaint_feedback`).
- **Escalation policy**: always-escalate intents + account-specific signal detection
  in customer message + Delta response signals (e.g., "please DM us").
- **Difficulty estimate**: based on score gap and message length:
  - `easy` — clear winner with multiple strong signals
  - `hard` — no dominant intent or very short message
  - `ambiguous` — everything else

**Proposal limitations**: the proposer is deterministic and transparent but not
a trained model. It makes systematic errors on:
- Multi-intent messages where customer action and policy question co-occur
- Sarcasm or indirect phrasing
- Very short messages with minimal signal
These cases are flagged as `ambiguous` or `hard` in `proposed_difficulty`.

### 5.3 Fields Collected

| Field | Type | Allowed Values | Source |
|---|---|---|---|
| `gold_intent` | categorical | 10 approved intent codes | Human-approved |
| `escalate` | binary | `yes` \| `no` | Human-approved |
| `escalation_reason` | free text | Required if `escalate=yes` | Human-approved |
| `annotator_notes` | free text | Optional | Human-approved |
| `difficulty` | categorical | `easy` \| `ambiguous` \| `hard` | Human-approved |
| `annotation_source` | categorical | see below | Set automatically |

### 5.4 Annotation Source Values

| Value | Meaning |
|---|---|
| `human_approved_ai_proposal` | Annotator accepted the AI/rule-based proposal without changes |
| `human_edited_ai_proposal` | Annotator reviewed and modified at least one field |
| `human_annotated_direct` | Annotator used `annotate_golden.py` directly (predates proposal workflow) |

**No example is considered annotated unless one of the above values is present.**
The `proposal_method` column in `golden_proposals.csv` records the exact proposer
version (`rule_based_v1`) for reproducibility.

**AI/rule-based labels are proposals only. Final golden labels are human-approved or human-edited.**

### 5.5 Labelling Rules

1. **One intent per example** — the single best intent for the customer's primary need.
2. **Based on customer message**, not Delta's historical reply.
3. **Heuristic and proposed intents are non-binding** — the annotator must read
   the actual conversation independently before accepting or editing.
4. **Ambiguous examples** → best judgment + `difficulty=ambiguous` + `annotator_notes`.
5. **No automatic label copying** — the review tool requires explicit acceptance (key `A`)
   per example; batch acceptance still requires explicit `A` per row.


---

## 6. Leakage Prevention

| Measure | Implementation |
|---|---|
| Reserved pool isolated before index construction | Index fitted on `dev_corpus.parquet` only (seed=42 split, then `retrieval_index.pkl` fitted) |
| Reserved pool never used for classifier training | Heuristic classifier is purely rule-based; no learning on any split |
| Sampling only reads reserved pool | `sample_golden_set.py` loads only `reserved_golden_pool.parquet` |
| Sampling script is idempotent | Fails if output CSV already exists |
| Validation gate G4 | Checks all 200 conv_ids are from reserved pool |
| Validation gate G5 | Checks zero conv_id overlap with dev corpus |

### Text-level coincidence note

One customer message in the golden set
(*"I need my in-flight coffee @Delta, but I can't do black. Please add nondairy #vegan creamers."*)
also appears verbatim in 35 different dev-corpus conversations. This is a duplicated broadcast
tweet in the raw Twitter dataset — the tweet was retweeted and appeared under many different
conversation threads. The golden conv_id (460296) is not in the retrieval index.
This is a text coincidence, not a labelling or retrieval leakage issue. The gold label
is the human annotation, not anything derived from the dev corpus.

---

## 7. Quality Assurance

After annotation, `scripts/validate_golden_set.py` runs quality gates:

### Required gates (must all pass)

| Gate | Description |
|---|---|
| G1 | Exactly 200 rows |
| G2 | All `sample_id` values unique |
| G3 | All `conversation_id` values unique |
| G4 | All `conversation_id` values from reserved pool |
| G5 | No `conversation_id` overlap with dev corpus |
| G6 | All `gold_intent` filled with valid intent codes |
| G7 | All `escalate` values are `yes` \| `no` |
| G8 | `escalation_reason` non-empty for every `escalate=yes` row |
| G9 | All 10 intents represented at least once |
| G10 | Annotator disagreement rate with heuristic > 5% |

### Advisory checks (informational — do not block)

| Check | Description |
|---|---|
| A1 | `difficulty` column present |
| A2 | No blank `difficulty` values |
| A3 | All `difficulty` values are `easy \| ambiguous \| hard` |

---

## 8. Intended Evaluation Use

The completed golden set will be used in Phase 6 to evaluate:

1. **Intent classification accuracy** — predicted intent vs `gold_intent`
2. **Escalation precision/recall** — predicted escalation vs `escalate`
3. **Reply quality** — LLM-as-a-judge comparing generated reply against `delta_response`
4. **Baseline comparisons** — trivial (majority class) and simple (TF-IDF) baselines
5. **Difficulty-stratified error analysis** — metrics broken down by `difficulty` level

---

## 9. Files

| File | Description |
|---|---|
| `data/golden_eval/golden_annotation.csv` | The 200-row annotation file (human-approved labels) |
| `data/golden_eval/golden_proposals.csv` | Rule-based proposed labels (NOT ground truth) |
| `data/processed/reserved_golden_pool.parquet` | Source candidate pool (5,234 convs) |
| `src/proposer.py` | Rule-based proposer (`rule_based_v1`) |
| `scripts/sample_golden_set.py` | Reproducible stratified sampling script |
| `scripts/propose_golden_labels.py` | Generates `golden_proposals.csv` |
| `scripts/review_golden_proposals.py` | Interactive single + batch review tool |
| `scripts/annotate_golden.py` | Direct annotation CLI (bypasses proposal workflow) |
| `scripts/validate_golden_set.py` | Post-annotation quality gate validator |
| `docs/golden_annotation_guide.md` | Per-example labelling instructions |
| `docs/golden_set_methodology.md` | This document |
