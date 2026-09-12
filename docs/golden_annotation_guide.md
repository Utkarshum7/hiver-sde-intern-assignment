# Golden Evaluation Set — Annotator Guide

**Version:** 1.1  
**Project:** Delta Customer Support AI Agent — Hiver SDE Intern Assignment  
**File to annotate:** `data/golden_eval/golden_annotation.csv`  
**Tool:** `python scripts/annotate_golden.py`

---

## 1. Purpose

You are building 200 hand-labelled "gold standard" examples that will be used to:

1. **Measure intent classification accuracy** — does the AI assign the right intent?
2. **Evaluate reply quality** — does the AI draft a reply grounded in real Delta data?
3. **Test escalation decisions** — does the AI correctly flag cases for human handling?
4. **Stratify error analysis by difficulty** — where does the AI struggle most?

These 200 examples are the **only objective ground truth** in the entire project.  
They must be labelled by a human — **do not let any model or heuristic label them**.

---

## 2. Critical Warnings Before You Start

> [!IMPORTANT]
> **The `heuristic_intent` column is NOT ground truth.**
>
> It is a keyword-match estimate used only for stratified sampling. It is wrong
> approximately 15–25% of the time. The annotation tool will display it as a
> reference signal *before* the intent menu. You MUST independently read the
> customer message and decide the correct intent before consulting this value.
> Do not copy it without independent judgment.

> [!WARNING]
> **The historical Delta response is context/evidence — not automatically the correct answer.**
>
> The `delta_response` column shows what Delta actually replied to this customer.
> Historical responses may be deflective, incomplete, or misclassified themselves.
> Use the response to understand what happened, but base your **intent label** on the
> **customer message**, not on how Delta responded.

> [!NOTE]
> **Your progress is saved after every completed example.**
>
> You can quit with `Ctrl+C` at any time and resume by re-running the tool.
> Only blank rows will be shown on the next session.

---

## 3. What You Will See

For each of the 200 conversations, the annotation tool displays:

| What | Source | How to use |
|---|---|---|
| **CUSTOMER MESSAGE** | Combined customer tweet text | Primary basis for all labels |
| **DELTA RESPONSE** | Historical @Delta reply tweets | Context only — not a label signal |
| **Heuristic reference** | Keyword-match estimate | Weak reference — verify independently |

You will then answer five questions:

| Field | Type | Required |
|---|---|---|
| `gold_intent` | 1 of 10 codes | Always |
| `escalate` | `yes` or `no` | Always |
| `escalation_reason` | Free text | Required if `escalate=yes` |
| `annotator_notes` | Free text | Optional |
| `difficulty` | `easy`, `ambiguous`, or `hard` | Always |

---

## 4. The 10 Intent Codes

Assign the **single best** intent that describes the **customer's primary need**.  
When in doubt between two, choose the more specific/urgent one.

| # | Code | Label | Classify here when… |
|---|---|---|---|
| 1 | `flight_status_inquiry` | Flight Status & Schedule | Asking about gate, arrival/departure, tracking — **no rebooking action needed** |
| 2 | `flight_delay_rebooking` | Delay & Cancellation Rebooking | Cancelled flight / missed connection / needs a different flight |
| 3 | `baggage_allowance_policy` | Baggage Fee & Policy | Asks how much / rules / size limits for bags — **not a missing/damaged bag claim** |
| 4 | `lost_damaged_baggage` | Lost or Damaged Baggage | Reports a bag that didn't arrive, was damaged, or is missing |
| 5 | `seat_assignment_upgrade` | Seat Selection & Upgrade | Wants to pick, change, or upgrade a seat; wants to sit with travel companion |
| 6 | `skymiles_loyalty_program` | SkyMiles & Medallion | Balance, missing miles, Medallion status, SkyClub, partner miles |
| 7 | `refund_credit_voucher` | Refund, eCredit & Voucher | Wants a monetary refund, eCredit application, or flight voucher |
| 8 | `checkin_boarding_pass` | Check-in & Boarding Pass | App/online check-in, boarding pass errors, TSA PreCheck missing |
| 9 | `inflight_amenities_service` | In-Flight Amenities | Wi-Fi, entertainment, meals, power outlets, onboard crew service |
| 10 | `general_complaint_feedback` | General Complaint / Compliment | Praise for crew, vague non-actionable rant, or general feedback |

### Decision rules for common ambiguities

**Intent 1 vs 2 (status vs rebooking):**
> Is the customer trying to get on a *different* flight? → `flight_delay_rebooking`  
> Are they only asking *is my current flight on time*? → `flight_status_inquiry`

**Intent 3 vs 4 (policy vs claim):**
> Policy question: "How much is a checked bag?" → `baggage_allowance_policy`  
> Active claim: "My bag didn't come out at baggage claim" → `lost_damaged_baggage`

**Intent 5 vs 7 (seat vs refund):**
> "I was downgraded and want a refund of the fare difference" → `refund_credit_voucher`  
> "I want to change my seat" → `seat_assignment_upgrade`

**Intent 7 vs 2 (refund vs rebooking):**
> "My flight was cancelled and I want my money back" → `refund_credit_voucher`  
> "My flight was cancelled and I need to get on another flight today" → `flight_delay_rebooking`

**Catchall rule:** If you genuinely cannot determine the primary intent, use  
`general_complaint_feedback` and explain in `annotator_notes`.

---

## 5. Escalation Decision

**`escalate = yes`** when resolution requires account access, a booking record,  
a baggage claim system, or a human judgement call:

| Always escalate | Auto-handle OK |
|---|---|
| Flight rebooking / standby listing | General fee / policy questions |
| Lost or damaged baggage claim | Flight status (no account needed) |
| Refund or eCredit application | Generic check-in troubleshooting steps |
| Account-specific SkyMiles dispute | SkyMiles general benefit information |
| Safety, medical, or disability concerns | Compliments / general feedback |
| Complex multi-leg itinerary changes | |

When **`escalate = yes`**, you **must** fill in `escalation_reason` with 1–2 sentences  
describing what a human agent would need to do.

**Example:** *"Customer's bag didn't arrive at carousel after international flight; requires  
baggage claim lookup in Delta's baggage tracking system."*

---

## 6. Difficulty

Rate how difficult this example would be to classify for any reasonable annotator:

| Value | # | When to use |
|---|---|---|
| `easy` | 1 | Intent is unambiguous; most annotators would choose the same label |
| `ambiguous` | 2 | Could reasonably fit 2+ intents; correct label requires interpretation or context |
| `hard` | 3 | Genuinely borderline; high inter-annotator disagreement likely; substantial context needed |

**Difficulty reflects labelling difficulty, not resolution complexity.**

Examples:
- A clear "how much is a checked bag?" tweet → `easy`
- A tweet complaining about delays and also asking for a refund → `ambiguous`
- A very short, vague rant with no clear action requested → `hard`

The difficulty field is used in Phase 6 to stratify evaluation metrics. It does not affect whether the annotation is accepted.

---

## 7. Quality Rules

1. **Read the customer message first, before anything else.**
2. **The heuristic hint is non-binding.** It will be shown in a clearly labelled box before the intent menu. Treat it as a weak sampling reference — it is wrong ~15–25% of the time.
3. **One intent per example.** No multi-label.
4. **Base intent on the customer message**, not the Delta response.
5. **Ambiguous cases** → use best judgment, note in `annotator_notes`, set `difficulty=ambiguous` or `hard`.
6. **Short or vague messages** → `general_complaint_feedback` + note.
7. **Do not use the Delta response to infer the customer's intent.** A deflective response (e.g., "please DM us") does not mean the case is low-urgency.

---

## 8. How to Run — Accelerated Review Workflow

### Recommended: proposal-then-review (fastest)

```bash
# Step 1: Generate proposed labels (only needed once; never modifies golden_annotation.csv)
python scripts/propose_golden_labels.py

# Step 2a: Batch review — 10 examples at a time (fastest for bulk acceptance)
python scripts/review_golden_proposals.py --batch

# Step 2b: Single review — one example at a time (more control per example)
python scripts/review_golden_proposals.py

# Step 3: Validate after all 200 are reviewed
python scripts/validate_golden_set.py
```

### In batch mode (`--batch`)

The tool shows 10 examples at once with compact display:
- `A` = accept all proposed labels for that example
- `R` = send to single-example review for manual editing

Enter 10 choices space-separated (e.g. `A A R A A A A A A A`) or type `ALL` to accept all 10.

### In single mode

For each example, choose:
- **A** = accept all proposed labels
- **I** = change intent
- **E** = change escalation decision
- **D** = change difficulty
- **R** = change escalation reason
- **N** = edit notes
- **S** = skip (do not approve yet)

### Alternative: direct annotation (slow, bypasses proposals)

```bash
# Use only if proposal workflow is not desired for specific examples
python scripts/annotate_golden.py
```

The tool saves progress after every approved example. You can quit (`Ctrl+C`) and resume anytime.

---

## 9. Validation Gates

After annotation, `validate_golden_set.py` runs these checks:

### Required gates (all must pass)

| Gate | Check |
|---|---|
| G1 | Exactly 200 rows |
| G2 | All `sample_id` unique |
| G3 | All `conversation_id` unique |
| G4 | All `conversation_id` from reserved pool (leakage check) |
| G5 | No `conversation_id` in dev corpus (leakage check) |
| G6 | All `gold_intent` filled and valid |
| G7 | All `escalate` filled as `yes` or `no` |
| G8 | `escalation_reason` non-empty for every `escalate=yes` row |
| G9 | All 10 intents represented at least once |
| G10 | Annotator disagreement rate with heuristic > 5% |

### Advisory checks (informational — do not block the set)

| Check | Description |
|---|---|
| A1 | `difficulty` column present |
| A2 | No blank `difficulty` values |
| A3 | All `difficulty` values are `easy`, `ambiguous`, or `hard` |
