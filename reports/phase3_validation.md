# Phase 3 Validation Audit Report

## 1. Delta Count Reconciliation

| Metric | Phase 1 Profiler | Phase 3 Processed Dataset | Discrepancy Cause | Exact Verified Value |
|---|---:|---:|---|---:|
| **Strict Delta + Customer Messages** | 87,549 | 87,549 | None (Exact match) | **87,549** |
| **All Thread Messages (incl. third-party)** | N/A | 87,994 | Phase 3 BFS collected 445 third-party user tweets in multi-branch threads | **87,994** |
| **2-Way Conversations (Customer + Delta)** | 26,164 | 26,166 | 2 edge-case multi-root threads linked upon full graph traversal | **26,166** |
| **Total Reconstructed Conversations** | N/A | 26,168 | Includes 2 one-way Delta support announcement threads without customer replies | **26,168** |

**Conclusion**: Both phases are 100% mathematically consistent. The 44-message difference arises because Phase 3 preserved all 445 third-party tweets in multi-author conversation trees, whereas Phase 1 strictly counted `author_id == 'Delta'` (42,253) + customer inbound (45,296) = 87,549.

---

## 2. Methodology & Intent Assignment Audit

> **Crucial Disclosure**: All intent classifications and resolution quality metrics reported in Phase 2 and Phase 3 were generated via **automated exploratory pattern matchers and keyword heuristics**. They are **candidate exploratory topic distributions**, NOT human hand-labelled ground truth.

* **Intent Assignment Method**: Rule-based regex pattern matchers (`scripts/analyze_delta_intents.py`).
* **Resolution Quality Method**: Keyword heuristic matching (`scripts/sample_top_brands.py`).
* **Human Ground Truth**: Will be created strictly in **Phase 4** when hand-labelling the 150–250 Golden Evaluation Set.

---

## 3. Intent Taxonomy Refinements & Boundary Disambiguation

* **Flight Status vs. Rebooking**: `flight_status_inquiry` (44.58%) is general timing/format questions. `flight_delay_rebooking` (2.30%) is actionable requests for urgent rebooking after cancellations.
* **Baggage Allowance vs. Lost Baggage**: `baggage_allowance_policy` (0.71%) is pre-flight fee rules. `lost_damaged_baggage` (0.23%) is post-flight lost/damaged bag claims.
* **General Complaint & Feedback**: Comprises 37.50% of conversations (praise for crew, general non-actionable feedback).

---

## 4. Escalation Policy Corrections & System Boundaries

* **System Capability Boundary**: The AI agent operates **WITHOUT live Delta API access** (no live flight radar API, no live GDS seat modification engine, no refund payment gateway).
* **Auto-Handle Definition**: Providing verified static policy rules and canonical URL links (`delta.com/...`).
* **Escalate Definition**: Any request requiring live real-time flight status, PNR modification, cash refunds, or lost luggage claims.

---

## 5. Golden Evaluation Set Sampling Strategy (Phase 4 Allocation)

A **stratified minimum-per-intent + oversampling of rare/high-risk intents** strategy is recommended for the 200-example Golden Set:

| Intent ID | Candidate Volume | % of Dataset | Proposed Golden Set Allocation (200 total) |
|---|---:|---:|---:|
| `flight_status_inquiry` | 11,666 | 44.58% | **30** |
| `general_complaint_feedback` | 9,814 | 37.50% | **30** |
| `seat_assignment_upgrade` | 1,502 | 5.74% | **20** |
| `skymiles_loyalty_program` | 955 | 3.65% | **20** |
| `checkin_boarding_pass` | 806 | 3.08% | **20** |
| `flight_delay_rebooking` | 601 | 2.30% | **20** (Oversampled) |
| `inflight_amenities_service` | 409 | 1.56% | **15** |
| `baggage_allowance_policy` | 186 | 0.71% | **15** |
| `refund_credit_voucher` | 170 | 0.65% | **15** (Oversampled) |
| `lost_damaged_baggage` | 59 | 0.23% | **15** (Oversampled) |
| **TOTAL** | **26,168** | **100.00%** | **200** |

---

## 6. Remaining Risks & Mitigation
1. **Low Volume for Rare Intents**: `lost_damaged_baggage` has 59 candidate threads; oversampling 15 ensures robust evaluation.
2. **Boilerplate Redirection**: RAG retrieval must exclude DM redirect boilerplate and filter for policy-grounded replies.

---
*Validation audit complete.*
