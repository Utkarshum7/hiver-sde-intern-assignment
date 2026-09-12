# Delta Customer Support AI Agent — Escalation Policy & System Governance

> [!IMPORTANT]
> **System Capability Boundary**: The AI agent operates **WITHOUT live integration to internal Delta APIs** (no live flight tracking feed, no live GDS seat modification engine, no live NetTracer baggage system, no refund gateway).
> 
> * **AUTO-HANDLE** means providing verified, factual information and canonical URLs based on public policy and historical resolution evidence.
> * **ESCALATE** means transferring the user to a human agent whenever live data, account actions, financial changes, or sensitive circumstances are required.

---

## 1. Conservative Escalation Principles

### AUTO-HANDLE Criteria
The AI agent will automatically draft and handle customer queries **ONLY when ALL** of the following conditions are met:
1. **Public Policy Evidence**: The request can be fully answered using public Delta policy rules (e.g. baggage pricing, standard check-in rules, SkyMiles tier guidelines).
2. **High Classifier Confidence**: Intent classification confidence score exceeds $\ge 0.85$.
3. **No Account Action Required**: The query does NOT require modifying a live PNR, issuing refunds, changing seats, or checking real-time flight status.
4. **No Emergency / High-Risk**: The message does NOT report safety hazards, active airport stranding, lost luggage, or flight cancellations.

### ESCALATE Criteria
The AI agent MUST immediately escalate the conversation to a human support agent when **ANY** of the following apply:
1. **Live Data Needed**: Requests requiring live real-time flight tracking or current gate updates.
2. **Flight Cancellation or Missed Connection**: Urgent airport rebooking needed.
3. **Account / Financial Data Needed**: Passenger PNR, credit card details, SkyMiles account PIN, or refund processing.
4. **Lost or Damaged Baggage Claims**: Physical damage claims or active baggage tracing.
5. **Ambiguous or Low Confidence**: Intent classification confidence $< 0.85$.
6. **Customer Distress / Safety Issue**: High frustration, medical emergencies, or safety complaints.

---

## 2. Intent-Level Escalation Decision Matrix

| Intent ID | Auto-Handle Examples (Public Info) | Escalate Examples (Human Action Needed) | Escalation Rationale |
|---|---|---|---|
| `flight_status_inquiry` | "Where can I check flight status?" / "What is the standard arrival format?" | "Is DL123 delayed right now? I have 15 mins to connect!" | Static format info is auto-handleable; live real-time tracking requires live API / human lookup. |
| `flight_delay_rebooking` | *None* | "Flight cancelled, need next flight" / "Stranded in ATL with kids" | **100% Escalate**: Rebooking requires live PNR & agent seat reservation capabilities. |
| `baggage_allowance_policy` | "How much for a second checked bag?" / "Carry-on size limits" | "Agent charged me $75 for bag on a free bag Medallion ticket, refund me!" | Fee policy is public knowledge; refund/fee disputes require financial account access. |
| `lost_damaged_baggage` | *None* | "Landed in MIA 2 hrs ago and my bag is missing" / "Suitcase wheel ripped off" | **100% Escalate**: Lost baggage claims require NetTracer filing & compensation authority. |
| `seat_assignment_upgrade` | "How many miles to upgrade to First Class?" / "What is Comfort+ seating?" | "Traveling with 6yo child and assigned separate seats 12A and 34F, change us!" | Policy inquiry is auto-handleable; active seat swap requires PNR modification. |
| `skymiles_loyalty_program` | "What are Gold Medallion lounge rules?" / "How do MQMs accumulate?" | "Flew ATL to LHR last week but my 5,000 miles didn't post, add them" | General program rules auto-handleable; posting missing credit requires account access. |
| `refund_credit_voucher` | "What is the 24-hour risk-free cancellation policy?" | "Cancel ticket DL987 for full cash refund to original credit card" | **100% Escalate**: Financial transactions & cash refunds must be executed by human billing agents. |
| `checkin_boarding_pass` | "What is the deadline for airport check-in?" | "App gives error code 500 when checking in 24 hrs prior" | General rules auto-handleable; technical app error requires agent override / desk check-in. |
| `inflight_amenities_service` | "Are vegetarian meals served on international flights?" | "Inflight Wi-Fi dropped for entire flight after paying $19.95, refund me" | Food/amenity info is public; Wi-Fi refund requests require payment verification. |
| `general_complaint_feedback` | "Shoutout to flight attendant Mary on DL204!" / "Great flight today!" | "Worst experience ever with Delta staff in ATL, demand manager call back" | General compliments/thanks auto-handled; formal escalations routed to management. |

---

## 3. Data Leakage Prevention Plan & Golden Set Feasibility

### Proposed Golden Evaluation Set Sampling Strategy (Phase 4 Allocation)
To construct a statistically balanced **200-example hand-labelled Golden Evaluation Set** in Phase 4:

| Intent ID | Heuristic Candidate Volume | % of Dataset | Golden Set Sample Allocation (200 total) | Allocation Rationale |
|---|---:|---:|---:|---|
| `flight_status_inquiry` | 11,666 | 44.58% | **30** | High-volume baseline intent |
| `general_complaint_feedback` | 9,814 | 37.50% | **30** | High-volume non-actionable baseline |
| `seat_assignment_upgrade` | 1,502 | 5.74% | **20** | Moderate volume policy & upgrade intent |
| `skymiles_loyalty_program` | 955 | 3.65% | **20** | Moderate volume loyalty & benefits intent |
| `checkin_boarding_pass` | 806 | 3.08% | **20** | App troubleshooting & check-in intent |
| `flight_delay_rebooking` | 601 | 2.30% | **20** | **Oversampled high-risk escalation intent** |
| `inflight_amenities_service` | 409 | 1.56% | **15** | Onboard amenities & Wi-Fi intent |
| `baggage_allowance_policy` | 186 | 0.71% | **15** | Policy inquiry intent |
| `refund_credit_voucher` | 170 | 0.65% | **15** | **Oversampled financial escalation intent** |
| `lost_damaged_baggage` | 59 | 0.23% | **15** | **Oversampled rare high-risk escalation intent** |
| **TOTAL** | **26,168** | **100.00%** | **200** | **Balanced, statistically sound coverage** |

---
*Document updated following Phase 3 Validation Audit.*
