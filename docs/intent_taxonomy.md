# Delta Customer Support AI Agent — Intent Taxonomy & Validation Audit Specification

> [!IMPORTANT]
> **Methodological Disclosure**: All initial intent distributions and topic classifications reported in this taxonomy were generated using **heuristic rule-based pattern matchers and exploratory data profiling**, NOT human hand-labelled ground truth. Human hand-labelling will be performed strictly in Phase 4 during Golden Evaluation Set construction.

---

## 1. Intent Taxonomy Overview & Heuristic Class Distribution

Empirical exploratory topic distribution across all **26,168 reconstructed Delta conversations**:

| Intent ID | Intent Name | Heuristic Available Examples | % of Dataset | Auto-Handleable? | Primary Handling Mode |
|---|---|---:|---:|:---:|---|
| `flight_status_inquiry` | Flight Status & Schedule Inquiry | 11,666 | 44.58% | **Yes (Info only)** | Provide public flight status format / help link |
| `flight_delay_rebooking` | Flight Delay & Cancellation Rebooking | 601 | 2.30% | **No** | **Escalate to Human** for urgent flight rebooking |
| `baggage_allowance_policy` | Baggage Allowance & Fee Policy | 186 | 0.71% | **Yes** | Provide official Delta baggage fee & dimension rules |
| `lost_damaged_baggage` | Lost or Damaged Baggage Claim | 59 | 0.23% | **No** | **Escalate to Human** for baggage tracking & claim filing |
| `seat_assignment_upgrade` | Seat Selection & Upgrade Request | 1,502 | 5.74% | **Yes (Info only)** | Explain upgrade policy or direct to seat map tool |
| `skymiles_loyalty_program` | SkyMiles & Medallion Loyalty Program | 955 | 3.65% | **Yes** | Answer Medallion benefits & SkyMiles redemption queries |
| `refund_credit_voucher` | Ticket Refund, eCredit & Voucher Request | 170 | 0.65% | **No** | **Escalate to Human** for financial refund processing |
| `checkin_boarding_pass` | Check-in & Mobile Boarding Pass Troubleshooting | 806 | 3.08% | **Yes** | Guide user through app check-in & TSA PreCheck rules |
| `inflight_amenities_service` | In-Flight Amenities & Onboard Experience | 409 | 1.56% | **Yes** | Provide Wi-Fi, entertainment, & onboard meal info |
| `general_complaint_feedback` | General Service Complaint & Compliment | 9,814 | 37.50% | **Yes** | Empathize & acknowledge feedback or thank user |

---

## 2. Realistic System Capabilities & Auto-Handling Boundaries

> [!WARNING]
> **No Live API Access**: The AI agent operates **without live connection to internal Delta APIs** (no live flight tracking radar, no live airport gate feed, no GDS seat booking system, no live NetTracer baggage database, no financial refund gateway).

### Auto-Handle Definition
Auto-handling means providing **accurate, static policy information and canonical help links** based on historical evidence.

* **Allowed Auto-Handle**:
  * Answering general baggage fee questions ($30 1st bag, $40 2nd bag).
  * Explaining general SkyMiles Medallion tier requirements.
  * Explaining 24-hour risk-free cancellation policy rules.
  * Providing canonical help links (`delta.com/baggage`, `delta.com/flightstatus`).
* **Mandatory Human Escalation**:
  * Real-time flight tracking or live gate updates.
  * Flight rebooking due to delays or cancellations.
  * Cash refunds to credit cards or eCredit extensions.
  * Lost or broken baggage claims.
  * Seat changes or PNR modifications.

---

## 3. Detailed Intent Definitions & Representative Examples

### 1. `flight_status_inquiry`
* **Definition**: General inquiries regarding flight schedules, how to check flight status, or standard terminal/gate guidelines.
* **What Belongs**: General questions on flight timing format, standard schedule policies, static status link requests.
* **What Does NOT Belong**: Live real-time status lookup requests during active delays (`flight_delay_rebooking`).
* **Representative Examples**:
  1. `@Delta \u201cquick fix\u201d to a partially boarded plane. The 35 min delay is about the exact amount of time needed to cause me to miss my connection.`
  2. `@Delta Flight 1234 from MSP to LGA delayed again? Any update on new departure time?`
  3. `@Delta Can you tell me what terminal flight DL588 lands at in ATL?`

### 2. `flight_delay_rebooking`
* **Definition**: Customer experiencing cancelled flights, missed connections, or severe delays requiring immediate flight rebooking.
* **What Belongs**: Rebooking requests, missed connection assistance, standby requests.
* **What Does NOT Belong**: General flight status inquiries without cancellation (`flight_status_inquiry`).
* **Representative Examples**:
  1. `@Delta Flight DL499 cancelled due to crew. Need to get to BOS tonight for a funeral. Please rebook me!`
  2. `@Delta Missed my connection in DTW because DL112 was 2 hours late. Next flight is tomorrow, need help now.`

### 3. `baggage_allowance_policy`
* **Definition**: Inquiries regarding standard checked bag fees, carry-on size/weight limits, military baggage policy, or special equipment rules.
* **What Belongs**: Checked bag pricing, carry-on size limits, military luggage rules.
* **What Does NOT Belong**: Lost or broken luggage claims (`lost_damaged_baggage`).
* **Representative Examples**:
  1. `@Delta How much is a second checked bag for international flight to London?`
  2. `@Delta Can I bring my golf bag as checked baggage without extra oversize fee?`

### 4. `lost_damaged_baggage`
* **Definition**: Customer reporting delayed, missing, lost, or physically broken/damaged baggage after arrival.
* **What Belongs**: Missing bag on carousel, broken suitcase handle/wheels, lost baggage tracing.
* **What Does NOT Belong**: Pre-flight baggage pricing questions (`baggage_allowance_policy`).
* **Representative Examples**:
  1. `@Delta Landed in Miami 2 hours ago and my bag is missing. Baggage claim office is closed!`
  2. `@Delta My hardshell suitcase came out with a huge crack and missing wheel. How do I file a claim?`

---

## 4. Confusing Intent Pairs & Boundary Disambiguation

### Pair 1: `flight_status_inquiry` vs. `flight_delay_rebooking`
* **Difference**: `flight_status_inquiry` is an informational check about general timing/format. `flight_delay_rebooking` is an actionable request for urgent rebooking after cancellation.
* **Disambiguation**: Detect action verbs like `rebook`, `cancelled`, `missed connection`, `standby`.

### Pair 2: `baggage_allowance_policy` vs. `lost_damaged_baggage`
* **Difference**: `baggage_allowance_policy` is pre-flight informational queries. `lost_damaged_baggage` is post-flight reporting of missing/broken luggage.
* **Disambiguation**: Post-flight damage/missing terms (`lost`, `missing`, `damaged`, `broken`) signal `lost_damaged_baggage`.

---
*Document updated following Phase 3 Validation Audit.*
