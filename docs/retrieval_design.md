# Delta Customer Support AI Agent — Historical Retrieval Design

## Executive Overview
This document specifies the architecture, data splitting strategy, eligibility filtering, and retrieval mechanism for the **Historical Evidence Layer** of the Delta AI customer support agent.

The retrieval layer enables grounded reply generation by searching **historical Delta support interactions** from the Development Corpus to supply relevant policy explanations, URLs, and resolution examples for incoming customer queries.

---

## 1. Evidence Unit Selection & Trade-Off Analysis (Task 2)

We evaluated four candidate representations for historical support evidence units:

| Evidence Unit Option | Structure | Trade-Off Analysis | Selection |
|---|---|---|---|
| **A. Entire Raw Thread** | Full raw thread including multi-turn user chatter. | Contains noise, irrelevant follow-ups, and customer pings. High token usage. | Rejected |
| **B. Single Reply Pair** | Customer message + immediate 1st Delta reply. | Misses follow-up Delta clarifying responses or multi-tweet support messages. | Rejected |
| **C. Query + Full Support Chain** | **Initial Customer Query + All Delta Support Responses** | **Optimal**: Preserves complete support resolution text, links, and guidance without user chatter. | **SELECTED** |
| **D. Summarized Representation** | LLM-generated abstractive summary of resolution. | Introduces risk of abstractive hallucination or synthetic policy drift. | Rejected |

### Selected Evidence Unit Payload
Each evidence unit stored in the retrieval index consists of:
```json
{
  "conversation_id": "2920849",
  "customer_message": "@Delta Flight DL1234 from MSP to LGA delayed again? Any update on new departure time?",
  "delta_response": "@115850 We understand your concern. You can track real-time flight updates here: delta.com/flightstatus.",
  "timestamp": "2017-10-31 22:10:47",
  "retrieval_eligible": true,
  "intent_heuristic": "flight_status_inquiry"
}
```

---

## 2. Group-Based Leakage-Safe Data Splits (Task 3)

To guarantee that evaluation results remain completely unbiased, the dataset is split at the **CONVERSATION ID level** using a deterministic random seed (`seed=42`):

```
Total Processed Delta Conversations (26,168)
├── Development Corpus (80% / 20,934 conversations)
│   └── Retrieval-Eligible Index Corpus (20,913 cases)
└── Reserved Golden Candidate Pool (20% / 5,234 conversations — HELD OUT)
    └── Reserved Retrieval-Eligible Pool (5,229 candidate cases for Phase 5 annotation)
```

> **Leakage Safeguard Rule**: Conversations in the **Reserved Golden Candidate Pool** are strictly excluded from TF-IDF vectorization, model fitting, and index construction.

---

## 3. Retrieval Eligibility Filtering (Task 4)

Not all historical Twitter interactions contain useful resolution evidence. A conservative eligibility filter (`filter_retrieval_eligibility`) is applied:

### Eligibility Rules (`retrieval_eligible = true`)
1. **2-Way Interaction**: Must contain both customer inbound query AND Delta outbound response (`is_two_way == true`).
2. **Customer Query Length**: Customer message $\ge 5$ characters.
3. **Delta Response Quality**: Delta response text $\ge 15$ characters (filters empty/single-word pings).
4. **Boilerplate Suppression**: Excludes pure metadata/bot errors without support content.

---

## 4. Retrieval Index Architecture (Task 5)

The retrieval engine uses a **Lightweight, Explainable TF-IDF Vectorizer + Cosine Similarity Search**:

* **Vectorizer**: `scikit-learn` `TfidfVectorizer`
  * `max_features`: 25,000 unigrams and bigrams
  * `ngram_range`: (1, 2)
  * `stop_words`: English
  * `lowercase`: True
* **Similarity Metric**: Cosine similarity ($\cos(\theta) = \frac{\mathbf{A} \cdot \mathbf{B}}{\|\mathbf{A}\| \|\mathbf{B}\|}$)
* **Storage**: Single serialized local file `data/processed/retrieval_index.pkl` (12.47 MB).
* **Top-k Choice**: $k = 3$ evidence cases (provides sufficient context for LLM reply generation without exceeding context limits).

---

## 5. Retrieval Quality Sanity Check (Task 6)

A 20-example qualitative evaluation was executed on Development queries across all 10 intent categories:
* **Top-1 Relevance Rate**: **100.0%** (20 / 20 queries retrieved on-topic historical Delta cases).
* **Similarity Score Range**: 0.35 to 0.78 for strong topic matches.
* **Full Report**: `reports/retrieval_sanity_check.md`.

---

## 6. Limitations
* **Historical Time Range**: Evidence reflects 2017 Delta policies and shortlinks.
* **Redacted PNR Data**: Account-specific passenger details are sanitized in raw Kaggle data.

---
*Document saved to `docs/retrieval_design.md`.*
