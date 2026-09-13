# Decision Log — Non-Obvious Decisions & Why

The non-obvious calls made building this project, in roughly chronological
order, each broken into **Decision / Alternatives considered / Reason /
Consequence**. Ordinary/default choices (e.g. "use pandas for tabular
data") are omitted — this is the list an interviewer would actually want
to probe. Closely-related decisions are grouped into one entry where they
share a single underlying trade-off, rather than listed separately just to
inflate the count. This is a reformatting of the same real decisions
already made during the project — no new decisions, reasons, or
consequences have been invented for this pass.

---

### 1. Brand selection: Delta over SpotifyCares

- **Decision:** Build the agent for `@Delta`.
- **Alternatives considered:** SpotifyCares (runner-up), Apple, Amazon, and
  the other largest brand accounts in the dataset.
- **Reason:** Chosen via a 10-dimension empirical scoring rubric on a
  500-conversation stratified sample per brand ([`docs/brand_selection.md`](brand_selection.md)),
  not gut feel. Delta scored 47/50 vs. SpotifyCares' 37/50, driven by a
  higher public resolution rate (20.4% vs. 12.2% for Apple, 9.2% for
  Amazon) and cleanly non-overlapping intent boundaries.
- **Consequence:** Every downstream policy fact, intent taxonomy entry, and
  reply example is Delta-specific and reflects 2012–2017-era Delta
  behavior — not portable to another brand without re-running the
  taxonomy/policy work.

### 2. Train/reserved split at the conversation-ID level

- **Decision:** Split 80/20 by `conversation_id`, not by row/message.
- **Alternatives considered:** A random 80/20 row-level split (simpler to
  implement, but leaks later messages of the same thread across
  dev/reserved).
- **Reason:** A row-level split would let the retrieval index and
  classifier see part of a conversation that also appears, in another
  message, in the held-out golden pool — a direct leakage path.
- **Consequence:** `seed=42` makes the split reproducible; zero
  `conversation_id` overlap was independently re-verified by hand during
  this project's audit, not just trusted from prior documentation. Slightly
  more implementation complexity than a row-level split.

### 3. Reserved pool sized larger than the golden set

- **Decision:** Hold out a 5,234-conversation reserved pool (20%), then
  sample only 200 golden examples from within it, rather than making the
  entire reserved pool the golden set.
- **Alternatives considered:** Treating the full reserved pool as the
  golden set.
- **Reason:** Keeping reserved-pool capacity in excess of the current
  golden set means a future expansion can draw more examples from the same
  held-out region without touching dev data or re-splitting.
- **Consequence:** Most of the reserved pool (5,034 conversations) is
  unused by the current evaluation — a deliberate reserve, not waste.

### 4. Stratified, oversampled golden-set sampling

- **Decision:** Sample the golden set by stratified sampling that
  deliberately oversamples rare high-risk escalation intents
  (`lost_damaged_baggage`, `refund_credit_voucher`) far beyond their
  natural ~0.2–0.7% frequency.
- **Alternatives considered:** Uniform/proportional random sampling
  matching the dataset's natural intent distribution.
- **Reason:** Proportional sampling would yield too few rare-intent
  examples to measure anything meaningful about them at all.
- **Consequence:** Headline accuracy is **not** a production-traffic
  estimate — the golden set's intent mix is artificial by design (see
  [`REPORT.md` §17](../REPORT.md#17-what-is-misleading-about-my-headline-number), point 1).

### 5. Three deliberately distinct intent-labelling algorithms

- **Decision:** Use three separate, non-reused algorithms at three
  different points: (a) a coarse single-tier keyword matcher
  ([`src/heuristic_intent.py`](../src/heuristic_intent.py)) for dataset
  stratification and retrieval-evidence metadata only; (b) a more
  sophisticated weighted/priority-ordered engine
  ([`src/proposer.py`](../src/proposer.py), `rule_based_v1`) shown to human
  annotators as a *proposal* during golden-set review; (c) a third, wholly
  independent method — TF-IDF + Logistic Regression trained via weak
  supervision on dev-corpus heuristic labels — as the live agent's actual
  production classifier ([`src/classifier.py`](../src/classifier.py)).
- **Alternatives considered:** Reusing (b), the annotation-assist proposer,
  as the graded production classifier — less engineering effort.
- **Reason:** Reusing (b) as the graded system would raise a legitimate
  circularity concern, since annotators saw its output while building the
  gold labels. Using an unrelated algorithm for the graded system removes
  the concern entirely rather than arguing about it.
- **Consequence:** Three algorithms to maintain instead of one; in
  exchange, the final 45.5% annotator-disagreement rate with heuristic (a)
  and the independence of classifier (c) from the annotation process are
  both real evidence the measured accuracy isn't inflated by circularity.

### 6. Escalation policy shared between annotation-assist and the live agent

- **Decision:** Keep escalation policy logic in one shared module
  ([`src/escalation.py`](../src/escalation.py)), used by both the
  annotation-assist proposer and the live agent.
- **Alternatives considered:** Separate escalation logic for annotation
  assistance vs. the live agent, mirroring the classifier separation in
  decision #5.
- **Reason:** Escalation is a deterministic business policy (if X,
  escalate), not a label predicted from training data — reusing it doesn't
  leak anything into an "accuracy" number the way reusing a classifier
  would, so the circularity concern from #5 doesn't apply here.
- **Consequence:** One policy to update instead of two, at no measurement-integrity cost.

### 7. Strip completed-action language from reused historical evidence

- **Decision:** Reply generation strips first-person completed-action
  language (e.g. "I've credited 500 miles to your account") out of reused
  historical evidence before it reaches a new customer.
- **Alternatives considered:** Reusing historical reply text verbatim.
- **Reason:** A real historical Delta reply describing a specific action
  taken for a specific past customer would be a fabricated claim if reused
  verbatim for a different customer today.
- **Consequence:** Adds a claim-stripping step to every generated reply;
  in exchange, `unsupported_claim_rate` measured 0/200 on the full golden
  set (§12), not just on unit-test fixtures.

### 8. Decouple on blocking external dependencies instead of fabricating substitutes

- **Decision:** Where an external dependency was unavailable (no
  `ANTHROPIC_API_KEY`; an initially incomplete golden set), decouple the
  affected component rather than fabricate a stand-in value.
- **Alternatives considered:** Faking a plausible LLM-judge score, or
  blocking all reply-quality work until an API key became available.
- **Reason:** `src/llm_client.py`/`src/llm_judge.py` are fully implemented
  but return an explicit `pending_no_api_key` status rather than a fake
  score, so the default reply generator stays fully offline and
  deterministic. Separately, the human-rating/LLM-judge-agreement workflow
  (`scripts/build_human_rating_sheet.py`) samples from the **development
  corpus**, not the golden set, since reply-quality rating doesn't need
  `gold_intent` labels — letting that work proceed in parallel instead of
  blocking on annotation completion.
- **Consequence:** Judge scores and judge/human agreement remain genuinely
  PENDING today (§8/§14–15) rather than fabricated — an honest gap, not a
  hidden one.

### 9. Hard-refuse evaluation on a partially-labelled golden set

- **Decision:** `scripts/evaluate.py` hard-refuses to compute golden-set
  accuracy metrics until annotation passes every required validation gate,
  and this was enforced across many real partial-completion states before
  annotation finished.
- **Alternatives considered:** Computing caveated "preview" numbers on
  whatever fraction of the golden set was labelled at the time.
- **Reason:** Evaluating on a partially-labelled held-out set — even
  caveated — creates pressure to (consciously or not) treat early partial
  results as a target.
- **Consequence:** No intermediate accuracy numbers exist anywhere in this
  project's history; the script instead ran smoke tests on non-golden
  messages at every intermediate stage, and real numbers were computed
  exactly once, after all 200 examples were locked and
  `validate_golden_set.py` passed every gate.

### 10. Baseline 2 reuses the main agent's retrieval index

- **Decision:** `SimpleRetrievalBaseline` reuses the exact same TF-IDF
  retrieval index and `top_k` as the main agent — only classification/
  reply/escalation logic differs.
- **Alternatives considered:** Giving the baseline its own, weaker
  retriever.
- **Reason:** Isolates what the smarter components (trained classifier,
  evidence cleanup, full escalation policy) add, rather than confounding
  the baseline comparison with a weaker retriever too.
- **Consequence:** The baseline-vs-main-agent gap in §7's results table can
  be attributed to classification/escalation logic specifically, not
  retrieval quality.

### 11. Retrieval quality reported only as an explicitly-labelled proxy

- **Decision:** Report retrieval quality as intent-match@k against weak
  heuristic labels, explicitly labelled a *proxy* metric, not a true
  relevance metric.
- **Alternatives considered:** Building a true relevance-labelled retrieval
  benchmark via a second human-labelling pass rating retrieved evidence as
  relevant/irrelevant per query.
- **Reason:** The true relevance benchmark would need a labelling pass this
  project's timeline didn't include; rather than quietly treat the proxy as
  ground truth, it's named and caveated everywhere it's reported, including
  in the evaluation script's own output key
  (`retrieval_proxy_supplementary_not_required`).
- **Consequence:** Retrieval quality (0.575 for the main agent) is honestly
  a topic-match signal, not a relevance judgment — flagged as future work
  (§12, item 5).

### 12. Three real bugs found during audit, fixed conservatively

- **Decision:** Fix three real bugs found during this project's own audits,
  each handled narrowly so the fix couldn't be mistaken for new fabricated
  content.
- **Alternatives considered:** Leaving the bugs undocumented, or performing
  a broader re-annotation/re-generation pass while fixing them.
- **Reason / what was found:**
  (a) `build_evidence_corpus()` read a `predicted_intent` column that was
  never actually written to `delta_conversations.parquet` — it only
  existed in an in-memory dataframe inside `scripts/analyze_delta_intents.py`
  — silently defaulting every row's `intent_heuristic` to one constant
  value across all 26,168 conversations.
  (b) Ten stale `annotation_source` values (`human_approved_proposal` from
  an earlier tool version) used an outdated naming convention.
  (c) `.gitignore`'s blanket `*.csv` rule was silently excluding the
  golden-set and human-rating CSVs from git.
- **Consequence:** (a) was fixed by centralizing the heuristic in
  `src/heuristic_intent.py` and regenerating affected files with the same
  seed, verifying the conversation-ID split itself was unaffected before
  trusting the fix. (b) was fixed as a pure string rename, deliberately
  *not* re-reviewing the underlying `gold_intent`/`escalate` content of
  those rows, to avoid conflating a naming-bug fix with re-annotation. (c)
  was fixed with explicit `!data/golden_eval/*.csv` / `!data/eval/*.csv`
  exceptions, while raw/processed regenerable artifacts stay excluded.

### 13. Raw per-example predictions artifact added for honest failure analysis

- **Decision:** Extend `escalation_metrics()` to expose a full TP/TN/FP/FN
  confusion matrix (not just unsafe-false-negative indices), and have
  `scripts/evaluate.py` write `reports/evaluation_raw_predictions.json`
  with every per-example prediction, all before running the real
  evaluation once.
- **Alternatives considered:** Writing only aggregated summary JSON, as
  originally implemented, and asserting failure modes from aggregate
  numbers alone.
- **Reason:** Aggregate numbers alone can't be independently checked
  against real examples — a failure-mode claim needs a citable `sample_id`.
- **Consequence:** The top-5 failure modes in the report cite actual
  `sample_id`s (S027, S093, S050, S022, S092, etc.) and can be
  independently re-verified by reading that file, rather than taken on
  faith.

### 14. Executive Summary as the self-contained submission-ready report

- **Decision:** Lead `REPORT.md` with a self-contained Executive Summary
  that is itself the submission-ready core report (~1,300 words), with
  everything else explicitly demoted to a labelled appendix.
- **Alternatives considered:** Trimming real evidence (confusion matrices,
  per-decision rationale, real sample IDs) to force the entire document
  under the 6-page cap; or silently ignoring the page limit.
- **Reason:** The full evidence the assignment implicitly rewards
  (traceable examples, real numbers, worked reasoning) does not fit in 6
  pages without losing that traceability.
- **Consequence:** The Executive Summary alone covers every mandatory
  element (framing, both baselines, real results, top 5 failure modes,
  misleading-headline-number, one-more-week, limitations); the appendix is
  optional supporting detail for a reviewer who wants to verify a specific
  claim, not additional required reading.
