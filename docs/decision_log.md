# Decision Log — Non-Obvious Decisions & Why

A plain list of the non-obvious calls made building this project, in
roughly chronological order. Ordinary/default choices (e.g. "use pandas
for tabular data") are omitted — this is the list an interviewer would
actually want to probe. Closely-related decisions are grouped into one
entry where they share a single underlying trade-off, rather than listed
separately just to inflate the count.

1. **Delta over SpotifyCares (runner-up)** — chosen via a 10-dimension scoring
   rubric on a 500-conversation stratified sample per brand (`docs/brand_selection.md`),
   not gut feel. Delta scored highest on public resolution rate (20.4% vs
   12.2%) and clean, non-overlapping intent boundaries.

2. **Splitting happens at the conversation-ID level, not the message or row
   level.** A random 80/20 row split would leak later messages of the same
   thread across dev/reserved. `seed=42`, verified independently (zero
   conversation_id overlap re-checked by hand during audit, not just trusted
   from docs).

3. **The reserved pool (20%, 5,234 conversations) is deliberately larger than
   the 200-example golden set.** The golden set is *sampled from* the
   reserved pool rather than being the entire reserved pool, so a future
   golden-set expansion can draw more examples from the same held-out region
   without ever touching dev data or re-splitting.

4. **Golden-set sampling is stratified, not uniform, and deliberately
   oversamples rare high-risk escalation intents** (`lost_damaged_baggage`,
   `refund_credit_voucher`) far beyond their natural ~0.2-0.7% frequency.
   This is necessary to be able to measure anything about rare-but-important
   failure modes at all, but it means headline accuracy is **not** a
   production-traffic estimate (REPORT.md, "What is misleading...").

5. **Three deliberately distinct algorithms classify intent at different
   points in the pipeline, and that's on purpose, not duplication.** (a) A
   coarse single-tier keyword matcher (`src/heuristic_intent.py`) is used
   only for dataset stratification and retrieval-evidence metadata. (b) A
   more sophisticated weighted/tie-broken/priority-ordered engine
   (`src/proposer.py`, `rule_based_v1`) is shown to human annotators as a
   *proposal* during golden-set review. (c) The live agent's production
   classifier (`src/classifier.py`) is a *third* method again — TF-IDF +
   Logistic Regression trained via weak supervision on dev-corpus heuristic
   labels — chosen specifically over reusing (b) as the "main system"
   classifier, because reusing it would raise a legitimate circularity
   concern (annotators saw its output while building the gold labels). The
   final 45.5% annotator disagreement rate with heuristic (a) is some
   evidence annotators weren't rubber-stamping, but using an unrelated
   algorithm for the graded system (c) removes the concern entirely rather
   than arguing about it.

6. **Escalation policy logic lives in one shared module
   (`src/escalation.py`)** used by both the annotation-assist proposer and
   the live agent. This is *not* the same kind of sharing as #5 above:
   escalation is a deterministic business policy (if X, escalate), not a
   label being predicted from training data, so reusing it doesn't leak
   anything into an "accuracy" number the way reusing a classifier would.

7. **Reply generation strips first-person completed-action language out of
   reused historical evidence before it reaches a new customer.** A real
   historical Delta reply might say "I've credited 500 miles to your
   account" — reusing that sentence verbatim for a *different* customer
   today would be a fabricated claim. This is a subtle grounding-safety bug
   class that's easy to miss if you only think about "is the reply
   relevant" and not "did I just imply an action I didn't take."

8. **Two blocking external dependencies (no LLM API key, an initially
   incomplete golden set) were worked around by decoupling, never by
   fabricating a substitute.** With no `ANTHROPIC_API_KEY` available, the
   default reply generator (`src/reply_generator.py`) is fully offline and
   deterministic — grounded template replies from retrieved evidence +
   documented policy facts — while `src/llm_client.py`/`src/llm_judge.py`
   are fully implemented but return an explicit `pending_no_api_key` status
   rather than a fake score. Separately, the human-rating/LLM-judge-agreement
   workflow (`scripts/build_human_rating_sheet.py`) deliberately samples from
   the **development corpus**, not the golden set — reply-quality rating
   doesn't need `gold_intent` labels at all, so that work could proceed in
   parallel instead of blocking on annotation completion.

9. **`scripts/evaluate.py` was built to hard-refuse computing golden-set
   accuracy metrics until annotation passed every required validation
   gate — and it actually enforced that, across many partial-completion
   states, before annotation finished.** It would have been easy to compute
   "preview" numbers on whatever fraction was labelled at the time and
   caveat them in the report. That was explicitly ruled out: evaluating on
   a partially-labelled held-out set — even caveated — creates pressure to
   (consciously or not) treat early partial results as a target. The script
   instead ran a smoke test on non-golden example messages at every
   intermediate stage, and only computed real numbers once all 200 examples
   were locked and `validate_golden_set.py` passed every gate.

10. **Baseline 2 (`SimpleRetrievalBaseline`) reuses the exact same TF-IDF
    retrieval index and `top_k` as the main agent** — only the
    classification/reply/escalation logic differs. This is deliberate: it
    isolates what the smarter components (trained classifier, evidence
    cleanup, full escalation policy) add, rather than confounding the
    comparison with a weaker retriever too.

11. **Retrieval quality is reported as an explicitly-labelled *proxy* metric
    (intent-match@k against weak heuristic labels), not a true relevance
    metric.** A real relevance-labelled retrieval eval would need a second
    human-labelling pass (rating retrieved evidence as relevant/irrelevant
    per query) that this project's timeline doesn't include. Rather than
    quietly treat the proxy as ground truth, it's named and caveated
    everywhere it's reported, including in the evaluation script's own
    output key (`retrieval_proxy_supplementary_not_required`).

12. **Three real bugs were found during this project's own audits and
    fixed rather than left alone**, each handled conservatively so the fix
    itself couldn't be mistaken for new fabricated content: (a)
    `build_evidence_corpus()` read a `predicted_intent` column that is
    never actually written to `delta_conversations.parquet` (it only ever
    existed in an in-memory dataframe inside `scripts/analyze_delta_intents.py`),
    silently defaulting every row's `intent_heuristic` to one constant value
    across all 26,168 conversations — fixed by centralizing the heuristic in
    `src/heuristic_intent.py` and regenerating affected files with the
    *same seed*, verifying the conversation-ID split itself was unaffected
    before trusting the fix. (b) Ten stale `annotation_source` values
    (`human_approved_proposal` from an earlier tool version) were renamed to
    the current convention as a pure string-rename, deliberately *not*
    re-reviewing the underlying `gold_intent`/`escalate` content of those
    rows, to avoid conflating a naming-bug fix with re-annotation. (c)
    `.gitignore`'s blanket `*.csv` rule was silently excluding the golden-set
    and human-rating CSVs from git — fixed with explicit
    `!data/golden_eval/*.csv` / `!data/eval/*.csv` exceptions, while
    raw/processed regenerable artifacts stay excluded.

13. **A raw per-example predictions artifact was added specifically to make
    failure-mode analysis honest, not just possible.** `escalation_metrics()`
    originally only exposed "unsafe false negative" indices, and
    `scripts/evaluate.py` only wrote aggregated JSON. Both were extended
    (full TP/TN/FP/FN confusion matrix; `reports/evaluation_raw_predictions.json`
    with every prediction) *before* running the real evaluation once — so
    the top-5 failure modes in the report cite actual `sample_id`s and can
    be independently checked by re-reading that file, rather than being
    asserted from aggregate numbers alone.

14. **The final report leads with a self-contained Executive Summary that
    is itself the submission-ready core report, with everything else
    explicitly demoted to a labelled appendix — rather than trimming real
    evidence to fit a page count, or silently ignoring the limit.** The
    assignment caps the report at 6 pages; the full evidence requested
    (confusion matrices, per-decision rationale, real sample IDs for every
    failure mode) does not fit in that space without losing traceability.
    The Executive Summary alone covers every mandatory element (framing,
    both baselines, real results, top 5 failure modes, misleading-headline-number,
    one-more-week, limitations) in ~1,300 words; the appendix is optional
    supporting detail for a reviewer who wants to verify a specific claim,
    not additional required reading.
