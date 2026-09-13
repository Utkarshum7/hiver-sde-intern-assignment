# Hiver SDE Intern Assignment — AI Customer Support Agent (@Delta)

End-to-end AI customer-support agent for **Delta Air Lines** (`@Delta`),
built on the Kaggle *Customer Support on Twitter* dataset
(`thoughtvector/customer-support-on-twitter`).

**Full report (executive summary + full evidence appendix — problem
framing, results, failure analysis, decision log, "what's misleading about
my headline number," etc.): [REPORT.md](REPORT.md).** This README covers
setup, reproduction, architecture, and where things live.

---

## Live Demo

**[https://delta-support-intelligence-demo.onrender.com](https://delta-support-intelligence-demo.onrender.com)**

Hosted on Render's free tier, built from this repository's
[`deploy/render-demo`](https://github.com/Utkarshum7/hiver-sde-intern-assignment/tree/deploy/render-demo)
branch — same 5-field agent contract (`intent`, `reply`, `escalate`,
`escalation_reason`, `evidence`) as the local demo below, running fully
offline against the same classifier/retrieval artifacts described in this
README (no live Delta system access, no LLM API calls). Free-tier services
sleep after ~15 minutes of inactivity, so the first request after a quiet
period can take up to ~1 minute to wake up — the UI shows a notice about
this. See ["Deploying to Render"](#deploying-to-render) below for how it's
configured, and
[`data/processed/DEPLOYMENT_ARTIFACT_NOTE.md`](data/processed/DEPLOYMENT_ARTIFACT_NOTE.md)
for the narrow, display-only text redaction applied to the deployed copy
of the retrieval evidence (phone numbers / claim-reference codes only —
does not affect classification, ranking, or any locked evaluation metric
below).

---

## Project Status (real, as of this writing)

* **Brand:** `@Delta` — 26,168 reconstructed conversation threads (87,994 messages)
* **Intent taxonomy:** 10 empirically-derived categories ([docs/intent_taxonomy.md](docs/intent_taxonomy.md))
* **Retrieval:** leakage-safe TF-IDF + cosine similarity over 20,913 historical evidence units ([docs/retrieval_design.md](docs/retrieval_design.md))
* **Agent:** intent classification → retrieval → grounded reply generation → escalation decision ([src/agent.py](src/agent.py))
* **Baselines:** trivial + simple-retrieval, both implemented and evaluated ([src/baselines.py](src/baselines.py))
* **Golden evaluation set:** 200 stratified examples sampled from held-out data — **200/200 (100%) human-annotated and locked**, all 14 required validation gates pass ([REPORT.md §9](REPORT.md#9-golden-set-methodology))
* **Headline results:** **real, measured once** against the locked golden set — intent accuracy **56.5%**, macro F1 **58.3%**, escalation F1 **63.9%**, escalation **recall 51.7%** (the important safety-relevant weakness — see [REPORT.md §12/§16](REPORT.md#12-results)), 0% unsupported-claim rate in generated replies
* **LLM-as-judge:** infrastructure complete, **PENDING** real scores (no `ANTHROPIC_API_KEY` configured)
* **Human-agreement:** infrastructure complete, **PENDING** (0/20 rating-sheet rows filled in)

---

## Architecture

```
CUSTOMER MESSAGE
      |
      v
INTENT CLASSIFICATION   src/classifier.py   (TF-IDF + Logistic Regression,
      |                                      weak-supervised on dev corpus)
      v
HISTORICAL RETRIEVAL    src/retrieval.py    (TF-IDF + cosine similarity,
      |                                      dev corpus only, top-k=3)
      v
EVIDENCE                top-k historical (customer_message, delta_response) pairs
      v
REPLY GENERATION        src/reply_generator.py  (grounded, offline, strips
      |                                          unsupported action claims)
      v
ESCALATION DECISION     src/escalation.py   (deterministic policy)
      v
FINAL RESPONSE          { intent, reply, escalate, escalation_reason, evidence }
```

Every stage has its own module and its own test file (`tests/test_classifier.py`,
`tests/test_retrieval.py`, `tests/test_reply_generator.py`, `tests/test_escalation.py`,
plus `tests/test_agent.py` for the end-to-end wiring). Full diagram + design
rationale for each stage: [REPORT.md §6](REPORT.md#6-architecture).

**Example agent behavior** (real output, non-golden example messages — run yourself with `python scripts/evaluate.py`, which includes this smoke test):

```
MSG: My suitcase came out with a broken wheel, how do I file a claim?
  -> intent=general_complaint_feedback (conf=0.45)  escalate=yes
     reason: Customer feedback escalated due to account-specific reference detected.
     reply: Thanks for reaching out, and sorry for the trouble. This one needs
            a specialist who can pull up your booking/account details...

MSG: How much does a second checked bag cost on an international flight?
  -> intent=flight_status_inquiry (conf=0.75)  escalate=no
     reply: Thanks for reaching out to Delta! Hey Charlotte! That's right, all
            of your luggage is arranged for during your check-in window.
```

Note the first example: intent was misclassified (should be
`lost_damaged_baggage`) but the escalation policy's account-signal
detection still caught it and escalated correctly — real defense-in-depth,
not a cherry-picked success. The second example shows a real, measured
weakness left in on purpose: the intent was misclassified (should be
`baggage_allowance_policy`), so the grounded reply answers the wrong
question — a live instance of the `flight_status_inquiry` over-prediction
bug quantified in [REPORT.md §16](REPORT.md#16-top-5-failure-modes) failure mode 3.

---

## Environment & Dependency Setup

**Requirements:** Python 3.12 (or 3.11+), Git.

```powershell
# Windows PowerShell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

The optional LLM judge / LLM reply-generation path needs an Anthropic API
key, never required for the rest of the pipeline:

```powershell
$env:ANTHROPIC_API_KEY = "sk-ant-..."   # optional; omit to run fully offline
```

---

## Reproduction — Full Data Preparation

**Only needed once**, or after changing the raw dataset / splitting logic.
Requires `data/raw/twcs.csv` (~500 MB, download from Kaggle — not committed
to this repo; see `.gitignore`). **Takes a few minutes** on the full
26,168-conversation Delta subset (the assignment explicitly does not
require running against the full 3M-row raw dataset beyond this one
extraction pass).

```powershell
# 1. Extract Delta conversation threads from the raw Kaggle CSV (~1-2 min)
python scripts/build_delta_dataset.py
# -> data/processed/delta_conversations.parquet, delta_messages.parquet

# 2. Leakage-safe 80/20 conversation-level split + fit the TF-IDF retrieval index (~30s)
python scripts/build_retrieval_index.py
# -> data/processed/dev_corpus.parquet, reserved_golden_pool.parquet, retrieval_index.pkl

# 3. Train the live agent's intent classifier (dev corpus only, weak-supervised) (~10-20s)
python scripts/train_classifier.py
# -> data/processed/intent_classifier.pkl

# 4. (Optional, already run once) Retrieval sanity check
python scripts/test_retrieval.py
# -> reports/retrieval_sanity_check.md
```

---

## Reproduction — Headline Evaluation (target: <15 minutes; actual: seconds)

This step only **loads** the artifacts built above — it does not touch the
raw dataset and finishes in a few seconds once step 1-3 above have run.

```powershell
python scripts/evaluate.py
```

Since the golden set is complete and locked, this **computes and writes real
results**:
- `reports/evaluation_results.json` — full metrics (intent accuracy/macro-F1/per-intent
  P-R-F1/support/confusion matrix; escalation accuracy/precision/recall/F1/confusion
  matrix/FP-FN; a supplementary retrieval proxy metric; automated reply-safety rate),
  a fresh leakage recheck, and LLM-judge/human-agreement status — for the main agent
  and both baselines, side by side.
- `reports/evaluation_raw_predictions.json` — every one of the 200×3 raw
  predictions (gold vs. predicted intent/escalate, full reply text, evidence
  used) — this is what the failure-mode analysis in REPORT.md §16 is read from.
- `reports/evaluation_report.md` — human-readable summary of the above.

If you delete/reset a portion of `golden_annotation.csv`'s `gold_intent`
column, this script will instead refuse to compute results and print the
exact validation-gate status plus a non-golden pipeline smoke test — see
[REPORT.md §11](REPORT.md#11-evaluation-methodology) for why that guardrail exists.

---

## Local Demo

A minimal local FastAPI server ([scripts/demo_server.py](scripts/demo_server.py))
so you can exercise the live agent from a browser instead of the Python
REPL. It calls `DeltaSupportAgent.handle()` and returns the same 5 fields
the pipeline already produces — `intent`, `reply`, `escalate`,
`escalation_reason`, `evidence`.

**Demo-only safety override — not part of the evaluated agent.** The demo
server adds one additional, narrow, deterministic check on top of (never
instead of) the real agent's own escalation decision: if the agent says
`escalate: no` but the customer message contains specific claim/account
phrasing (e.g. "file a claim", "baggage claim", "lost baggage", "missing
luggage", "damaged suitcase", "broken wheel", "refund my booking",
"booking reference", "reservation number"), the demo forces `escalate:
yes` and replaces the reply with a safe escalation notice, so the UI never
shows a confident policy answer for what looks like a real claim. This
logic lives **entirely in `scripts/demo_server.py`** — it does not touch
`src/agent.py`, `src/classifier.py`, `src/escalation.py`, or
`scripts/evaluate.py` in any way. **The headline evaluation numbers in
[REPORT.md](REPORT.md) and `reports/evaluation_results.json` were measured
against the real agent only and do NOT include this override** — they are
a strictly lower bound on the demo's actual escalation safety, not an
overstatement of it. When the override activates, the UI visibly labels it
("⚠ Safety override applied") so it's never mistaken for the model's own
decision.

**Install** (adds two small dependencies, `fastapi` and `uvicorn`, on top
of the base `requirements.txt`):

```powershell
pip install -r requirements.txt
```

**Launch** (requires `data/processed/intent_classifier.pkl` and
`retrieval_index.pkl` to already exist — see "Full Data Preparation" above
if you haven't run that yet):

```powershell
python -m uvicorn scripts.demo_server:app --host 127.0.0.1 --port 8000
```

**Open in a browser:** [http://127.0.0.1:8000](http://127.0.0.1:8000)

The model/retrieval-index load happens **once, at server startup**, and
takes **approximately 17 seconds** the first time (cold OS file cache) —
the page won't be reachable until that finishes. Every request after that
is fast (~15ms). Sample messages (also available as one-click buttons on
the page itself):

- `How much does a second checked bag cost on an international flight?`
- `My flight was cancelled, I need to get to Boston tonight!`
- `My suitcase came out with a broken wheel, how do I file a claim?`

**This is an offline demonstration only.** Like the underlying agent, it
never performs a live Delta action — no flight-status lookup, no
booking/PNR change, no refund, no baggage trace, no account access. Every
reply is either grounded in retrieved historical evidence / static policy
facts, or an escalation notice explaining why a human is needed.

---

## Deploying to Render

The demo server ([scripts/demo_server.py](scripts/demo_server.py)) can be
deployed as a persistent Python web service on [Render](https://render.com)
using the included [`render.yaml`](render.yaml). This section covers only
the demo — it has no relationship to, and no effect on, the evaluation
pipeline, golden dataset, or measured results (§ below).

**Required model artifacts.** The demo needs exactly two small, derived
files to run — `data/processed/intent_classifier.pkl` (~2.3 MB) and
`data/processed/retrieval_index.pkl` (~12.4 MB), together ~15 MB. These are
normally gitignored (regenerable via `scripts/train_classifier.py` /
`scripts/build_retrieval_index.py`), but `.gitignore` carries a narrow,
explicit exception for exactly these two filenames so a host that builds
from this repository has them available. They contain only a fitted TF-IDF
vectorizer/classifier and the already-public dev-corpus evidence text (no
secrets, no held-out/golden data — independently re-verified: zero overlap
between the indexed conversation IDs and the reserved/golden pool).

`retrieval_index.pkl` additionally has a narrow, display-only redaction
applied (`scripts/prepare_deployment_artifacts.py`): a small number of
phone-number- and claim-reference-shaped substrings in the evidence text
are replaced with a placeholder, while Delta's own repeated public support
numbers are left as-is. **The TF-IDF vectorizer and matrix that drive
retrieval ranking are fitted before this step and are completely
unaffected by it** — see `data/processed/DEPLOYMENT_ARTIFACT_NOTE.md`
(generated alongside the artifact) for the full disclosure, including
exactly what did and didn't change and why the locked evaluation results
are unaffected.

**Steps:**
1. Push this repository to GitHub (already done) with the two `.pkl`
   artifacts committed (see above).
2. In the Render dashboard: New → Blueprint → select this repo. Render
   reads `render.yaml` automatically and provisions one free-tier Python
   web service from it.
3. Render builds with `pip install -r requirements-deploy.txt` — a
   **deployment-only** dependency list (see "Dependencies for deployment"
   below) — and starts it with
   `uvicorn scripts.demo_server:app --host 0.0.0.0 --port $PORT`.
4. Render provides the `$PORT` environment variable automatically; nothing
   needs to be configured for it manually. Do not hardcode a port — the
   start command in `render.yaml` already reads `$PORT` at runtime.
5. Once deployed, Render gives you a free `https://<service-name>.onrender.com`
   URL automatically — no separate TLS/domain setup needed.

**No secrets required.** The service declares no API keys or credentials.
`ANTHROPIC_API_KEY` is never referenced by `render.yaml` and isn't needed —
the demo's reply generation is fully offline and deterministic.

**Expected cold-start behavior.** Render's free tier puts web services to
sleep after ~15 minutes of inactivity. The next visitor triggers a cold
start: Render re-launches the container (roughly 30-60s), and then the
app's own model/index load adds its usual ~17s on top — so the *first*
request after a quiet period can take up to roughly a minute before
responding, even though the service is healthy and will keep responding
quickly after that. The UI shows a small, permanent notice about this
("Hosted demo may take a moment to wake up after inactivity.") so a first
visitor doesn't mistake the wait for a broken deployment.

**Dependencies for deployment.** `requirements.txt` (used for local dev,
data preparation, and the test suite) is unchanged — removing packages
from it isn't safe to do without proving every existing workflow still
works (e.g. `pyarrow` for parquet I/O in `scripts/build_delta_dataset.py`,
`kaggle` for dataset download, `tqdm` for build-script progress bars,
`pytest` for the test suite). Instead, [`requirements-deploy.txt`](requirements-deploy.txt)
is a separate, deployment-only file containing just the packages the demo
server's code actually imports at request time — traced directly from
`scripts/demo_server.py`'s import graph (`pandas`, `numpy`, `scikit-learn`,
`scipy`, `joblib`, `fastapi`, `uvicorn`). This keeps the deployed
install smaller/faster without touching the file every other workflow
depends on.

**Running locally** is unchanged from the "Local Demo" section above —
`python -m uvicorn scripts.demo_server:app --host 127.0.0.1 --port 8000`
still works exactly as before; `render.yaml` only affects what happens on
Render, not local usage.

---

## Reproduction — Golden-Set Annotation (complete; commands kept for reference/audit)

```powershell
python scripts/propose_golden_labels.py       # generates AI/rule proposals (never touches gold labels)
python scripts/review_golden_proposals.py --batch   # human batch review, 10 at a time
python scripts/validate_golden_set.py         # 14 quality gates incl. leakage checks
```

Full annotator protocol: [docs/golden_annotation_guide.md](docs/golden_annotation_guide.md).
Full sampling/labelling methodology write-up: [docs/golden_set_methodology.md](docs/golden_set_methodology.md).

---

## Reproduction — Reply-Quality Judge & Human Agreement (infrastructure ready, PENDING real inputs)

Samples from the **development corpus** (never the golden set), so this
runs independently of golden-set status:

```powershell
python scripts/build_human_rating_sheet.py --n 20
# -> data/eval/human_rating_sheet.csv  (fill in the human_* columns yourself)
# -> data/eval/llm_judge_results.json  (real scores if ANTHROPIC_API_KEY is set,
#                                        otherwise honestly marked "pending_no_api_key")

python scripts/compute_judge_human_agreement.py
# -> reports/judge_human_agreement.json (only once BOTH real human ratings
#                                          and real judge scores exist; prints
#                                          exactly what's missing otherwise)
```

---

## Tests

```powershell
pytest
```

**121 tests, all passing**, covering intent classification, escalation
policy, retrieval leakage isolation, reply-generation grounding/safety,
baselines, evaluation metrics (on synthetic fixtures only — never the
golden set), annotation provenance, the end-to-end agent pipeline
contract, and the local demo server (with the model load mocked, so the
suite never pays the real ~17s load cost).

---

## Where Artifacts Live

| What | Path | Committed? |
|---|---|---|
| Raw dataset | `data/raw/twcs.csv` | No (download from Kaggle; ~500MB) |
| Processed conversations/splits/index/classifier | `data/processed/*.parquet`, `*.pkl` | No (regenerable via the scripts above) |
| **Golden evaluation set (the human-labelled deliverable)** | `data/golden_eval/golden_annotation.csv` | **Yes** |
| AI/rule-based proposals (not ground truth) | `data/golden_eval/golden_proposals.csv` | Yes |
| Human rating sheet + judge cache/results | `data/eval/*.csv`, `data/eval/*.json` | Yes (CSV); judge cache/results JSON yes |
| Evaluation results, raw predictions, reports | `reports/*.json`, `reports/*.md` | Yes |
| Source code | `src/`, `scripts/`, `tests/` | Yes |

---

## Known Limitations / Not Implemented

The agent **never** claims to do the following — every one of these is
handled by escalating to a human, not by pretending to perform it:

- No live Delta flight-status API (no real-time gate/delay lookup)
- No PNR/booking modification (seat, itinerary, reservation changes)
- No refund/payment gateway
- No baggage-tracing system
- No customer-account access (SkyMiles balance, Medallion status, booking history)

Whether the escalation policy reliably catches every case that needs one of
the above is a *measured, not assumed* question — see the real 51.7%
escalation recall and the quantified regex-coverage gap in
[REPORT.md §12/§16](REPORT.md#12-results). Full limitations list: [REPORT.md §18](REPORT.md#18-limitations).

---

## Project Documentation

* [REPORT.md](REPORT.md) — executive summary + full evidence appendix (problem framing, architecture, evaluation methodology, real results, failure analysis, "what's misleading about my headline number," limitations, one-more-week plan)
* [docs/evaluation_report.md](docs/evaluation_report.md) — the same report content, reorganized under the assignment's exact requested section headings, for quick reviewer cross-checking
* [docs/requirements_checklist.md](docs/requirements_checklist.md) — final adversarial review, requirement-by-requirement, PASS/PENDING/FAIL with evidence
* [docs/decision_log.md](docs/decision_log.md) — 14 non-obvious engineering decisions, each with alternatives considered, reason, and trade-off
* [docs/intent_taxonomy.md](docs/intent_taxonomy.md) — intent definitions, real examples, capability boundaries
* [docs/escalation_policy.md](docs/escalation_policy.md) — escalation decision matrix
* [docs/retrieval_design.md](docs/retrieval_design.md) — retrieval evidence design & leakage prevention
* [docs/brand_selection.md](docs/brand_selection.md) — brand candidate empirical analysis
* [docs/golden_set_methodology.md](docs/golden_set_methodology.md) — sampling & annotation methodology
* [docs/golden_annotation_guide.md](docs/golden_annotation_guide.md) — annotator instructions

---

## Citations & References

**Dataset.** Kaggle, *Customer Support on Twitter*
(`thoughtvector/customer-support-on-twitter`), 2,811,774 tweets, 516.5 MB —
[kaggle.com/datasets/thoughtvector/customer-support-on-twitter](https://www.kaggle.com/datasets/thoughtvector/customer-support-on-twitter).
Used for all `@Delta` conversation reconstruction, retrieval evidence, and
golden-set sampling in this project; no other external dataset is used.

**Libraries this project depends on** (see [requirements.txt](requirements.txt)
for exact version constraints) — all used through their public APIs, no
vendored or copy-pasted library source:
- [scikit-learn](https://scikit-learn.org/) — `TfidfVectorizer` + cosine
  similarity for retrieval ([src/retrieval.py](src/retrieval.py));
  `TfidfVectorizer` + `LogisticRegression` for intent classification
  ([src/classifier.py](src/classifier.py)); `sklearn.metrics` for accuracy,
  macro-F1, precision/recall/F1, confusion matrices, and Cohen's kappa
  ([src/metrics.py](src/metrics.py), [scripts/compute_judge_human_agreement.py](scripts/compute_judge_human_agreement.py))
- [SciPy](https://scipy.org/) — `pearsonr`/`spearmanr` for judge/human
  correlation ([scripts/compute_judge_human_agreement.py](scripts/compute_judge_human_agreement.py))
- [pandas](https://pandas.pydata.org/) / [NumPy](https://numpy.org/) — all
  tabular data handling and numeric computation
- [FastAPI](https://fastapi.tiangolo.com/) / [Uvicorn](https://www.uvicorn.org/) — the demo server ([scripts/demo_server.py](scripts/demo_server.py))
- [Anthropic Claude API](https://docs.anthropic.com/) — optional LLM-as-judge
  reply scoring ([src/llm_judge.py](src/llm_judge.py), [src/llm_client.py](src/llm_client.py));
  the agent's own reply generation is template/retrieval-based and never
  calls an LLM, so this dependency is judge-only and optional (see
  "LLM-as-judge" above — real scores are PENDING without a configured key)
- [joblib](https://joblib.readthedocs.io/) — persisting the trained
  classifier and retrieval index; [pyarrow](https://arrow.apache.org/) —
  Parquet I/O for intermediate datasets
- [pytest](https://pytest.org/) — the test suite (198 tests)

**Methods.** TF-IDF + cosine-similarity retrieval and TF-IDF +
logistic-regression classification are standard, well-established IR/ML
techniques, not attributed to a specific paper; no novel algorithm is
claimed anywhere in this project. Cohen's kappa (weighted) and
Pearson/Spearman correlation are standard inter-rater-agreement statistics,
computed via the library implementations cited above, not reimplemented.

**Code originality.** No source file in [`src/`](src/) or [`scripts/`](scripts/)
was copied or adapted from an external tutorial, Stack Overflow answer, or
another repository — a repo-wide search for such attributions found none
because none exist. All architecture, prompts, regexes, and heuristics are
original to this project.
