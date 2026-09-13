"""
Deployment-artifact preparation — narrow, display-only redaction.

WHAT THIS DOES: loads the already-built data/processed/intent_classifier.pkl
and data/processed/retrieval_index.pkl, and applies a narrow, deterministic
redaction to exactly two TEXT columns of the retrieval index's corpus
(`customer_message`, `delta_response`) — the columns the demo's Evidence
panel displays. It replaces phone-number-shaped substrings (except ones
that recur across many different conversations, treated as Delta's own
published support numbers) and claim/case-reference-shaped codes with a
fixed placeholder string.

WHAT THIS DOES NOT DO — by construction, not by promise:
  - Does NOT touch intent_classifier.pkl at all (loaded only to prove it's
    loadable; never re-saved).
  - Does NOT refit, rebuild, or otherwise touch the TF-IDF vectorizer or
    tfidf_matrix inside retrieval_index.pkl — only corpus_df's two text
    columns are replaced; retrieval ranking/similarity scores are
    unaffected (verified by tests: vocabulary and matrix shape unchanged).
  - Does NOT touch conversation_id, timestamp, is_two_way,
    retrieval_eligible, or intent_heuristic — copied through unchanged.
  - Does NOT change row count or which conversations are indexed.
  - Does NOT read, write, or reference data/golden_eval/,
    reserved_golden_pool.parquet, reports/, or scripts/evaluate.py.
  - NEVER prints a full message or an unredacted matched value — only
    counts, regex source strings, and file paths.

Usage:
    python scripts/prepare_deployment_artifacts.py           # dry run (default): report only, writes nothing
    python scripts/prepare_deployment_artifacts.py --apply   # actually write the redacted artifact + backup + metadata note

Deterministic and idempotent: re-running (with --apply) against an
already-redacted artifact makes zero further changes, because the
replacement placeholders never themselves match the redaction patterns.
"""

import argparse
import os
import re
import shutil
import sys
from collections import Counter
from datetime import datetime, timezone

import joblib
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

CLASSIFIER_PATH = "data/processed/intent_classifier.pkl"
RETRIEVAL_INDEX_PATH = "data/processed/retrieval_index.pkl"
BACKUP_PATH = RETRIEVAL_INDEX_PATH + ".pre_redaction_backup"
METADATA_PATH = "data/processed/DEPLOYMENT_ARTIFACT_NOTE.md"

TEXT_COLUMNS = ("customer_message", "delta_response")

# ---------------------------------------------------------------------------
# Redaction patterns — identical to the ones used in the prior read-only
# privacy review, so results are directly comparable to that report.
# ---------------------------------------------------------------------------

PHONE_PATTERN = re.compile(r"\b(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]\d{3}[-.\s]\d{4}\b")
# Deliberately requires 5-10 digits after 2-4 letters, which naturally
# excludes typical Delta flight numbers (2 letters + up to 4 digits, e.g.
# "DL1234") while still catching claim/case-reference-shaped codes like the
# "FCODL20752" / "IR..." examples found during the privacy review.
CLAIM_REF_PATTERN = re.compile(r"\b[A-Z]{2,4}\d{5,10}\b")

PHONE_REDACTION_TEXT = "[phone number redacted]"
CLAIM_REF_REDACTION_TEXT = "[reference number redacted]"

# A phone-shaped value seen in this many or more DISTINCT rows across the
# whole corpus is treated as a widely-repeated / official number (Delta's
# own published support lines show up dozens of times each) and is left
# untouched. This is inferred from the corpus itself, not a hardcoded
# phone-number allowlist, so it can't go stale. Values seen in fewer rows
# are treated as more likely to be a single customer's own number and are
# redacted.
OFFICIAL_NUMBER_MIN_ROWS = 2


# ---------------------------------------------------------------------------
# Pure, unit-testable redaction logic (no I/O) — see tests/test_deployment_artifacts.py
# ---------------------------------------------------------------------------

def rows_containing_each_phone_value(df: pd.DataFrame) -> Counter:
    """For each distinct phone-shaped string, counts the number of DISTINCT
    rows it appears in (across either text column) anywhere in the corpus."""
    counts = Counter()
    for _, row in df.iterrows():
        seen_this_row = set()
        for col in TEXT_COLUMNS:
            for m in PHONE_PATTERN.finditer(str(row[col])):
                seen_this_row.add(m.group(0))
        for val in seen_this_row:
            counts[val] += 1
    return counts


def redact_text(text: str, official_phone_values: set) -> "tuple[str, int]":
    """Applies both redaction patterns to one string. Returns
    (redacted_text, replacement_count). Never logs the matched value."""
    text = str(text)
    replacements = [0]

    def _phone_sub(m):
        if m.group(0) in official_phone_values:
            return m.group(0)
        replacements[0] += 1
        return PHONE_REDACTION_TEXT

    text = PHONE_PATTERN.sub(_phone_sub, text)

    def _claim_sub(m):
        replacements[0] += 1
        return CLAIM_REF_REDACTION_TEXT

    text = CLAIM_REF_PATTERN.sub(_claim_sub, text)

    return text, replacements[0]


def apply_redaction(corpus_df: pd.DataFrame) -> "tuple[pd.DataFrame, dict]":
    """Returns a NEW dataframe (input is never mutated) with customer_message
    and delta_response redacted; all other columns copied through
    unchanged. Also returns a summary dict for reporting (counts only —
    never the matched values themselves)."""
    row_counts = rows_containing_each_phone_value(corpus_df)
    official_values = {v for v, c in row_counts.items() if c >= OFFICIAL_NUMBER_MIN_ROWS}

    new_df = corpus_df.copy(deep=True)
    total_replacements = 0
    changed_row_mask = pd.Series(False, index=new_df.index)

    for col in TEXT_COLUMNS:
        new_values = []
        for idx, val in new_df[col].items():
            redacted, n = redact_text(val, official_values)
            new_values.append(redacted)
            total_replacements += n
            if n > 0:
                changed_row_mask.loc[idx] = True
        new_df[col] = new_values

    summary = {
        "distinct_phone_values_seen": len(row_counts),
        "distinct_phone_values_treated_as_official": len(official_values),
        "distinct_phone_values_redacted": len(row_counts) - len(official_values),
        "total_replacements": total_replacements,
        "rows_changed": int(changed_row_mask.sum()),
    }
    return new_df, summary


# ---------------------------------------------------------------------------
# Metadata note
# ---------------------------------------------------------------------------

def _metadata_note_text(summary: dict, row_count: int) -> str:
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    return f"""# Deployment Artifact Note — retrieval_index.pkl

Generated by `scripts/prepare_deployment_artifacts.py` on {ts}.

**This file describes a deployment-only property of `retrieval_index.pkl`.
It does not describe, and has no effect on, the locked golden-set
evaluation results in `reports/`.**

## What was done

The TF-IDF vectorizer and TF-IDF matrix inside `retrieval_index.pkl` were
**fitted before this redaction step, from the unmodified dev corpus, and
are byte-for-byte unchanged by it.** Retrieval ranking and similarity
scoring are therefore identical to before this step — this redaction only
rewrites the *display* text (`customer_message`, `delta_response`) shown by
the demo's Evidence panel for a small number of rows.

- Rows in the corpus: {row_count} (unchanged)
- Distinct phone-shaped values found: {summary['distinct_phone_values_seen']}
- Treated as official/repeated (seen in >= {OFFICIAL_NUMBER_MIN_ROWS} distinct
  rows — e.g. Delta's own published support lines) and left as-is:
  {summary['distinct_phone_values_treated_as_official']}
- Treated as singleton/rare and redacted: {summary['distinct_phone_values_redacted']}
- Total substring replacements (phone + claim-reference patterns combined):
  {summary['total_replacements']}
- Rows with at least one replacement: {summary['rows_changed']} of {row_count}

## What this means for evaluation

- `reports/evaluation_results.json` and `reports/evaluation_raw_predictions.json`
  are static files already committed from a run against the pre-redaction
  artifact. **They are not recomputed, not touched, and remain the locked
  record of that measurement.**
- This artifact is written to the same path the local agent and
  `scripts/evaluate.py` both read from (`data/processed/retrieval_index.pkl`)
  — there is currently no separate deployment-only path in this codebase,
  and adding one would require modifying `src/agent.py`, which this change
  deliberately avoids.
- If `scripts/evaluate.py` were re-run in the future against this
  redacted artifact: intent classification, escalation decisions, and
  which evidence is retrieved (ranking) would be **provably identical**
  (the vectorizer and tfidf_matrix that drive all of that are untouched —
  see the automated tests). The only theoretical difference is the exact
  wording of a *generated reply* on the small number of rows where a
  redacted phrase happened to be part of quoted evidence text — a
  cosmetic difference, not a change in any measured metric.
- This artifact is **deployment-only**. It is not used to produce, and
  does not retroactively change, the locked evaluation results.

## Reversibility

A byte-for-byte backup of the pre-redaction file was saved to
`retrieval_index.pkl.pre_redaction_backup` before this file was
overwritten (not committed to git; local safety net only).
"""


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--apply", action="store_true",
        help="Actually write the redacted artifact, backup, and metadata note. "
             "Without this flag, only a dry-run report is printed and nothing is written.",
    )
    args = parser.parse_args()

    print("=" * 70)
    print("DEPLOYMENT ARTIFACT PREPARATION")
    print("=" * 70)

    print(f"\nLoading {CLASSIFIER_PATH} (read-only; this script never modifies it)...")
    joblib.load(CLASSIFIER_PATH)  # loaded only to prove it's a valid, loadable artifact
    print("  OK — loaded successfully, will not be touched or re-saved.")

    print(f"\nLoading {RETRIEVAL_INDEX_PATH}...")
    retriever = joblib.load(RETRIEVAL_INDEX_PATH)
    original_row_count = len(retriever.corpus_df)
    original_vocab_size = len(retriever.vectorizer.vocabulary_)
    original_matrix_shape = retriever.tfidf_matrix.shape
    print(f"  corpus_df rows: {original_row_count}")
    print(f"  vectorizer vocabulary size: {original_vocab_size}")
    print(f"  tfidf_matrix shape: {original_matrix_shape}")

    print("\nComputing redaction (in memory; nothing written yet)...")
    redacted_df, summary = apply_redaction(retriever.corpus_df)

    print("\n" + "-" * 70)
    print("REPORT (counts only — no message text or matched values shown)")
    print("-" * 70)
    print(f"Regex — phone pattern     : {PHONE_PATTERN.pattern}")
    print(f"Regex — claim/ref pattern : {CLAIM_REF_PATTERN.pattern}")
    print(f"Official-number rule      : a phone-shaped value seen in >= "
          f"{OFFICIAL_NUMBER_MIN_ROWS} distinct rows is treated as an official/"
          f"repeated Delta number and left unredacted.")
    print()
    print(f"Distinct phone-shaped values found      : {summary['distinct_phone_values_seen']}")
    print(f"  -> treated as official, left as-is    : {summary['distinct_phone_values_treated_as_official']}")
    print(f"  -> treated as singleton, redacted     : {summary['distinct_phone_values_redacted']}")
    print(f"Total substring replacements            : {summary['total_replacements']}")
    print(f"Rows with >=1 replacement               : {summary['rows_changed']} of {original_row_count}")
    print()
    print("Files that will change if --apply is used:")
    print(f"  {RETRIEVAL_INDEX_PATH}  (corpus_df text columns only — vectorizer/tfidf_matrix untouched)")
    print(f"  {BACKUP_PATH}  (new — untouched pre-redaction backup)")
    print(f"  {METADATA_PATH}  (new)")
    print(f"\n{CLASSIFIER_PATH} will NOT be modified.")

    if not args.apply:
        print("\n" + "=" * 70)
        print("DRY RUN — no files were written.")
        print("Re-run with --apply to actually write the redacted artifact.")
        print("=" * 70)
        return

    print("\nApplying (--apply given)...")
    shutil.copy2(RETRIEVAL_INDEX_PATH, BACKUP_PATH)
    print(f"  Backed up original to {BACKUP_PATH}")

    retriever.corpus_df = redacted_df
    joblib.dump(retriever, RETRIEVAL_INDEX_PATH)
    print(f"  Wrote redacted artifact to {RETRIEVAL_INDEX_PATH}")

    # Self-check: reload and verify the invariants this script promises.
    reloaded = joblib.load(RETRIEVAL_INDEX_PATH)
    assert len(reloaded.corpus_df) == original_row_count, "row count changed!"
    assert len(reloaded.vectorizer.vocabulary_) == original_vocab_size, "vocabulary changed!"
    assert reloaded.tfidf_matrix.shape == original_matrix_shape, "matrix shape changed!"
    assert set(reloaded.corpus_df["conversation_id"].astype(str)) == \
        set(retriever.corpus_df["conversation_id"].astype(str)), "conversation IDs changed!"
    print("  Self-check passed: row count, vocabulary size, matrix shape, "
          "and conversation IDs all confirmed unchanged after reload.")

    with open(METADATA_PATH, "w", encoding="utf-8") as f:
        f.write(_metadata_note_text(summary, original_row_count))
    print(f"  Wrote {METADATA_PATH}")

    print("\nDone. Nothing has been committed, pushed, or deployed.")


if __name__ == "__main__":
    main()
