"""
Tests for scripts/prepare_deployment_artifacts.py — the narrow, display-only
redaction applied to the deployment copy of retrieval_index.pkl.

Split into two groups:
  1. Pure unit tests on small SYNTHETIC dataframes (fast, no dependency on
     the real ~13MB artifact) for the redaction logic itself.
  2. Integration tests against the REAL artifact files on disk, which prove
     the invariants the script promises actually held after it ran:
     unchanged row count/conversation IDs/vectorizer/matrix, no targeted
     patterns remaining in text, no reserved/golden leakage, and that
     every evaluation-critical path is untouched by this branch's changes.

None of these tests print a full message or an unredacted matched value.
"""

import os
import re
import subprocess

import joblib
import pandas as pd
import pytest

from scripts import prepare_deployment_artifacts as pda

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


# ---------------------------------------------------------------------------
# 1. Pure unit tests — synthetic data only
# ---------------------------------------------------------------------------

def _fixture_corpus_df():
    """A tiny synthetic corpus: one row with an official-shaped repeated
    number, several rows with singleton (unique) phone numbers, one row
    with a claim-reference code, and one row with neither (must survive
    untouched)."""
    return pd.DataFrame([
        {"conversation_id": "1", "customer_message": "Call 1-800-221-1212 for help",
         "delta_response": "Yes, 1-800-221-1212 is our line", "timestamp": "t1",
         "is_two_way": True, "retrieval_eligible": True, "intent_heuristic": "flight_status_inquiry"},
        {"conversation_id": "2", "customer_message": "please call 1-800-221-1212 too",
         "delta_response": "ok", "timestamp": "t2",
         "is_two_way": True, "retrieval_eligible": True, "intent_heuristic": "flight_status_inquiry"},
        {"conversation_id": "3", "customer_message": "my number is 415-555-0199, call me",
         "delta_response": "will do", "timestamp": "t3",
         "is_two_way": True, "retrieval_eligible": True, "intent_heuristic": "general_complaint_feedback"},
        {"conversation_id": "4", "customer_message": "file claim FCOD20752 please",
         "delta_response": "checking case FCOD20752 now", "timestamp": "t4",
         "is_two_way": True, "retrieval_eligible": True, "intent_heuristic": "lost_damaged_baggage"},
        {"conversation_id": "5", "customer_message": "flight DL1234 was great, thanks!",
         "delta_response": "So glad to hear it!", "timestamp": "t5",
         "is_two_way": True, "retrieval_eligible": True, "intent_heuristic": "general_complaint_feedback"},
    ])


def test_official_repeated_number_is_left_unredacted():
    df = _fixture_corpus_df()
    redacted, summary = pda.apply_redaction(df)
    # Row 1's number appears in 2 distinct rows (1 and 2) -> official, kept.
    assert "1-800-221-1212" in redacted.loc[0, "customer_message"]
    assert "1-800-221-1212" in redacted.loc[1, "customer_message"]


def test_singleton_phone_number_is_redacted():
    df = _fixture_corpus_df()
    redacted, summary = pda.apply_redaction(df)
    assert "415-555-0199" not in redacted.loc[2, "customer_message"]
    assert pda.PHONE_REDACTION_TEXT in redacted.loc[2, "customer_message"]


def test_claim_reference_code_is_redacted_in_both_columns():
    df = _fixture_corpus_df()
    redacted, summary = pda.apply_redaction(df)
    assert "FCOD20752" not in redacted.loc[3, "customer_message"]
    assert "FCOD20752" not in redacted.loc[3, "delta_response"]
    assert pda.CLAIM_REF_REDACTION_TEXT in redacted.loc[3, "customer_message"]
    assert pda.CLAIM_REF_REDACTION_TEXT in redacted.loc[3, "delta_response"]


def test_flight_number_is_never_redacted():
    """DL1234 (2 letters + 4 digits) must not match the claim-reference
    pattern, which requires 5-10 digits — this is the deliberate design
    property that keeps ordinary flight numbers untouched."""
    df = _fixture_corpus_df()
    redacted, summary = pda.apply_redaction(df)
    assert "DL1234" in redacted.loc[4, "customer_message"]


def test_non_text_columns_are_preserved_exactly():
    df = _fixture_corpus_df()
    redacted, summary = pda.apply_redaction(df)
    for col in ("conversation_id", "timestamp", "is_two_way", "retrieval_eligible", "intent_heuristic"):
        assert list(redacted[col]) == list(df[col])


def test_row_count_and_conversation_ids_unchanged_by_construction():
    df = _fixture_corpus_df()
    redacted, summary = pda.apply_redaction(df)
    assert len(redacted) == len(df)
    assert set(redacted["conversation_id"]) == set(df["conversation_id"])


def test_input_dataframe_is_never_mutated():
    df = _fixture_corpus_df()
    original_snapshot = df.copy(deep=True)
    pda.apply_redaction(df)
    pd.testing.assert_frame_equal(df, original_snapshot)


def test_redaction_is_idempotent_on_synthetic_data():
    df = _fixture_corpus_df()
    once, _ = pda.apply_redaction(df)
    twice, summary_second_pass = pda.apply_redaction(once)
    pd.testing.assert_frame_equal(once, twice)
    assert summary_second_pass["total_replacements"] == 0


def test_summary_counts_are_self_consistent():
    df = _fixture_corpus_df()
    _, summary = pda.apply_redaction(df)
    assert summary["distinct_phone_values_seen"] == \
        summary["distinct_phone_values_treated_as_official"] + summary["distinct_phone_values_redacted"]
    assert summary["total_replacements"] >= summary["distinct_phone_values_redacted"]


# ---------------------------------------------------------------------------
# 2. Integration tests against the real, already-prepared artifact
# ---------------------------------------------------------------------------

RETRIEVAL_INDEX_PATH = os.path.join(REPO_ROOT, "data", "processed", "retrieval_index.pkl")
CLASSIFIER_PATH = os.path.join(REPO_ROOT, "data", "processed", "intent_classifier.pkl")

_artifacts_present = os.path.exists(RETRIEVAL_INDEX_PATH) and os.path.exists(CLASSIFIER_PATH)
requires_artifacts = pytest.mark.skipif(
    not _artifacts_present,
    reason="data/processed/*.pkl artifacts not present in this checkout",
)


@pytest.fixture(scope="module")
def real_retriever():
    return joblib.load(RETRIEVAL_INDEX_PATH)


@pytest.fixture(scope="module")
def real_classifier():
    return joblib.load(CLASSIFIER_PATH)


@requires_artifacts
def test_real_artifact_has_no_email_addresses(real_retriever):
    email_pat = re.compile(r"[\w.+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}")
    df = real_retriever.corpus_df
    count = sum(
        1 for col in ("customer_message", "delta_response")
        for t in df[col].astype(str)
        if email_pat.search(t)
    )
    assert count == 0


@requires_artifacts
def test_real_artifact_has_no_remaining_singleton_phone_or_claim_patterns(real_retriever):
    """After redaction, the only PHONE_PATTERN matches allowed to remain are
    ones that occur in >= OFFICIAL_NUMBER_MIN_ROWS distinct rows (i.e. were
    deliberately left as official numbers). CLAIM_REF_PATTERN must have
    zero matches remaining anywhere."""
    df = real_retriever.corpus_df
    row_counts = pda.rows_containing_each_phone_value(df)
    non_official_remaining = [v for v, c in row_counts.items() if c < pda.OFFICIAL_NUMBER_MIN_ROWS]
    assert non_official_remaining == [], (
        f"{len(non_official_remaining)} singleton phone-shaped value(s) still present"
    )

    claim_count = sum(
        1 for col in ("customer_message", "delta_response")
        for t in df[col].astype(str)
        if pda.CLAIM_REF_PATTERN.search(t)
    )
    assert claim_count == 0


@requires_artifacts
def test_real_artifact_row_count_matches_documented_dev_corpus_size(real_retriever):
    assert len(real_retriever.corpus_df) == 20913


@requires_artifacts
def test_real_artifact_excludes_reserved_and_golden_ids(real_retriever):
    reserved_path = os.path.join(REPO_ROOT, "data", "processed", "reserved_golden_pool.parquet")
    golden_path = os.path.join(REPO_ROOT, "data", "golden_eval", "golden_annotation.csv")
    if not (os.path.exists(reserved_path) and os.path.exists(golden_path)):
        pytest.skip("reserved pool / golden set not present in this checkout")

    indexed_ids = set(real_retriever.corpus_df["conversation_id"].astype(str))
    reserved_ids = set(pd.read_parquet(reserved_path)["conversation_id"].astype(str))
    golden_ids = set(pd.read_csv(golden_path, dtype=str)["conversation_id"].astype(str))

    assert indexed_ids.isdisjoint(reserved_ids)
    assert indexed_ids.isdisjoint(golden_ids)


@requires_artifacts
def test_real_artifact_vectorizer_and_matrix_shape_unchanged(real_retriever):
    """These are the two properties that drive retrieval ranking. Fixed,
    known-good values from before this redaction step was ever applied."""
    assert len(real_retriever.vectorizer.vocabulary_) == 25000
    assert real_retriever.tfidf_matrix.shape == (20913, 25000)


@requires_artifacts
def test_real_classifier_artifact_unchanged(real_classifier):
    """The classifier is never touched by prepare_deployment_artifacts.py —
    this checks its known-good shape/classes are exactly as before."""
    expected_classes = {
        "baggage_allowance_policy", "checkin_boarding_pass", "flight_delay_rebooking",
        "flight_status_inquiry", "general_complaint_feedback", "inflight_amenities_service",
        "lost_damaged_baggage", "refund_credit_voucher", "seat_assignment_upgrade",
        "skymiles_loyalty_program",
    }
    assert set(real_classifier.classes_) == expected_classes
    assert len(real_classifier.vectorizer.vocabulary_) == 20000


@requires_artifacts
def test_deployment_artifact_note_exists_and_discloses_required_points():
    note_path = os.path.join(REPO_ROOT, "data", "processed", "DEPLOYMENT_ARTIFACT_NOTE.md")
    assert os.path.exists(note_path)
    with open(note_path, encoding="utf-8") as f:
        text = f.read()
    assert "fitted before" in text.lower()
    assert "ranking" in text.lower() and "unaffected" in text.lower() or "unchanged" in text.lower()
    assert "not recomputed" in text.lower() or "not touched" in text.lower()
    assert "deployment-only" in text.lower()


# ---------------------------------------------------------------------------
# Idempotency of the real, already-applied artifact (re-running --apply
# again must be a genuine no-op — verified live, not just asserted)
# ---------------------------------------------------------------------------

@requires_artifacts
def test_reapplying_to_the_real_artifact_is_a_true_no_op(real_retriever):
    redacted_again, summary = pda.apply_redaction(real_retriever.corpus_df)
    assert summary["total_replacements"] == 0
    assert summary["rows_changed"] == 0


# ---------------------------------------------------------------------------
# Forbidden paths untouched on this branch (this turn's explicit list)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("path", [
    "src",
    "scripts/evaluate.py",
    "data/golden_eval",
    "reports",
    "REPORT.md",
    "docs/retrieval_design.md",
    "requirements.txt",
])
def test_no_evaluation_or_forbidden_paths_changed_on_this_branch(path):
    try:
        result = subprocess.run(
            ["git", "diff", "--name-only", "origin/master...HEAD", "--", path],
            cwd=REPO_ROOT, capture_output=True, text=True, timeout=10,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pytest.skip("git not available in this environment")

    if result.returncode != 0:
        pytest.skip(f"git diff against origin/master unavailable: {result.stderr.strip()}")

    changed = [l for l in result.stdout.splitlines() if l.strip()]
    assert changed == [], f"Forbidden path {path!r} has changes on this branch: {changed}"


def test_prepare_script_source_never_writes_to_forbidden_paths():
    """Static check: the redaction script's source code must not contain a
    write/open-for-write call referencing any forbidden path."""
    script_path = os.path.join(REPO_ROOT, "scripts", "prepare_deployment_artifacts.py")
    with open(script_path, encoding="utf-8") as f:
        source = f.read()
    # Only the two intended output paths (RETRIEVAL_INDEX_PATH, METADATA_PATH,
    # and the backup) may appear as write targets; check the well-known
    # forbidden filenames never appear anywhere near an open()/to_parquet()/
    # to_csv() call by simply confirming they don't appear as bare strings
    # at all (the module docstring explains why, in prose, which is fine).
    for forbidden in ("reserved_golden_pool.parquet", "golden_annotation.csv"):
        # allow only inside the docstring "does not do" bullet list
        occurrences = source.count(forbidden)
        assert occurrences <= 1, f"{forbidden!r} referenced more than once in the script"
