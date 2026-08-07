"""Verification gate tests — ADR 0002 self-consistency + schema validity."""

from forge.schema import PIIRecord, PIISpan, PIIType
from forge.verify import (
    RejectReason,
    VerificationStats,
    VerifiedRecord,
    _validate_schema,
    majority_vote_spans,
    update_stats,
    verify_record,
)


def _span(start: int, end: int, label: PIIType, text: str) -> PIISpan:
    return PIISpan(start=start, end=end, label=label, text=text)


def _rec(text: str, spans: list[PIISpan], rid: str = "t1") -> PIIRecord:
    return PIIRecord(id=rid, text=text, spans=spans, split="train")


# --- majority_vote_spans ---


def test_unanimous_agreement():
    text = "Call Alice at alice@x.com"
    spans = [_span(5, 10, PIIType.PERSON, "Alice"), _span(14, 25, PIIType.EMAIL, "alice@x.com")]
    samples = [_rec(text, spans) for _ in range(3)]
    consensus, agreement = majority_vote_spans(samples)
    assert len(consensus) == 2
    assert agreement == 1.0


def test_majority_keeps_span():
    text = "Hello Bob"
    span_bob = _span(6, 9, PIIType.PERSON, "Bob")
    samples = [
        _rec(text, [span_bob]),
        _rec(text, [span_bob]),
        _rec(text, []),
    ]
    consensus, agreement = majority_vote_spans(samples)
    assert len(consensus) == 1
    assert consensus[0].text == "Bob"
    assert abs(agreement - 2 / 3) < 1e-9


def test_minority_drops_span():
    text = "Hello Bob"
    span_bob = _span(6, 9, PIIType.PERSON, "Bob")
    samples = [
        _rec(text, [span_bob]),
        _rec(text, []),
        _rec(text, []),
    ]
    consensus, _agreement = majority_vote_spans(samples)
    assert len(consensus) == 0


def test_no_samples():
    consensus, agreement = majority_vote_spans([])
    assert consensus == []
    assert agreement == 0.0


def test_all_empty_spans():
    text = "No PII here"
    samples = [_rec(text, []) for _ in range(3)]
    consensus, agreement = majority_vote_spans(samples)
    assert consensus == []
    assert agreement == 1.0


def test_different_labels_same_offsets():
    text = "Number 12345"
    samples = [
        _rec(text, [_span(7, 12, PIIType.SSN, "12345")]),
        _rec(text, [_span(7, 12, PIIType.BANK_ACCOUNT, "12345")]),
        _rec(text, [_span(7, 12, PIIType.SSN, "12345")]),
    ]
    consensus, _agreement = majority_vote_spans(samples)
    assert len(consensus) == 1
    assert consensus[0].label == PIIType.SSN


def test_single_sample_minority_not_enough():
    text = "Hi Alice"
    span = _span(3, 8, PIIType.PERSON, "Alice")
    samples = [
        _rec(text, [span]),
        _rec(text, []),
        _rec(text, []),
    ]
    consensus, _agreement = majority_vote_spans(samples)
    assert len(consensus) == 0


def test_k1_requires_one_vote():
    text = "Hello Bob"
    span = _span(6, 9, PIIType.PERSON, "Bob")
    consensus, agreement = majority_vote_spans([_rec(text, [span])])
    assert len(consensus) == 1
    assert agreement == 1.0


def test_k2_majority_at_half():
    text = "Hello Bob"
    span = _span(6, 9, PIIType.PERSON, "Bob")
    samples = [_rec(text, [span]), _rec(text, [])]
    consensus, _agreement = majority_vote_spans(samples)
    assert len(consensus) == 1


def test_k5_needs_three_votes():
    text = "Hello Bob"
    span = _span(6, 9, PIIType.PERSON, "Bob")
    samples = [_rec(text, [span])] * 3 + [_rec(text, [])] * 2
    consensus, agreement = majority_vote_spans(samples)
    assert len(consensus) == 1
    assert abs(agreement - 3 / 5) < 1e-9


# --- _validate_schema (uses model_construct to bypass Pydantic validators) ---


def _bad_rec(text: str, spans: list[PIISpan]) -> PIIRecord:
    """Construct a PIIRecord bypassing validators — for testing _validate_schema."""
    return PIIRecord.model_construct(
        id="t1", text=text, spans=spans, lang="en",
        source="synthetic:faker", split="train",
    )


def test_validate_schema_ok():
    rec = _rec("Hello Alice", [_span(6, 11, PIIType.PERSON, "Alice")])
    assert _validate_schema(rec) == []


def test_validate_schema_out_of_bounds():
    bad_span = PIISpan.model_construct(start=0, end=10, label=PIIType.PERSON, text="Hi")
    rec = _bad_rec("Hi", [bad_span])
    issues = _validate_schema(rec)
    assert len(issues) == 1
    assert "out of bounds" in issues[0]


def test_validate_schema_inverted_span():
    bad_span = PIISpan.model_construct(start=11, end=6, label=PIIType.PERSON, text="Alice")
    rec = _bad_rec("Hello Alice", [bad_span])
    issues = _validate_schema(rec)
    assert len(issues) == 1
    assert "inverted" in issues[0]


def test_validate_schema_text_mismatch():
    bad_span = PIISpan.model_construct(start=6, end=11, label=PIIType.PERSON, text="Bob")
    rec = _bad_rec("Hello Alice", [bad_span])
    issues = _validate_schema(rec)
    assert len(issues) == 1
    assert "mismatch" in issues[0]


def test_validate_schema_no_cascading_errors():
    bad_span = PIISpan.model_construct(start=0, end=100, label=PIIType.PERSON, text="wrong")
    rec = _bad_rec("Hi", [bad_span])
    issues = _validate_schema(rec)
    assert len(issues) == 1


# --- verify_record ---


def test_verify_all_valid_unanimous():
    text = "SSN 123-45-6789"
    span = _span(4, 15, PIIType.SSN, "123-45-6789")
    samples = [_rec(text, [span]) for _ in range(3)]
    result = verify_record("r1", text, samples, [True, True, True])
    assert result.accepted
    assert len(result.record.spans) == 1
    assert result.agreement_ratio == 1.0


def test_verify_majority_schema_invalid():
    text = "Hello"
    samples = [_rec(text, []) for _ in range(3)]
    result = verify_record("r1", text, samples, [False, False, True])
    assert not result.accepted
    assert RejectReason.MAJORITY_SCHEMA_INVALID in result.reject_reasons


def test_verify_all_schema_invalid():
    text = "Hello"
    samples = [_rec(text, []) for _ in range(3)]
    result = verify_record("r1", text, samples, [False, False, False])
    assert not result.accepted
    assert RejectReason.MAJORITY_SCHEMA_INVALID in result.reject_reasons


def test_verify_no_samples_at_all():
    result = verify_record("r1", "Hello", [], [], min_samples=1)
    assert not result.accepted
    assert RejectReason.NO_VALID_SAMPLES in result.reject_reasons


def test_verify_low_agreement_rejected():
    text = "Name Alice email alice@x.com"
    s1 = _span(5, 10, PIIType.PERSON, "Alice")
    s2 = _span(17, 28, PIIType.EMAIL, "alice@x.com")
    s3 = _span(5, 10, PIIType.USERNAME, "Alice")
    samples = [
        _rec(text, [s1, s2]),
        _rec(text, [s3]),
        _rec(text, []),
    ]
    result = verify_record("r1", text, samples, [True, True, True], min_agreement=0.9)
    assert not result.accepted
    assert RejectReason.LOW_AGREEMENT in result.reject_reasons


def test_verify_preserves_record_id_and_split():
    text = "Hello Bob"
    span = _span(6, 9, PIIType.PERSON, "Bob")
    samples = [_rec(text, [span]) for _ in range(3)]
    result = verify_record("custom_id", text, samples, [True, True, True], split="dev")
    assert result.record.id == "custom_id"
    assert result.record.split == "dev"


def test_verify_too_few_samples():
    text = "Hello Bob"
    span = _span(6, 9, PIIType.PERSON, "Bob")
    samples = [_rec(text, [span])]
    result = verify_record("r1", text, samples, [True], min_samples=2)
    assert not result.accepted
    assert RejectReason.TOO_FEW_SAMPLES in result.reject_reasons


def test_verify_constraint_violation_excluded_from_vote():
    text = "Hello Bob"
    good_span = _span(6, 9, PIIType.PERSON, "Bob")
    bad_span = PIISpan.model_construct(start=6, end=9, label=PIIType.PERSON, text="WRONG")
    bad_rec = PIIRecord.model_construct(
        id="t1", text=text, spans=[bad_span], lang="en",
        source="synthetic:faker", split="train",
    )
    samples = [
        _rec(text, [good_span]),
        _rec(text, [good_span]),
        bad_rec,
    ]
    result = verify_record("r1", text, samples, [True, True, True])
    assert RejectReason.CONSTRAINT_VIOLATION in result.reject_reasons
    assert not result.accepted


# --- update_stats ---


def test_stats_accepted():
    stats = VerificationStats()
    text = "Hi Alice"
    span = _span(3, 8, PIIType.PERSON, "Alice")
    rec = PIIRecord(id="r1", text=text, spans=[span], split="train")
    vr = VerifiedRecord(record=rec, accepted=True, n_samples=3, agreement_ratio=1.0)
    update_stats(stats, vr)
    assert stats.total == 1
    assert stats.accepted == 1
    assert stats.per_type_accepted["PERSON"] == 1


def test_stats_rejected_consistency():
    stats = VerificationStats()
    rec = PIIRecord(id="r1", text="Hello", spans=[], split="train")
    vr = VerifiedRecord(record=rec, accepted=False, n_samples=3, agreement_ratio=0.3,
                        reject_reasons=(RejectReason.LOW_AGREEMENT,))
    update_stats(stats, vr)
    assert stats.total == 1
    assert stats.accepted == 0
    assert stats.rejected_consistency == 1


def test_stats_rejected_schema():
    stats = VerificationStats()
    rec = PIIRecord(id="r1", text="Hello", spans=[], split="train")
    vr = VerifiedRecord(record=rec, accepted=False, n_samples=3, agreement_ratio=0.0,
                        reject_reasons=(RejectReason.MAJORITY_SCHEMA_INVALID,))
    update_stats(stats, vr)
    assert stats.rejected_schema == 1


def test_stats_rejected_empty():
    stats = VerificationStats()
    rec = PIIRecord(id="r1", text="Hello", spans=[], split="train")
    vr = VerifiedRecord(record=rec, accepted=False, n_samples=3, agreement_ratio=0.0,
                        reject_reasons=(RejectReason.NO_VALID_SAMPLES,))
    update_stats(stats, vr)
    assert stats.rejected_empty == 1


def test_stats_summary():
    stats = VerificationStats()
    stats.total = 10
    stats.accepted = 7
    stats.rejected_consistency = 2
    stats.rejected_schema = 1
    s = stats.summary()
    assert s["accept_rate"] == 0.7
    assert s["total"] == 10
