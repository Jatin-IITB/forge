"""Schema invariants: offsets, overlap, derived redaction, severity."""

import pytest
from pydantic import ValidationError

from forge.schema import HIGH_SEVERITY, PIIRecord, PIISpan, PIIType


def test_redaction_is_derived_and_correct():
    r = PIIRecord(
        id="t1",
        text="Email me at a@b.com now",
        spans=[PIISpan(start=12, end=19, label=PIIType.EMAIL, text="a@b.com")],
    )
    assert r.redacted == "Email me at [EMAIL] now"


def test_span_text_must_match_offsets():
    with pytest.raises(ValidationError):
        PIIRecord(
            id="t2",
            text="hello world",
            spans=[PIISpan(start=0, end=5, label=PIIType.PERSON, text="WRONG")],
        )


def test_span_cannot_exceed_text():
    with pytest.raises(ValidationError):
        PIIRecord(
            id="t3",
            text="abc",
            spans=[PIISpan(start=0, end=10, label=PIIType.PERSON, text="abc")],
        )


def test_overlapping_spans_rejected():
    with pytest.raises(ValidationError):
        PIIRecord(
            id="t4",
            text="John Smith",
            spans=[
                PIISpan(start=0, end=4, label=PIIType.PERSON, text="John"),
                PIISpan(start=0, end=10, label=PIIType.PERSON, text="John Smith"),
            ],
        )


def test_high_severity_flag():
    s = PIISpan(start=0, end=4, label=PIIType.SSN, text="1234")
    assert s.is_high_severity
    assert PIIType.SSN in HIGH_SEVERITY
    assert PIIType.PERSON not in HIGH_SEVERITY


def test_empty_spans_is_valid_negative_example():
    r = PIIRecord(id="t5", text="no pii here", spans=[])
    assert r.redacted == "no pii here"
