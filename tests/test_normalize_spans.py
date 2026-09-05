"""Span normalization: what it trims, what it refuses to touch.

These test the pure trimming logic with no tokenizer, so they run in the
eval-only venv. The tokenizer-dependent half -- which records get dropped as
unrepresentable -- is covered by exercising the real round-trip against
hand-built offsets, which pins the behaviour without pulling in torch.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from normalize_spans import trim_record

from forge.schema import PIIRecord, PIISpan, PIIType


def _record(text: str, spans: list[tuple[int, int, PIIType]]) -> PIIRecord:
    return PIIRecord(
        id="t-1",
        text=text,
        spans=[PIISpan(start=s, end=e, label=lab, text=text[s:e]) for s, e, lab in spans],
        split="train",
    )


def test_trailing_space_is_trimmed_off_a_span():
    """The train-v3-00768 defect: Faker's address value ends in a space."""
    # Two spaces, as in the real record: one from Faker's address value and one
    # from the template. BOTH must go -- trimming a single character would leave
    # the span still unrepresentable and the gate would fire again at hour two.
    text = "incident at 12/074, Ganguly Zila, Khora  (near East April)"
    start = text.index("12/074")
    end = text.index("(near")  # past both spaces
    record = _record(text, [(start, end, PIIType.STREET_ADDRESS)])
    assert record.spans[0].text.endswith("  ")

    fixed, changed = trim_record(record)

    assert changed == 1
    assert fixed.spans[0].text == "12/074, Ganguly Zila, Khora"
    assert fixed.spans[0].end == text.index("Khora") + len("Khora")
    assert fixed.spans[0].label is PIIType.STREET_ADDRESS


def test_leading_whitespace_is_trimmed_too():
    text = "Name:  Tristan Batta reported"
    record = _record(text, [(text.index(":") + 1, text.index(" reported"), PIIType.PERSON)])

    fixed, changed = trim_record(record)

    assert changed == 1
    assert fixed.spans[0].text == "Tristan Batta"


def test_span_text_is_resliced_not_merely_relabelled():
    """The stored `text` must follow the offsets, or eval compares stale strings."""
    text = "at 12/074, Khora  next"
    record = _record(text, [(3, text.index(" next"), PIIType.STREET_ADDRESS)])

    fixed, _ = trim_record(record)
    span = fixed.spans[0]

    assert span.text == text[span.start : span.end]


def test_whitespace_only_span_is_dropped_entirely():
    text = "value:    end"
    record = _record(text, [(6, 10, PIIType.PERSON)])

    fixed, changed = trim_record(record)

    assert changed == 1
    assert fixed.spans == []


def test_clean_record_is_returned_untouched():
    """No-op must be identity: the gold sets rely on this reporting 0 changes."""
    text = "Contact Tristan Batta at 10.0.0.1 today"
    record = _record(
        text,
        [
            (text.index("Tristan"), text.index(" at"), PIIType.PERSON),
            (text.index("10.0"), text.index(" today"), PIIType.IP_ADDRESS),
        ],
    )

    fixed, changed = trim_record(record)

    assert changed == 0
    assert fixed is record
    assert [s.model_dump() for s in fixed.spans] == [s.model_dump() for s in record.spans]


def test_other_spans_in_the_record_survive_a_trim():
    text = "Tristan Batta lives at 12/074, Khora  nearby"
    record = _record(
        text,
        [
            (0, 13, PIIType.PERSON),
            (text.index("12/074"), text.index(" nearby"), PIIType.STREET_ADDRESS),
        ],
    )

    fixed, changed = trim_record(record)

    assert changed == 1
    assert len(fixed.spans) == 2
    assert fixed.spans[0].text == "Tristan Batta"
    assert fixed.spans[1].text == "12/074, Khora"


class TestRepresentability:
    """The drop path: a span boundary buried inside a token cannot be marked."""

    def test_boundary_inside_a_merged_token_fails_the_round_trip(self):
        """The train-v3-0317x defect: Qwen merges '/)' so the URL end is not a
        token boundary. Gold is correct; BIOES simply cannot express it."""
        from forge.token_classifier import assert_bioes_round_trip

        text = "see https://price.com/) includes"
        record = _record(text, [(4, text.index(")"), PIIType.URL)])
        # "https://price.com" | "/)" -- the span ends mid-token, at 22 not 23.
        offsets = [(0, 3), (3, 4), (4, 21), (21, 23), (23, 32)]

        with pytest.raises(ValueError, match="round-trip lost exact gold boundaries"):
            assert_bioes_round_trip(record, offsets)

    def test_same_span_round_trips_when_the_boundary_is_a_token_edge(self):
        """Control: the label is fine, so only the tokenization can be at fault."""
        from forge.token_classifier import assert_bioes_round_trip

        text = "see https://price.com/ includes"
        record = _record(text, [(4, 22, PIIType.URL)])
        offsets = [(0, 3), (3, 4), (4, 21), (21, 22), (22, 31)]

        assert_bioes_round_trip(record, offsets)
