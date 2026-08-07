"""Inference adapter tests — prompt construction, JSON extraction, offset reconstruction."""

import json

import pytest

from forge.inference import (
    _extract_json,
    build_messages,
    parse_response,
    reconstruct_offsets,
)
from forge.schema import PIISpan, PIIType

# --- build_messages ---

def test_build_messages_structure():
    msgs = build_messages("Hello world")
    assert len(msgs) == 2
    assert msgs[0]["role"] == "system"
    assert msgs[1]["role"] == "user"
    assert "Hello world" in msgs[1]["content"]


def test_build_messages_preserves_text_exactly():
    text = "SSN: 123-45-6789\nDOB: 01/01/1990"
    msgs = build_messages(text)
    assert text in msgs[1]["content"]


def test_build_messages_teacher_mode():
    msgs = build_messages("Hello world", teacher_mode=True)
    assert "rationale" in msgs[0]["content"]
    assert "training data" in msgs[0]["content"]


def test_build_messages_default_is_not_teacher():
    msgs = build_messages("Hello world")
    assert "rationale" not in msgs[0]["content"]


# --- _extract_json ---

def test_extract_json_plain():
    raw = '{"spans": [{"label": "PERSON", "text": "Alice"}]}'
    result = _extract_json(raw)
    assert result["spans"][0]["label"] == "PERSON"


def test_extract_json_with_code_fence():
    raw = '```json\n{"spans": [{"label": "EMAIL", "text": "a@b.com"}]}\n```'
    result = _extract_json(raw)
    assert result["spans"][0]["text"] == "a@b.com"


def test_extract_json_with_bare_fence():
    raw = '```\n{"spans": []}\n```'
    result = _extract_json(raw)
    assert result["spans"] == []


def test_extract_json_with_surrounding_text():
    raw = 'Here is the result:\n```json\n{"spans": []}\n```\nDone.'
    result = _extract_json(raw)
    assert result["spans"] == []


def test_extract_json_invalid_raises():
    with pytest.raises(json.JSONDecodeError):
        _extract_json("not json at all")


def test_extract_json_whitespace():
    raw = '  \n  {"spans": []}  \n  '
    result = _extract_json(raw)
    assert result["spans"] == []


# --- reconstruct_offsets ---

def test_reconstruct_simple():
    text = "My name is Alice and email alice@x.com"
    raw_spans = [
        {"label": "PERSON", "text": "Alice"},
        {"label": "EMAIL", "text": "alice@x.com"},
    ]
    spans = reconstruct_offsets(text, raw_spans)
    assert len(spans) == 2
    assert spans[0] == PIISpan(start=11, end=16, label=PIIType.PERSON, text="Alice")
    assert spans[1] == PIISpan(start=27, end=38, label=PIIType.EMAIL, text="alice@x.com")


def test_reconstruct_invalid_label_skipped():
    text = "My name is Bob"
    raw_spans = [{"label": "INVALID_TYPE", "text": "Bob"}]
    spans = reconstruct_offsets(text, raw_spans)
    assert len(spans) == 0


def test_reconstruct_empty_text_skipped():
    text = "Hello world"
    raw_spans = [{"label": "PERSON", "text": ""}]
    spans = reconstruct_offsets(text, raw_spans)
    assert len(spans) == 0


def test_reconstruct_text_not_found_skipped():
    text = "Hello world"
    raw_spans = [{"label": "PERSON", "text": "NotInText"}]
    spans = reconstruct_offsets(text, raw_spans)
    assert len(spans) == 0


def test_reconstruct_duplicate_text_resolved_in_order():
    text = "Alice met Alice again"
    raw_spans = [
        {"label": "PERSON", "text": "Alice"},
        {"label": "PERSON", "text": "Alice"},
    ]
    spans = reconstruct_offsets(text, raw_spans)
    assert len(spans) == 2
    assert spans[0].start == 0
    assert spans[1].start == 10


def test_reconstruct_overlapping_spans_deduped():
    text = "Name: Alice Smith here"
    raw_spans = [
        {"label": "PERSON", "text": "Alice Smith"},
        {"label": "PERSON", "text": "Alice"},
    ]
    spans = reconstruct_offsets(text, raw_spans)
    assert len(spans) == 1
    assert spans[0].text == "Alice Smith"


def test_reconstruct_sorted_by_offset():
    text = "email: a@b.com name: Bob"
    raw_spans = [
        {"label": "PERSON", "text": "Bob"},
        {"label": "EMAIL", "text": "a@b.com"},
    ]
    spans = reconstruct_offsets(text, raw_spans)
    assert spans[0].label == PIIType.EMAIL
    assert spans[1].label == PIIType.PERSON


def test_reconstruct_missing_fields_skipped():
    text = "Hello Alice"
    raw_spans = [
        {"label": "PERSON"},
        {"text": "Alice"},
        {},
    ]
    spans = reconstruct_offsets(text, raw_spans)
    assert len(spans) == 0


def test_reconstruct_all_19_types():
    labels = [t.value for t in PIIType]
    parts = [f"[{lbl}]" for lbl in labels]
    text = " ".join(parts)
    raw_spans = [{"label": lbl, "text": f"[{lbl}]"} for lbl in labels]
    spans = reconstruct_offsets(text, raw_spans)
    assert len(spans) == 19
    found_labels = {s.label for s in spans}
    assert found_labels == set(PIIType)


# --- parse_response ---

def test_parse_response_valid():
    text = "Contact: alice@x.com"
    raw = json.dumps({"spans": [{"label": "EMAIL", "text": "alice@x.com"}]})
    record, valid = parse_response("r1", text, raw)
    assert valid is True
    assert len(record.spans) == 1
    assert record.spans[0].label == PIIType.EMAIL
    assert record.id == "r1"


def test_parse_response_empty_spans():
    text = "No PII here"
    raw = json.dumps({"spans": []})
    record, valid = parse_response("r1", text, raw)
    assert valid is True
    assert len(record.spans) == 0


def test_parse_response_invalid_json():
    record, valid = parse_response("r1", "some text", "not json")
    assert valid is False
    assert len(record.spans) == 0


def test_parse_response_missing_spans_key():
    record, valid = parse_response("r1", "some text", '{"entities": []}')
    assert valid is False
    assert len(record.spans) == 0


def test_parse_response_spans_not_list():
    record, valid = parse_response("r1", "some text", '{"spans": "wrong"}')
    assert valid is False
    assert len(record.spans) == 0


def test_parse_response_code_fenced():
    text = "Hello Bob"
    raw = '```json\n{"spans": [{"label": "PERSON", "text": "Bob"}]}\n```'
    record, valid = parse_response("r1", text, raw)
    assert valid is True
    assert len(record.spans) == 1
    assert record.spans[0].text == "Bob"


def test_parse_response_split_preserved():
    text = "Hello"
    raw = '{"spans": []}'
    record, _ = parse_response("r1", text, raw, split="dev")
    assert record.split == "dev"


def test_parse_response_graceful_on_empty_string():
    record, valid = parse_response("r1", "text", "")
    assert valid is False
    assert record.spans == []


def test_parse_response_mixed_valid_invalid_spans():
    text = "Alice at alice@x.com"
    raw = json.dumps({"spans": [
        {"label": "PERSON", "text": "Alice"},
        {"label": "INVALID", "text": "alice@x.com"},
        {"label": "EMAIL", "text": "alice@x.com"},
    ]})
    record, valid = parse_response("r1", text, raw)
    assert valid is True
    assert len(record.spans) == 2
    labels = {s.label for s in record.spans}
    assert PIIType.PERSON in labels
    assert PIIType.EMAIL in labels


# --- Audit-driven tests (Findings 1, 2, 5, 7) ---

def test_parse_response_non_dict_json_array():
    raw = '[{"label": "PERSON", "text": "Alice"}]'
    record, valid = parse_response("r1", "Hello Alice", raw)
    assert valid is False
    assert record.spans == []


def test_parse_response_non_dict_json_null():
    record, valid = parse_response("r1", "Hello", "null")
    assert valid is False
    assert record.spans == []


def test_parse_response_non_dict_json_number():
    record, valid = parse_response("r1", "Hello", "42")
    assert valid is False
    assert record.spans == []


def test_extract_json_unfenced_with_preamble():
    raw = 'Sure! Here are the PII entities:\n{"spans": [{"label": "PERSON", "text": "Bob"}]}'
    result = _extract_json(raw)
    assert result["spans"][0]["text"] == "Bob"


def test_extract_json_unfenced_with_trailing_text():
    raw = '{"spans": []}\nI found 0 entities.'
    result = _extract_json(raw)
    assert result["spans"] == []


def test_extract_json_unfenced_with_surrounding_text():
    raw = 'Result:\n{"spans": []}\nNo PII found.'
    result = _extract_json(raw)
    assert result["spans"] == []


def test_reconstruct_unicode_emoji():
    text = "Contact \U0001f600 Alice at a@b.com"
    raw_spans = [
        {"label": "PERSON", "text": "Alice"},
        {"label": "EMAIL", "text": "a@b.com"},
    ]
    spans = reconstruct_offsets(text, raw_spans)
    assert len(spans) == 2
    assert text[spans[0].start:spans[0].end] == "Alice"
    assert text[spans[1].start:spans[1].end] == "a@b.com"


def test_reconstruct_cjk():
    text = "张三的邮箱是 zhang@x.com"
    raw_spans = [
        {"label": "PERSON", "text": "张三"},
        {"label": "EMAIL", "text": "zhang@x.com"},
    ]
    spans = reconstruct_offsets(text, raw_spans)
    assert len(spans) == 2
    assert text[spans[0].start:spans[0].end] == "张三"


def test_parse_response_extra_keys_in_spans():
    text = "Hello Alice"
    raw = json.dumps({"spans": [
        {"label": "PERSON", "text": "Alice", "confidence": 0.99, "extra": True},
    ]})
    record, valid = parse_response("r1", text, raw)
    assert valid is True
    assert len(record.spans) == 1


def test_parse_response_whitespace_only():
    record, valid = parse_response("r1", "text", "   \n  ")
    assert valid is False
    assert record.spans == []


def test_reconstruct_span_text_is_substring_of_another():
    text = "SSN 123-45-6789 and partial 123-45"
    raw_spans = [
        {"label": "SSN", "text": "123-45-6789"},
        {"label": "SSN", "text": "123-45"},
    ]
    spans = reconstruct_offsets(text, raw_spans)
    assert len(spans) == 2
    assert spans[0].text == "123-45-6789"
    assert spans[1].text == "123-45"
    assert text[spans[1].start:spans[1].end] == "123-45"
