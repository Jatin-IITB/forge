"""Inference adapter — prompt construction, response parsing, offset reconstruction.

The model outputs PII spans as ``{"label": ..., "text": ...}`` pairs (no offsets).
Offsets are reconstructed by finding each text substring in the original input,
scanning left-to-right so duplicate substrings are resolved in document order.

This keeps the model's job simple (detect + classify, not count characters) while
producing the exact ``PIIRecord`` format the eval harness expects.
"""

from __future__ import annotations

import json
import re

from forge.schema import PIIRecord, PIISpan, PIIType

VALID_LABELS: set[str] = {t.value for t in PIIType}

SYSTEM_PROMPT = """\
You are a PII (Personally Identifiable Information) detection system.

Given the input text, identify ALL PII entities and return them as a JSON object.

For each PII entity, provide:
- "label": the PII type (see valid types below)
- "text": the EXACT substring from the input text (copy-paste, do not rephrase)

Valid PII types:
PERSON, EMAIL, PHONE, STREET_ADDRESS, USERNAME, URL, IP_ADDRESS,
LOCATION, DATE_OF_BIRTH, AGE, CREDIT_CARD, BANK_ACCOUNT, SSN,
AADHAAR, PAN, PASSPORT, DRIVER_LICENSE, PASSWORD, API_KEY

Return format (strict JSON, no markdown):
{"spans": [{"label": "PERSON", "text": "Jane Doe"}, ...]}

If NO PII is found, return:
{"spans": []}

Rules:
- Return ONLY the JSON object, nothing else.
- The "text" field must be an EXACT substring of the input — do not modify spacing, \
capitalization, or punctuation.
- Report every occurrence; do not skip duplicates.
- Generic place names used as context (e.g. "weather in Mumbai") are NOT PII.
"""

COMPACT_SYSTEM_PROMPT = """\
You are a PII (Personally Identifiable Information) detection system.

Given the input text, identify ALL PII entities and return them as a JSON object.

For each PII entity, provide:
- "l": the PII type (see valid types below)
- "t": the EXACT substring from the input text (copy-paste, do not rephrase)

Valid PII types:
PERSON, EMAIL, PHONE, STREET_ADDRESS, USERNAME, URL, IP_ADDRESS,
LOCATION, DATE_OF_BIRTH, AGE, CREDIT_CARD, BANK_ACCOUNT, SSN,
AADHAAR, PAN, PASSPORT, DRIVER_LICENSE, PASSWORD, API_KEY

Return format (strict JSON, no markdown, no spaces):
{"s":[{"l":"PERSON","t":"Jane Doe"}]}

If NO PII is found, return:
{"s":[]}

Rules:
- Return ONLY the JSON object, nothing else.
- The "t" field must be an EXACT substring of the input — do not modify spacing, \
capitalization, or punctuation.
- Report every occurrence; do not skip duplicates.
- Generic place names used as context (e.g. "weather in Mumbai") are NOT PII.
"""

TEACHER_SYSTEM_PROMPT = """\
You are a PII (Personally Identifiable Information) detection system used \
to generate training data. You must be thorough and explain your reasoning.

Given the input text, identify ALL PII entities and return them as a JSON object.

For each PII entity, provide:
- "label": the PII type (see valid types below)
- "text": the EXACT substring from the input text (copy-paste, do not rephrase)
- "rationale": a brief explanation of WHY this is PII of this type (one sentence)

Valid PII types:
PERSON, EMAIL, PHONE, STREET_ADDRESS, USERNAME, URL, IP_ADDRESS,
LOCATION, DATE_OF_BIRTH, AGE, CREDIT_CARD, BANK_ACCOUNT, SSN,
AADHAAR, PAN, PASSPORT, DRIVER_LICENSE, PASSWORD, API_KEY

Return format (strict JSON, no markdown):
{"spans": [{"label": "PERSON", "text": "Jane Doe", "rationale": "full name in email greeting"}, ...]}

If NO PII is found, return:
{"spans": []}

Rules:
- Return ONLY the JSON object, nothing else.
- The "text" field must be an EXACT substring of the input — do not modify spacing, \
capitalization, or punctuation.
- Report every occurrence; do not skip duplicates.
- Generic place names used as context (e.g. "weather in Mumbai") are NOT PII.
- Be thorough — missing a PII entity is worse than a false positive.
"""

USER_TEMPLATE = "Detect all PII in this text:\n\n{text}"


def build_messages(
    text: str,
    teacher_mode: bool = False,
    *,
    compact: bool = False,
) -> list[dict[str, str]]:
    """Build the chat prompt, optionally requesting the compact serving format."""
    if teacher_mode and compact:
        raise ValueError("compact output is only defined for student inference")
    prompt = (
        TEACHER_SYSTEM_PROMPT
        if teacher_mode
        else COMPACT_SYSTEM_PROMPT
        if compact
        else SYSTEM_PROMPT
    )
    return [
        {"role": "system", "content": prompt},
        {"role": "user", "content": USER_TEMPLATE.format(text=text)},
    ]


_SPANS_ALIASES = ("spans", "s", "pii", "entities", "results", "pii_entities", "data")


def _extract_json(raw: str) -> dict:
    """Extract JSON from model output, handling markdown code fences and surrounding text."""
    # Strip think tags — handle truncated (no closing tag) case too
    raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL)
    raw = re.sub(r"<think>[^{]*", "", raw)
    raw = raw.strip()
    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw, re.DOTALL)
    if fence:
        return json.loads(fence.group(1))
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        bare = re.search(r"\{.*\}", raw, re.DOTALL)
        if bare:
            return json.loads(bare.group(0))
        raise


def reconstruct_offsets(text: str, raw_spans: list[dict]) -> list[PIISpan]:
    """Match model-output text fragments against the input and compute offsets.

    Scans left-to-right so duplicate substrings are resolved in document order.
    Skips spans whose text is not found or whose label is invalid.
    """
    spans: list[PIISpan] = []
    used_positions: set[int] = set()

    for raw in raw_spans:
        label_str = raw.get("label", raw.get("l", ""))
        span_text = raw.get("text", raw.get("t", ""))
        if not span_text or label_str not in VALID_LABELS:
            continue

        search_start = 0
        while search_start <= len(text) - len(span_text):
            idx = text.find(span_text, search_start)
            if idx == -1:
                break
            if idx not in used_positions:
                spans.append(PIISpan(
                    start=idx,
                    end=idx + len(span_text),
                    label=PIIType(label_str),
                    text=span_text,
                ))
                used_positions.add(idx)
                break
            search_start = idx + 1

    spans.sort(key=lambda s: s.start)

    # Drop overlapping spans (keep the earlier one)
    deduped: list[PIISpan] = []
    prev_end = -1
    for s in spans:
        if s.start >= prev_end:
            deduped.append(s)
            prev_end = s.end

    return deduped


def parse_response(record_id: str, text: str, raw_response: str, split: str = "test") -> tuple[PIIRecord | None, bool]:
    """Parse a model response into a PIIRecord.

    Returns:
        (record, schema_valid): the parsed record and whether the response was
        valid JSON with the expected structure. If schema_valid is False, record
        has empty spans (graceful degradation, not crash).
    """
    try:
        data = _extract_json(raw_response)
        if not isinstance(data, dict):
            raise TypeError("expected JSON object, got " + type(data).__name__)
        raw_spans = None
        for alias in _SPANS_ALIASES:
            if alias in data:
                raw_spans = data[alias]
                break
        if raw_spans is None:
            raise KeyError("missing 'spans' key (tried: " + ", ".join(_SPANS_ALIASES) + ")")
        if not isinstance(raw_spans, list):
            raise TypeError("'spans' is not a list")
        spans = reconstruct_offsets(text, raw_spans)
        return PIIRecord(id=record_id, text=text, spans=spans, split=split), True
    except (json.JSONDecodeError, ValueError, KeyError, TypeError):
        return PIIRecord(id=record_id, text=text, spans=[], split=split), False
