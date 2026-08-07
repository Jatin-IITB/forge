#!/usr/bin/env python3
"""Build a tiny ILLUSTRATIVE gold sample (data/gold/sample.jsonl).

This is NOT the frozen gold set. It exists so the record format is concrete in the
repo and so tests have fixtures. The real dev/test sets are built by the synthetic
data engine (Faker, fixed seed) and then HUMAN-VERIFIED and frozen — see
data/gold/PROTOCOL.md.

Records are assembled from segments so character offsets are computed, never
hand-counted: a segment is either plain text or a (value, label) PII tuple.
"""

from __future__ import annotations

from pathlib import Path

from forge.schema import PIIRecord, PIISpan, PIIType

# A segment is plain text, or a PII value tagged with its type.
Segment = str | tuple[str, PIIType]

EXAMPLES: list[tuple[str, str, list[Segment]]] = [
    (
        "sample-0001",
        "en",
        [
            "Hi, I'm ",
            ("Rajesh Kumar", PIIType.PERSON),
            " and you can reach me at ",
            ("rajesh.k@example.com", PIIType.EMAIL),
            " or ",
            ("+91 98765 43210", PIIType.PHONE),
            ".",
        ],
    ),
    (
        "sample-0002",
        "en",
        [
            "Please charge card ",
            ("4111 1111 1111 1111", PIIType.CREDIT_CARD),
            " exp 04/27 for the order shipping to ",
            ("221B Baker Street, London", PIIType.STREET_ADDRESS),
            ".",
        ],
    ),
    (
        "sample-0003",
        "en",
        [
            "KYC update: Aadhaar ",
            ("4123 4567 8901", PIIType.AADHAAR),
            ", PAN ",
            ("ABCDE1234F", PIIType.PAN),
            ", account ",
            ("000123456789", PIIType.BANK_ACCOUNT),
            ".",
        ],
    ),
    (
        "sample-0004",
        "en",
        [
            "Login failed for user ",
            ("jgupta", PIIType.USERNAME),
            " from ",
            ("203.0.113.42", PIIType.IP_ADDRESS),
            " with token ",
            ("sk-live-9f8a7b6c5d4e", PIIType.API_KEY),
            ".",
        ],
    ),
    (
        "sample-0005",
        "en",
        [
            "The weather in Mumbai is pleasant today and the trains are on time.",
        ],  # negative example: no PII (LOCATION 'Mumbai' is generic context, left unlabelled per PROTOCOL)
    ),
]


def build(example_id: str, lang: str, segments: list[Segment]) -> PIIRecord:
    text_parts: list[str] = []
    spans: list[PIISpan] = []
    cursor = 0
    for seg in segments:
        if isinstance(seg, str):
            text_parts.append(seg)
            cursor += len(seg)
        else:
            value, label = seg
            start = cursor
            end = start + len(value)
            spans.append(PIISpan(start=start, end=end, label=label, text=value))
            text_parts.append(value)
            cursor = end
    return PIIRecord(
        id=example_id,
        text="".join(text_parts),
        spans=spans,
        lang=lang,
        source="synthetic:handwritten",
        split="dev",
    )


def main() -> None:
    out = Path(__file__).resolve().parents[1] / "data" / "gold" / "sample.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    records = [build(*ex) for ex in EXAMPLES]
    with out.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(r.model_dump_json() + "\n")
    print(f"wrote {len(records)} records -> {out}")
    # Show the redaction for the first record as a sanity demo.
    print("demo redaction:", records[0].redacted)


if __name__ == "__main__":
    main()
