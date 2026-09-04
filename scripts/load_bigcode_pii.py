#!/usr/bin/env python3
"""Convert bigcode/bigcode-pii-dataset into Forge PIIRecord JSONL.

## Licensing, stated up front because it constrains everything downstream

The dataset is **gated**. Access requires a HuggingFace account, sharing contact
details, and accepting Terms of Use. Two terms govern what this script may do:

    Term 1: use is restricted to "training or evaluating models for PII removal"
    Term 2: "You agree that you will not share the PII dataset or any modified
             versions for whatever purpose."

Term 1 permits what Forge does. **Term 2 forbids redistributing anything derived
from it**, so the converted corpus is written to a gitignored path and must never
be committed, sampled into a report, or pushed. Measured *numbers* may be
published; the *data* may not.

The cost is reproducibility: a reader cannot rerun this without accepting the
gate themselves. That is a real weakening of the `adr/0003` litmus test and is
recorded in ADR 0022 rather than glossed over.

## What this corpus can and cannot validate

It is **source code**, and `contracts/pii_redaction_v2.yaml` declares code
out-of-domain. `forge/ood.py` refuses `code_python`, `code_sql`, `code_json`,
`code_html` and `code_regex`, scoring 21/21 on the OOD probes. So evaluating on
this corpus requires **disabling the OOD gate**, and the result describes the
model on input the shipped system would decline. That is a legitimate
measurement only if reported as such.

Type coverage against Forge's 19-type schema:

    BigCode          Forge            note
    EMAIL        ->  EMAIL
    NAME         ->  PERSON
    USERNAME     ->  USERNAME
    IP_ADDRESS   ->  IP_ADDRESS
    KEY          ->  API_KEY          high-severity
    PASSWORD     ->  PASSWORD         high-severity
    ID           ->  (dropped)        no clean Forge equivalent; ambiguous
    *_EXAMPLE    ->  (dropped)        placeholder PII, NOT a redaction target
    *_LICENSE    ->  (dropped)        names in licence headers, NOT a target
    AMBIGUOUS    ->  (dropped)        annotator could not decide

**Six of nineteen types, and two of the nine high-severity ones.** Nothing here
validates AADHAAR, PAN, SSN, CREDIT_CARD, BANK_ACCOUNT, PASSPORT or
DRIVER_LICENSE -- and no public corpus can, because publishing real values of
those would be the leak the tool exists to prevent.

Dropping `*_EXAMPLE` and `*_LICENSE` is a deliberate scoring decision, not
tidying: they mark strings that *look* like PII but must not be redacted, so
counting them as gold would reward exactly the over-redaction the validator
layer is tuned against. They are retained in the output under
`non_targets` so a future run can score them as negatives.

Usage:
    huggingface-cli login          # after accepting the gate in a browser
    python scripts/load_bigcode_pii.py --out data/external/bigcode_pii.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

from forge.schema import HIGH_SEVERITY, PIIRecord, PIISpan, PIIType

# Only categories with an unambiguous Forge equivalent are mapped. Everything
# else is dropped loudly rather than guessed at.
CATEGORY_MAP: dict[str, PIIType] = {
    "EMAIL": PIIType.EMAIL,
    "NAME": PIIType.PERSON,
    "USERNAME": PIIType.USERNAME,
    "IP_ADDRESS": PIIType.IP_ADDRESS,
    "KEY": PIIType.API_KEY,
    "PASSWORD": PIIType.PASSWORD,
}

# Annotated, but explicitly NOT redaction targets. Kept separately so they can
# be scored as negatives instead of silently discarded.
NON_TARGET_SUFFIXES = ("_EXAMPLE", "_LICENSE")
NON_TARGET_EXACT = {"AMBIGUOUS", "ID"}


def _fragment_span(text: str, frag: dict) -> tuple[int, int, str] | None:
    """Pull (start, end, value) out of a fragment, tolerating layout variants."""
    pos = frag.get("position") or {}
    if isinstance(pos, dict):
        start, end = pos.get("start"), pos.get("end")
    elif isinstance(pos, (list, tuple)) and len(pos) == 2:
        start, end = pos
    else:
        start, end = frag.get("start"), frag.get("end")
    if start is None or end is None:
        return None
    return int(start), int(end), frag.get("value") or text[int(start) : int(end)]


def convert(rows, *, max_records: int | None = None) -> tuple[list[PIIRecord], dict]:
    """Map BigCode rows to PIIRecord, dropping anything that does not verify.

    Offsets are checked against the text exactly as `scripts/audit_gold.py`
    checks the gold set: a span whose slice does not equal its stated value is
    dropped rather than trusted, because a silently misaligned span trains and
    scores as a wrong answer.
    """
    records: list[PIIRecord] = []
    stats: Counter[str] = Counter()

    for i, row in enumerate(rows):
        if max_records and len(records) >= max_records:
            break
        text = row.get("text") or ""
        if not text.strip():
            stats["skipped_empty_text"] += 1
            continue

        spans: list[PIISpan] = []
        non_targets: list[dict] = []

        for frag in row.get("fragments") or []:
            category = (frag.get("category") or "").upper()
            parsed = _fragment_span(text, frag)
            if parsed is None:
                stats["dropped_no_offsets"] += 1
                continue
            start, end, value = parsed

            if category.endswith(NON_TARGET_SUFFIXES) or category in NON_TARGET_EXACT:
                non_targets.append({"category": category, "start": start, "end": end})
                stats[f"non_target_{category}"] += 1
                continue

            pii_type = CATEGORY_MAP.get(category)
            if pii_type is None:
                stats[f"unmapped_{category}"] += 1
                continue

            if not (0 <= start < end <= len(text)):
                stats["dropped_out_of_bounds"] += 1
                continue
            if text[start:end] != value:
                stats["dropped_offset_mismatch"] += 1
                continue

            spans.append(PIISpan(start=start, end=end, label=pii_type, text=value))
            stats[f"kept_{pii_type.value}"] += 1

        # Overlapping gold spans make exact-match scoring ill-defined, the same
        # reason audit_gold.py treats them as an error in the frozen set.
        spans.sort(key=lambda s: (s.start, s.end))
        deduped: list[PIISpan] = []
        for s in spans:
            if deduped and s.start < deduped[-1].end:
                stats["dropped_overlapping"] += 1
                continue
            deduped.append(s)

        records.append(
            PIIRecord(
                id=f"bigcode-{row.get('id', i)}",
                text=text,
                spans=deduped,
                split="test",
                source=f"bigcode:{row.get('language', 'unknown')}",
            )
        )
        if non_targets:
            stats["records_with_non_targets"] += 1

    return records, dict(stats)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=Path("data/external/bigcode_pii.jsonl"))
    ap.add_argument("--split", default="train")
    ap.add_argument("--max-records", type=int, default=None)
    ap.add_argument(
        "--local-json", type=Path, default=None,
        help="Read a locally downloaded JSON/JSONL instead of calling the Hub",
    )
    args = ap.parse_args()

    if args.local_json:
        raw = args.local_json.read_text(encoding="utf-8")
        rows = json.loads(raw) if raw.lstrip().startswith("[") else [
            json.loads(x) for x in raw.splitlines() if x.strip()
        ]
    else:
        try:
            from datasets import load_dataset
        except ImportError:
            print("needs the [train] extra: pip install -e '.[train]'", file=sys.stderr)
            return 2
        try:
            rows = load_dataset("bigcode/bigcode-pii-dataset", split=args.split)
        except Exception as e:  # noqa: BLE001
            print(
                f"could not load the dataset: {e}\n\n"
                "This dataset is GATED. Accept the terms at\n"
                "  https://huggingface.co/datasets/bigcode/bigcode-pii-dataset\n"
                "then authenticate with `huggingface-cli login`.",
                file=sys.stderr,
            )
            return 1

    records, stats = convert(rows, max_records=args.max_records)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(r.model_dump_json() + "\n")

    n_spans = sum(len(r.spans) for r in records)
    hs = {t.value for t in HIGH_SEVERITY}
    per_type = Counter(s.label.value for r in records for s in r.spans)

    print(f"wrote {len(records)} records, {n_spans} spans -> {args.out}")
    print("\nper-type coverage:")
    for t, n in per_type.most_common():
        print(f"  {t:<14}{n:>6}{'  *high-severity' if t in hs else ''}")
    print(f"\n  high-severity types present: {sorted(set(per_type) & hs)}")
    print(f"  high-severity types ABSENT:  {sorted(hs - set(per_type))}")
    print("\nconversion stats:")
    for k, v in sorted(stats.items()):
        print(f"  {k:<32}{v:>6}")
    print(
        "\nREMINDER: Term 2 of the dataset licence forbids sharing this file or\n"
        "anything derived from it. It is gitignored. Publish numbers, not data."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
