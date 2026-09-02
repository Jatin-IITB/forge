#!/usr/bin/env python3
"""Mechanical audit of the frozen gold set.

This is **not** human verification and must never be described as such. It
checks the properties a machine can check — offset exactness, span disjointness,
type/value plausibility, duplicate and leakage detection — and says nothing
about whether a human agreed the labels are right. `PROTOCOL.md` §5 is still
owed. See `docs/HONEST_ASSESSMENT.md`.

What it *is* good for: the gold set has already shipped one silent defect (a
clock-dependent generator, ADR 0011) that no test caught because every test
regenerated the data the same wrong way. These checks read the committed bytes
and assert properties that must hold regardless of how they were produced.

    python scripts/audit_gold.py                    # dev + test
    python scripts/audit_gold.py --split test --json
"""

from __future__ import annotations

import argparse
import itertools
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

from forge.schema import HIGH_SEVERITY, PIIRecord

# Shape constraints that must hold for a label to be *plausible*. Deliberately
# loose: this catches a SSN labelled onto an email address, not a wrong digit.
# A checksum failure is NOT an error here — the synthetic generator emits random
# digits, so most AADHAAR values legitimately fail Verhoeff (see validators.py).
SHAPE: dict[str, re.Pattern[str]] = {
    "EMAIL": re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$"),
    "AADHAAR": re.compile(r"^\d{4}[\s-]?\d{4}[\s-]?\d{4}$"),
    "PAN": re.compile(r"^[A-Z]{5}\d{4}[A-Z]$", re.IGNORECASE),
    "SSN": re.compile(r"^\d{3}-?\d{2}-?\d{4}$"),
    "CREDIT_CARD": re.compile(r"^[\d\s-]{12,23}$"),
    "IP_ADDRESS": re.compile(r"^[\d.]+$|^[0-9a-f:]+$", re.IGNORECASE),
    "URL": re.compile(r"^(https?://|www\.)", re.IGNORECASE),
    "PHONE": re.compile(r"^[\d\s()+\-.]{7,}$"),
    "AGE": re.compile(r"^\d{1,3}( years?( old)?)?$", re.IGNORECASE),
}


class Audit:
    def __init__(self) -> None:
        self.errors: list[str] = []
        self.warnings: list[str] = []

    def err(self, msg: str) -> None:
        self.errors.append(msg)

    def warn(self, msg: str) -> None:
        self.warnings.append(msg)


def _load(path: Path) -> list[PIIRecord]:
    return [
        PIIRecord.model_validate_json(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def audit_records(records: list[PIIRecord], split: str, a: Audit) -> dict:
    """Per-record structural checks. Errors here mean the data is wrong."""
    seen_ids: set[str] = set()
    seen_text: dict[str, str] = {}
    type_counts: Counter[str] = Counter()
    empty_records = 0

    for r in records:
        if r.id in seen_ids:
            a.err(f"{split}: duplicate record id {r.id}")
        seen_ids.add(r.id)

        if r.text in seen_text:
            a.err(f"{split}: {r.id} has text identical to {seen_text[r.text]}")
        seen_text[r.text] = r.id

        if not r.spans:
            empty_records += 1

        # Offsets must index the committed text exactly. This is the check that
        # would have caught a generator whose text and spans drifted apart.
        for s in r.spans:
            type_counts[s.label.value] += 1
            actual = r.text[s.start : s.end]
            if actual != s.text:
                a.err(
                    f"{split}: {r.id} span [{s.start}:{s.end}] {s.label.value} — "
                    f"text[{s.start}:{s.end}]={actual!r} but span.text={s.text!r}"
                )
            if not s.text.strip():
                a.err(f"{split}: {r.id} {s.label.value} span is blank")
            if s.text != s.text.strip():
                a.warn(f"{split}: {r.id} {s.label.value} span has edge whitespace: {s.text!r}")
            if s.start < 0 or s.end > len(r.text) or s.start >= s.end:
                a.err(f"{split}: {r.id} span [{s.start}:{s.end}] out of bounds (len={len(r.text)})")

            pat = SHAPE.get(s.label.value)
            if pat and not pat.match(s.text.strip()):
                a.warn(f"{split}: {r.id} {s.label.value}={s.text!r} does not match expected shape")

        # Overlapping spans make exact-match scoring ill-defined: one predicted
        # span could satisfy two gold spans.
        ordered = sorted(r.spans, key=lambda s: (s.start, s.end))
        for x, y in itertools.pairwise(ordered):
            if y.start < x.end:
                a.err(
                    f"{split}: {r.id} overlapping spans "
                    f"{x.label.value}[{x.start}:{x.end}] / {y.label.value}[{y.start}:{y.end}]"
                )

    return {
        "n_records": len(records),
        "n_spans": sum(type_counts.values()),
        "empty_records": empty_records,
        "type_counts": dict(type_counts.most_common()),
    }


def audit_coverage(stats: dict, split: str, a: Audit) -> None:
    """Is each type represented densely enough for its claim to mean anything?"""
    for t in sorted(x.value for x in HIGH_SEVERITY):
        n = stats["type_counts"].get(t, 0)
        if n == 0:
            a.err(f"{split}: high-severity type {t} has NO gold instances — its floor is unmeasurable")
        elif n < 20:
            # With 0 misses, the 95% lower bound on recall is alpha**(1/n).
            bound = 0.05 ** (1.0 / n)
            a.warn(
                f"{split}: {t} has only {n} instances — even a perfect score "
                f"supports just >={bound:.3f} recall at 95% confidence"
            )


def audit_leakage(train_path: Path, gold: dict[str, list[PIIRecord]], a: Audit) -> dict:
    """Any gold carrier text appearing in training data invalidates the split."""
    if not train_path.exists():
        return {"checked": False}
    train_texts = {r.text for r in _load(train_path)}
    hits = defaultdict(list)
    for split, records in gold.items():
        for r in records:
            if r.text in train_texts:
                hits[split].append(r.id)
                a.err(f"LEAKAGE: {split} record {r.id} text appears verbatim in {train_path.name}")
    return {"checked": True, "train_records": len(train_texts), "leaked": {k: v for k, v in hits.items()}}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gold-dir", type=Path, default=Path("data/gold"))
    ap.add_argument("--split", choices=["dev", "test", "both"], default="both")
    ap.add_argument("--train", type=Path, default=Path("data/train.jsonl"))
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    splits = ["dev", "test"] if args.split == "both" else [args.split]
    a = Audit()
    gold: dict[str, list[PIIRecord]] = {}
    stats: dict[str, dict] = {}

    for split in splits:
        path = args.gold_dir / f"{split}.jsonl"
        if not path.exists():
            a.err(f"missing {path}")
            continue
        gold[split] = _load(path)
        stats[split] = audit_records(gold[split], split, a)
        audit_coverage(stats[split], split, a)

    leak = audit_leakage(args.train, gold, a)

    if args.json:
        print(json.dumps({
            "splits": stats, "leakage": leak,
            "errors": a.errors, "warnings": a.warnings,
        }, indent=2))
    else:
        for split, s in stats.items():
            print(f"\n{split}: {s['n_records']} records, {s['n_spans']} spans, "
                  f"{s['empty_records']} with no spans")
            for t, n in s["type_counts"].items():
                mark = " *" if t in {x.value for x in HIGH_SEVERITY} else ""
                print(f"    {t:<18}{n:>5}{mark}")
        print("\n  * = high-severity (gated at 0.99 recall)")
        if leak.get("checked"):
            print(f"\nleakage check vs {args.train.name}: "
                  f"{sum(len(v) for v in leak['leaked'].values())} hits "
                  f"across {leak['train_records']} training records")

        print(f"\n{'=' * 58}")
        print(f"  AUDIT: {len(a.errors)} errors, {len(a.warnings)} warnings")
        print(f"{'=' * 58}")
        for e in a.errors:
            print(f"  [ERROR] {e}")
        for w in a.warnings[:25]:
            print(f"  [warn]  {w}")
        if len(a.warnings) > 25:
            print(f"  ... and {len(a.warnings) - 25} more warnings")
        print("\n  Mechanical checks only. PROTOCOL.md section 5 (human")
        print("  verification) is a separate obligation and remains unmet.")

    return 1 if a.errors else 0


if __name__ == "__main__":
    sys.exit(main())
