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


# Types whose *value* does not identify them: a bare number or date only means
# what the surrounding sentence says it means. AADHAAR, PAN, SSN, CREDIT_CARD,
# EMAIL and URL are self-identifying by shape and need no anchor.
_ANCHOR: dict[str, re.Pattern[str]] = {
    "AGE": re.compile(r"\b(age[ds]?|years? old|yrs?|y/?o)\b", re.IGNORECASE),
    "DATE_OF_BIRTH": re.compile(r"\b(born|d\.?o\.?b\.?|date of birth|birthday|birth date)\b", re.IGNORECASE),
    "BANK_ACCOUNT": re.compile(r"\b(account|a/c|acct|iban|bank|routing|deposit|statement|transaction)\b", re.IGNORECASE),
    "PASSWORD": re.compile(r"\b(password|passwd|pwd|passphrase|security code|otp|credential)\b", re.IGNORECASE),
}

# Positive evidence that the span means something *else*. Distinct from a missing
# anchor: absence of "born" is weak, presence of "invoice ... on <date>" is strong.
_CONTRADICTS: dict[str, re.Pattern[str]] = {
    "AGE": re.compile(r"\b(within|days?|hours?|weeks?|months?|amount|status|qty|quantity|\$|USD|INR)\b", re.IGNORECASE),
    "DATE_OF_BIRTH": re.compile(
        r"\b(invoice|order|confirmed|filed|effective|issued|expires?|due|shipped|posted"
        r"|created|scheduled|appointment|meeting|delivery|transaction|payment|follow.?up)\b", re.IGNORECASE
    ),
}

_ANCHOR_WINDOW = 45


def audit_semantics(records: list[PIIRecord], split: str, a: Audit) -> dict:
    """Are context-dependent labels supported by the sentence around them?

    Construction-based generation makes labels exact by *offset* — the filler
    knows where it put the value — while nothing checks they are right by
    *meaning*. A {DATE_OF_BIRTH} slot dropped into "Order #1234 confirmed on
    <date>" is offset-perfect and semantically false, and training on it teaches
    that any date is a birth date. That is a precision failure manufactured on
    purpose, in a corpus whose point is to fix a precision/recall frontier.

    Two signals, deliberately not conflated:

    * **contradicted** — positive evidence of another meaning. Strong; an error.
    * **unanchored** — merely no supporting keyword. Weak on its own: "Rachita
      Thakkar (53) requested an upgrade" is a perfectly good AGE with no anchor
      word, and "unfamiliar transaction on my statement from <digits>" is a good
      BANK_ACCOUNT. Reported as a warning, never an error.

    Found by comparing the frozen gold (0 contradicted) against a teacher-written
    carrier corpus (17% contradicted on AGE and DATE_OF_BIRTH). The difference is
    structural: gold templates are hand-written with the anchor built in
    ("age {AGE}", "born {DOB}"); generated prose has no such guarantee.
    """
    out: dict[str, dict] = {}
    for label, anchor in _ANCHOR.items():
        total = anchored = contradicted = 0
        example = None
        for r in records:
            for s in r.spans:
                if s.label.value != label:
                    continue
                total += 1
                lo = max(0, s.start - _ANCHOR_WINDOW)
                hi = min(len(r.text), s.end + _ANCHOR_WINDOW)
                window = r.text[lo : s.start] + " " + r.text[s.end : hi]
                if anchor.search(window):
                    anchored += 1
                    continue
                bad = _CONTRADICTS.get(label)
                if bad and bad.search(window):
                    contradicted += 1
                    if example is None:
                        example = r.text[lo:hi]
        if not total:
            continue
        out[label] = {"total": total, "anchored": anchored, "contradicted": contradicted}
        if contradicted:
            a.err(
                f"{split}: {label} — {contradicted}/{total} spans sit in a context that "
                f"contradicts the label, e.g. {example!r}"
            )
        unanchored = total - anchored - contradicted
        if total >= 10 and unanchored / total > 0.5:
            a.warn(
                f"{split}: {label} — {unanchored}/{total} spans have no supporting keyword. "
                "Weak signal on its own, but worth reading if the type is trained on."
            )
    return out


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
    ap.add_argument("--split", choices=["dev", "val", "test", "all"], default="all")
    ap.add_argument("--train", type=Path, default=Path("data/train.jsonl"))
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    # "val" is skipped silently when absent so this still runs on a clone that
    # predates WP-0d; every other named split missing is an error.
    splits = ["dev", "val", "test"] if args.split == "all" else [args.split]
    if args.split == "all" and not (args.gold_dir / "val.jsonl").exists():
        splits.remove("val")
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
        stats[split]["semantics"] = audit_semantics(gold[split], split, a)

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
