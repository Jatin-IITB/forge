#!/usr/bin/env python3
"""Drop records whose labels are contradicted by their own carrier text.

`data/train_v3.jsonl` is construction-filled, so its spans are exact by OFFSET
and unchecked by MEANING. ADR 0015 and commit 6a592c6 measured the consequence:

    AGE            20 / 145 spans sit in a context that contradicts the label
    DATE_OF_BIRTH  88 / 325   e.g. "Order #1234 confirmed on 25/02/1971"

Only 8 of 325 DATE_OF_BIRTH spans carry a birth anchor at all. Training on that
teaches that any date is a birth date -- a precision failure manufactured on
purpose, in a corpus whose whole point is fixing a precision/recall frontier.

The frozen gold set has zero contradicted labels, and `data/train_v2.jsonl` has
zero, because both come from hand-written templates with the anchor built into
the template ("Patient X, age {AGE}", "born {DOB}"). Generated prose carries no
such guarantee, which is why this filter exists for v3 and was never needed
before.

Whole records are dropped, not individual spans. A record whose AGE is wrong is
evidence that its carrier was filled carelessly, and the remaining spans in it
are not independently trustworthy; keeping them would preserve exactly the
records most likely to carry an undetected second error.

    python scripts/filter_train_v3.py --in data/train_v3.jsonl \
        --out data/train_v3_clean.jsonl
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from audit_gold import _ANCHOR, _ANCHOR_WINDOW, _CONTRADICTS

from forge.schema import HIGH_SEVERITY, PIIRecord


def contradicted_spans(rec: PIIRecord) -> list[tuple[str, str]]:
    """Spans whose surrounding text positively indicates a different meaning.

    Absence of a supporting keyword is NOT enough: "Rachita Thakkar (53)
    requested an upgrade" is a perfectly good AGE with no anchor word. Only
    positive counter-evidence counts, which is the same distinction
    `audit_semantics` draws between `contradicted` (an error) and `unanchored`
    (a warning).
    """
    out = []
    for s in rec.spans:
        bad = _CONTRADICTS.get(s.label.value)
        if not bad:
            continue
        lo = max(0, s.start - _ANCHOR_WINDOW)
        hi = min(len(rec.text), s.end + _ANCHOR_WINDOW)
        window = rec.text[lo : s.start] + " " + rec.text[s.end : hi]
        anchor = _ANCHOR.get(s.label.value)
        if anchor and anchor.search(window):
            continue
        if bad.search(window):
            out.append((s.label.value, rec.text[lo:hi]))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in", dest="src", type=Path, default=Path("data/train_v3.jsonl"))
    ap.add_argument("--out", type=Path, default=Path("data/train_v3_clean.jsonl"))
    args = ap.parse_args()

    if not args.src.exists():
        print(f"missing {args.src}", file=sys.stderr)
        return 1

    records = [
        PIIRecord.model_validate_json(line)
        for line in args.src.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    kept, dropped = [], []
    reasons: Counter[str] = Counter()
    examples: dict[str, str] = {}
    for r in records:
        bad = contradicted_spans(r)
        if bad:
            dropped.append(r)
            for label, ctx in bad:
                reasons[label] += 1
                examples.setdefault(label, ctx)
        else:
            kept.append(r)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        "".join(r.model_dump_json() + "\n" for r in kept), encoding="utf-8"
    )

    # Re-read the written bytes and re-verify, rather than trusting the in-memory
    # filter. Same belt-and-braces as scripts/build_validation.py.
    reread = [
        PIIRecord.model_validate_json(line)
        for line in args.out.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    residual = sum(len(contradicted_spans(r)) for r in reread)
    if residual:
        print(f"FATAL: {residual} contradicted spans survived the filter", file=sys.stderr)
        return 1

    hs = {t.value for t in HIGH_SEVERITY}
    per_type = Counter(s.label.value for r in kept for s in r.spans)
    sha = hashlib.sha256(args.out.read_bytes()).hexdigest()

    print(f"in  {len(records):>5} records, {sum(len(r.spans) for r in records):>5} spans")
    print(f"out {len(kept):>5} records, {sum(len(r.spans) for r in kept):>5} spans   "
          f"({len(kept)/len(records)*100:.1f}% kept)")
    print(f"\ndropped {len(dropped)} records, by contradicted label:")
    for label, n in reasons.most_common():
        print(f"  {label:<16}{n:>4}   e.g. ...{examples[label][:70]!r}")
    print("\nsurviving per-type coverage:")
    for t, n in per_type.most_common():
        print(f"  {t:<16}{n:>5}{'  *high-severity' if t in hs else ''}")
    print("\nverified on the written bytes: 0 contradicted spans remain")
    print(f"wrote  {args.out}")
    print(f"sha256 {sha}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
