#!/usr/bin/env python3
"""Union v2 and v3, because the v3 experiment failed on composition, not volume.

## What the v3 run actually showed

Training on v3 alone moved model-only micro-F1 on the frozen test set from
0.8845 to 0.8517 -- backwards. The pre-registered prediction P1 (F1 rises) is
refuted, and the pre-registered *inference* attached to it was wrong too: it said
"if P1 fails, more data is not the lever, and the gap is capacity or task
formulation." That was a false dichotomy. It treated the corpus axis as *amount*
when v3 changed the *mixture*:

    type             v2 share   v3 share   test share
    high-severity      38.8%       9.9%       33.4%
    PASSPORT           38 spans    4 spans    23 spans
    DRIVER_LICENSE     76          37         15
    BANK_ACCOUNT       84          53         29
    STREET_ADDRESS     71         518         32
    LOCATION           48         585         22

v3 tripled the span count while *absolutely reducing* five of the nine
high-severity types. It is not "more of v2"; it is a different task distribution
-- 90% prose PII against a test set that is a third structured identifiers.

Every per-type movement follows from that, which is why this is a mechanism and
not a story fitted after the fact:

    STREET_ADDRESS  0.8438 -> 0.9412   7x more data      (P2 predicted, held)
    PASSPORT        ?      -> 0.0000   38 -> 4 spans     (P3 predicted, held)
    CREDIT_CARD     recall    0.4878   the numeric identifier types collapse
    BANK_ACCOUNT    prec      0.4667   into mutual confusion once none of them
    DRIVER_LICENSE  prec      0.5357   has enough support to separate

## Why the union, and not a distribution matched to the test set

The obvious move is to reweight training until it mirrors the test mixture. That
is quietly a form of peeking: the frozen set is the specification, and tuning
proportions against it reverse-engineers the eval rather than the task.

The union needs no such knowledge. It uses only the fact that both corpora are
valid training data and that they are disjoint -- verified here, not assumed:
0 shared texts, 0 gold leakage from either. It restores identifier density as a
side effect of adding v2 back, rather than as a target.

The union lands at 16.8% high-severity against the test set's 33.4%. Still
short. That is deliberate: if a corpus built without reference to the test
distribution closes most of the gap, composition was the binding constraint. If
it does not, the next move is capacity, and this run is what licenses that claim.

## PRE-REGISTERED

    P4  micro-F1 > 0.8845 -- beats BOTH parents, not just v3. The union has
        v3's prose enrichment and v2's identifier density; if composition is
        the mechanism it should beat the better parent, not split the two.
    P5  PASSPORT F1 > 0.0000. 42 spans against v3's 4.
    P6  STREET_ADDRESS F1 >= 0.9412 holds. 589 spans, more than v3's 518, so
        the v3 gain must survive -- if it regresses, the types are competing
        for capacity and P4's mechanism is wrong.
    P7  CREDIT_CARD recall > 0.4878.

    FALSIFIER: if P4 fails while P5 holds, identifier density was recovered and
    F1 still did not move, so composition is NOT the binding constraint and the
    remaining gap is capacity or task formulation. That is the fork this run
    exists to resolve, and it is the claim the failed v3 prediction was not
    entitled to make.

    python scripts/build_train_v4.py --out data/train_v4.jsonl
    python scripts/normalize_spans.py --in data/train_v4.jsonl \
        --out data/train_v4_aligned.jsonl
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from collections import Counter
from pathlib import Path

from forge.schema import HIGH_SEVERITY, PIIRecord

GOLD_PATHS = (Path("data/gold/test.jsonl"), Path("data/gold/val.jsonl"))


def load(path: Path) -> list[PIIRecord]:
    return [
        PIIRecord.model_validate_json(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--v2", type=Path, default=Path("data/train_v2.jsonl"))
    ap.add_argument("--v3", type=Path, default=Path("data/train_v3_aligned.jsonl"))
    ap.add_argument("--out", type=Path, default=Path("data/train_v4.jsonl"))
    args = ap.parse_args()

    for path in (args.v2, args.v3):
        if not path.exists():
            print(f"missing {path}", file=sys.stderr)
            return 1

    v2, v3 = load(args.v2), load(args.v3)

    # Gold leakage is checked against the COMMITTED bytes of both splits, the
    # same way audit_gold.py checks the frozen set. A training corpus that
    # contains a test text does not fail loudly; it just reports a score that
    # means nothing, which is the failure mode that cost this project a week
    # when the dedup pass was handed the wrong split.
    gold_texts: set[str] = set()
    for path in GOLD_PATHS:
        if path.exists():
            gold_texts |= {r.text for r in load(path)}

    kept: list[PIIRecord] = []
    seen: set[str] = set()
    stats: Counter[str] = Counter()
    for source, records in (("v2", v2), ("v3", v3)):
        for record in records:
            if record.text in gold_texts:
                stats["dropped_gold_leak"] += 1
                continue
            if record.text in seen:
                stats[f"dropped_duplicate_{source}"] += 1
                continue
            seen.add(record.text)
            kept.append(record)
            stats[f"kept_{source}"] += 1

    if stats["dropped_gold_leak"]:
        print(
            f"FATAL: {stats['dropped_gold_leak']} training records share text with "
            "the gold set. Neither parent had any; investigate before training.",
            file=sys.stderr,
        )
        return 1

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8", newline="\n") as handle:
        for record in kept:
            handle.write(record.model_dump_json() + "\n")

    reread = load(args.out)
    if len(reread) != len(kept):
        print(f"FATAL: wrote {len(kept)}, read back {len(reread)}", file=sys.stderr)
        return 1
    if len({r.text for r in reread}) != len(reread):
        print("FATAL: duplicates survived in the written bytes", file=sys.stderr)
        return 1

    hs = {t.value for t in HIGH_SEVERITY}
    per_type = Counter(s.label.value for r in reread for s in r.spans)
    total = sum(per_type.values())
    hs_total = sum(per_type[t] for t in hs)

    test = Counter()
    if GOLD_PATHS[0].exists():
        test = Counter(s.label.value for r in load(GOLD_PATHS[0]) for s in r.spans)
    test_total = sum(test.values()) or 1

    print(f"v2 {len(v2):>5} records -> kept {stats['kept_v2']}")
    print(f"v3 {len(v3):>5} records -> kept {stats['kept_v3']}")
    for key, n in sorted(stats.items()):
        if key.startswith("dropped"):
            print(f"  {key:<28}{n:>5}")
    print(f"\nout {len(reread)} records, {total} spans")
    print(f"high-severity {hs_total} spans = {hs_total / total * 100:.1f}% "
          f"(v3 alone was 9.9%, test is {sum(test[t] for t in hs) / test_total * 100:.1f}%)")

    print(f"\n{'type':<16}{'train':>7}{'share':>8}{'test share':>12}")
    for label, n in per_type.most_common():
        flag = "  *high-severity" if label in hs else ""
        print(f"{label:<16}{n:>7}{n / total * 100:>7.1f}%"
              f"{test.get(label, 0) / test_total * 100:>11.1f}%{flag}")

    missing = sorted(hs - set(per_type))
    if missing:
        print(f"\n  high-severity types ABSENT: {missing}")

    print(f"\nverified on the written bytes: {len(reread)} records, 0 duplicates, 0 gold leaks")
    print(f"wrote  {args.out}")
    print(f"sha256 {hashlib.sha256(args.out.read_bytes()).hexdigest()}")
    print("\nNext: scripts/normalize_spans.py, then train. v2 has never been through")
    print("the BIOES normalizer, so run it before committing GPU time.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
