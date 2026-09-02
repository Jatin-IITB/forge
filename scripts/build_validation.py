#!/usr/bin/env python3
"""Build a clean held-out validation split (WP-0d).

**Why this exists.** `data/gold/dev.jsonl` cannot be used for model selection:
`scripts/audit_gold.py` shows 150 of its 189 records (79.4%) appear verbatim in
`data/train.jsonl`. The data engine was seeded from dev, and `forge/dedup.py`
was handed the *test* split to check leakage against, so dev contamination was
never detected. Selecting a LoRA rank on dev would score memorised training text
and would reliably pick the most overfit configuration.

**Why not the obvious alternatives.**

- *Carve a slice out of test* — breaks the freeze. `test.jsonl` is the
  denominator of every published number and is never re-partitioned.
- *Use the 39 uncontaminated dev records* — far too few. With ~2 high-severity
  instances per type, a perfect score would support almost no lower bound.
- *Reuse dev and subtract the leaked records* — the survivors are not a random
  sample; they are whatever the data engine happened not to consume.

So: regenerate from the same deterministic builder under a **different seed**.
Same template distribution (which is what a validation set should share with
test), different Faker values, therefore different carrier text. Disjointness is
then **asserted, not assumed** — the script refuses to write if any record
overlaps train, dev or test.

    python scripts/build_validation.py                 # seed 4242 -> data/gold/val.jsonl
    python scripts/build_validation.py --seed 9001 --max-records 300
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

from build_gold import TEMPLATES, PIIValueGenerator, build_record, expand_template

from forge.schema import PIIRecord

# Deliberately not 42 (the frozen gold seed) and not 1337 (the ADR 0009
# augmentation seed), so a mix-up is visible rather than silent.
VALIDATION_SEED = 4242


def _texts(path: Path) -> set[str]:
    if not path.exists():
        return set()
    return {
        PIIRecord.model_validate_json(line).text
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    }


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--seed", type=int, default=VALIDATION_SEED)
    ap.add_argument("--output", type=Path, default=Path("data/gold/val.jsonl"))
    ap.add_argument("--train", type=Path, default=Path("data/train.jsonl"))
    ap.add_argument("--gold-dir", type=Path, default=Path("data/gold"))
    ap.add_argument(
        "--max-records", type=int, default=None,
        help="Cap the split size (default: keep every disjoint record).",
    )
    args = ap.parse_args()

    if args.seed == 42:
        print("refusing: seed 42 is the frozen gold seed and would reproduce test", file=sys.stderr)
        return 2

    gen = PIIValueGenerator(args.seed)
    rng = random.Random(args.seed + 1)

    records: list[PIIRecord] = []
    seq = 0
    for tpl, weight in TEMPLATES:
        for _ in range(weight):
            seq += 1
            records.append(build_record(f"pii-{seq:04d}", expand_template(tpl, gen), split="val"))
    rng.shuffle(records)

    forbidden = {
        "train": _texts(args.train),
        "dev": _texts(args.gold_dir / "dev.jsonl"),
        "test": _texts(args.gold_dir / "test.jsonl"),
    }

    kept: list[PIIRecord] = []
    seen: set[str] = set()
    dropped = dict.fromkeys(forbidden, 0)
    dropped["self_duplicate"] = 0

    for r in records:
        if r.text in seen:
            dropped["self_duplicate"] += 1
            continue
        hit = next((name for name, texts in forbidden.items() if r.text in texts), None)
        if hit:
            dropped[hit] += 1
            continue
        seen.add(r.text)
        kept.append(r)
        if args.max_records and len(kept) >= args.max_records:
            break

    kept = [
        r.model_copy(update={"id": f"pii-val-{i:04d}", "split": "val"})
        for i, r in enumerate(kept, 1)
    ]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        for r in kept:
            f.write(r.model_dump_json() + "\n")

    n_spans = sum(len(r.spans) for r in kept)
    print(f"generated {len(records)} candidates from seed {args.seed}")
    for name, n in dropped.items():
        print(f"  dropped {n:>4} overlapping {name}")
    print(f"wrote {len(kept)} records ({n_spans} spans) -> {args.output}")

    # Belt and braces: re-read what was written and verify the invariant on the
    # bytes, not on the in-memory objects that were just filtered.
    written = _texts(args.output)
    for name, texts in forbidden.items():
        overlap = written & texts
        if overlap:
            print(f"FATAL: {len(overlap)} written records overlap {name}", file=sys.stderr)
            return 1
    print("verified disjoint from train, dev and test")
    return 0


if __name__ == "__main__":
    sys.exit(main())
