#!/usr/bin/env python3
"""Use Track B's teacher disagreements to audit Track A's construction labels.

ADR 0015 predicted the teacher would *discover* entities in its own prose that
construction had not injected (P3). It discovered exactly zero. But the same
comparison surfaced something the ADR did not predict and that matters more.

Construction labels are exact **by offset** — `forge.carriers.fill` accumulates
them during the fill, so `text[start:end]` always equals the span. They are not
guaranteed correct **by semantics**. The carrier writes a slot and the filler
drops a value into it, and nothing checks that the surrounding prose means what
the placeholder claims:

    "...please settle it within 46 days."            46 labelled AGE
    "...last seen at the sorting facility at 27 hours."  27 labelled AGE
    "Transaction row shows ... amount $25 ..."       25 labelled AGE
    "...for the claim filed on 20/12/1961."          labelled DATE_OF_BIRTH
    "Claim at Gerardport before 05 Apr 1986."        labelled DATE_OF_BIRTH

None of those are the entity the label claims. Training on them teaches the model
that any small integer is an age and any date is a date of birth, which is a
precision failure manufactured on purpose.

The teacher found them because it labels the text it is given without knowing what
was injected — so on Track B records it silently declines these spans, and they
show up as `anchor.missing`. Track B and Track A are filled from the **same 456
carrier shapes**, so a disagreement observed on a Track B instance transfers to
every Track A record built from that shape. That is what this script counts.

Note the asymmetry, which is the whole reason the anchor keeps construction rather
than deferring to the teacher: on `LOCATION` the teacher is the one that is wrong
(exact recall 0.273 on the frozen test set, 0.100 on val, and it misses 93% of
injected `LOCATION` here), so a high miss rate there is evidence about the
teacher. On `AGE` and `DATE_OF_BIRTH` the teacher scores 1.000 on the frozen test
set, so a high miss rate there is evidence about the carrier. `--suspect-types`
sets which side is on trial; it defaults to the types where the teacher is known
to be reliable.

    python scripts/audit_carriers.py
    python scripts/audit_carriers.py --json reports/carrier_audit.json
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from build_train_v3 import (
    DATA_ENGINE_SEED,
    _spans_from_json,
    instantiate,
    load_carriers,
    read_cache,
)

from forge.carriers import anchor_against_construction, shape_of
from forge.schema import PIIType

# Types where the teacher is a credible judge of construction: it scores 1.000
# exact recall on both against the frozen gold set, so when it declines one of
# these in its own prose, the carrier is the suspect.
DEFAULT_SUSPECT = (PIIType.AGE, PIIType.DATE_OF_BIRTH)


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--carriers", type=Path, default=Path("data/carriers_v3.jsonl"))
    ap.add_argument("--cache", type=Path,
                    default=Path("data/train_v3.teacher_cache.jsonl"))
    ap.add_argument("--total", type=int, default=5000)
    ap.add_argument("--track-b-fraction", type=float, default=0.40)
    ap.add_argument("--track-b-overplan", type=float, default=1.3)
    ap.add_argument("--seed", type=int, default=DATA_ENGINE_SEED)
    ap.add_argument("--threshold", type=float, default=0.5,
                    help="Miss rate at or above which a (shape, type) pair is suspect")
    ap.add_argument("--suspect-types", nargs="*", default=[t.value for t in DEFAULT_SUSPECT])
    ap.add_argument("--examples", type=int, default=5)
    ap.add_argument("--json", type=Path, default=None)
    args = ap.parse_args()

    carriers, _ = load_carriers(args.carriers)
    planned = instantiate(carriers, args.total, args.track_b_fraction, args.seed,
                          overplan=args.track_b_overplan)
    cached, _ = read_cache(args.cache)
    suspect_types = {PIIType(t) for t in args.suspect_types}

    # (shape, type) -> [instances the teacher saw, instances it declined]
    tally: dict[tuple[str, PIIType], list[int]] = defaultdict(lambda: [0, 0])
    examples: dict[PIIType, list[dict]] = defaultdict(list)
    labelled = 0

    for rec, track in planned:
        if track != "B":
            continue
        entry = cached.get(rec.text)
        if not entry or not entry["gate_accepted"]:
            continue
        labelled += 1
        shape = shape_of(rec)
        anchor = anchor_against_construction(
            list(rec.spans), _spans_from_json(entry["teacher_spans"]))
        for s in rec.spans:
            tally[(shape, s.label)][0] += 1
        for s in anchor.missing:
            tally[(shape, s.label)][1] += 1
            if s.label in suspect_types and len(examples[s.label]) < args.examples:
                examples[s.label].append({"span": s.text, "text": rec.text})

    if not labelled:
        print("no labelled Track B records in the cache yet — nothing to audit",
              file=sys.stderr)
        return 2

    suspect = {k for k, (n, m) in tally.items()
               if k[1] in suspect_types and n and m / n >= args.threshold}
    audited_shapes = {k[0] for k in tally}

    track_a = [r for r, t in planned if t == "A"]
    affected = [r for r in track_a
                if any((shape_of(r), s.label) in suspect for s in r.spans)]

    print(f"Track B records audited:   {labelled}")
    print(f"carrier shapes audited:    {len(audited_shapes)} of {len(carriers)} "
          f"({len(audited_shapes) / len(carriers):.0%}) — the rest are UNAUDITED, not clean")
    print(f"suspect (shape, type) pairs at miss rate >= {args.threshold}: {len(suspect)}\n")

    print(f"{'type':<16}{'injected':>10}{'declined':>10}{'rate':>8}   verdict")
    print("-" * 62)
    per_type = {}
    for label in sorted({k[1] for k in tally}, key=lambda x: x.value):
        n = sum(v[0] for k, v in tally.items() if k[1] is label)
        m = sum(v[1] for k, v in tally.items() if k[1] is label)
        if not n:
            continue
        rate = m / n
        if label in suspect_types:
            verdict = "carrier suspect" if rate >= args.threshold else "ok"
        elif label is PIIType.LOCATION:
            verdict = "teacher blind spot (ADR 0015)"
        else:
            verdict = "not on trial"
        per_type[label.value] = {"injected": n, "declined": m, "rate": round(rate, 4),
                                 "verdict": verdict}
        print(f"{label.value:<16}{n:>10}{m:>10}{rate:>8.3f}   {verdict}")

    share = len(affected) / max(1, len(track_a))
    print(f"\nTrack A records built from a suspect (shape, type): "
          f"**{len(affected)} of {len(track_a)} = {share:.1%}**")
    print(f"Extrapolating over unaudited shapes, the true figure is roughly "
          f"{share * len(carriers) / max(1, len(audited_shapes)):.0%}.")

    for label, exs in examples.items():
        print(f"\n{label.value} declined by the teacher — construction's label, its prose:")
        for e in exs:
            print(f"  {e['span']!r} in {e['text'][:150]!r}")

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps({
            "track_b_audited": labelled,
            "shapes_audited": len(audited_shapes),
            "shapes_total": len(carriers),
            "threshold": args.threshold,
            "suspect_types": sorted(t.value for t in suspect_types),
            "suspect_pairs": len(suspect),
            "track_a_total": len(track_a),
            "track_a_affected": len(affected),
            "track_a_affected_share": round(share, 4),
            "per_type": per_type,
            "examples": {k.value: v for k, v in examples.items()},
        }, indent=2) + "\n", encoding="utf-8")
        print(f"\nwrote {args.json}")

    return 0


def suspect_pairs_summary(tally: dict, threshold: float) -> Counter:
    """Suspect pair count per type — kept for reuse by the data card."""
    out: Counter = Counter()
    for (_, label), (n, m) in tally.items():
        if n and m / n >= threshold:
            out[label.value] += 1
    return out


if __name__ == "__main__":
    sys.exit(main())
