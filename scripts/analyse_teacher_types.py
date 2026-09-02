#!/usr/bin/env python3
"""Per-type teacher/student agreement — the evidence table behind ADR 0015.

ADR 0015 splits the data engine by type, and the split is derived from one
question: **for each type, is the teacher actually a better labeller than what we
already have?** This regenerates that answer from committed prediction files, so
the ADR's table can be checked rather than trusted.

Two distinctions the standard evaluator does not draw, and that the routing
decision needs:

- **exact vs boundary-only.** A same-label span that overlaps gold but ends
  somewhere else is not a detection failure, it is a *convention* disagreement.
  `data/gold/PROTOCOL.md` §3 fixes the convention ("a full mailing address ->
  STREET_ADDRESS; a bare city used as context -> LOCATION"), and the teacher
  splits addresses. Training on its boundaries would train against our contract.
- **headroom = teacher F1 - student F1.** A type where the student already beats
  the teacher cannot be improved by distilling the teacher, and `LOCATION` is
  exactly that case.

    python scripts/analyse_teacher_types.py
    python scripts/analyse_teacher_types.py --json reports/teacher_type_analysis.json
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from forge.schema import HIGH_SEVERITY, PIIRecord


def load(path: Path) -> list[PIIRecord]:
    if not path.exists():
        return []
    return [PIIRecord.model_validate_json(ln)
            for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]


def prf(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    p = tp / max(1, tp + fp)
    r = tp / max(1, tp + fn)
    return p, r, 2 * p * r / max(1e-12, p + r)


def score(gold: dict[str, PIIRecord], preds: list[PIIRecord]) -> dict:
    """Exact-match tallies plus boundary-only near misses, per type."""
    tp, fp, fn, boundary = Counter(), Counter(), Counter(), Counter()
    examples = defaultdict(list)
    matched = 0

    for pr in preds:
        g = gold.get(pr.text)
        if g is None:
            continue
        matched += 1
        gk = {(s.start, s.end, s.label.value): s for s in g.spans}
        pk = {(s.start, s.end, s.label.value) for s in pr.spans}
        for key, s in gk.items():
            if key in pk:
                tp[s.label.value] += 1
                continue
            overlap = next((p for p in pr.spans if p.label == s.label
                            and p.start < s.end and s.start < p.end), None)
            fn[s.label.value] += 1
            if overlap is not None:
                boundary[s.label.value] += 1
                if len(examples[s.label.value]) < 3:
                    examples[s.label.value].append({"gold": s.text, "pred": overlap.text})
        for key in pk - set(gk):
            fp[key[2]] += 1

    return {"tp": tp, "fp": fp, "fn": fn, "boundary": boundary,
            "examples": examples, "matched": matched}


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gold", type=Path, default=Path("data/gold/test.jsonl"))
    ap.add_argument("--teacher", type=Path,
                    default=Path("data/predictions_teacher_120b_test.jsonl"))
    ap.add_argument("--student", type=Path,
                    default=Path("data/predictions_student_run_002.jsonl"))
    ap.add_argument("--json", type=Path, default=None)
    args = ap.parse_args()

    gold_recs = load(args.gold)
    if not gold_recs:
        print(f"no gold at {args.gold}", file=sys.stderr)
        return 2
    gold = {r.text: r for r in gold_recs}
    gold_counts = Counter(s.label.value for r in gold_recs for s in r.spans)

    t = score(gold, load(args.teacher))
    s = score(gold, load(args.student))
    print(f"gold {args.gold}: {len(gold_recs)} records, {sum(gold_counts.values())} spans")
    print(f"teacher matched {t['matched']}, student matched {s['matched']}\n")

    header = (f"{'type':<16}{'n':>5}{'T exact':>9}{'T bnd':>7}{'T F1':>8}"
              f"{'S F1':>8}{'headroom':>10}  owner")
    print(header)
    print("-" * len(header))

    rows = {}
    for label in sorted(gold_counts, key=lambda x: -gold_counts[x]):
        n = gold_counts[label]
        _, t_rec, t_f1 = prf(t["tp"][label], t["fp"][label], t["fn"][label])
        _, _, s_f1 = prf(s["tp"][label], s["fp"][label], s["fn"][label])
        owner = "validator" if any(h.value == label for h in HIGH_SEVERITY) else "model"
        head = t_f1 - s_f1
        rows[label] = {
            "n": n, "teacher_exact_recall": round(t_rec, 4),
            "teacher_boundary_only": t["boundary"][label],
            "teacher_f1": round(t_f1, 4), "student_f1": round(s_f1, 4),
            "headroom": round(head, 4), "owner": owner,
            "boundary_examples": t["examples"].get(label, []),
        }
        flag = "  <- distilling this would REGRESS" if head < 0 and owner == "model" else ""
        print(f"{label:<16}{n:>5}{t_rec:>9.3f}{t['boundary'][label]:>7}"
              f"{t_f1:>8.4f}{s_f1:>8.4f}{head:>+10.4f}  {owner}{flag}")

    model_rows = {k: v for k, v in rows.items() if v["owner"] == "model"}
    regressive = [k for k, v in model_rows.items() if v["headroom"] < 0]
    print(f"\nModel-owned types where the teacher is WORSE than the student: "
          f"{regressive or 'none'}")
    print("These cannot be Track B: distillation transfers the teacher's behaviour, so a "
          "type the student already wins can only get worse.")

    print("\nBoundary disagreements (gold | teacher) — convention, not detection:")
    for label, ex in sorted(t["examples"].items()):
        if not ex:
            continue
        print(f"  {label}:")
        for e in ex:
            print(f"    gold={e['gold']!r}")
            print(f"    tchr={e['pred']!r}")

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps({
            "gold": str(args.gold), "teacher": str(args.teacher),
            "student": str(args.student), "per_type": rows,
            "regressive_model_owned": regressive,
        }, indent=2) + "\n", encoding="utf-8")
        print(f"\nwrote {args.json}")
    return 0


def carrier_shape_counts(paths: list[Path]) -> dict[str, int]:
    """Distinct span-masked skeletons per file — the diversity metric ADR 0015 targets."""
    out = {}
    for p in paths:
        recs = load(p)
        if recs:
            out[str(p)] = len({re.sub(r"\s+", " ", r.redacted).strip() for r in recs})
    return out


if __name__ == "__main__":
    sys.exit(main())
