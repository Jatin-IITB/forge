#!/usr/bin/env python3
"""Does the 'be thorough' teacher prompt fix the teacher's LOCATION blind spot?

ADR 0015 routes `LOCATION` to Track A (construction) rather than Track B
(distillation) because the committed teacher baseline misses 16 of 22 `LOCATION`
instances on the frozen test set — exact recall 0.273, worse than the student's
own 0.7805. That decision rests on `data/predictions_teacher_120b_test.jsonl`,
which `scripts/run_inference.py` produced with the **plain** system prompt
(`build_messages(text)`, no `teacher_mode`).

But `scripts/run_data_engine.py` labels training data with a *different* prompt —
`TEACHER_SYSTEM_PROMPT`, which adds "Be thorough — missing a PII entity is worse
than a false positive" and asks for a rationale per span. That instruction is
aimed squarely at recall. If it lifts `LOCATION`, the Track A/B split in ADR 0015
is derived from the wrong distribution and `LOCATION` belongs in Track B after
all.

Designing against one prompt's measurements and generating with another is
exactly the attribution failure ADR 0013 diagnosed in run_002, so this is
settled with ~40 calls rather than an argument.

**Run on `data/gold/val.jsonl`, never on test.** val is the clean split built by
`scripts/build_validation.py` (seed 4242, disjointness asserted), it has never
been used for a teacher measurement, and using it here keeps the frozen test set
out of a configuration choice entirely.

    python scripts/probe_teacher_prompt.py --api-key-env CEREBRAS_API_KEY --n 20
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from forge.inference import build_messages, parse_response
from forge.schema import PIIRecord, PIIType
from forge.teacher_client import ThrottledTeacher

try:
    from openai import OpenAI
except ImportError:  # pragma: no cover
    print("Install the openai package: pip install 'openai>=1.0'", file=sys.stderr)
    sys.exit(1)

# The types the probe is actually about: ADR 0015's Track B roster plus LOCATION,
# which is the type under dispute.
WATCHED = [PIIType.LOCATION, PIIType.STREET_ADDRESS, PIIType.PERSON,
           PIIType.USERNAME, PIIType.AGE]


def load(path: Path) -> list[PIIRecord]:
    return [PIIRecord.model_validate_json(ln)
            for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--split", type=Path, default=Path("data/gold/val.jsonl"))
    ap.add_argument("--output", type=Path, default=Path("reports/teacher_prompt_probe.json"))
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--model", default="gpt-oss-120b")
    ap.add_argument("--base-url", default="https://api.cerebras.ai/v1")
    ap.add_argument("--api-key-env", default="CEREBRAS_API_KEY")
    ap.add_argument("--rpm", type=float, default=5.0)
    ap.add_argument("--max-tokens", type=int, default=1024)
    ap.add_argument("--timeout", type=float, default=90.0)
    ap.add_argument("--reasoning-effort", default="low")
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=7717)
    args = ap.parse_args()

    if "test" in args.split.name:
        print("refusing: the frozen test set is not used to choose a configuration",
              file=sys.stderr)
        return 2

    key = os.environ.get(args.api_key_env)
    if not key:
        print(f"Environment variable {args.api_key_env} is not set.", file=sys.stderr)
        return 1

    records = load(args.split)
    # Bias the sample toward records that actually carry a watched type, so 20
    # records buy a usable number of instances rather than a usable number of rows.
    with_loc = [r for r in records if any(s.label is PIIType.LOCATION for s in r.spans)]
    others = [r for r in records if r not in with_loc]
    rng = random.Random(args.seed)
    rng.shuffle(with_loc)
    rng.shuffle(others)
    sample = (with_loc[: max(1, args.n // 2)] + others[: args.n - max(1, args.n // 2)])[: args.n]

    gold_counts = Counter(s.label.value for r in sample for s in r.spans)
    print(f"probe sample: {len(sample)} records from {args.split}")
    print(f"gold instances: {dict(gold_counts)}\n")

    teacher = ThrottledTeacher(
        # max_retries=0: ThrottledTeacher is the only retrier, so the SDK cannot
        # silently spend requests against an exhausted quota before we see it.
        OpenAI(base_url=args.base_url, api_key=key, max_retries=0), rpm=args.rpm,
        on_retry=lambda a, e, d: print(f"    {type(e).__name__}, retry {a} in {d:.0f}s",
                                       flush=True),
    )
    results: dict[str, dict] = {}
    scored: dict[str, set[str]] = {}

    for mode_name, teacher_mode in (("plain", False), ("teacher_mode", True)):
        preds: dict[str, set] = {}
        for i, rec in enumerate(sample, 1):
            try:
                resp, _ = teacher.complete(
                    model=args.model,
                    messages=build_messages(rec.text, teacher_mode=teacher_mode),
                    max_tokens=args.max_tokens, temperature=args.temperature,
                    timeout=args.timeout,
                    extra_body={"reasoning_effort": args.reasoning_effort},
                )
                pred, _ = parse_response(rec.id, rec.text,
                                         resp.choices[0].message.content or "")
            except Exception as exc:  # noqa: BLE001
                print(f"  [{mode_name} {i}/{len(sample)}] DROPPED: {type(exc).__name__}",
                      flush=True)
                continue
            preds[rec.id] = {(s.start, s.end, s.label.value) for s in pred.spans}
            print(f"  [{mode_name} {i}/{len(sample)}] {len(pred.spans)} spans "
                  f"(gold {len(rec.spans)})", flush=True)
        results[mode_name] = preds
        scored[mode_name] = set(preds)
        print()

    # Score only records BOTH arms returned. Scoring each arm against a denominator
    # of all sampled records would understate recall by whatever fraction was
    # dropped, and comparing two arms with different denominators is meaningless.
    common = scored["plain"] & scored["teacher_mode"]
    by_id = {r.id: r for r in sample}
    common_gold = Counter(s.label.value for rid in common for s in by_id[rid].spans)
    print("=" * 68)
    print(f"scored in both arms: {len(common)}/{len(sample)} records "
          f"(plain {len(scored['plain'])}, thorough {len(scored['teacher_mode'])})")
    if len(common) < len(sample):
        print(f"  {len(sample) - len(common)} record(s) dropped after retries; the table "
              f"below uses only the {len(common)} common to both arms.")

    def counts(mode):
        tp, fn, fp = Counter(), Counter(), Counter()
        for rid in common:
            gk = {(s.start, s.end, s.label.value) for s in by_id[rid].spans}
            pk = results[mode][rid]
            for k in gk & pk:
                tp[k[2]] += 1
            for k in gk - pk:
                fn[k[2]] += 1
            for k in pk - gk:
                fp[k[2]] += 1
        return tp, fn, fp

    tallies = {m: counts(m) for m in results}
    print(f"\n{'type':<16}{'gold':>6}{'plain R':>10}{'thorough R':>13}{'delta':>9}")
    verdict_rows = {}
    for t in WATCHED:
        g = common_gold.get(t.value, 0)
        if not g:
            continue
        r_plain = tallies["plain"][0].get(t.value, 0) / g
        r_teach = tallies["teacher_mode"][0].get(t.value, 0) / g
        verdict_rows[t.value] = {"gold": g, "plain_recall": round(r_plain, 4),
                                 "teacher_mode_recall": round(r_teach, 4)}
        print(f"{t.value:<16}{g:>6}{r_plain:>10.3f}{r_teach:>13.3f}{r_teach - r_plain:>+9.3f}")

    def micro(mode):
        tp, fn, fp = tallies[mode]
        TP, FP, FN = sum(tp.values()), sum(fp.values()), sum(fn.values())
        return 2 * TP / max(1, 2 * TP + FP + FN)

    identical = sum(1 for rid in common
                    if results["plain"][rid] == results["teacher_mode"][rid])
    print(f"\nmicro-F1  plain={micro('plain'):.4f}  thorough={micro('teacher_mode'):.4f}")
    print(f"records where the two prompts returned identical span sets: "
          f"{identical}/{len(common)}")
    print(f"tokens spent: {teacher.stats.total_tokens} "
          f"({teacher.stats.calls_ok} ok, {teacher.stats.retries} retries, "
          f"{teacher.stats.calls_failed} gave up)")

    payload = {
        "split": str(args.split),
        "n_sampled": len(sample),
        "n_scored_both_arms": len(common),
        "gold_instances_sampled": dict(gold_counts),
        "gold_instances_scored": dict(common_gold),
        "per_type": verdict_rows,
        "micro_f1": {m: round(micro(m), 4) for m in results},
        "identical_span_sets": identical,
        "per_type_counts": {
            m: {"tp": dict(t[0]), "fn": dict(t[1]), "fp": dict(t[2])}
            for m, t in tallies.items()
        },
        "teacher_stats": teacher.stats.summary(),
        "model": args.model, "temperature": args.temperature, "seed": args.seed,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
