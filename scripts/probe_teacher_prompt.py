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
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from forge.inference import build_messages, parse_response
from forge.schema import PIIRecord, PIIType

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

    client = OpenAI(base_url=args.base_url, api_key=key)
    min_interval = 60.0 / args.rpm if args.rpm else 0.0
    last = 0.0
    tokens = 0
    results: dict[str, dict] = {}

    for mode_name, teacher_mode in (("plain", False), ("teacher_mode", True)):
        tp, fn, fp = Counter(), Counter(), Counter()
        n_pred = 0
        for i, rec in enumerate(sample, 1):
            try:
                if min_interval:
                    wait = min_interval - (time.monotonic() - last)
                    if wait > 0:
                        time.sleep(wait)
                last = time.monotonic()
                resp = client.chat.completions.create(
                    model=args.model,
                    messages=build_messages(rec.text, teacher_mode=teacher_mode),
                    max_tokens=args.max_tokens, temperature=args.temperature,
                    timeout=args.timeout,
                    extra_body={"reasoning_effort": args.reasoning_effort},
                )
                if resp.usage:
                    tokens += resp.usage.total_tokens
                pred, _ = parse_response(rec.id, rec.text,
                                         resp.choices[0].message.content or "")
            except Exception as exc:  # noqa: BLE001
                print(f"  [{mode_name} {i}/{len(sample)}] api error: {type(exc).__name__}")
                continue
            n_pred += 1
            gk = {(s.start, s.end, s.label.value) for s in rec.spans}
            pk = {(s.start, s.end, s.label.value) for s in pred.spans}
            for k in gk & pk:
                tp[k[2]] += 1
            for k in gk - pk:
                fn[k[2]] += 1
            for k in pk - gk:
                fp[k[2]] += 1
            print(f"  [{mode_name} {i}/{len(sample)}] {len(pred.spans)} spans "
                  f"(gold {len(rec.spans)})", flush=True)

        results[mode_name] = {
            "records_scored": n_pred,
            "tp": dict(tp), "fn": dict(fn), "fp": dict(fp),
        }
        print()

    print("=" * 68)
    print(f"{'type':<16}{'gold':>6}{'plain R':>10}{'thorough R':>13}{'delta':>9}")
    verdict_rows = {}
    for t in WATCHED:
        g = gold_counts.get(t.value, 0)
        if not g:
            continue
        r_plain = results["plain"]["tp"].get(t.value, 0) / g
        r_teach = results["teacher_mode"]["tp"].get(t.value, 0) / g
        verdict_rows[t.value] = {"gold": g, "plain_recall": round(r_plain, 4),
                                 "teacher_mode_recall": round(r_teach, 4)}
        print(f"{t.value:<16}{g:>6}{r_plain:>10.3f}{r_teach:>13.3f}{r_teach - r_plain:>+9.3f}")

    def micro(res):
        TP = sum(res["tp"].values())
        FP = sum(res["fp"].values())
        FN = sum(res["fn"].values())
        return 2 * TP / max(1, 2 * TP + FP + FN)

    print(f"\nmicro-F1  plain={micro(results['plain']):.4f}  "
          f"thorough={micro(results['teacher_mode']):.4f}")
    print(f"tokens spent: {tokens}")

    payload = {
        "split": str(args.split), "n_records": len(sample),
        "gold_instances": dict(gold_counts), "per_type": verdict_rows,
        "micro_f1": {k: round(micro(v), 4) for k, v in results.items()},
        "raw": results, "tokens": tokens, "model": args.model,
        "temperature": args.temperature, "seed": args.seed,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
