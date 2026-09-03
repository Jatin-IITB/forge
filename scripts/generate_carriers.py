#!/usr/bin/env python3
"""Generate carrier shapes with the teacher (ADR 0015, stage 1 of 2).

The student's worst types are the ones a fixed template pool cannot teach:
`STREET_ADDRESS` F1 0.0923, `PERSON` 0.5000 with 110 false negatives. Both eval
splits and every training record so far come from the same 110 hand-written
templates in `scripts/build_gold.py` — 109 distinct shapes in `test.jsonl`, 208
in `train_v2.jsonl`. A template that always puts the address in the same
syntactic slot cannot teach where an address ends.

So the teacher writes the shapes. It is asked for *skeletons* with `{{TYPE}}`
placeholders, never for labels, which keeps this stage cheap (~10 shapes per
call) and keeps the resulting spans exact by construction once filled.

Provenance: the shapes are project-owned synthetic text produced by an
Apache-2.0 open-weight model, so they clear the ADR 0003 litmus test without a
per-corpus licence review. That is the reason this design was preferred to
seeding from a public corpus.

Every shape is checked against the shapes of all four splits (train, dev, val,
test) and dropped on an exact match — stricter than the contract's
carrier-sentence rule, and free.

    python scripts/generate_carriers.py --api-key-env CEREBRAS_API_KEY \
        --model gpt-oss-120b --base-url https://api.cerebras.ai/v1 \
        --target 400 --rpm 5 --resume
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

from forge.carriers import (
    Carrier,
    CarrierError,
    known_given_names,
    shape_of,
    validate_shape,
)
from forge.schema import PIIRecord, PIIType
from forge.teacher_client import ThrottledTeacher

try:
    from openai import OpenAI
except ImportError:  # pragma: no cover - import guard
    print("Install the openai package: pip install 'openai>=1.0'", file=sys.stderr)
    sys.exit(1)


# Registers deliberately spread beyond the "one tidy sentence" shape of the
# existing template pool. Multi-sentence and multi-line registers are the ones
# that produce entities in unfamiliar syntactic positions.
REGISTERS = [
    "a customer support ticket body, 2-3 sentences, slightly informal",
    "an internal chat message thread of 2-4 short turns, with speaker prefixes",
    "an email body with a greeting and a sign-off",
    "a filled-in web form rendered as 'Field: value' lines",
    "an application or server log excerpt with a timestamp prefix",
    "a clinical or appointment note written by staff, terse and abbreviated",
    "an invoice or order confirmation with line items",
    "an SMS or push notification, under 25 words",
    "an incident report paragraph written in the past tense",
    "a handover note between two shifts, using fragments not full sentences",
    "a CRM comment field, run-on and lightly punctuated",
    "a bug report with reproduction steps as a numbered list",
    "a bank or insurance letter paragraph, formal register",
    "a job application or HR onboarding note",
    "a delivery or logistics status update",
    "a legal or compliance memo sentence with a parenthetical aside",
    "a social media style post or comment",
    "a voicemail transcription with disfluencies",
    "a spreadsheet row flattened to prose",
    "a customer review mentioning the reviewer's own details",
]

TYPE_MENU = ", ".join(t.value for t in PIIType)

SYSTEM = """\
You write TEMPLATE SKELETONS for a PII-detection training corpus.

A skeleton is ordinary English text in which every piece of personal information \
has been replaced by a typed placeholder of the form {{TYPE}}.

Valid placeholder types (use these EXACTLY, uppercase, double braces):
""" + TYPE_MENU + """

Hard rules:
- Output ONLY a JSON object: {"shapes": ["...", "..."]}
- NEVER write a real or invented PII value. Names, addresses, numbers, emails, \
handles and ages must ALWAYS be placeholders, never literal text.
- This includes SPEAKER LABELS and SIGNATURES. Write "{{PERSON}}: message" for a \
chat turn, never "Alice: message". Write "Thanks, {{PERSON}}" never "Thanks, Sam". \
A literal name anywhere in the text makes the example unusable.
- Never place two placeholders next to each other without intervening words or \
punctuation.
- Vary where the placeholder sits in the sentence: subject, object, mid-clause, \
inside a parenthetical, at the very start, at the very end, inside a list.
- Vary sentence length and structure. Do not reuse a phrasing you already used.
- Plain text only. No markdown, no code fences.
"""

USER = """\
Write {n} DIFFERENT skeletons in this register: {register}

Each skeleton must contain between {lo} and {hi} placeholders, and must include \
at least one of: {emphasis}.

{extra}

Return: {{"shapes": ["skeleton one", "skeleton two", ...]}}
"""

EXTRA_MULTI = (
    "Make at least half of them mention two or more DIFFERENT people, so the text "
    "contains several PERSON placeholders in different grammatical roles."
)
EXTRA_ADDRESS = (
    "Where you use {{STREET_ADDRESS}}, surround it with text that makes its end "
    "boundary non-obvious: a following clause, a nearby {{LOCATION}}, or a "
    "parenthetical. Treat a full mailing address as ONE {{STREET_ADDRESS}} "
    "placeholder and a bare city used as context as {{LOCATION}}."
)
EXTRA_PLAIN = "Keep the surrounding prose natural and specific, not generic filler."


def _split_shapes(paths: list[Path]) -> set[str]:
    """Shapes already present in an evaluation or training split."""
    shapes: set[str] = set()
    for p in paths:
        if not p.exists():
            continue
        for line in p.read_text(encoding="utf-8").splitlines():
            if line.strip():
                shapes.add(shape_of(PIIRecord.model_validate_json(line)))
    return shapes


def _load_existing(path: Path) -> list[Carrier]:
    if not path.exists():
        return []
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            d = json.loads(line)
            out.append(Carrier(shape=d["shape"], source=d["source"], register=d["register"]))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--output", type=Path, default=Path("data/carriers_v3.jsonl"))
    ap.add_argument("--model", default="gpt-oss-120b")
    ap.add_argument("--base-url", default="https://api.cerebras.ai/v1")
    ap.add_argument("--api-key-env", default="CEREBRAS_API_KEY", metavar="VAR",
                    help="Read the API key from this environment variable (never logged)")
    ap.add_argument("--target", type=int, default=400, help="Distinct shapes to collect")
    ap.add_argument("--per-call", type=int, default=10)
    ap.add_argument("--max-calls", type=int, default=200)
    ap.add_argument("--temperature", type=float, default=1.0,
                    help="High by default: diversity is the entire product of this stage")
    ap.add_argument("--max-tokens", type=int, default=2048)
    ap.add_argument("--timeout", type=float, default=120.0)
    ap.add_argument("--rpm", type=float, default=5.0)
    ap.add_argument("--reasoning-effort", default="low", choices=["low", "medium", "high"])
    ap.add_argument("--seed", type=int, default=7717)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--splits", type=Path, nargs="+", default=[
        Path("data/train.jsonl"), Path("data/train_v2.jsonl"),
        Path("data/gold/dev.jsonl"), Path("data/gold/val.jsonl"),
        Path("data/gold/test.jsonl"),
    ])
    args = ap.parse_args()

    key = os.environ.get(args.api_key_env)
    if not key:
        print(f"Environment variable {args.api_key_env} is not set.", file=sys.stderr)
        return 1

    rng = random.Random(args.seed)
    forbidden = _split_shapes(args.splits)
    print(f"shapes already used by existing splits: {len(forbidden)}")
    known_names = known_given_names()
    print(f"given-name screen: {len(known_names)} names from Faker en_US/en_GB/en_IN")

    existing = _load_existing(args.output) if args.resume else []
    seen = {c.normalised() for c in existing}
    print(f"resuming with {len(existing)} shapes" if existing else "starting fresh")

    teacher = ThrottledTeacher(
        # max_retries=0: ThrottledTeacher is the only retrier, so the SDK cannot
        # silently spend requests against an exhausted quota before we see it.
        OpenAI(base_url=args.base_url, api_key=key, max_retries=0), rpm=args.rpm,
        on_retry=lambda a, e, d: print(f"    {type(e).__name__}, retry {a} in {d:.0f}s",
                                       flush=True),
    )
    out = args.output.open("a" if existing else "w", encoding="utf-8")

    rejects: Counter[str] = Counter()
    calls = 0

    while len(seen) < args.target and calls < args.max_calls:
        register = REGISTERS[calls % len(REGISTERS)]
        lo, hi = rng.choice([(1, 2), (2, 4), (3, 5), (4, 7), (0, 0)])
        if (lo, hi) == (0, 0):
            # Negative controls: prose that looks like it should contain PII but
            # does not. train_v2 has almost none, and precision is the half of
            # F1 that run_002 was not short of - keep the ratio small.
            lo, hi = 0, 0
            emphasis = "nothing"
            extra = ("These skeletons must contain ZERO placeholders: realistic business "
                     "prose with no personal information at all. Mention companies, "
                     "products, dates or amounts instead.")
        else:
            focus = rng.choice([
                ["PERSON", "LOCATION"],
                ["STREET_ADDRESS", "LOCATION"],
                ["PERSON", "AGE", "DATE_OF_BIRTH"],
                ["USERNAME", "EMAIL", "URL"],
                ["PERSON", "STREET_ADDRESS"],
                ["PERSON", "USERNAME", "IP_ADDRESS"],
                ["LOCATION", "PHONE", "PERSON"],
                ["PERSON", "PERSON"],
                ["STREET_ADDRESS", "PHONE"],
                ["AGE", "PERSON", "LOCATION"],
                # Validator-owned types appear, but at a deliberately low rate:
                # they are solved by forge/validators.py at 1.0000 recall.
                ["PERSON", "CREDIT_CARD"],
                ["PERSON", "AADHAAR", "PAN"],
            ])
            emphasis = ", ".join("{{" + f + "}}" for f in focus)
            extra = rng.choice([EXTRA_MULTI, EXTRA_ADDRESS, EXTRA_PLAIN, EXTRA_PLAIN])

        prompt = USER.format(n=args.per_call, register=register, lo=lo, hi=hi,
                             emphasis=emphasis, extra=extra)
        try:
            resp, dt = teacher.complete(
                model=args.model,
                messages=[{"role": "system", "content": SYSTEM},
                          {"role": "user", "content": prompt}],
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                timeout=args.timeout,
                extra_body={"reasoning_effort": args.reasoning_effort},
            )
            calls += 1
            raw = resp.choices[0].message.content or ""
        except Exception as exc:  # noqa: BLE001
            calls += 1
            rejects["api_error"] += 1
            print(f"  [call {calls}] gave up: {type(exc).__name__}", flush=True)
            continue

        try:
            data = json.loads(raw[raw.index("{"): raw.rindex("}") + 1])
            shapes = data["shapes"]
        except (ValueError, KeyError):
            rejects["unparseable_response"] += 1
            print(f"  [call {calls}] unparseable response", flush=True)
            continue

        added = 0
        for s in shapes:
            try:
                carrier = validate_shape(s, known_names=known_names)
            except CarrierError as exc:
                rejects[str(exc).split("(")[0].strip()] += 1
                continue
            norm = carrier.normalised()
            if norm in seen:
                rejects["duplicate_shape"] += 1
                continue
            if norm in forbidden:
                rejects["collides_with_eval_split"] += 1
                continue
            seen.add(norm)
            out.write(json.dumps({
                "shape": carrier.shape,
                "source": f"teacher:{args.model}",
                "register": register,
            }) + "\n")
            added += 1
        out.flush()
        print(f"  [call {calls}] {register[:38]:<38} +{added:>2}/{len(shapes):>2} "
              f"-> {len(seen)}/{args.target} ({dt:.1f}s)", flush=True)

    out.close()
    print("\n--- carrier generation ---")
    print(f"distinct shapes: {len(seen)}  (target {args.target})")
    print(f"teacher: {json.dumps(teacher.stats.summary())}")
    print(f"rejects: {dict(rejects)}")

    meta = {
        "distinct_shapes": len(seen),
        "target": args.target,
        "api_calls": calls,
        "total_tokens": teacher.stats.total_tokens,
        "teacher_stats": teacher.stats.summary(),
        "rejects": dict(rejects),
        "model": args.model,
        "temperature": args.temperature,
        "seed": args.seed,
        "registers": len(REGISTERS),
        "checked_against_splits": [str(p) for p in args.splits],
    }
    args.output.with_suffix(".meta.json").write_text(
        json.dumps(meta, indent=2) + "\n", encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
