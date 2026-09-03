#!/usr/bin/env python3
"""A/B/C the decoding constraint: unconstrained vs permissive vs strict grammar.

`docs/DESIGN.md` has always named constrained decoding as the mechanism for the
reliability gate. It was never built, and G2 failed at 0.9974 (384/385) — one
malformed response, against a threshold with zero margin at n=385.

The permissive grammar fixes G2 and costs **-0.0162 micro-F1**, larger than the
-0.0151 that disqualified Q4_K_M under the <=0.01 exit gate. This script exists
to test *why*, and whether the cost is avoidable.

**The hypothesis.** Grammar-constrained sampling filters at the *token* level.
With optional whitespace the decoder will accept ``{`` and ``"spans"`` as two
tokens, so when the model's highest-probability continuation is the single
merged token ``{"spans":`` — what it saw in every training target — the grammar
can push it onto a different tokenization of the same string. The output stays
valid; the model leaves the path it was trained on. If that is the mechanism,
pinning the grammar to the exact byte sequence in the system prompt should
recover the loss.

All three arms run against **one server process under identical flags**, because
batched decoding on Metal is not bit-deterministic and comparing arms measured
under different `-np`/`-c` settings would confound the constraint with the
serving config.

Usage:
    scripts/bench_grammar.py --base-url http://localhost:8080/v1 \
        --out reports/bench/grammar_abc.json
"""

from __future__ import annotations

import argparse
import json
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from forge import ci
from forge.eval import evaluate
from forge.grammar import spans_gbnf
from forge.inference import build_messages, parse_response
from forge.schema import PIIRecord


def _load(path: Path) -> list[PIIRecord]:
    return [
        PIIRecord.model_validate_json(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _call(base_url: str, rec: PIIRecord, grammar: str | None, max_tokens: int, retries: int = 3):
    body = {
        "model": "m",
        "messages": build_messages(rec.text),
        "max_tokens": max_tokens,
        "temperature": 0.0,
    }
    if grammar:
        body["grammar"] = grammar
    req = urllib.request.Request(
        f"{base_url}/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    last: Exception | None = None
    for _ in range(retries):
        try:
            d = json.loads(urllib.request.urlopen(req, timeout=300).read())
            return d["choices"][0]["message"]["content"], d["usage"]["completion_tokens"]
        except (urllib.error.URLError, TimeoutError, KeyError, ValueError) as e:
            last = e
            time.sleep(2)
    raise RuntimeError(f"{rec.id}: {last}")


def run_arm(name, grammar, gold, args):
    t0 = time.monotonic()
    with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
        outs = list(ex.map(lambda r: _call(args.base_url, r, grammar, args.max_tokens), gold))
    wall = time.monotonic() - t0

    preds, valid, toks = [], 0, 0
    for rec, (raw, n_tok) in zip(gold, outs):
        toks += n_tok
        pred, ok = parse_response(rec.id, rec.text, raw, split="test")
        valid += ok
        preds.append(pred or PIIRecord(id=rec.id, text=rec.text, spans=[], split="test"))

    report = evaluate(gold, preds, schema_valid_count=valid)
    interval = ci.micro_f1_ci(gold, preds, n_resamples=args.ci_resamples)
    return {
        "name": name,
        "micro_f1": report.micro_f1,
        "ci_lo": interval.lo,
        "ci_hi": interval.hi,
        "micro_precision": report.micro_precision,
        "micro_recall": report.micro_recall,
        "schema_valid": valid,
        "n": len(gold),
        "schema_rate": valid / len(gold),
        "tokens_per_record": toks / len(gold),
        "wall_s": wall,
        "s_per_record": wall / len(gold),
    }, preds


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base-url", default="http://localhost:8080/v1")
    ap.add_argument("--gold", type=Path, default=Path("data/gold/test.jsonl"))
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--max-tokens", type=int, default=1024)
    ap.add_argument("--ci-resamples", type=int, default=10_000)
    ap.add_argument("--out", type=Path, default=Path("reports/bench/grammar_abc.json"))
    ap.add_argument("--save-predictions", type=Path, default=None)
    args = ap.parse_args()

    gold = _load(args.gold)
    arms = [
        ("unconstrained", None),
        ("grammar_permissive", spans_gbnf(exact_spacing=False)),
        ("grammar_exact", spans_gbnf(exact_spacing=True)),
    ]

    rows, saved = [], {}
    for name, grammar in arms:
        print(f"running {name} ...", flush=True)
        row, preds = run_arm(name, grammar, gold, args)
        rows.append(row)
        saved[name] = preds

    base = rows[0]
    print()
    print(f"{'arm':<20}{'micro-F1':>10}{'95% CI':>20}{'schema':>11}{'tok/rec':>9}{'dF1':>9}")
    print("-" * 82)
    for r in rows:
        d = r["micro_f1"] - base["micro_f1"]
        print(
            f"{r['name']:<20}{r['micro_f1']:>10.4f}  [{r['ci_lo']:.4f},{r['ci_hi']:.4f}]"
            f"{r['schema_valid']:>7}/{r['n']}{r['tokens_per_record']:>9.1f}{d:>+9.4f}"
        )

    print()
    print("G2 (schema >= 0.999):")
    for r in rows:
        print(f"  {r['name']:<20}{r['schema_rate']:.4f}  {'PASS' if r['schema_rate'] >= 0.999 else 'FAIL'}")
    print()
    print("Exit gate on quality (|dF1| <= 0.01, the rule that rejected Q4_K_M):")
    for r in rows[1:]:
        d = r["micro_f1"] - base["micro_f1"]
        print(f"  {r['name']:<20}{d:>+8.4f}  {'within' if abs(d) <= 0.01 else 'OVER'}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"arms": rows}, indent=2) + "\n", encoding="utf-8")
    print(f"\nwrote {args.out}")

    if args.save_predictions:
        for name, preds in saved.items():
            p = args.save_predictions.with_name(f"{args.save_predictions.stem}_{name}.jsonl")
            p.write_text("".join(x.model_dump_json() + "\n" for x in preds), encoding="utf-8")
            print(f"wrote {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
