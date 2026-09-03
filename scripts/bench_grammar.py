#!/usr/bin/env python3
"""A/B/C(/D) decoding changes on one server and the full frozen test set.

Two experiments are supported:

``spacing`` reproduces the original unconstrained/permissive/exact-spacing
grammar comparison. ``compact`` isolates the two cheap output-shortening levers:

* compact grammar + original trained prompt (force an unseen shape);
* compact prompt + unconstrained decode (ask for the unseen shape).
* compact prompt + grammar (ask for the shape and make it structural).

All arms run against one server under identical flags. Repetitions are
round-robin and reverse arm order on alternating passes, so machine drift does
not always favour the same arm. Every pass records load average and swap; the
best records/s pass supplies the saved predictions and headline row, while all
passes remain in the artifact.

Usage:
    scripts/bench_grammar.py --base-url http://localhost:8080/v1 \
        --experiment compact --concurrency 32 --repeat 2 \
        --out reports/bench/compact_abc.json
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
from forge.grammar import compact_spans_gbnf, spans_gbnf
from forge.inference import build_messages, parse_response
from forge.schema import PIIRecord

try:
    from scripts.bench_serving import _contention, usd_per_1k
except ModuleNotFoundError:
    from bench_serving import _contention, usd_per_1k


def _load(path: Path) -> list[PIIRecord]:
    return [
        PIIRecord.model_validate_json(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _call(
    base_url: str,
    rec: PIIRecord,
    grammar: str | None,
    max_tokens: int,
    compact_prompt: bool,
    retries: int = 3,
):
    body = {
        "model": "m",
        "messages": build_messages(rec.text, compact=compact_prompt),
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


def run_arm(name, grammar, compact_prompt, gold, args, pass_number):
    contention_at_start = _contention()
    t0 = time.monotonic()
    with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
        outs = list(
            ex.map(
                lambda r: _call(
                    args.base_url,
                    r,
                    grammar,
                    args.max_tokens,
                    compact_prompt,
                ),
                gold,
            )
        )
    wall = time.monotonic() - t0

    preds, valid, toks = [], 0, 0
    for rec, (raw, n_tok) in zip(gold, outs):
        toks += n_tok
        pred, ok = parse_response(rec.id, rec.text, raw, split="test")
        valid += ok
        preds.append(pred or PIIRecord(id=rec.id, text=rec.text, spans=[], split="test"))

    report = evaluate(gold, preds, schema_valid_count=valid)
    interval = ci.micro_f1_ci(gold, preds, n_resamples=args.ci_resamples)
    sustained = wall / len(gold)
    cost = usd_per_1k(sustained)
    return {
        "name": name,
        "pass": pass_number,
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
        "s_per_record": sustained,
        "records_per_s": len(gold) / wall,
        "output_tok_s_aggregate": toks / wall,
        "usd_per_1k": cost["usd_per_1k"],
        "contention_at_start": contention_at_start,
        "contention_at_end": _contention(),
    }, preds


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base-url", default="http://localhost:8080/v1")
    ap.add_argument("--gold", type=Path, default=Path("data/gold/test.jsonl"))
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--max-tokens", type=int, default=1024)
    ap.add_argument("--ci-resamples", type=int, default=10_000)
    ap.add_argument("--repeat", type=int, default=1)
    ap.add_argument("--experiment", choices=["spacing", "compact"], default="spacing")
    ap.add_argument("--server-cmd", default=None)
    ap.add_argument("--llama-cpp-commit", default=None)
    ap.add_argument("--out", type=Path, default=Path("reports/bench/grammar_abc.json"))
    ap.add_argument("--save-predictions", type=Path, default=None)
    args = ap.parse_args()

    gold = _load(args.gold)
    if args.experiment == "compact":
        arms = [
            ("verbose_baseline", None, False),
            ("compact_grammar", compact_spans_gbnf(), False),
            ("compact_prompt", None, True),
            ("compact_prompt_grammar", compact_spans_gbnf(), True),
        ]
    else:
        arms = [
            ("unconstrained", None, False),
            ("grammar_permissive", spans_gbnf(exact_spacing=False), False),
            ("grammar_exact", spans_gbnf(exact_spacing=True), False),
        ]

    passes: dict[str, list[dict]] = {name: [] for name, _, _ in arms}
    saved: dict[str, list[PIIRecord]] = {}
    best: dict[str, dict] = {}
    for pass_idx in range(args.repeat):
        ordered = arms if pass_idx % 2 == 0 else list(reversed(arms))
        for name, grammar, compact_prompt in ordered:
            print(f"running {name} pass {pass_idx + 1}/{args.repeat} ...", flush=True)
            row, preds = run_arm(
                name,
                grammar,
                compact_prompt,
                gold,
                args,
                pass_idx + 1,
            )
            passes[name].append(row)
            if name not in best or row["records_per_s"] > best[name]["records_per_s"]:
                best[name] = row
                saved[name] = preds

    rows = []
    for name, _, _ in arms:
        row = dict(best[name])
        row["passes"] = passes[name]
        rows.append(row)

    base = rows[0]
    print()
    print(
        f"{'arm':<20}{'F1':>8}{'precision':>11}{'recall':>9}{'schema':>11}"
        f"{'tok/rec':>9}{'tok/s':>9}{'s/rec':>9}{'$/1k':>10}{'dF1':>9}"
    )
    print("-" * 105)
    for r in rows:
        d = r["micro_f1"] - base["micro_f1"]
        print(
            f"{r['name']:<20}{r['micro_f1']:>8.4f}{r['micro_precision']:>11.4f}"
            f"{r['micro_recall']:>9.4f}{r['schema_valid']:>7}/{r['n']}"
            f"{r['tokens_per_record']:>9.1f}{r['output_tok_s_aggregate']:>9.1f}"
            f"{r['s_per_record']:>9.4f}{r['usd_per_1k']:>10.5f}{d:>+9.4f}"
        )
        starts = [p["contention_at_start"]["loadavg_1m"] for p in r["passes"]]
        ends = [p["contention_at_end"]["loadavg_1m"] for p in r["passes"]]
        rates = [round(p["output_tok_s_aggregate"], 2) for p in r["passes"]]
        print(f"  passes tok/s={rates}; loadavg_1m start={starts} end={ends}")

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
    artifact = {
        "experiment": args.experiment,
        "repeat_passes": args.repeat,
        "selection": "best records/s pass per arm; every pass retained below",
        "server_cmd": args.server_cmd,
        "llama_cpp_commit": args.llama_cpp_commit,
        "arms": rows,
    }
    args.out.write_text(json.dumps(artifact, indent=2) + "\n", encoding="utf-8")
    print(f"\nwrote {args.out}")

    if args.save_predictions:
        for name, preds in saved.items():
            p = args.save_predictions.with_name(f"{args.save_predictions.stem}_{name}.jsonl")
            p.write_text("".join(x.model_dump_json() + "\n" for x in preds), encoding="utf-8")
            meta = p.with_suffix(".meta.json")
            meta.write_text(json.dumps(best[name], indent=2) + "\n", encoding="utf-8")
            print(f"wrote {p} (+ .meta.json)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
