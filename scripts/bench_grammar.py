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
from forge.grammar import compact_spans_gbnf, compact_spans_json_schema, spans_gbnf
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
    json_schema: dict | None,
    retry_constraint: str,
    max_tokens: int,
    compact_prompt: bool,
    retries: int = 3,
):
    base_body = {
        "model": "m",
        "messages": build_messages(rec.text, compact=compact_prompt),
        "max_tokens": max_tokens,
        "temperature": 0.0,
    }

    def request(*, grammar_value: str | None = None, schema_value: dict | None = None):
        body = dict(base_body)
        if grammar_value:
            body["grammar"] = grammar_value
        if schema_value:
            body["json_schema"] = schema_value
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

    raw, n_tok = request(grammar_value=grammar, schema_value=json_schema)
    attempt_details = [
        {
            "constraint": "json-schema" if json_schema else "gbnf" if grammar else "none",
            "raw": raw,
            "completion_tokens": n_tok,
        }
    ]
    _, valid = parse_response(rec.id, rec.text, raw, split=rec.split)
    attempts = 1
    if not valid and retry_constraint != "none":
        raw, retry_tok = request(
            grammar_value=compact_spans_gbnf() if retry_constraint == "gbnf" else None,
            schema_value=(
                compact_spans_json_schema()
                if retry_constraint == "json-schema"
                else None
            ),
        )
        attempt_details.append(
            {
                "constraint": retry_constraint,
                "raw": raw,
                "completion_tokens": retry_tok,
            }
        )
        n_tok += retry_tok
        attempts += 1
    return raw, n_tok, attempts, attempt_details


def run_arm(
    name,
    grammar,
    json_schema,
    retry_constraint,
    compact_prompt,
    gold,
    args,
    pass_number,
):
    contention_at_start = _contention()
    t0 = time.monotonic()
    with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
        outs = list(
            ex.map(
                lambda r: _call(
                    args.base_url,
                    r,
                    grammar,
                    json_schema,
                    retry_constraint,
                    args.max_tokens,
                    compact_prompt,
                ),
                gold,
            )
        )
    wall = time.monotonic() - t0

    preds, valid, toks, request_attempts = [], 0, 0, 0
    retry_details = []
    for rec, (raw, n_tok, attempts, attempt_details) in zip(gold, outs):
        toks += n_tok
        request_attempts += attempts
        if attempts > 1:
            retry_details.append({"rec_id": rec.id, "attempts": attempt_details})
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
        "records_retried": request_attempts - len(gold),
        "request_attempts": request_attempts,
        "retry_details": retry_details,
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
            ("verbose_baseline", None, None, "none", False),
            ("compact_prompt_grammar", compact_spans_gbnf(), None, "none", True),
            ("compact_prompt_retry", None, None, "gbnf", True),
            (
                "compact_prompt_json_schema",
                None,
                compact_spans_json_schema(),
                "none",
                True,
            ),
            ("compact_prompt_json_retry", None, None, "json-schema", True),
        ]
    else:
        arms = [
            ("unconstrained", None, None, "none", False),
            ("grammar_permissive", spans_gbnf(exact_spacing=False), None, "none", False),
            ("grammar_exact", spans_gbnf(exact_spacing=True), None, "none", False),
        ]

    passes: dict[str, list[dict]] = {name: [] for name, *_ in arms}
    saved: dict[str, list[PIIRecord]] = {}
    best: dict[str, dict] = {}
    for pass_idx in range(args.repeat):
        ordered = arms if pass_idx % 2 == 0 else list(reversed(arms))
        for name, grammar, json_schema, retry_constraint, compact_prompt in ordered:
            print(f"running {name} pass {pass_idx + 1}/{args.repeat} ...", flush=True)
            row, preds = run_arm(
                name,
                grammar,
                json_schema,
                retry_constraint,
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
    for name, *_ in arms:
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
