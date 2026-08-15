#!/usr/bin/env python3
"""Data engine — generate verified training data from a teacher model.

For each seed text (from a seed corpus or generated), queries the teacher
k times, runs the verification gate, deduplicates, and writes train.jsonl.

Usage:
    # Generate from gold dev set as seed texts
    python scripts/run_data_engine.py \
        --seed-texts data/gold/dev.jsonl \
        --gold data/gold/test.jsonl \
        --output data/train.jsonl \
        --model Qwen/Qwen2.5-32B-Instruct \
        --base-url http://localhost:8000/v1 \
        --k 3

    # Generate from a custom seed corpus
    python scripts/run_data_engine.py \
        --seed-texts data/seeds/corpus.jsonl \
        --gold data/gold/test.jsonl data/gold/dev.jsonl \
        --output data/train.jsonl \
        --model Qwen/Qwen2.5-32B-Instruct
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

from forge.dedup import dedup_training_data
from forge.inference import build_messages, parse_response
from forge.schema import PIIRecord
from forge.verify import VerificationStats, update_stats, verify_record

try:
    from openai import OpenAI
except ImportError:
    print("Install the openai package: pip install 'openai>=1.0'", file=sys.stderr)
    sys.exit(1)


def load_records(path: Path) -> list[PIIRecord]:
    records = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            records.append(PIIRecord.model_validate_json(line))
    return records


def load_seed_texts(path: Path) -> list[tuple[str, str]]:
    """Load seed texts. Returns list of (id, text) pairs."""
    records = load_records(path)
    return [(r.id, r.text) for r in records]


def query_teacher(
    client: OpenAI,
    text: str,
    model: str,
    max_tokens: int,
    temperature: float,
    timeout: float,
    reasoning_effort: str | None = None,
) -> tuple[PIIRecord | None, bool, dict]:
    """Query the teacher once and parse the response.

    Returns (record, schema_valid, trace) where trace captures the raw
    response and any reasoning content for diagnostics.
    """
    messages = build_messages(text, teacher_mode=True)
    resp = client.chat.completions.create(
        model=model,
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature,
        timeout=timeout,
        extra_body={"reasoning_effort": reasoning_effort} if reasoning_effort else None,
    )
    msg = resp.choices[0].message
    raw = msg.content or ""
    reasoning = getattr(msg, "reasoning_content", None) or getattr(msg, "reasoning", None) or ""
    rec, valid = parse_response("tmp", text, raw, split="train")
    trace = {"raw": raw, "valid": valid}
    if reasoning:
        trace["reasoning"] = reasoning
    return rec, valid, trace


def main() -> int:
    ap = argparse.ArgumentParser(description="Generate verified training data.")
    ap.add_argument("--seed-texts", type=Path, required=True, help="JSONL with seed texts")
    ap.add_argument("--gold", type=Path, nargs="+", required=True, help="Gold JSONL(s) for leakage check")
    ap.add_argument("--output", type=Path, required=True, help="Output train.jsonl")
    ap.add_argument("--model", required=True, help="Teacher model name")
    ap.add_argument("--base-url", default="http://localhost:8000/v1", help="API base URL")
    ap.add_argument("--api-key", default="not-needed", help="API key")
    ap.add_argument("--k", type=int, default=3, help="Number of teacher samples for consistency")
    ap.add_argument("--max-tokens", type=int, default=1024, help="Max output tokens")
    ap.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature for diversity")
    ap.add_argument("--timeout", type=float, default=60.0, help="Per-request timeout (seconds)")
    ap.add_argument("--consistency-threshold", type=float, default=0.5)
    ap.add_argument("--min-agreement", type=float, default=0.6)
    ap.add_argument("--dedup-threshold", type=float, default=0.85, help="Jaccard threshold for near-dedup")
    ap.add_argument("--limit", type=int, default=None, help="Process only first N seed texts")
    ap.add_argument("--resume", action="store_true", help="Resume from existing output, skip already-processed seeds")
    ap.add_argument(
        "--api-key-env", default=None, metavar="VAR",
        help="Read the API key from this environment variable (never logged)",
    )
    ap.add_argument(
        "--rpm", type=float, default=None,
        help="Throttle: max requests per minute (free tiers cap this)",
    )
    ap.add_argument(
        "--reasoning-effort", default=None, choices=["low", "medium", "high"],
        help="For reasoning teachers (gpt-oss): passed as extra_body",
    )
    args = ap.parse_args()

    if args.api_key_env:
        key = os.environ.get(args.api_key_env)
        if not key:
            print(f"Environment variable {args.api_key_env} is not set.", file=sys.stderr)
            return 1
        args.api_key = key

    seeds = load_seed_texts(args.seed_texts)
    if args.limit:
        seeds = seeds[:args.limit]

    resumed_texts: set[str] = set()
    resumed_records: list[PIIRecord] = []
    if args.resume and args.output.exists():
        resumed_records = load_records(args.output)
        resumed_texts = {r.text for r in resumed_records}
        print(f"resuming: {len(resumed_records)} records already in output")

    gold_records: list[PIIRecord] = []
    for gp in args.gold:
        gold_records.extend(load_records(gp))

    client = OpenAI(base_url=args.base_url, api_key=args.api_key)
    stats = VerificationStats()
    verified_records: list[PIIRecord] = list(resumed_records)
    total_api_calls = 0
    total_latency = 0.0

    log_path = args.output.with_suffix(".log.jsonl")
    log_file = log_path.open("a", encoding="utf-8")

    seeds_to_process = [(sid, txt) for sid, txt in seeds if txt not in resumed_texts]
    print(f"seed texts: {len(seeds)} total, {len(seeds_to_process)} to process, k={args.k}, teacher={args.model}")
    print(f"gold records for leakage check: {len(gold_records)}")
    print(f"teacher log: {log_path}")
    print()

    # Free-tier teachers cap requests per minute; k samples per seed multiplies
    # the call count, so the throttle is enforced per call, not per seed.
    min_interval = 60.0 / args.rpm if args.rpm else 0.0
    last_call = 0.0

    for i, (seed_id, text) in enumerate(seeds_to_process, len(resumed_records) + 1):
        samples = []
        valid_flags = []
        errors = 0
        traces = []

        for sample_j in range(args.k):
            try:
                if min_interval:
                    wait = min_interval - (time.monotonic() - last_call)
                    if wait > 0:
                        time.sleep(wait)
                last_call = time.monotonic()
                t0 = time.monotonic()
                rec, valid, trace = query_teacher(
                    client, text, args.model,
                    args.max_tokens, args.temperature, args.timeout,
                    reasoning_effort=args.reasoning_effort,
                )
                trace["latency_s"] = round(time.monotonic() - t0, 2)
                total_latency += trace["latency_s"]
                total_api_calls += 1
                samples.append(rec)
                valid_flags.append(valid)
                traces.append(trace)
            except Exception as exc:  # noqa: BLE001
                errors += 1
                total_api_calls += 1
                traces.append({"error": str(exc), "valid": False})

        record_id = f"train_{i:05d}"
        result = verify_record(
            record_id=record_id,
            text=text,
            samples=samples,
            schema_valid_flags=valid_flags,
            consistency_threshold=args.consistency_threshold,
            min_agreement=args.min_agreement,
            split="train",
        )
        update_stats(stats, result)

        if result.accepted:
            verified_records.append(result.record)

        status = "ACCEPT" if result.accepted else "REJECT"
        n_spans = len(result.record.spans)
        reasons = "; ".join(r.value for r in result.reject_reasons) if result.reject_reasons else ""
        errs = f", {errors} api_errors" if errors else ""
        total_target = len(seeds_to_process) + len(resumed_records)
        print(f"  [{i}/{total_target}] {record_id}: {status} ({n_spans} spans, agreement={result.agreement_ratio:.2f}{errs}){' — ' + reasons if reasons else ''}")

        log_entry = {
            "id": record_id,
            "seed_id": seed_id,
            "accepted": result.accepted,
            "n_spans": n_spans,
            "agreement": result.agreement_ratio,
            "reject_reasons": [r.value for r in result.reject_reasons],
            "traces": traces,
        }
        log_file.write(json.dumps(log_entry) + "\n")
        log_file.flush()

    log_file.close()
    print("\n--- Verification ---")
    print(f"total: {stats.total}, accepted: {stats.accepted} ({stats.accept_rate:.1%})")
    print(f"rejected: consistency={stats.rejected_consistency}, schema={stats.rejected_schema}, empty={stats.rejected_empty}")

    dedup_result = dedup_training_data(
        verified_records,
        gold=gold_records,
        near_threshold=args.dedup_threshold,
    )

    print("\n--- Dedup ---")
    print(f"kept: {len(dedup_result.kept)}, removed: exact={dedup_result.removed_exact}, near={dedup_result.removed_near}, leakage={dedup_result.removed_leakage}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        for rec in dedup_result.kept:
            f.write(rec.model_dump_json() + "\n")

    meta = {
        "verification": stats.summary(),
        "dedup": dedup_result.summary(),
        "total_api_calls": total_api_calls,
        "avg_latency_s": total_latency / total_api_calls if total_api_calls else 0,
        "model": args.model,
        "k": args.k,
        "temperature": args.temperature,
        "consistency_threshold": args.consistency_threshold,
        "min_agreement": args.min_agreement,
        "dedup_threshold": args.dedup_threshold,
    }
    meta_path = args.output.with_suffix(".meta.json")
    meta_path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")

    print(f"\nwrote {len(dedup_result.kept)} training records -> {args.output}")
    print(f"metadata -> {meta_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
