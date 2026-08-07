#!/usr/bin/env python3
"""Run a model against the gold set and produce predictions.jsonl.

Uses an OpenAI-compatible API endpoint (works with vLLM, Ollama,
text-generation-inference, or any provider that serves /v1/chat/completions).

Usage:
    # Score the teacher
    python scripts/run_inference.py data/gold/test.jsonl predictions_teacher.jsonl \
        --model Qwen/Qwen2.5-32B-Instruct --base-url http://localhost:8000/v1

    # Score the base (student zero-shot)
    python scripts/run_inference.py data/gold/test.jsonl predictions_base.jsonl \
        --model Qwen/Qwen2.5-1.5B-Instruct --base-url http://localhost:8000/v1

    # Then evaluate:
    python scripts/run_eval.py data/gold/test.jsonl predictions_teacher.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from forge.inference import build_messages, parse_response
from forge.schema import PIIRecord

try:
    from openai import OpenAI
except ImportError:
    print("Install the openai package: pip install 'openai>=1.0'", file=sys.stderr)
    sys.exit(1)


def load_gold_texts(path: Path) -> list[PIIRecord]:
    records = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            records.append(PIIRecord.model_validate_json(line))
    return records


def main() -> int:
    ap = argparse.ArgumentParser(description="Run model inference on gold set.")
    ap.add_argument("gold", type=Path, help="Gold JSONL (for input texts + ids)")
    ap.add_argument("output", type=Path, help="Output predictions JSONL")
    ap.add_argument("--model", required=True, help="Model name/path")
    ap.add_argument("--base-url", default="http://localhost:8000/v1", help="API base URL")
    ap.add_argument("--api-key", default="not-needed", help="API key (default: not-needed for local)")
    ap.add_argument("--max-tokens", type=int, default=1024, help="Max output tokens")
    ap.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature")
    ap.add_argument("--timeout", type=float, default=60.0, help="Per-request timeout (seconds)")
    ap.add_argument("--limit", type=int, default=None, help="Process only first N records")
    args = ap.parse_args()

    gold = load_gold_texts(args.gold)
    if args.limit:
        gold = gold[: args.limit]

    client = OpenAI(base_url=args.base_url, api_key=args.api_key)

    predictions: list[PIIRecord] = []
    schema_valid = 0
    total_latency = 0.0
    errors = 0

    for i, rec in enumerate(gold, 1):
        messages = build_messages(rec.text)
        try:
            t0 = time.monotonic()
            resp = client.chat.completions.create(
                model=args.model,
                messages=messages,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                timeout=args.timeout,
            )
            latency = time.monotonic() - t0
            total_latency += latency

            raw = resp.choices[0].message.content or ""
            pred, valid = parse_response(rec.id, rec.text, raw, split=rec.split)
            predictions.append(pred)
            if valid:
                schema_valid += 1

            n_spans = len(pred.spans) if pred else 0
            status = "OK" if valid else "PARSE_FAIL"
            print(f"  [{i}/{len(gold)}] {rec.id}: {status} ({n_spans} spans, {latency:.1f}s)")

        except Exception as e:  # noqa: BLE001
            errors += 1
            predictions.append(PIIRecord(id=rec.id, text=rec.text, spans=[], split=rec.split))
            print(f"  [{i}/{len(gold)}] {rec.id}: ERROR ({e})")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        for p in predictions:
            f.write(p.model_dump_json() + "\n")

    meta = {
        "schema_valid": schema_valid,
        "total": len(gold),
        "errors": errors,
        "avg_latency_s": total_latency / len(gold) if gold else 0,
        "model": args.model,
    }
    meta_path = args.output.with_suffix(".meta.json")
    meta_path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")

    valid_pct = schema_valid / len(gold) * 100 if gold else 0
    print(f"\nwrote {len(predictions)} predictions -> {args.output}")
    print(f"metadata -> {meta_path}")
    print(f"schema valid: {schema_valid}/{len(gold)} ({valid_pct:.1f}%)")
    print(f"errors: {errors}")
    print(f"avg latency: {meta['avg_latency_s']:.2f}s")

    return 0


if __name__ == "__main__":
    sys.exit(main())
