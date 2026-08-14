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
import os
import sys
import time
from pathlib import Path

from forge.inference import build_messages, parse_response
from forge.schema import PIIRecord


def load_gold_texts(path: Path) -> list[PIIRecord]:
    records = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            records.append(PIIRecord.model_validate_json(line))
    return records


def run_api(args, gold: list[PIIRecord], sink=None):
    """Run inference via an OpenAI-compatible API.

    Transport robustness (throttle, retry) lives here; scoring semantics —
    prompts, parsing, metrics — are unchanged. Recorded latency is the
    successful attempt's server round-trip only, never throttle sleeps.
    Each prediction is flushed to `sink` immediately so an interrupted run
    loses nothing already processed (restart with --resume).
    """
    try:
        import openai
        from openai import OpenAI
    except ImportError:
        print("Install the openai package: pip install 'openai>=1.0'", file=sys.stderr)
        sys.exit(1)

    client = OpenAI(base_url=args.base_url, api_key=args.api_key)
    extra_body = {"reasoning_effort": args.reasoning_effort} if args.reasoning_effort else None
    min_interval = 60.0 / args.rpm if args.rpm else 0.0

    predictions: list[PIIRecord] = []
    schema_valid = 0
    latencies: list[float] = []
    errors = 0
    last_start = 0.0

    for i, rec in enumerate(gold, 1):
        messages = build_messages(rec.text)

        if min_interval:
            wait = min_interval - (time.monotonic() - last_start)
            if wait > 0:
                time.sleep(wait)

        pred = None
        for attempt in range(1, args.max_retries + 1):
            last_start = time.monotonic()
            try:
                t0 = time.monotonic()
                resp = client.chat.completions.create(
                    model=args.model,
                    messages=messages,
                    max_tokens=args.max_tokens,
                    temperature=args.temperature,
                    timeout=args.timeout,
                    extra_body=extra_body,
                )
                latency = time.monotonic() - t0
                latencies.append(latency)

                raw = resp.choices[0].message.content or ""
                pred, valid = parse_response(rec.id, rec.text, raw, split=rec.split)
                if valid:
                    schema_valid += 1

                n_spans = len(pred.spans) if pred else 0
                status = "OK" if valid else "PARSE_FAIL"
                print(f"  [{i}/{len(gold)}] {rec.id}: {status} ({n_spans} spans, {latency:.1f}s)")
                break

            except (openai.RateLimitError, openai.APIConnectionError, openai.InternalServerError) as e:
                if attempt == args.max_retries:
                    print(f"  [{i}/{len(gold)}] {rec.id}: GAVE UP after {attempt} attempts ({type(e).__name__})")
                    break
                backoff = min(15.0 * (2 ** (attempt - 1)), 120.0)
                print(f"  [{i}/{len(gold)}] {rec.id}: {type(e).__name__}, retry {attempt}/{args.max_retries} in {backoff:.0f}s")
                time.sleep(backoff)
            except Exception as e:  # noqa: BLE001
                print(f"  [{i}/{len(gold)}] {rec.id}: ERROR ({type(e).__name__}: {e})")
                break

        if pred is None:
            errors += 1
            pred = PIIRecord(id=rec.id, text=rec.text, spans=[], split=rec.split)
        predictions.append(pred)
        if sink is not None:
            sink.write(pred.model_dump_json() + "\n")
            sink.flush()

    return predictions, schema_valid, latencies, errors


def run_local(args, gold: list[PIIRecord], sink=None):
    """Run inference with a local model (base + optional LoRA adapter)."""
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"loading tokenizer: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    use_mps = torch.backends.mps.is_available()
    device = "mps" if use_mps else ("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if use_mps else torch.bfloat16

    print(f"loading base model: {args.model} (device={device}, dtype={dtype})")
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=dtype, trust_remote_code=True,
    ).to(device)

    if args.adapter:
        print(f"loading LoRA adapter: {args.adapter}")
        model = PeftModel.from_pretrained(model, args.adapter)
        model = model.merge_and_unload()

    model.eval()

    predictions: list[PIIRecord] = []
    schema_valid = 0
    latencies: list[float] = []
    errors = 0

    for i, rec in enumerate(gold, 1):
        messages = build_messages(rec.text)
        try:
            t0 = time.monotonic()
            encoded = tokenizer.apply_chat_template(
                messages, add_generation_prompt=True, return_tensors="pt",
            )
            if hasattr(encoded, "input_ids"):
                input_ids = encoded["input_ids"].to(device)
            else:
                input_ids = encoded.to(device)
            with torch.no_grad():
                outputs = model.generate(
                    input_ids,
                    max_new_tokens=args.max_tokens,
                    do_sample=args.temperature > 0,
                    temperature=args.temperature if args.temperature > 0 else None,
                    pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
                )
            raw = tokenizer.decode(outputs[0][input_ids.shape[-1]:], skip_special_tokens=True)
            latency = time.monotonic() - t0
            latencies.append(latency)

            pred, valid = parse_response(rec.id, rec.text, raw, split=rec.split)
            predictions.append(pred)
            if valid:
                schema_valid += 1

            n_spans = len(pred.spans) if pred else 0
            status = "OK" if valid else "PARSE_FAIL"
            print(f"  [{i}/{len(gold)}] {rec.id}: {status} ({n_spans} spans, {latency:.1f}s)")

        except Exception as e:  # noqa: BLE001
            errors += 1
            pred = PIIRecord(id=rec.id, text=rec.text, spans=[], split=rec.split)
            predictions.append(pred)
            print(f"  [{i}/{len(gold)}] {rec.id}: ERROR ({type(e).__name__}: {e})")

        if sink is not None:
            sink.write(predictions[-1].model_dump_json() + "\n")
            sink.flush()

    return predictions, schema_valid, latencies, errors


def main() -> int:
    ap = argparse.ArgumentParser(description="Run model inference on gold set.")
    ap.add_argument("gold", type=Path, help="Gold JSONL (for input texts + ids)")
    ap.add_argument("output", type=Path, help="Output predictions JSONL")
    ap.add_argument("--model", required=True, help="Model name/path (HF name or local path)")
    ap.add_argument("--adapter", type=Path, default=None, help="LoRA adapter path (enables local inference)")
    ap.add_argument("--base-url", default="http://localhost:8000/v1", help="API base URL")
    ap.add_argument("--api-key", default="not-needed", help="API key (prefer --api-key-env)")
    ap.add_argument(
        "--api-key-env", default=None, metavar="VAR",
        help="Read the API key from this environment variable (never logged)",
    )
    ap.add_argument("--max-tokens", type=int, default=1024, help="Max output tokens")
    ap.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature")
    ap.add_argument("--timeout", type=float, default=60.0, help="Per-request timeout (seconds)")
    ap.add_argument("--limit", type=int, default=None, help="Process only first N records")
    ap.add_argument(
        "--rpm", type=float, default=None,
        help="Throttle: max requests per minute (e.g. 5 for Cerebras free tier)",
    )
    ap.add_argument("--max-retries", type=int, default=5, help="Retries on rate-limit/transient errors")
    ap.add_argument(
        "--reasoning-effort", default=None, choices=["low", "medium", "high"],
        help="For reasoning models (gpt-oss): passed as extra_body",
    )
    ap.add_argument(
        "--resume", action="store_true",
        help="Skip records whose ids are already in the output file; append to it",
    )
    args = ap.parse_args()

    if args.api_key_env:
        key = os.environ.get(args.api_key_env)
        if not key:
            print(f"Environment variable {args.api_key_env} is not set.", file=sys.stderr)
            return 1
        args.api_key = key

    gold = load_gold_texts(args.gold)
    if args.limit:
        gold = gold[: args.limit]

    done_ids: set[str] = set()
    if args.resume and args.output.exists():
        for line in args.output.read_text(encoding="utf-8").splitlines():
            if line.strip():
                done_ids.add(PIIRecord.model_validate_json(line).id)
        gold = [r for r in gold if r.id not in done_ids]
        print(f"resume: {len(done_ids)} records already done, {len(gold)} remaining")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if (args.resume and done_ids) else "w"
    with args.output.open(mode, encoding="utf-8") as sink:
        if args.adapter:
            print(f"local inference: {args.model} + adapter {args.adapter}")
            predictions, schema_valid, latencies, errors = run_local(args, gold, sink=sink)
        else:
            print(f"API inference: {args.model} via {args.base_url}")
            predictions, schema_valid, latencies, errors = run_api(args, gold, sink=sink)

    def pct(sorted_vals: list[float], q: float) -> float:
        if not sorted_vals:
            return 0.0
        idx = round(q * (len(sorted_vals) - 1))
        return sorted_vals[idx]

    lat_sorted = sorted(latencies)
    meta = {
        "schema_valid": schema_valid,
        "total": len(gold),
        "resumed_skipped": len(done_ids),
        "errors": errors,
        "avg_latency_s": sum(latencies) / len(latencies) if latencies else 0,
        "p50_latency_s": pct(lat_sorted, 0.50),
        "p95_latency_s": pct(lat_sorted, 0.95),
        "model": args.model,
        "adapter": str(args.adapter) if args.adapter else None,
        "base_url": None if args.adapter else args.base_url,
        "reasoning_effort": args.reasoning_effort,
        "rpm_throttle": args.rpm,
    }
    meta_path = args.output.with_suffix(".meta.json")
    meta_path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")

    valid_pct = schema_valid / len(gold) * 100 if gold else 0
    print(f"\nwrote {len(predictions)} predictions -> {args.output}")
    print(f"schema valid: {schema_valid}/{len(gold)} ({valid_pct:.1f}%)")
    print(f"errors: {errors}")
    print(f"avg latency: {meta['avg_latency_s']:.2f}s | p50: {meta['p50_latency_s']:.2f}s | p95: {meta['p95_latency_s']:.2f}s")

    return 0


if __name__ == "__main__":
    sys.exit(main())
