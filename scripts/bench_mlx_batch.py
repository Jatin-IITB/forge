#!/usr/bin/env python3
"""Direct MLX-LM continuous-batching benchmark on the frozen test prompts."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from forge.inference import build_messages, parse_response
from forge.schema import PIIRecord

try:
    from scripts.bench_serving import _contention, usd_per_1k
except ModuleNotFoundError:
    from bench_serving import _contention, usd_per_1k


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--gold", type=Path, default=Path("data/gold/test.jsonl"))
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--completion-batch-size", type=int, required=True)
    parser.add_argument("--prefill-batch-size", type=int, default=8)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--config-name", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--save-predictions", type=Path, required=True)
    args = parser.parse_args()

    try:
        import mlx.core as mx
        from mlx_lm import load
        from mlx_lm.generate import BatchGenerator
    except ImportError as exc:
        parser.error(f"MLX-LM is required: {exc}")

    records = [
        PIIRecord.model_validate_json(line)
        for line in args.gold.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if args.limit:
        records = records[: args.limit]
    model, tokenizer = load(args.model)
    prompts = [
        tokenizer.apply_chat_template(
            build_messages(record.text, compact=True),
            add_generation_prompt=True,
            tokenize=True,
        )
        for record in records
    ]
    stop_tokens = getattr(tokenizer, "eos_token_ids", None)
    if stop_tokens is None:
        eos = getattr(tokenizer, "eos_token_id", None)
        stop_tokens = [[eos]] if eos is not None else None

    def run_once(selected_records, selected_prompts):
        generator = BatchGenerator(
            model,
            max_tokens=args.max_tokens,
            stop_tokens=stop_tokens,
            completion_batch_size=args.completion_batch_size,
            prefill_batch_size=args.prefill_batch_size,
        )
        uids = generator.insert(selected_prompts)
        generated = {uid: [] for uid in uids}
        started = time.perf_counter()
        while responses := generator.next():
            for response in responses:
                token = getattr(response, "token", None)
                if token is not None and int(token) >= 0:
                    generated[response.uid].append(int(token))
        mx.synchronize()

        predictions = []
        valid = 0
        token_total = 0
        for record, uid in zip(selected_records, uids):
            tokens = generated[uid]
            token_total += len(tokens)
            raw = tokenizer.decode(tokens, skip_special_tokens=True)
            pred, ok = parse_response(record.id, record.text, raw, split=record.split)
            valid += int(ok)
            predictions.append(pred)
        wall = time.perf_counter() - started
        return wall, predictions, valid, token_total, generator.stats()

    warmup_size = min(len(records), args.completion_batch_size)
    run_once(records[:warmup_size], prompts[:warmup_size])
    passes = []
    best = None
    for pass_number in range(1, args.repeat + 1):
        contention_start = _contention()
        wall, predictions, valid, token_total, stats = run_once(records, prompts)
        sustained = wall / len(records)
        row = {
            "pass": pass_number,
            "wall_clock_s": wall,
            "sustained_s_per_record": sustained,
            "records_per_s": len(records) / wall,
            "schema_valid": valid,
            "completion_tokens": token_total,
            "output_tok_s": token_total / wall,
            "mlx_generation_tps": getattr(stats, "generation_tps", None),
            "contention_at_start": contention_start,
            "contention_at_end": _contention(),
        }
        passes.append(row)
        if best is None or row["records_per_s"] > best[0]["records_per_s"]:
            best = (row, predictions)
        print(
            f"pass {pass_number}: {sustained:.5f} s/record, "
            f"schema {valid}/{len(records)}",
            flush=True,
        )

    assert best is not None
    best_row, best_predictions = best
    artifact = {
        "config_name": args.config_name,
        "backend": "mlx-lm-batch-generator",
        "model": args.model,
        "completion_batch_size": args.completion_batch_size,
        "prefill_batch_size": args.prefill_batch_size,
        "n_records": len(records),
        "repeat_passes": args.repeat,
        "selection": "best records/s pass; all passes retained",
        **best_row,
        "usd_per_1k": usd_per_1k(best_row["sustained_s_per_record"])["usd_per_1k"],
        "passes": passes,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(artifact, indent=2) + "\n", encoding="utf-8")
    args.save_predictions.parent.mkdir(parents=True, exist_ok=True)
    args.save_predictions.write_text(
        "".join(record.model_dump_json() + "\n" for record in best_predictions),
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
