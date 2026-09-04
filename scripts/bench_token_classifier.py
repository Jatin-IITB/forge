#!/usr/bin/env python3
"""Benchmark and emit predictions from the one-pass Qwen PII classifier."""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import time
from pathlib import Path

import torch
from transformers import AutoConfig, AutoTokenizer

from forge.schema import PIIRecord
from forge.token_classifier import constrained_viterbi, decode_bioes
from forge.token_model import ForgeQwen2ForTokenClassification

try:
    from scripts.bench_serving import _contention, usd_per_1k
except ModuleNotFoundError:
    from bench_serving import _contention, usd_per_1k


def load_records(path: Path) -> list[PIIRecord]:
    return [
        PIIRecord.model_validate_json(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--gold", type=Path, default=Path("data/gold/test.jsonl"))
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--config-name", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--save-predictions", type=Path, required=True)
    args = parser.parse_args()

    records = load_records(args.gold)
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    config = AutoConfig.from_pretrained(args.model)
    config._attn_implementation = "eager"
    device = "mps" if torch.backends.mps.is_available() else (
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    dtype = torch.float16 if device == "mps" else torch.bfloat16
    model = ForgeQwen2ForTokenClassification.from_pretrained(
        args.model,
        config=config,
        dtype=dtype,
    ).to(device)
    model.eval()

    def run_once() -> tuple[float, list[PIIRecord], int, list[float]]:
        predictions: list[PIIRecord] = []
        token_total = 0
        batch_latencies: list[float] = []
        started = time.perf_counter()
        for start in range(0, len(records), args.batch_size):
            batch = records[start : start + args.batch_size]
            encoded = tokenizer(
                [record.text for record in batch],
                add_special_tokens=False,
                padding=True,
                truncation=True,
                max_length=args.max_length,
                return_offsets_mapping=True,
                return_tensors="pt",
            )
            offsets = encoded.pop("offset_mapping")
            attention_mask = encoded["attention_mask"]
            lengths = attention_mask.sum(dim=1).tolist()
            token_total += int(attention_mask.sum())
            model_inputs = {key: value.to(device) for key, value in encoded.items()}

            batch_started = time.perf_counter()
            with torch.inference_mode():
                logits = model(**model_inputs).logits
            if device == "mps":
                torch.mps.synchronize()

            for index, record in enumerate(batch):
                length = int(lengths[index])
                path = constrained_viterbi(logits[index, :length].float().cpu().tolist())
                token_offsets = [tuple(pair) for pair in offsets[index, :length].tolist()]
                predictions.append(
                    decode_bioes(
                        record.id,
                        record.text,
                        token_offsets,
                        path,
                        split=record.split,
                    )
                )
            batch_latencies.append(time.perf_counter() - batch_started)
        return time.perf_counter() - started, predictions, token_total, batch_latencies

    for _ in range(args.warmup):
        original = records
        records = records[: args.batch_size]
        run_once()
        records = original

    contention_start = _contention()
    passes = []
    best = None
    for pass_number in range(1, args.repeat + 1):
        wall, predictions, token_total, batch_latencies = run_once()
        row = {
            "pass": pass_number,
            "wall_clock_s": wall,
            "sustained_s_per_record": wall / len(records),
            "records_per_s": len(records) / wall,
            "input_tokens": token_total,
            "input_tok_s": token_total / wall,
            "batch_latency_mean_s": statistics.fmean(batch_latencies),
            "batch_latency_max_s": max(batch_latencies),
            "contention_at_end": _contention(),
        }
        passes.append(row)
        if best is None or row["records_per_s"] > best[0]["records_per_s"]:
            best = (row, predictions)
        print(
            f"pass {pass_number}: {row['sustained_s_per_record']:.5f} s/record, "
            f"{row['records_per_s']:.2f} records/s",
            flush=True,
        )

    assert best is not None
    best_row, best_predictions = best
    economics = usd_per_1k(best_row["sustained_s_per_record"])
    artifact = {
        "config_name": args.config_name,
        "backend": "transformers-token-classification",
        "model": args.model,
        "batch_size": args.batch_size,
        "n_records": len(records),
        "repeat_passes": args.repeat,
        "selection": "best records/s pass; all passes retained",
        "sustained_s_per_record": best_row["sustained_s_per_record"],
        "records_per_s": best_row["records_per_s"],
        "usd_per_1k": economics["usd_per_1k"],
        "passes": passes,
        "contention_at_start": contention_start,
        "contention_at_end": _contention(),
        "env": {
            "platform": platform.platform(),
            "machine": platform.machine(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "device": device,
            "dtype": str(dtype),
        },
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(artifact, indent=2) + "\n", encoding="utf-8")
    args.save_predictions.parent.mkdir(parents=True, exist_ok=True)
    args.save_predictions.write_text(
        "".join(record.model_dump_json() + "\n" for record in best_predictions),
        encoding="utf-8",
    )
    args.save_predictions.with_suffix(".meta.json").write_text(
        json.dumps(artifact, indent=2) + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
