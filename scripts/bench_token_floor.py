#!/usr/bin/env python3
"""Measure the bare one-pass Qwen token-classification serving floor."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from transformers import AutoConfig, AutoTokenizer

from forge.schema import PIIRecord
from forge.token_classifier import ID2LABEL
from forge.token_model import ForgeQwen2ForTokenClassification

try:
    from scripts.bench_serving import _contention, usd_per_1k
except ModuleNotFoundError:
    from bench_serving import _contention, usd_per_1k


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--gold", type=Path, default=Path("data/gold/test.jsonl"))
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[8, 16, 32])
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    records = [
        PIIRecord.model_validate_json(line)
        for line in args.gold.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    config = AutoConfig.from_pretrained(args.model)
    config.num_labels = len(ID2LABEL)
    config.forge_full_attention = True
    config.use_cache = False
    config._attn_implementation = "eager"
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    dtype = torch.float16 if device == "mps" else torch.float32
    model = ForgeQwen2ForTokenClassification.from_pretrained(
        args.model,
        config=config,
        dtype=dtype,
        ignore_mismatched_sizes=True,
    ).to(device)
    model.eval()

    rows = []
    for batch_size in args.batch_sizes:
        encoded_batches = []
        for start in range(0, len(records), batch_size):
            encoded = tokenizer(
                [record.text for record in records[start : start + batch_size]],
                add_special_tokens=False,
                padding=True,
                return_tensors="pt",
            )
            encoded_batches.append({key: value.to(device) for key, value in encoded.items()})
        with torch.inference_mode():
            model(**encoded_batches[0])
        if device == "mps":
            torch.mps.synchronize()

        for pass_number in range(1, args.repeat + 1):
            contention_start = _contention()
            started = time.perf_counter()
            with torch.inference_mode():
                for encoded in encoded_batches:
                    model(**encoded)
            if device == "mps":
                torch.mps.synchronize()
            wall = time.perf_counter() - started
            sustained = wall / len(records)
            rows.append(
                {
                    "batch_size": batch_size,
                    "pass": pass_number,
                    "wall_clock_s": wall,
                    "sustained_s_per_record": sustained,
                    "records_per_s": len(records) / wall,
                    "usd_per_1k": usd_per_1k(sustained)["usd_per_1k"],
                    "contention_at_start": contention_start,
                    "contention_at_end": _contention(),
                }
            )
            print(
                f"batch={batch_size} pass={pass_number}: "
                f"{sustained:.5f} s/record",
                flush=True,
            )

    artifact = {
        "model": args.model,
        "n_records": len(records),
        "full_attention": True,
        "device": device,
        "dtype": str(dtype),
        "rows": rows,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(artifact, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
