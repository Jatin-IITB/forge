#!/usr/bin/env python3
"""Train a one-pass Qwen2 BIOES PII classifier with LoRA."""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path

try:
    import torch
    from datasets import Dataset
    from peft import LoraConfig, TaskType, get_peft_model
    from transformers import (
        AutoConfig,
        AutoTokenizer,
        DataCollatorForTokenClassification,
        Trainer,
        TrainingArguments,
    )
except ImportError as exc:
    print(f"missing training dependency: {exc}", file=sys.stderr)
    raise SystemExit(1)

from forge.schema import PIIRecord
from forge.token_classifier import ID2LABEL, LABEL2ID, encode_bioes
from forge.token_model import ForgeQwen2ForTokenClassification


def load_records(paths: list[Path]) -> list[PIIRecord]:
    records: list[PIIRecord] = []
    for path in paths:
        records.extend(
            PIIRecord.model_validate_json(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
    return records


def encode_records(records, tokenizer, max_length):
    rows = []
    for record in records:
        encoded = tokenizer(
            record.text,
            add_special_tokens=False,
            truncation=True,
            max_length=max_length,
            return_offsets_mapping=True,
        )
        offsets = [tuple(pair) for pair in encoded.pop("offset_mapping")]
        if offsets and offsets[-1][1] < len(record.text):
            raise ValueError(f"{record.id}: truncation would discard labelled text")
        encoded["labels"] = encode_bioes(record, offsets)
        rows.append(encoded)
    return Dataset.from_list(rows)


def class_weights(dataset) -> torch.Tensor:
    counts: Counter[int] = Counter(
        label
        for labels in dataset["labels"]
        for label in labels
        if label >= 0
    )
    outside = max(counts.get(LABEL2ID["O"], 1), 1)
    weights = []
    for label_id in range(len(ID2LABEL)):
        count = max(counts.get(label_id, 1), 1)
        weights.append(min(5.0, math.sqrt(outside / count)))
    weights[LABEL2ID["O"]] = 1.0
    return torch.tensor(weights, dtype=torch.float32)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-data", type=Path, action="append", required=True)
    parser.add_argument("--val-data", type=Path, required=True)
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--full-attention", action="store_true")
    parser.add_argument("--epochs", type=float, default=3)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.base_model, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    train_records = load_records(args.train_data)
    val_records = load_records([args.val_data])
    train_dataset = encode_records(train_records, tokenizer, args.max_length)
    val_dataset = encode_records(val_records, tokenizer, args.max_length)

    config = AutoConfig.from_pretrained(args.base_model)
    config.num_labels = len(ID2LABEL)
    config.id2label = dict(enumerate(ID2LABEL))
    config.label2id = LABEL2ID
    config.forge_full_attention = args.full_attention
    config.use_cache = False
    config._attn_implementation = "eager"

    use_mps = torch.backends.mps.is_available()
    dtype = torch.float16 if use_mps else torch.bfloat16
    model = ForgeQwen2ForTokenClassification.from_pretrained(
        args.base_model,
        config=config,
        dtype=dtype,
        ignore_mismatched_sizes=True,
    )
    model.set_class_weights(class_weights(train_dataset))
    model.enable_input_require_grads()

    lora = LoraConfig(
        task_type=TaskType.TOKEN_CLS,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.05,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        modules_to_save=["score"],
    )
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()

    training_args = TrainingArguments(
        output_dir=str(args.output_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
        logging_steps=10,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        save_total_limit=2,
        fp16=use_mps,
        bf16=not use_mps,
        gradient_checkpointing=True,
        seed=args.seed,
        report_to="none",
        remove_unused_columns=False,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=DataCollatorForTokenClassification(tokenizer),
        processing_class=tokenizer,
    )

    resume = True if args.resume and args.output_dir.exists() else None
    trainer.train(resume_from_checkpoint=resume)

    adapter_dir = args.output_dir / "final"
    trainer.save_model(str(adapter_dir))
    tokenizer.save_pretrained(str(adapter_dir))

    merged = model.merge_and_unload()
    merged_dir = args.output_dir / "final-merged"
    merged.save_pretrained(merged_dir, safe_serialization=True)
    tokenizer.save_pretrained(merged_dir)

    metadata = {
        "base_model": args.base_model,
        "train_data": [str(path) for path in args.train_data],
        "val_data": str(args.val_data),
        "train_records": len(train_records),
        "val_records": len(val_records),
        "full_attention": args.full_attention,
        "labels": ID2LABEL,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "grad_accum": args.grad_accum,
        "lr": args.lr,
        "seed": args.seed,
    }
    (args.output_dir / "train_meta.json").write_text(
        json.dumps(metadata, indent=2) + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
