#!/usr/bin/env python3
"""Train a one-pass Qwen2 BIOES PII classifier with LoRA."""

from __future__ import annotations

import argparse
import importlib.metadata
import importlib.util
import json
import math
import sys
from collections import Counter
from pathlib import Path

from forge.schema import PIIRecord
from forge.token_classifier import (
    ID2LABEL,
    LABEL2ID,
    assert_bioes_round_trip,
    constrained_viterbi_batch,
)


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
    n_spans = 0
    n_multi_token_spans = 0
    multi_token_by_type: Counter[str] = Counter()
    subtoken_boundary_by_type: Counter[str] = Counter()
    label_counts: Counter[int] = Counter()
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
        labels = assert_bioes_round_trip(record, offsets)
        encoded["labels"] = labels
        label_counts.update(labels)
        n_spans += len(record.spans)
        for span in record.spans:
            overlapping_tokens = sum(
                end > start and end > span.start and start < span.end
                for start, end in offsets
            )
            if overlapping_tokens > 1:
                n_multi_token_spans += 1
                multi_token_by_type[span.label.value] += 1
            if (
                not any(start == span.start for start, _ in offsets)
                or not any(end == span.end for _, end in offsets)
            ):
                subtoken_boundary_by_type[span.label.value] += 1
        rows.append(encoded)
    token_count = sum(label_counts.values())
    outside_tokens = label_counts[LABEL2ID["O"]]
    stats = {
        "records": len(records),
        "spans": n_spans,
        "round_trip_failures": 0,
        "overlapping_or_nested_spans": 0,
        "multi_token_spans": n_multi_token_spans,
        "multi_token_spans_by_type": dict(sorted(multi_token_by_type.items())),
        "subtoken_boundary_spans": sum(subtoken_boundary_by_type.values()),
        "subtoken_boundary_spans_by_type": dict(
            sorted(subtoken_boundary_by_type.items())
        ),
        "tokens": token_count,
        "outside_tokens": outside_tokens,
        "outside_token_fraction": outside_tokens / token_count if token_count else 0.0,
        "label_counts": {
            ID2LABEL[label_id]: count
            for label_id, count in sorted(label_counts.items())
        },
    }
    return rows, stats


def class_weights(rows):
    import torch

    counts: Counter[int] = Counter(
        label
        for row in rows
        for labels in [row["labels"]]
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


def _entities(label_ids: list[int]) -> set[tuple[int, int, str]]:
    """Extract token-index entities for validation-set model selection."""
    entities: set[tuple[int, int, str]] = set()
    index = 0
    while index < len(label_ids):
        label = ID2LABEL[label_ids[index]]
        if label == "O":
            index += 1
            continue
        tag, pii_type = label.split("-", 1)
        if tag == "S":
            entities.add((index, index + 1, pii_type))
            index += 1
            continue
        if tag != "B":
            index += 1
            continue
        end = index + 1
        while end < len(label_ids):
            end_label = ID2LABEL[label_ids[end]]
            end_tag, _, end_type = end_label.partition("-")
            if end_type != pii_type:
                break
            if end_tag == "E":
                entities.add((index, end + 1, pii_type))
                end += 1
                break
            if end_tag != "I":
                break
            end += 1
        index = end
    return entities


def span_metrics(eval_prediction) -> dict[str, float]:
    """Exact BIOES entity metrics on the clean validation split."""
    logits, labels = eval_prediction
    tp = fp = fn = 0
    lengths = [
        sum(int(label_id) != -100 for label_id in row_labels)
        for row_labels in labels
    ]
    predicted_paths = constrained_viterbi_batch(logits, lengths)
    for predicted, row_labels, length in zip(predicted_paths, labels, lengths):
        gold = [int(label_id) for label_id in row_labels[:length]]
        predicted_entities = _entities(predicted)
        gold_entities = _entities(gold)
        tp += len(predicted_entities & gold_entities)
        fp += len(predicted_entities - gold_entities)
        fn += len(gold_entities - predicted_entities)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"span_f1": f1, "span_precision": precision, "span_recall": recall}


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
    parser.add_argument(
        "--qlora",
        action="store_true",
        help=(
            "Load the 1.5B backbone in 4-bit NF4 for low-VRAM CUDA training. "
            "The final artifact is reloaded and merged as fp16 on CPU."
        ),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--verify-alignment-only",
        action="store_true",
        help="assert exact train/validation BIOES round trips, then exit before loading the model",
    )
    args = parser.parse_args()

    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        print(f"missing training dependency: {exc}", file=sys.stderr)
        return 1

    tokenizer = AutoTokenizer.from_pretrained(args.base_model, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    train_records = load_records(args.train_data)
    val_records = load_records([args.val_data])
    train_rows, train_alignment = encode_records(
        train_records, tokenizer, args.max_length
    )
    val_rows, val_alignment = encode_records(val_records, tokenizer, args.max_length)
    print(
        "BIOES round-trip verified: "
        f"train={train_alignment['records']} records/{train_alignment['spans']} spans, "
        f"val={val_alignment['records']} records/{val_alignment['spans']} spans; "
        "unsupported overlaps/nesting=0"
    )
    if args.verify_alignment_only:
        return 0

    try:
        import torch
        from datasets import Dataset
        from peft import (
            LoraConfig,
            PeftModel,
            TaskType,
            get_peft_model,
            prepare_model_for_kbit_training,
        )
        from transformers import (
            AutoConfig,
            BitsAndBytesConfig,
            DataCollatorForTokenClassification,
            Trainer,
            TrainingArguments,
        )
    except ImportError as exc:
        print(f"missing training dependency: {exc}", file=sys.stderr)
        return 1

    from forge.token_model import ForgeQwen2ForTokenClassification

    train_dataset = Dataset.from_list(train_rows)
    val_dataset = Dataset.from_list(val_rows)
    config = AutoConfig.from_pretrained(args.base_model)
    config.num_labels = len(ID2LABEL)
    config.id2label = dict(enumerate(ID2LABEL))
    config.label2id = LABEL2ID
    config.forge_full_attention = args.full_attention
    config.use_cache = False
    config._attn_implementation = "eager"

    use_mps = torch.backends.mps.is_available()
    use_cuda = torch.cuda.is_available()
    if args.qlora and not use_cuda:
        parser.error("--qlora requires an NVIDIA CUDA GPU")
    if args.qlora and importlib.util.find_spec("bitsandbytes") is None:
        parser.error("--qlora requires bitsandbytes; install the project [train] extra")

    use_bf16 = use_cuda and torch.cuda.is_bf16_supported()
    use_fp16 = use_mps or (use_cuda and not use_bf16)
    dtype = (
        torch.float16
        if use_fp16
        else torch.bfloat16
        if use_bf16
        else torch.float32
    )
    if use_cuda:
        total_vram_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(
            f"CUDA training: {torch.cuda.get_device_name(0)}, "
            f"{total_vram_gb:.1f} GiB VRAM, precision={dtype}, qlora={args.qlora}"
        )
    elif use_mps:
        print(f"MPS training: precision={dtype}, qlora=False")
    else:
        print(f"CPU training: precision={dtype}, qlora=False")

    quantization_config = None
    if args.qlora:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=dtype,
            bnb_4bit_use_double_quant=True,
        )

    model = ForgeQwen2ForTokenClassification.from_pretrained(
        args.base_model,
        config=config,
        dtype=dtype,
        quantization_config=quantization_config,
        device_map={"": 0} if args.qlora else None,
        ignore_mismatched_sizes=True,
    )
    if args.qlora:
        model = prepare_model_for_kbit_training(
            model,
            use_gradient_checkpointing=True,
        )
    model.set_class_weights(class_weights(train_rows))
    if not args.qlora:
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
        logging_nan_inf_filter=False,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="span_f1",
        greater_is_better=True,
        save_total_limit=2,
        fp16=use_fp16,
        bf16=use_bf16,
        optim="paged_adamw_8bit" if args.qlora else "adamw_torch",
        gradient_checkpointing=True,
        seed=args.seed,
        report_to="none",
        remove_unused_columns=False,
        eval_accumulation_steps=4,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=DataCollatorForTokenClassification(tokenizer),
        processing_class=tokenizer,
        compute_metrics=span_metrics,
    )

    resume = True if args.resume and args.output_dir.exists() else None
    trainer.train(resume_from_checkpoint=resume)

    adapter_dir = args.output_dir / "final"
    trainer.save_model(str(adapter_dir))
    tokenizer.save_pretrained(str(adapter_dir))

    if args.qlora:
        # Merging directly into a bitsandbytes model would make the benchmark
        # consume a training-only quantized proxy. Release trainer references,
        # reload the same 1.5B backbone as fp16 on CPU, then apply the selected
        # adapter. The shipped artifact is therefore independent of bnb.
        import gc

        del model
        del trainer
        gc.collect()
        torch.cuda.empty_cache()

        merge_config = AutoConfig.from_pretrained(args.base_model)
        merge_config.num_labels = len(ID2LABEL)
        merge_config.id2label = dict(enumerate(ID2LABEL))
        merge_config.label2id = LABEL2ID
        merge_config.forge_full_attention = args.full_attention
        merge_config.use_cache = False
        merge_config._attn_implementation = "eager"
        merge_base = ForgeQwen2ForTokenClassification.from_pretrained(
            args.base_model,
            config=merge_config,
            dtype=torch.float16,
            ignore_mismatched_sizes=True,
            low_cpu_mem_usage=True,
        )
        merged = PeftModel.from_pretrained(
            merge_base,
            str(adapter_dir),
        ).merge_and_unload()
        merged_dtype = "torch.float16"
    else:
        merged = model.merge_and_unload()
        merged_dtype = str(dtype)

    merged_dir = args.output_dir / "final-merged"
    merged.save_pretrained(merged_dir, safe_serialization=True)
    tokenizer.save_pretrained(merged_dir)

    metadata = {
        "base_model": args.base_model,
        "base_model_commit": getattr(config, "_commit_hash", None),
        "train_data": [str(path) for path in args.train_data],
        "val_data": str(args.val_data),
        "train_records": len(train_records),
        "val_records": len(val_records),
        "alignment": {
            "train": train_alignment,
            "validation": val_alignment,
        },
        "full_attention": args.full_attention,
        "qlora": args.qlora,
        "training_dtype": str(dtype),
        "merged_dtype": merged_dtype,
        "cuda_device": torch.cuda.get_device_name(0) if use_cuda else None,
        "cuda_vram_gb": (
            round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 2)
            if use_cuda
            else None
        ),
        "labels": ID2LABEL,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "grad_accum": args.grad_accum,
        "lr": args.lr,
        "seed": args.seed,
        "versions": {
            package: importlib.metadata.version(package)
            for package in (
                "torch",
                "transformers",
                "peft",
                "datasets",
                "accelerate",
                *(("bitsandbytes",) if args.qlora else ()),
            )
        },
    }
    (args.output_dir / "train_meta.json").write_text(
        json.dumps(metadata, indent=2) + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
