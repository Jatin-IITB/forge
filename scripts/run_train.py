#!/usr/bin/env python3
"""Train a student model with LoRA SFT on verified training data.

Usage:
    python scripts/run_train.py \
        --train-data data/train.jsonl \
        --base-model Qwen/Qwen2.5-1.5B-Instruct \
        --output-dir checkpoints/run_001

    # With QLoRA (4-bit base):
    python scripts/run_train.py \
        --train-data data/train.jsonl \
        --base-model Qwen/Qwen2.5-1.5B-Instruct \
        --output-dir checkpoints/run_001 \
        --qlora

    # Resume from checkpoint:
    python scripts/run_train.py \
        --train-data data/train.jsonl \
        --base-model Qwen/Qwen2.5-1.5B-Instruct \
        --output-dir checkpoints/run_001 \
        --resume
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from forge.train import LORA_DEFAULTS, SFT_DEFAULTS, load_training_data

try:
    import torch
    from datasets import Dataset
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
    )
    from trl import SFTConfig, SFTTrainer
except ImportError as e:
    print(
        f"Missing training dependency: {e}\n"
        "Install with: pip install 'forge[train]'\n"
        "Or: pip install torch transformers peft trl datasets accelerate bitsandbytes",
        file=sys.stderr,
    )
    sys.exit(1)


def main() -> int:
    ap = argparse.ArgumentParser(description="Train student model with LoRA SFT.")
    ap.add_argument("--train-data", type=Path, required=True, help="Verified train.jsonl")
    ap.add_argument("--base-model", required=True, help="Base model name/path")
    ap.add_argument("--output-dir", type=Path, required=True, help="Checkpoint output directory")
    ap.add_argument("--epochs", type=int, default=SFT_DEFAULTS["num_train_epochs"])
    ap.add_argument("--batch-size", type=int, default=SFT_DEFAULTS["per_device_train_batch_size"])
    ap.add_argument("--grad-accum", type=int, default=SFT_DEFAULTS["gradient_accumulation_steps"])
    ap.add_argument("--lr", type=float, default=SFT_DEFAULTS["learning_rate"])
    ap.add_argument("--max-seq-length", type=int, default=SFT_DEFAULTS["max_length"])
    ap.add_argument("--lora-r", type=int, default=LORA_DEFAULTS["r"])
    ap.add_argument("--lora-alpha", type=int, default=LORA_DEFAULTS["lora_alpha"])
    ap.add_argument("--qlora", action="store_true", help="Use 4-bit QLoRA")
    ap.add_argument("--seed", type=int, default=SFT_DEFAULTS["seed"])
    ap.add_argument("--resume", action="store_true", help="Resume from latest checkpoint")
    args = ap.parse_args()

    print(f"loading training data from {args.train_data}")
    conversations = load_training_data(args.train_data)
    print(f"loaded {len(conversations)} training examples")

    if not conversations:
        print("ERROR: no training examples found", file=sys.stderr)
        return 1

    print(f"loading tokenizer: {args.base_model}")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    quantization_config = None
    if args.qlora:
        print("using 4-bit QLoRA quantization")
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )

    use_mps = torch.backends.mps.is_available()
    if use_mps:
        print("detected Apple Silicon MPS — using fp16 (bf16 causes NaN on MPS)")
        model_dtype = torch.float16
        use_bf16 = False
        use_fp16 = True
    else:
        model_dtype = torch.bfloat16 if not args.qlora else None
        use_bf16 = SFT_DEFAULTS["bf16"]
        use_fp16 = False

    print(f"loading base model: {args.base_model}")
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        quantization_config=quantization_config,
        dtype=model_dtype,
        device_map="auto" if not use_mps else None,
        trust_remote_code=True,
    )

    if args.qlora:
        model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=LORA_DEFAULTS["lora_dropout"],
        target_modules=LORA_DEFAULTS["target_modules"],
        bias=LORA_DEFAULTS["bias"],
        task_type=LORA_DEFAULTS["task_type"],
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    def format_conversation(example):
        return {"text": tokenizer.apply_chat_template(example["messages"], tokenize=False)}

    dataset = Dataset.from_dict({"messages": conversations})
    dataset = dataset.map(format_conversation)

    sft_config = SFTConfig(
        output_dir=str(args.output_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        warmup_ratio=SFT_DEFAULTS["warmup_ratio"],
        lr_scheduler_type=SFT_DEFAULTS["lr_scheduler_type"],
        logging_steps=SFT_DEFAULTS["logging_steps"],
        save_strategy=SFT_DEFAULTS["save_strategy"],
        bf16=use_bf16,
        fp16=use_fp16,
        seed=args.seed,
        report_to="none",
        remove_unused_columns=False,
        max_length=args.max_seq_length,
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=dataset,
        processing_class=tokenizer,
    )

    resume_from = None
    if args.resume and args.output_dir.exists():
        checkpoints = sorted(args.output_dir.glob("checkpoint-*"))
        if checkpoints:
            resume_from = str(checkpoints[-1])
            print(f"resuming from {resume_from}")

    print("starting training")
    trainer.train(resume_from_checkpoint=resume_from)

    final_dir = args.output_dir / "final"
    trainer.save_model(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))
    print(f"saved adapter to {final_dir}")

    meta = {
        "base_model": args.base_model,
        "lora_r": args.lora_r,
        "lora_alpha": args.lora_alpha,
        "qlora": args.qlora,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "grad_accum": args.grad_accum,
        "lr": args.lr,
        "max_length": args.max_seq_length,
        "seed": args.seed,
        "train_examples": len(conversations),
    }
    meta_path = args.output_dir / "train_meta.json"
    meta_path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print(f"metadata -> {meta_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
