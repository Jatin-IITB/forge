"""Training utilities — data formatting and LoRA configuration for SFT.

Converts verified PIIRecords into chat-format training examples and provides
LoRA configuration defaults for fine-tuning a small student model.
"""

from __future__ import annotations

import json
from pathlib import Path

from forge.inference import SYSTEM_PROMPT, USER_TEMPLATE
from forge.schema import PIIRecord


def record_to_chat(record: PIIRecord) -> list[dict[str, str]]:
    """Convert a PIIRecord to a chat-format training example.

    The assistant response is the JSON that the student should produce:
    {"spans": [{"label": "...", "text": "..."}]}

    Offsets are not included — the model outputs {label, text} pairs and
    offset reconstruction happens post-hoc in parse_response.
    """
    spans_out = [{"label": s.label.value, "text": s.text} for s in record.spans]
    assistant_content = json.dumps({"spans": spans_out}, ensure_ascii=False)

    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": USER_TEMPLATE.format(text=record.text)},
        {"role": "assistant", "content": assistant_content},
    ]


def load_training_data(path: Path) -> list[list[dict[str, str]]]:
    """Load train.jsonl and convert to chat-format examples."""
    conversations = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            record = PIIRecord.model_validate_json(line)
            conversations.append(record_to_chat(record))
    return conversations


LORA_DEFAULTS = {
    "r": 16,
    "lora_alpha": 32,
    "lora_dropout": 0.05,
    "target_modules": ["q_proj", "v_proj", "k_proj", "o_proj"],
    "bias": "none",
    "task_type": "CAUSAL_LM",
}

SFT_DEFAULTS = {
    "num_train_epochs": 3,
    "per_device_train_batch_size": 4,
    "gradient_accumulation_steps": 4,
    "learning_rate": 2e-4,
    "warmup_ratio": 0.1,
    "lr_scheduler_type": "cosine",
    "logging_steps": 10,
    "save_strategy": "epoch",
    "bf16": True,
    "seed": 42,
    "max_length": 2048,
}
