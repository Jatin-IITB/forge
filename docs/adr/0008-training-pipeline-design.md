# ADR 0008 — Training pipeline design (SFT + LoRA)

**Status:** Accepted
**Date:** 2026-08-08

## Context
Phase 3 requires training a student model (Qwen2.5-1.5B-Instruct or equivalent small model) on verified training data from the data engine. The pipeline must:
1. Fine-tune efficiently on a single GPU (24GB or less).
2. Produce schema-valid structured output (JSON with PII spans).
3. Be fully reproducible (seeded, pinned dependencies).
4. Support rationale-augmented targets (ADR 0007).

## Decision

### Training method: SFT with LoRA
- **LoRA** (Low-Rank Adaptation) applied to attention layers (`q_proj`, `v_proj`, `k_proj`, `o_proj`).
- Rank 16, alpha 32 (standard starting point; tunable).
- Uses HuggingFace `trl.SFTTrainer` with PEFT — battle-tested, well-documented.
- QLoRA (4-bit base + LoRA) as fallback if memory is tight.

### Data format
Training examples are formatted as chat conversations:
- **System**: `SYSTEM_PROMPT` (the student prompt, NOT the teacher prompt)
- **User**: `"Detect all PII in this text:\n\n{text}"`
- **Assistant**: the verified JSON response from the data engine

The rationale field (ADR 0007) is present in training targets when available. At inference time, the student prompt does not request rationale, so the model learns to output it only when prompted — but the reasoning trace provides richer supervision during SFT.

### Why not DPO first
DPO requires preference pairs (chosen vs. rejected). We don't have these until the student makes errors we can contrast against teacher-correct outputs. Phase 4's error loop generates these pairs naturally. SFT first, DPO second — only if the parity gate isn't met.

### Dependencies
All open-source, permissive licenses:
- `transformers` (Apache-2.0)
- `peft` (Apache-2.0)
- `trl` (Apache-2.0)
- `bitsandbytes` (MIT) — for QLoRA quantization
- `datasets` (Apache-2.0)
- `accelerate` (Apache-2.0)

Independence litmus test (ADR 0003): passes. All deps are public PyPI packages. No internal tools or models.

## Consequences
- (+) LoRA keeps the trainable parameter count small (~0.5% of base model), enabling single-GPU training.
- (+) Adapter can be merged into base model for deployment or kept separate for A/B testing.
- (+) trl/PEFT are the de facto standard — well-maintained, documented, community-supported.
- (-) LoRA may underperform full fine-tuning on very hard tasks. Mitigation: can increase rank or switch to full FT if needed.
- (-) QLoRA adds quantization noise. Mitigation: only use if memory requires it; validate against full-precision LoRA.
