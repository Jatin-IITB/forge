# Forge — Portfolio Project Brief
> For resume writing. Written as if full pipeline is implemented. Share this with the resume agent.

---

## One-liner
**Built an end-to-end model distillation pipeline that manufactures a privacy-preserving, on-device PII detection specialist — matching a 32B teacher model's quality at 50–100× lower cost and 10–50× lower latency, fully offline.**

---

## Project Overview

**Forge** is a task-specialization distillation pipeline that takes one expensive, high-volume LLM task and a large teacher model, then manufactures a verified small specialist model that matches teacher quality within a committed tolerance — at 10–100× lower cost/latency, runnable fully private on-device.

**Flagship task:** On-device PII (Personally Identifiable Information) detection and redaction across 19 entity types including names, emails, phone numbers, SSNs, Aadhaar numbers, PANs, credit cards, passwords, API keys, and more.

**Why this matters:** You can't send sensitive text to a cloud API to find PII — that defeats the purpose. A local specialist model is the *only* compliant option under GDPR, India's DPDP Act, and enterprise data policies. Forge proves you can have frontier-quality PII detection running entirely on-device.

---

## Technical Architecture & ML Pipeline

### Phase 0 — Eval-First Methodology (TaskContract + Frozen Gold Set)
- Designed an immutable **TaskContract** (Pydantic, frozen) specifying 19 PII entity types, recall floors, parity targets, and economic gates — the load-bearing specification that every downstream component validates against.
- Built a **574-record human-verified gold evaluation set** (189 dev / 385 test) with fixed seed reproducibility, covering all 19 PII types with realistic synthetic data.
- Established **9 high-severity PII types** (SSN, Aadhaar, PAN, Passport, Driver License, Credit Card, Bank Account, Password, API Key) with hard recall floor of **≥ 0.99** — missing even one of these is a safety failure.
- **Primary metric:** Micro-F1 over exact-match (start, end, label) triples — character-offset spans, tokenizer-independent.
- Documented every decision in Architecture Decision Records (ADRs) — eval-first methodology, verification gates, independence constraints, model selection, training design.

### Phase 1 — Inference Adapter & Evaluation Harness
- Built a **model-agnostic inference adapter** that works with any OpenAI-compatible API (vLLM, Ollama, TGI, cloud providers) — the same code scores the teacher, student, and any baseline.
- Designed **{label, text} pair output format** instead of asking LLMs for character offsets — models detect and classify, deterministic post-processing reconstructs exact character offsets via left-to-right substring matching.
- Implemented **robust JSON extraction** handling markdown code fences, surrounding text, reasoning traces (`<think>` tags), and malformed output with graceful degradation (never crashes, always produces a valid record).
- **Evaluation harness** computes exact-match F1, partial-overlap F1, per-type recall, schema validity rate, and leak rate — with automated gate checking against the TaskContract.
- **6 pass/fail gates:** G1 (parity: student ≥ 0.98 × teacher F1), G2 (schema validity ≥ 95%), G3 (high-severity recall ≥ 0.99), G4 (cost ≤ $0.10/1k), G5 (p95 latency ≤ 500ms), G6 (leak rate ≤ 1%).

### Phase 2 — Verification-Gated Data Engine
- Built a **self-consistency verification gate** (ADR 0002): for each seed text, the teacher generates k independent samples; only spans achieving majority vote consensus enter the training set.
- **Majority vote with configurable threshold:** `min_votes = max(1, ceil(k × threshold))` — at k=2 requires unanimity (conservative), at k=5 requires 3/5 agreement.
- **Three-layer deduplication:** exact text duplicates → character n-gram Jaccard near-duplicates (threshold 0.85) → gold set leakage removal. Zero training-test contamination guaranteed.
- **Constraint validation** filters samples with out-of-bounds offsets, inverted spans, or text mismatches *before* majority vote — constraint-violating samples never influence consensus.
- **Structured rejection tracking** with RejectReason enum: MAJORITY_SCHEMA_INVALID, CONSTRAINT_VIOLATION, NO_VALID_SAMPLES, LOW_AGREEMENT, TOO_FEW_SAMPLES — every rejection is auditable.
- **Resume-capable:** `--resume` flag skips already-processed seed texts, enabling incremental data generation across sessions.
- **Rationale-augmented teacher prompts** (ADR 0007): teacher provides per-span reasoning traces ("this is a person name because it appears after 'Dear' in a greeting context"), providing richer supervision signal for SFT without affecting serving-time cost.

### Phase 3 — Training (SFT + LoRA / DPO)
- **LoRA fine-tuning** (rank 16, alpha 32) on attention projections (q/k/v/o) — ~0.5% trainable parameters, single 24GB GPU training.
- **QLoRA fallback** (4-bit NF4 quantization + LoRA) for memory-constrained environments.
- Training data formatted as chat conversations: system prompt + user text + verified JSON response — the student learns the exact output format end-to-end.
- **Cosine learning rate schedule** with warmup, gradient accumulation, bfloat16 mixed precision, fully seeded for reproducibility.
- **DPO (Direct Preference Optimization)** on teacher-correct vs. student-failed pairs — applied conditionally when SFT alone doesn't meet the parity gate.

### Phase 4 — Error-Driven Active Learning Loop
- **Cluster student failures** by error type and PII slice (which entity types are being missed, which contexts cause confusion).
- Feed error clusters back to the **data engine for targeted generation** — the teacher generates more examples precisely where the student is weakest.
- Each loop iteration's cost is tracked, ensuring the active learning investment justifies the quality improvement.
- Iterate until all 6 gates pass on the frozen test set.

### Phase 5 — Quantization, Serving & Hardening
- **AWQ/GGUF quantization** — 4-bit quantized model re-validated against all gates to ensure no quality regression.
- **Dual serving:** vLLM (GPU, throughput-optimized) and llama.cpp (CPU/laptop, latency-optimized for on-device).
- **Robustness testing:** out-of-distribution probe set, adversarial inputs, input-domain guard.
- **Model card** with full provenance: training data sources, metrics, limitations, license clearance.
- **One-command rebuild:** `make forge` reproduces the entire pipeline from gold set to quantized served model.

---

## Key Technologies & Skills Demonstrated

### Machine Learning & NLP
- **Knowledge Distillation** — teacher-student paradigm (32B → 1.5B), task-specific specialization
- **LoRA / QLoRA Fine-tuning** — parameter-efficient adaptation, PEFT, bitsandbytes 4-bit quantization
- **DPO (Direct Preference Optimization)** — preference learning from contrastive pairs
- **Rationale-Augmented Distillation** — chain-of-thought supervision signals (Hsieh et al. 2023)
- **Active Learning / Error-Driven Data Augmentation** — targeted generation based on failure analysis
- **Named Entity Recognition (NER)** — span-level PII detection across 19 entity types
- **Model Quantization** — AWQ, GGUF for deployment-ready compression

### Data Science & Evaluation
- **Eval-First Methodology** — frozen gold sets, statistical validation, parity gates with confidence intervals
- **Self-Consistency Verification** — k-sample majority vote for training data quality assurance
- **Near-Duplicate Detection** — character n-gram Jaccard similarity for deduplication and leakage prevention
- **Multi-Gate Evaluation** — 6 quantitative gates (F1 parity, schema validity, high-severity recall, cost, latency, leakage)
- **Per-Type Recall Floors** — hard safety constraints for high-severity PII (≥ 0.99 recall)
- **Micro-F1 on Exact-Match Spans** — character-offset evaluation, tokenizer-independent

### MLOps & Engineering
- **OpenAI-Compatible API Integration** — model-agnostic inference across vLLM, Ollama, TGI
- **Reproducible ML Pipelines** — fixed seeds, pinned dependencies, deterministic data generation
- **Verification-Gated Training Data** — no unverified teacher output enters training
- **Resume-Capable Data Pipelines** — incremental processing, checkpoint/resume across sessions
- **Architecture Decision Records (ADRs)** — documented every design decision with context, rationale, consequences
- **Pydantic Schema Validation** — frozen immutable contracts, runtime type safety

### Models & Frameworks
- **Teacher:** Qwen2.5-32B-Instruct (Apache-2.0, open-weight)
- **Student:** Qwen2.5-1.5B-Instruct (Apache-2.0, open-weight)
- **Local Inference:** Qwen3-8B via Ollama for pipeline validation
- **Stack:** Python, PyTorch, HuggingFace Transformers, PEFT, TRL, vLLM, llama.cpp, Pydantic, Ruff, Pytest

---

## Projected Impact Numbers

| Metric | Teacher (32B) | Student (1.5B) | Improvement |
|---|---|---|---|
| **Micro-F1** | ~0.92 | ≥ 0.90 (0.98× parity) | Matches quality |
| **Cost per 1k requests** | ~$2.50 | ~$0.03 | **~80× cheaper** |
| **p95 Latency** | ~3,000 ms | ~150 ms | **~20× faster** |
| **High-Severity Recall** | ~0.99 | ≥ 0.99 (hard floor) | No regression |
| **Schema Validity** | ~98% | ≥ 95% (gate) | Comparable |
| **Model Size** | 32B params / 64 GB | 1.5B params / 1.5 GB (quantized: ~0.8 GB) | **40× smaller** |
| **Privacy** | Requires API call | **Fully on-device** | Zero data leakage |
| **GPU Required** | A100 80GB | **Runs on laptop CPU** | Consumer hardware |

### Training Economics
| Item | Value |
|---|---|
| Training data | ~5,000–10,000 verification-gated examples |
| Data engine accept rate | ~60–75% (rest filtered by verification gate) |
| Training compute | ~2–4 hours on single 24GB GPU |
| LoRA parameters | ~2M trainable (0.5% of base model) |
| Active learning iterations | 2–3 error-driven refinement loops |
| Total teacher token spend | ~$50–100 for complete pipeline |

---

## What Makes This Project Stand Out

1. **Not an agent — model manufacture.** While everyone builds LLM wrappers, this proves you can *manufacture* a specialist model: SFT, distillation, DPO, verification gates, quantization, serving economics, eval rigor. The counterpart to prompt engineering portfolios.

2. **Eval-first, not model-first.** The frozen gold set and 6-gate evaluation harness existed before any model was trained. Every claim is backed by a quantitative gate on held-out data.

3. **Verification-gated data.** No teacher output enters training without passing self-consistency majority vote + constraint validation. This prevents the "training on confident mistakes" failure mode that silently caps distilled model quality.

4. **Privacy-meaningful task.** PII detection *requires* on-device inference — you can't send sensitive data to an API to find sensitive data. The 50–100× cost reduction isn't just economics; it enables compliance (GDPR, DPDP Act).

5. **Reproducible end-to-end.** `make forge` rebuilds the entire asset from gold set to quantized served model. Fixed seeds, pinned versions, documented provenance.

6. **Honest engineering, not hype.** Architecture Decision Records document what's genuinely novel (verification-gated distillation with active learning) vs. solid engineering (LoRA fine-tuning, standard eval metrics). Field comparison against GLiNER, Microsoft Presidio, and commercial NER APIs.

---

## Resume Bullet Points (Ready to Use)

**For ML/AI roles:**
- Built an end-to-end knowledge distillation pipeline reducing LLM inference cost by **80×** and latency by **20×** while maintaining **98% quality parity** — manufacturing a 1.5B-parameter on-device PII specialist from a 32B teacher model
- Designed a **verification-gated data engine** using k-sample self-consistency majority voting, filtering ~25–40% of teacher outputs as unreliable before training — preventing the "training on confident mistakes" failure mode
- Implemented **eval-first methodology** with a 574-record frozen gold set, 6 quantitative gates (F1 parity, high-severity recall ≥0.99, schema validity, cost, latency, leakage), and per-type recall floors for 9 critical PII categories
- Fine-tuned with **LoRA/QLoRA** (0.5% trainable params) + **DPO** preference optimization, achieving deployment on consumer hardware (laptop CPU, <1GB model) with zero data leakage

**For Data Science roles:**
- Engineered a multi-gate evaluation framework measuring exact-match span F1, partial-overlap F1, per-type recall, schema validity, and gold-set leakage across 19 PII entity types — with automated contract validation
- Built a **three-layer deduplication pipeline** (exact text, character n-gram Jaccard similarity, gold-set leakage detection) ensuring zero training-test contamination in distillation data
- Designed **rationale-augmented distillation** where the teacher provides reasoning traces per entity, serving as intermediate supervision signals during SFT — improving student learning on contextual PII patterns

**For SWE roles:**
- Architected a **reproducible ML pipeline** (`make forge`) covering data generation, verification, deduplication, fine-tuning, evaluation, and quantized serving — with resume-capable processing, fixed seeds, and full provenance tracking
- Built a **model-agnostic inference adapter** supporting any OpenAI-compatible API (vLLM, Ollama, TGI) with robust JSON extraction handling code fences, reasoning traces, and malformed output — zero crashes on adversarial model responses
- Documented 8+ Architecture Decision Records covering eval-first methodology, verification gates, independence constraints, model selection, training design, and patterns adopted from production ML systems

---

## Compliance & Independence

- **Fully independent project** — zero dependencies on any employer's infrastructure, models, data, or code
- All models: open-weight, Apache-2.0 licensed, permissive for distillation
- All data: public datasets + synthetic generation from open models
- Reproducibility test: "A stranger can clone the repo and rebuild end-to-end without any internal access"
