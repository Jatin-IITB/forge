# ADR 0006 — Patterns informed by observation, not import

**Status:** Accepted
**Date:** 2026-08-07

## Context
We observed the architecture of Prism (an internal enterprise NL-to-SQL system) to identify techniques that could improve Forge's distillation pipeline. Several patterns in Prism align with well-known techniques in the distillation and ML-ops literature. This ADR documents which patterns we adopt, how they map to public prior art, and what the independence boundary is.

**ADR 0003 is absolute:** no code, data, configuration, model weights, or structural imports from Prism or any internal codebase enter Forge. The litmus test remains: "Could a stranger clone this repo and rebuild end-to-end without internal access?"

## Patterns adopted (all publicly known)

### 1. Error-driven teacher loop
**Observed in:** Prism's OptimizationPipeline (student fails → teacher generates → feedback → retry).
**Public prior art:** Active learning / error-driven data augmentation is standard (Settles 2009, "Active Learning Literature Survey"; Anthropic's Constitutional AI uses iterative refinement). Our Phase 4 error-cluster → targeted data-engine loop already implements this per ACTION_PLAN.md.
**What changes:** Nothing — we already designed this. Confirms the approach.

### 2. Multi-signal weighted evaluation
**Observed in:** Prism's SQLEvaluator (semantic similarity 70% + execution match 25% + performance 5%).
**Public prior art:** Multi-metric evaluation is standard in NLP (SemEval shared tasks routinely combine precision, recall, and task-specific signals). Our eval harness already has exact-match F1 + partial-overlap F1 + leak rate + per-type recall floors.
**What changes:** Consider adding a **weighted composite score** that combines our signals into a single optimization target for Phase 3 DPO reward modeling. The weighting would be: exact-match F1 (primary), high-severity recall penalty (hard gate, not weighted), leak rate (secondary). Document in a future ADR when Phase 3 design is finalized.

### 3. Checkpoint + resume for long runs
**Observed in:** Prism's optimization checkpoint system for recovery across iterations.
**Public prior art:** Checkpointing is universal in ML training (PyTorch `save_state_dict`, HuggingFace `Trainer` resume_from_checkpoint). Our data engine and training pipeline should support resumable runs.
**What changes:** Add `--resume` flag to `run_data_engine.py` (append to existing train.jsonl, skip already-processed seed IDs). Add to Phase 3 training script when built.

### 4. Chain-of-thought rationale augmentation
**Observed in:** Prism's DSPy ChainOfThought signatures throughout.
**Public prior art:** CoT prompting (Wei et al. 2022, "Chain-of-Thought Prompting Elicits Reasoning"); rationale-augmented distillation (Hsieh et al. 2023, "Distilling Step-by-Step"). ACTION_PLAN Phase 2 already calls for "labeled examples with rationales."
**What changes:** Enhance the teacher prompt in `forge/inference.py` to request a brief rationale per span (why this substring is PII of this type). The rationale enters training data but is stripped at inference time (student learns the reasoning trace during SFT but only outputs spans at serving). This is a Phase 3 design decision.

### 5. Tolerant output parsing
**Observed in:** Prism's ReActWithTolerantFinish — catch malformed model responses without hard crash.
**Public prior art:** Robust LLM output parsing is universal (LangChain, guidance, outlines all implement fallback parsing). Our `parse_response` already does this (graceful degradation to empty spans on parse failure).
**What changes:** Nothing — already implemented.

## Patterns explicitly NOT adopted

### KB patching / metadata reload
Prism patches a knowledge base between optimization iterations. Forge's task (PII detection) has no equivalent KB — the "knowledge" is in the model weights, not an external config. This pattern does not transfer.

### DSPy framework
Prism uses DSPy for prompt optimization. Forge deliberately avoids framework dependencies (ADR 0003 independence principle). Our prompts are hand-written and version-controlled. If prompt optimization is needed, it will be a simple grid search over prompt variants evaluated against the gold set, not a framework dependency.

### Multi-agent swarm
Prism uses swarm orchestration for complex multi-step reasoning. PII detection is a single-step extraction task — multi-agent coordination adds complexity without benefit.

## Consequences
- (+) Validates several Forge design decisions were already aligned with battle-tested patterns.
- (+) Identifies two concrete improvements: composite reward score for DPO, and rationale-augmented teacher prompts.
- (+) Clear documentation of what came from where, maintaining the independence guarantee.
- (−) None — no code or data was imported.
