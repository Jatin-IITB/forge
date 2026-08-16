# ADR 0006 — Patterns adopted from public prior art

**Status:** Accepted
**Date:** 2026-08-07
**Revised:** 2026-08-16 — rewritten to cite public sources directly (see Note on revision)

## Context

Forge's pipeline needs several well-established techniques from the distillation and
ML-ops literature. This ADR records which ones we adopt, the public prior art each rests
on, and what each changes in the codebase — so a reader can trace every design decision to
a citable source rather than to taste.

**ADR 0003 remains absolute:** no code, data, configuration, model weights, or structural
imports from any non-public source enter Forge. The litmus test is unchanged: *"Could a
stranger clone this repo and rebuild end-to-end without access to anything private?"*

## Patterns adopted

### 1. Error-driven teacher loop
**Prior art:** Active learning and error-driven data augmentation — Settles 2009,
*Active Learning Literature Survey*; iterative-refinement approaches such as Constitutional
AI. The student fails, the teacher generates targeted examples, retrain, repeat.
**Status:** Implemented as the Phase 4 error-cluster → targeted data-engine loop
(`scripts/error_analysis.py` → `scripts/generate_targeted_seeds.py`).

### 2. Multi-signal weighted evaluation
**Prior art:** Multi-metric evaluation is standard in NLP; SemEval shared tasks routinely
combine precision, recall, and task-specific signals into a reported suite.
**Status:** The eval harness reports exact-match F1, partial-overlap F1, leak rate, and
per-type recall floors. A weighted composite for DPO reward modelling remains deferred —
if built, exact-match F1 is primary, high-severity recall stays a **hard gate rather than a
weighted term** (a breach is not tradeable against average quality), leak rate secondary.

### 3. Checkpoint + resume for long runs
**Prior art:** Universal in ML training — PyTorch `save_state_dict`, HuggingFace `Trainer`
`resume_from_checkpoint`.
**Status:** Implemented across every long-running stage after machine sleep destroyed three
separate runs: `--resume` on the data engine, on inference (per-record flush), and on
training (`--save-steps`, plus the fix in ADR 0013's neighbourhood for resumed runs
inheriting a stale save schedule).

### 4. Chain-of-thought rationale augmentation
**Prior art:** Wei et al. 2022, *Chain-of-Thought Prompting Elicits Reasoning*; Hsieh et al.
2023, *Distilling Step-by-Step* (rationale-augmented distillation).
**Status:** The teacher prompt requests a brief rationale per span. Rationales enter the
training data and are stripped at inference, so the student sees the reasoning trace during
SFT but emits only spans at serving time.

### 5. Tolerant output parsing
**Prior art:** Robust LLM output parsing is standard practice — LangChain, guidance, and
outlines all implement fallback parsing.
**Status:** Implemented in `parse_response`, which degrades gracefully to an empty span list
on malformed output rather than raising. Schema validity is measured, not assumed.

## Patterns explicitly NOT adopted

**Knowledge-base patching between iterations.** Applies to systems whose task knowledge
lives in an external store. PII detection carries its knowledge in model weights, so there
is nothing to patch.

**Prompt-optimization frameworks (e.g. DSPy).** Rejected under ADR 0003's independence
principle — every framework is a dependency a stranger must also install and understand.
Prompts here are hand-written and version-controlled. If optimization becomes necessary it
will be a grid search over prompt variants scored against the gold set.

**Multi-agent orchestration.** PII detection is single-step span extraction; agent
coordination adds moving parts without addressing any measured failure.

## Consequences

- Every adopted technique traces to a citable public source, so the pipeline is auditable
  by a reader with no special context.
- Two items stayed deferred rather than being built speculatively (composite reward,
  prompt optimization) — both are now gated on evidence from a measured failure.
- The "not adopted" list is as load-bearing as the adopted one: it records what was
  considered and rejected, which is what stops a later reader assuming an omission was an
  oversight.

## Note on revision (2026-08-16)

The original version of this ADR framed these patterns as observations of a specific
third-party system and named its internal components. That framing was wrong on two counts.
It described architecture that is not ours to publish, and it was **misleading about
provenance** — every technique here is independently documented in public literature, which
is precisely what made each one safe to adopt. The citations above were always the real
justification; this revision makes them the only one. No technical content changed.
