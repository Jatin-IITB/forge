# ADR 0002 — Verification gate on all training data

**Status:** Accepted
**Date:** 2026-06-10

## Context
Distillation's #1 silent failure is training the student on the teacher's *confident mistakes*. A teacher model that is 92% correct will, unfiltered, teach the student its 8% errors as ground truth — permanently capping student quality below an already-imperfect ceiling. This is the analogue of persisting untrusted state: Forge refuses to train on what it cannot verify.

## Decision
No teacher output enters the training set unless it passes a **verification gate**:
1. **Self-consistency** — the teacher must agree with itself across `k` samples (majority/exact, task-dependent).
2. **Constraint validity** — output must satisfy the schema/grammar/type contract.
3. **Re-derivation (where cheap)** — an independent check (a verifier model, a rule, or execution) confirms the label.

Accept/reject rates are logged and published in the data card. Rejected examples are *not* discarded blindly — clusters of rejections are a signal about teacher weakness on a slice and feed the error analysis.

## Consequences
- (+) Removes the teacher's confident errors from the student's training signal.
- (+) Produces a measurable trust metric (accept rate) for the dataset.
- (−) Multiplies teacher-token cost (k-sampling) — the dominant cost; controlled by the per-iteration budget and the error-driven targeting (only generate where needed).
- (−) Over-aggressive gating can shrink coverage on genuinely hard slices; accept-rate-by-slice is monitored to catch this.
