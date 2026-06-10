# ADR 0001 — Eval-first: the gold set and parity gate exist before the model

**Status:** Accepted
**Date:** 2026-06-10

## Context
The most common self-deception in fine-tuning is choosing or relaxing the success target *after* seeing model results. This produces impressive-looking numbers that mean nothing. Aroha's AHSI enforces a structural discipline ("mandatory index-before-retrieve") rather than trusting the model/engineer to do the right thing; we adopt the same stance for evaluation.

## Decision
The held-out **gold test set** is built, human-verified, leakage-checked, and **frozen** in Phase 0 — before any training data is generated. The **parity gate** (`student_score ≥ 0.98 × teacher_score`, plus the cost/latency/safety gates) is committed to git in the `TaskContract` before modeling begins and is **immutable for a run**. The frozen test set is not inspected during data generation or training; it is touched only by the evaluation harness in Phase 4.

## Consequences
- (+) Results are honest by construction; the target cannot be moved to fit the model.
- (+) Forces clarity about *what success means* before spending compute.
- (−) Up-front human labeling cost before any model exists — accepted as the highest-leverage spend.
- If the contract is genuinely wrong (bad metric), the run is voided and a **new** contract is written — not edited mid-run.
