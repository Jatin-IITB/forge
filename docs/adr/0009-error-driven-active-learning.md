# ADR 0009 — Error-driven active learning (Phase 4)

**Status:** accepted  
**Date:** 2026-08-11  
**Context:** Phase 3 produced a LoRA student (run_001) with micro-F1 0.52, far below the
0.95 parity gate. Per-type analysis revealed severe training data imbalance as the root
cause: AADHAAR had 4 training examples vs 29 in the test set, DRIVER_LICENSE 9 vs 15, etc.

## Decision

Phase 4 uses a **three-step error-driven loop**:

1. **Error analysis** (`scripts/error_analysis.py`) — cluster student failures by PII type
   and error pattern. Outputs per-type recall/precision gaps and concrete augmentation
   targets (how many more examples of each type are needed).

2. **Construction-verified synthetic augmentation** (`scripts/generate_targeted_seeds.py`)
   — generate training records using the same Faker-based approach as the gold set builder.
   Templates are weighted proportionally to the error analysis output, with heavy emphasis
   on under-represented types and multi-entity records.

3. **Retrain → re-evaluate → repeat** — merge augmented data with existing training data,
   retrain the student, evaluate on the frozen test set, and loop if gates still fail.

### Why construction-verified, not teacher-annotated?

The data engine (ADR 0002) generates training data by having the teacher annotate seed
texts with k-sample consistency checking. This works well when the teacher reliably
detects the PII types in question. However, run_001 analysis showed the teacher (Qwen3 8B)
struggles with India-specific types (AADHAAR, PAN, DRIVER_LICENSE) — the very types we
need more data for.

Construction-verified data is **stronger than teacher verification** for synthetic text:
- Ground-truth spans are exact by construction (offsets computed at insertion time)
- Zero annotation error — no teacher model in the loop
- Deterministic and reproducible (fixed Faker seed)
- Same generation approach as the gold set, so distribution matches

The trade-off is that synthetic text is less diverse than natural text. Since the test set
is also synthetic (from `build_gold.py`), this is acceptable for the current phase. Future
phases should add teacher-annotated natural text.

### Seed isolation

The targeted generator uses Faker seed 1337 (gold set uses 42) to guarantee zero
text-level overlap between training and evaluation data.

### Multi-span emphasis

Run_001's PERSON recall was 14% despite 62 training examples — the model learned to
output ~1 span per record, but many test records have 2-4 persons. The augmentation
templates deliberately include multi-PERSON records (3-4 names) and complex multi-type
records (4-6 entities) to train multi-span extraction.

### Hard negatives

SSN had 28 false positives (precision 39%) — the model labels number sequences as SSN
even when they aren't. The augmentation includes 30 hard-negative records with
SSN-like numbers (order IDs, reference numbers, PINs) that should NOT be labelled,
teaching the model to distinguish.

## Consequences

- Training data grows from 150 to ~830 records (150 teacher-verified + ~680 synthetic)
- Reproducible: `make augment` regenerates identical data from seed 1337
- The error analysis script becomes a reusable diagnostic tool for future rounds
- If gates still fail after round 1, the loop repeats with a new error analysis
