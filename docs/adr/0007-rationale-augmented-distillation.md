# ADR 0007 — Rationale-augmented distillation

**Status:** Accepted
**Date:** 2026-08-07

## Context
Standard distillation trains the student to mimic the teacher's output labels. Rationale-augmented distillation (Hsieh et al. 2023, "Distilling Step-by-Step") additionally trains the student on the teacher's *reasoning trace* — why each label was chosen. The student learns to generate the rationale during SFT, which acts as an intermediate supervision signal, then the rationale is stripped at serving time (the student outputs only the spans).

For PII detection specifically: a rationale like "this is a person name because it appears after 'Dear' in a greeting context" provides richer signal than just the label PERSON. The student learns contextual patterns, not just surface patterns.

## Decision
Enhance the teacher prompt to request a brief rationale per span. The output format becomes:

```json
{"spans": [{"label": "PERSON", "text": "Jane Doe", "rationale": "full name in email greeting"}]}
```

The rationale field is:
- **Present in training data** (the student sees it during SFT).
- **Requested during teacher inference** for training data generation.
- **NOT requested during student inference** for evaluation — the student outputs the standard `{label, text}` format at serving time.
- **Optional** in the schema — eval and scoring ignore it entirely.

## Consequences
- (+) Richer supervision signal for SFT — the student learns contextual patterns.
- (+) No change to eval pipeline — rationale is ignored during scoring.
- (+) No serving cost — rationale is not generated at inference time.
- (−) Slightly longer teacher responses during data generation (more tokens per example).
- (−) Training data is larger per example (rationale text).
