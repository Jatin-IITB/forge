# run_002 — evaluation on the frozen test set

**Date:** 2026-08-15
**Student:** Qwen2.5-1.5B-Instruct + LoRA (r=16, attention-only), fp16 on MPS
**Training data:** `train_v2.jsonl` — 837 records (150 labelled by the 8B development
teacher + 687 construction-verified synthetic, ADR 0009)
**Teacher bar:** GPT-OSS-120B, micro-F1 0.9482 (`reports/baseline_120b.md`)
**Contract:** `pii_redaction_v2.yaml`

## Verdict: G1 fails, and not narrowly

| Metric | run_001 | **run_002** | target |
|---|---|---|---|
| micro-F1 | 0.5200 | **0.5750** | 0.9292 |
| micro-precision | 0.6600 | 0.6375 | — |
| micro-recall | 0.4300 | 0.5237 | — |
| partial-overlap F1 | — | 0.7209 | — |
| schema validity | 1.0000 | 0.9974 | 0.999 |
| redaction leak rate | — | 0.2186 | — |

**11 gates failed** — G1, G2 (0.9974 vs 0.999, marginal), and all nine high-severity
recall floors.

5.6× more targeted training data moved F1 by **+0.055**. `HONEST_ASSESSMENT.md` recorded in
advance that this result would mean the ADR 0009 imbalance diagnosis was wrong, so it is
rejected rather than reinterpreted. Full analysis in **ADR 0013**.

## The validator layer is doing the real work

| | model-only | system (+ validators) | Δ |
|---|---|---|---|
| micro-F1 | 0.5750 | **0.7334** | +0.1583 |
| micro-precision | 0.6375 | **0.7818** | +0.1443 |
| micro-recall | 0.5237 | **0.6906** | +0.1669 |
| min high-severity recall | **0.0000** | **1.0000** | +1.0000 |

Per-type, the gap is stark:

| High-severity type | model-only | system |
|---|---|---|
| DRIVER_LICENSE | **0.0000** | 1.0000 |
| AADHAAR | 0.1724 | 1.0000 |
| BANK_ACCOUNT | 0.2414 | 1.0000 |
| PASSWORD | 0.2632 | 1.0000 |
| API_KEY | 0.4483 | 1.0000 |
| PAN | 0.5862 | 1.0000 |
| SSN | 0.6111 | 1.0000 |
| CREDIT_CARD | 0.6341 | 1.0000 |
| PASSPORT | 0.6957 | 1.0000 |

**The model found zero of fifteen driver's licences.** Without ADR 0012's validator layer
this system would leak breach-severity identifiers at scale, which retrospectively makes
that decision load-bearing rather than a nicety.

Note the reporting discipline: 0.7334 is a *system* number and must never be quoted as the
distillation result. G1 is measured on model-only, and model-only is 0.5750.

## Where the model does work

Not everything is broken, and the pattern is informative:

| Type | F1 | Note |
|---|---|---|
| URL | 1.0000 | perfectly regular, unambiguous |
| EMAIL | 0.9265 | regular format, heavy pretraining exposure |
| IP_ADDRESS | 0.7931 | regular format |
| LOCATION | 0.7805 | contextual — and *better* than the teacher's 0.4138 |
| PHONE | 0.7381 | regular format |
| STREET_ADDRESS | **0.0923** | multi-token, fuzzy boundaries |
| PERSON | 0.5000 | 110 false negatives — the single largest error source |

The student **beats the teacher on LOCATION** (0.78 vs 0.41), so distillation is
transferring something real. It collapses on span types whose boundaries are ambiguous
(STREET_ADDRESS 0.09) and on high-frequency multi-entity types (PERSON, 110 FN).

## Diagnosis (ADR 0013)

Four measurements, each excluding a candidate cause:

1. **Not undertrained.** Loss reaches 1.16 by step 70 of 159 and stays flat (1.16–1.18) for
   the remaining 80 steps. More epochs cannot help.
2. **Not memorization or a diversity gap.** The student scores *higher* on test templates it
   never saw (0.6087) than on ones it did (0.5258). 52% of test template shapes are absent
   from training and this costs nothing.
3. **Not an output-format failure.** 0 of 116 sampled predicted spans were absent from the
   source text; the model copies exactly and emits valid JSON 99.7% of the time.
4. **The failure is under-detection plus boundary error.** 571 predicted spans against 695
   gold (0.82 ratio); exact-match 0.5750 against partial-overlap 0.7209.

The untested variable is **capacity**: both runs used r=16 on attention projections only —
4.36M trainable parameters, 0.28% of the model — leaving the MLP blocks unadapted. run_003
isolates it (identical data, r=64 + MLP, 73.9M trainable, 4.57%).

## Honest notes

- **G2 marginally failed** at 0.9974 vs the 0.999 threshold — one parse failure in 385. It
  is recorded as a failure rather than rounded up, because a threshold that bends under a
  single record is not a threshold.
- The 150 "teacher-annotated" records were labelled by the **8B development teacher**, which
  was later replaced for being too weak. No training record has ever been labelled by the
  120B teacher, so this project has not yet performed the distillation it is named for.
  That gap is separate from the capacity question and remains open.
