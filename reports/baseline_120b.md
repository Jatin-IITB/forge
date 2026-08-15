# Teacher baseline — GPT-OSS-120B on the frozen test set

**Date:** 2026-08-15
**Model:** `gpt-oss-120b` (Apache-2.0, MoE ~117B total / ~5.1B active), Cerebras, `reasoning_effort=low`
**Data:** `data/gold/test.jsonl` — 385 records, frozen, 695 gold spans
**Contract:** `contracts/pii_redaction_v2.yaml`
**Harness:** `scripts/run_inference.py` + `scripts/run_eval.py`, unchanged from Phase 1

This run establishes the **parity denominator**. It was measured *before* the student
finished training, so the G1 threshold cannot be back-fitted to a student result.

## Headline

| Metric | Value |
|---|---|
| **micro-F1 (primary)** | **0.9482** |
| micro-precision | 0.9615 |
| micro-recall | 0.9353 |
| partial-overlap F1 | 0.9729 |
| redaction leak rate | 0.0273 |
| avg latency | 2.61 s |
| p50 / p95 latency | 0.59 s / 8.02 s |

**⇒ G1 target for the student: micro-F1 ≥ 0.98 × 0.9482 = `0.9292`.**

run_001 scored 0.52. The gap to close is large.

### Schema validity caveat

The meta reports 302/302 (100%) because an earlier segment of this run was killed by
machine sleep before its meta was written; 83 records carry no recorded validity flag.
Auditing those 83 directly: 5 have empty predictions and **all 5 are true negatives in
gold**, so there are **zero detectable parse failures**. Effective schema validity is
385/385. The 0.7844 printed by the harness is a bookkeeping artifact, not a model failure —
recorded here rather than quietly corrected.

## The decisive finding: the teacher fails the high-severity gate

The contract sets an **absolute** recall floor of 0.99 on 9 high-severity types — missing
one of these is a reportable breach, so it is gated harder than overall parity. Measured on
the teacher:

| Type | Recall | vs 0.99 floor | Missed |
|---|---|---|---|
| CREDIT_CARD | 1.0000 | ✅ PASS | 0 / 41 |
| SSN | 1.0000 | ✅ PASS | 0 / 18 |
| API_KEY | 1.0000 | ✅ PASS | 0 / 29 |
| PASSWORD | 0.9474 | ❌ FAIL | 1 / 19 |
| PAN | 0.9310 | ❌ FAIL | 2 / 29 |
| PASSPORT | 0.9130 | ❌ FAIL | 2 / 23 |
| AADHAAR | 0.8276 | ❌ FAIL | 5 / 29 |
| BANK_ACCOUNT | 0.7931 | ❌ FAIL | 6 / 29 |
| DRIVER_LICENSE | 0.5333 | ❌ FAIL | 7 / 15 |

**6 of 9 high-severity types fail the floor at the teacher.**

### Why this is structurally decisive, not just disappointing

The student is trained to imitate the teacher. Distillation transfers the teacher's
behaviour, including its blind spots — the verification gate can discard teacher noise but
**cannot manufacture signal the teacher never produced** (the same argument that motivated
construction-verified augmentation in ADR 0009, now confirmed at the teacher level).

So the high-severity gate is **unreachable by distillation from this teacher**, no matter
how well the student trains. A student that achieved *perfect* parity would inherit
DRIVER_LICENSE recall of 0.53 against a 0.99 floor.

This is not a reason to move the gate. `SUCCESS.md` is explicit that renegotiating a
threshold after seeing results voids the run, and the floor encodes a real-world
consequence (an unredacted Aadhaar number is a breach) that does not become less true
because our teacher is bad at it.

### It is also the exact failure `HONEST_ASSESSMENT.md` predicted

That document, written before this measurement, stated:

> for well-formed identifiers (credit cards, SSN, Aadhaar), a well-written regex with a
> checksum is likely to beat a 1.5B model on both precision and latency, at zero cost.

The data now supports it. Every failing type is a **structured identifier with a known
format** — exactly the class where deterministic validation is strong and language models
are unreliable. The types the teacher aces (EMAIL, PHONE, URL, IP, SSN, API_KEY at 1.000)
are either equally structured *or* well represented in pretraining; the ones it fails are
India-specific or format-ambiguous (a driver's licence number looks like many other codes).

## Recommendation: the system must become a hybrid

The honest engineering answer is that **high-severity structured identifiers should not be
detected by a language model alone.** A deterministic validator layer — Verhoeff checksum
for Aadhaar, Luhn for credit cards, format regex for PAN/passport/driver's licence — run
alongside the model, with union recall, would plausibly take these types to ~1.00 recall at
near-zero cost and latency.

That reframes the deliverable from "specialist model" to **"specialist system: rules for
what rules do well, a distilled model for the contextual remainder"** — which is what a
privacy engineer would actually ship, and is a stronger portfolio claim than pretending a
1.5B model should memorize checksum arithmetic.

It also preserves every gate: G1 parity is still measured model-to-model on the same
harness; the validator layer is what carries the high-severity floor.

**Decision required before the parity loop continues** — recorded as ADR 0012.

## Secondary observations

- **LOCATION recall 0.2727** (16 of 22 missed) — the worst non-high-severity type. Likely a
  taxonomy-boundary problem: the teacher folds city names into STREET_ADDRESS or omits them.
  Worth a prompt clarification rather than a data fix.
- **CREDIT_CARD precision 0.8723** (6 FP) with perfect recall — the teacher over-flags
  number sequences, the same failure mode the student showed on SSN in run_001.
- **DRIVER_LICENSE has 7 FP and 7 FN** — not blindness but boundary/format confusion; it
  finds *something* in the right region and mislabels or mis-spans it.
- **p95 8.02 s vs p50 0.59 s** — a 13× spread, characteristic of reasoning-model
  variability. Relevant to G4: the student's p95 must beat 1.60 s.
