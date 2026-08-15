# ADR 0012 — High-severity identifiers move to a deterministic validator layer

**Status:** proposed — requires sign-off before the parity loop continues
**Date:** 2026-08-15
**Depends on:** `reports/baseline_120b.md` (the measurement that forces this)

## Context

The teacher baseline (GPT-OSS-120B, frozen 385-record test set) scored micro-F1 **0.9482**,
which is a good bar. But the per-type breakdown killed an assumption the project was built on:

| High-severity type | Teacher recall | Floor |
|---|---|---|
| DRIVER_LICENSE | 0.5333 | 0.99 |
| BANK_ACCOUNT | 0.7931 | 0.99 |
| AADHAAR | 0.8276 | 0.99 |
| PASSPORT | 0.9130 | 0.99 |
| PAN | 0.9310 | 0.99 |
| PASSWORD | 0.9474 | 0.99 |

**6 of 9 high-severity types fail the contract's recall floor at the teacher.**

Distillation transfers the teacher's behaviour, blind spots included. The verification gate
(ADR 0002) can discard teacher noise but cannot manufacture signal the teacher never
produced — the same reasoning that drove construction-verified augmentation in ADR 0009,
now confirmed one level up. Therefore:

> **The high-severity gate is unreachable by distillation from this teacher.** A student at
> *perfect* parity would inherit DRIVER_LICENSE recall of 0.53 against a 0.99 floor.

Three responses were possible. Only one is honest.

## Options considered

**A. Move the gate.** Rejected outright. `SUCCESS.md` states that renegotiating a threshold
after seeing results voids the run. The floor also encodes a real consequence — an
unredacted Aadhaar number is a reportable breach — which does not become less true because
our teacher is weak at it.

**B. Fix the teacher.** Better prompting, few-shot exemplars, or higher reasoning effort
might lift these types. Worth attempting and cheap to test, but it cannot be *relied* on:
asking a language model to be reliable at checksum arithmetic and jurisdiction-specific
formats is asking it to do the thing it is worst at. Even a large lift is unlikely to clear
0.99, and 0.99 is not a target one hits by nudging a prompt.

**C. Add a deterministic validator layer.** Adopted.

## Decision

**Detect high-severity structured identifiers with deterministic validators, and use the
distilled model for the contextual remainder.** Final spans are the union of both.

| Type | Deterministic signal |
|---|---|
| AADHAAR | 12-digit + **Verhoeff** checksum |
| CREDIT_CARD | 13–19 digit + **Luhn** checksum |
| PAN | `[A-Z]{5}[0-9]{4}[A-Z]` |
| PASSPORT | jurisdiction format set |
| DRIVER_LICENSE | jurisdiction format set (the worst LLM type, 0.53) |
| SSN | area/group/serial validity rules |
| BANK_ACCOUNT | length + context-keyword proximity |
| API_KEY | known prefixes + entropy threshold |
| PASSWORD | context-keyword proximity |

Checksummed types (Aadhaar, credit card, SSN) should reach ~1.00 recall with very high
precision. Format-only types (passport, driver's licence, bank account) trade precision for
recall — acceptable, because the contract gates **recall** on these and a false positive
costs a redundant redaction while a false negative costs a breach.

The model continues to own PERSON, LOCATION, STREET_ADDRESS, DATE_OF_BIRTH, AGE, USERNAME,
EMAIL, PHONE, URL, IP_ADDRESS — the types where context, not format, decides.

### What this changes about the project's claim

The deliverable is no longer "a specialist model." It is **a specialist system: rules for
what rules do well, a distilled model for the contextual remainder.** That is what a privacy
engineer would actually ship, and it is a *stronger* claim than pretending a 1.5B model
should learn Verhoeff arithmetic.

Every gate survives intact:

- **G1 parity** stays a model-to-model comparison on the same harness, so the distillation
  result remains honest and separately reportable.
- **High-severity floor** is carried by the validator layer.
- **G3/G4** improve — regex is effectively free and sub-millisecond.
- The **system-level** number is reported alongside the model-only number, never instead of
  it. Both appear in the results table.

### Non-negotiable reporting rule

Three numbers are always published together:

1. **model-only** F1 (the distillation result, vs teacher — this is G1),
2. **validator-only** recall on high-severity types,
3. **system** F1 (union).

Quoting (3) while implying it measures (1) would be exactly the dishonesty this repo's
gate discipline exists to prevent.

## Consequences

- New module `forge/validators.py`, unit-tested against known-valid and known-invalid
  identifiers, including real checksum test vectors.
- `run_eval.py` gains a `--validators` mode reporting all three numbers.
- The teacher's own weakness becomes a documented, quantified finding rather than a silent
  ceiling — arguably the most useful result the project has produced so far, because it was
  *predicted* by `HONEST_ASSESSMENT.md` before being measured.
- Option B (teacher prompt improvement) stays open as a parallel experiment; if it lifts the
  types materially it improves the distilled model too, and it is cheap to test.
- The gold set becomes the validator layer's test set as well, which is fine — it was never
  trained on and the validators contain no learned parameters.
