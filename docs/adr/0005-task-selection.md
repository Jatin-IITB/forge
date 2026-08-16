# ADR 0005 — Flagship task, teacher/base models, and data source

**Status:** Accepted
**Date:** 2026-06-10
**Closes:** Phase 0 model/data selection + licence-clearance portion of the exit gate
(`ACTION_PLAN.md` Phase 0; depends on `adr/0003`).

## Context
Phase 0 must lock the flagship task and pick a teacher, a base model, and a data source
that all pass the `adr/0003` independence litmus test ("a stranger can clone and rebuild
end-to-end") *and* permit distillation/redistribution. These choices are committed
before any modelling (`adr/0001`).

## Decision

### 1. Flagship task: on-device PII detection & redaction
Chosen over the alternatives because every constraint reinforces the others (DESIGN §0.2):
- **Meaning is airtight.** You cannot send sensitive text to a frontier API *to find
  sensitive text* — a local specialist is the *only* privacy-compliant option
  (GDPR; India DPDP Act 2023). The value prop is correctness-of-privacy, not just cost.
- **Crisply measurable** — entity/span-level P/R/F1, no fuzzy judging.
- **Independence-clean** — buildable entirely from synthetic + public data.
- **Demos viscerally** — redact, then pull the network cable; it still works.

Rejected for v1: **(B)** structured field extraction (strong, kept as the documented
second task) and **(C)** under-served-Indian-language specialist (generation eval is
harder; secondary).

### 2. Teacher: `Qwen/Qwen2.5-32B-Instruct` — **Apache-2.0**
Apache-2.0 imposes **no restriction on using outputs to train other models**, so
distillation and redistribution are unambiguously permitted. Verified per-size
licensing: Qwen2.5 `0.5B / 1.5B / 7B / 14B / 32B` are Apache-2.0; `72B` is the Qwen
research licence (avoided). 32B is the strongest Apache-2.0 size → best teacher quality
without a licence caveat.
- **Fallback:** `meta-llama/Llama-3.1-70B-Instruct`. The Llama 3.1/3.3 Community
  Licence was updated to **explicitly permit** distillation/synthetic-data generation
  to improve other models. Acceptable, but carries the >700M-MAU clause and a
  "name it Llama" requirement, so it is second choice behind clean Apache-2.0.

### 3. Base (student): `Qwen/Qwen2.5-1.5B-Instruct` — **Apache-2.0**
Fits the on-device target (≤3B params, runs on an Apple-silicon laptop / single 24GB GPU
for training) and is Apache-2.0 → free to fine-tune and release.
- **Larger option:** `Qwen2.5-3B-Instruct` (relicensed to Apache-2.0).
- **Fallback:** `meta-llama/Llama-3.2-1B/3B-Instruct` (Llama 3.2 Community Licence).

Same-family teacher/student (both Qwen2.5) also simplifies tokenizer/template alignment.

### 4. Data: self-generated synthetic (Faker, MIT) + human-verified freeze
- **Primary:** Faker injects fake PII *values* into carrier text at known offsets →
  exact ground-truth spans by construction, fully reproducible (`make gold`, fixed
  seed), fully redistributable (MIT + project-owned text). See `data/gold/PROTOCOL.md`.
- **Rejected:** `ai4privacy/pii-masking-200k` — academic-use only; commercial use gated
  behind `licensing@ai4privacy.com`. **Violates `adr/0003`** as a committed asset.
  Permitted only as an external academic benchmark for context.
- **Optional enrichment:** public-domain real text (e.g. Enron corpus) under a
  per-document licence check, re-labelled to this protocol.

## Independence litmus test — result: **PASS**
Open-weight Apache-2.0 teacher + open-weight Apache-2.0 base + MIT/self-generated data +
commodity compute + OSS-only code. If every private credential were revoked tomorrow, a stranger
could clone the repo and rebuild the model end-to-end. ✅

## Consequences
- (+) Zero licence caveats on the headline path (Apache-2.0 throughout) → cleanly
  open-source-able.
- (+) Synthetic-first data makes the *entire* pipeline reproducible by `make`, which is
  itself the Forge thesis.
- (−) Synthetic carrier text is less "messy" than real production text. Mitigated by the
  human-verification pass for realism and the optional public-domain enrichment; the OOD
  gate (`SUCCESS.md` G6) explicitly probes distribution shift.
- (−) A same-family teacher/student shares blind spots. Accepted for v1; the
  verification gate (`adr/0002`) still filters the teacher's confident mistakes.
