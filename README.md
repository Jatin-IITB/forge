# Forge

**Task-Specialization Distillation Pipeline** — give it one expensive, high-volume LLM task and a teacher model; it manufactures a verified specialist student that **matches teacher quality within a committed tolerance at 10–100× lower cost and latency**, runnable fully private.

Not an agent. The deliberate counterpart to an LLM-*orchestration* portfolio: this proves model *manufacture* — SFT/distillation/DPO, inference economics, and eval-first measurement rigor.

## Status
Planning. Design committed; no modeling yet (by design — see the eval-first contract).

## Read in this order
1. [`docs/DESIGN.md`](docs/DESIGN.md) — first-principles system design, field comparison, honest novelty calibration.
2. [`docs/SUCCESS.md`](docs/SUCCESS.md) — the six product gates + maturity rubric + what counts as failure.
3. [`docs/ACTION_PLAN.md`](docs/ACTION_PLAN.md) — phased plan with exit gates, budget, risk register.
4. [`docs/adr/`](docs/adr/) — architecture decision records.

## The one-sentence goal
> "A 1–3B model I built that matches GPT-4-class quality on **\<task\>** at ~1% of the cost, on a laptop — here's the benchmark, and `make forge` rebuilds it."

## Independence (non-negotiable)
Depends on **nothing internal** to any employer/internship: open-weight teacher, permissive base model, **public data**, own compute, OSS-only code. The asset must outlive any single job. **Litmus test:** if all internal access were cut tomorrow, a stranger could clone this repo and rebuild the model end-to-end. See [`docs/adr/0003`](docs/adr/0003-independence-and-public-data.md).

## Flagship task (v1)
**On-device PII detection & redaction** — meaning is airtight (you can't send PII to a frontier API *to find PII*; a local model is the only privacy-compliant option — GDPR / India DPDP Act), fully public data, crisply measurable, demos offline. Alternatives: public-document extraction; an under-served-Indian-language specialist. Locked in Phase 0.

## Principles (Aroha doc discipline — portable, not a dependency)
- Eval-first: the frozen gold set and parity gate exist **before** the model.
- Verification-gated data: no unverified teacher output trains the student.
- Error-driven loop: spend teacher tokens where the student is wrong.
- Economics are gates, not footnotes.
- Honest writeup: novel-vs-standard calibration, field comparison, weaknesses stated.
- Reproducible: one command rebuilds the asset.
