# ADR 0010 — Teacher scale-up: Qwen3-8B → Qwen3-32B via hosted open-weight endpoint

**Status:** proposed (pending endpoint selection)
**Date:** 2026-08-11
**Context:** The project's headline claim is a **32B→1.5B** distillation with ≥0.98× teacher
parity (gate G1). The current teacher is Qwen3-**8B** running locally via Ollama — chosen for
zero-cost iteration during Phases 1–3, not as the final bar.

Three forces make the 8B teacher untenable as the final teacher:

1. **Claim integrity.** The distillation claim names a 32B teacher. Parity measured against an
   8B is a different (weaker) claim. The claim ledger (`NORTH_STAR.md`) requires the repo to
   make the claim true or amend it — and a 32B teacher is achievable, so we make it true.
2. **Teacher quality ceiling.** On a 10-record dev sample the 8B teacher managed 7/10
   schema-valid responses at 116 s average latency (local, 16 GB MPS). A teacher that fails
   schema 30% of the time caps distillation quality: the verification gate discards its noise,
   but it cannot manufacture signal the teacher never produced — especially on India-specific
   types (AADHAAR, PAN, DRIVER_LICENSE) where run_001's analysis showed teacher weakness.
3. **Hardware reality.** The dev machine has 16 GB RAM. A 32B at Q4 needs ~20 GB for weights
   alone — local hosting is impossible. This is a constraint to route around, not to inherit
   into the claim.

## Decision

Adopt **Qwen3-32B** (Apache-2.0, open weights) as the teacher, served through a **hosted
OpenAI-compatible endpoint that serves the open checkpoint** (e.g. Together / Fireworks /
DeepInfra / OpenRouter — provider chosen by account access and price; the pipeline only needs
`--base-url`, `--model`, and a key via env var, which `run_inference.py` and the data engine
already support).

### Independence analysis (adr/0003 litmus)

- The **weights are open** (Apache-2.0). The hosted endpoint is a *convenience*, not a
  dependency: a stranger cloning this repo can point the same scripts at any provider serving
  the same checkpoint, or at their own vLLM instance on a rented GPU. Provider is fungible;
  model identity is pinned.
- ToS: open-weight hosting providers permit distillation of open checkpoints (the Qwen license
  itself permits it). Recorded here as checked; re-verify the chosen provider's ToS at signup.
- **Never** an employer-internal model or endpoint. Unchanged.

### What changes

1. **Teacher baseline (Phase 1 redo, once):** score Qwen3-32B on the frozen 385-record test
   set under the unchanged harness → `teacher_score`, `teacher_p95`, `teacher_$/1k`. This
   number becomes the G1/G3/G4 bar. The 8B numbers are kept in reports as history, clearly
   labelled "development teacher".
2. **Teacher-annotated data:** the 150-record tranche annotated by the 8B is regenerated with
   the 32B (k-sample self-consistency unchanged). Construction-verified synthetic data
   (ADR 0009) is teacher-independent and carries over untouched.
3. **Error-driven loop (Phase 4):** teacher tokens for targeted generation now spend against
   the 32B, concentrated on student-weak slices.

### Budget

Hosted 32B inference is ~$0.3–0.9 per 1M tokens. Scoring the test set is ~1M tokens (~$1).
A full k=5 data-engine round over a few thousand seeds is ~10–30M tokens (~$5–25).
**Hard cap: $25 per loop iteration, logged per run** (existing budget-discipline rule).

## Consequences

- The parity denominator gets *harder* — an honest raising of our own bar. Expect the measured
  gap to widen before it closes.
- $/1k and p95 comparisons become meaningful: hosted-32B pricing and latency vs on-device
  1.5B is exactly the deployment story the project sells.
- A small recurring cost enters the project (~$10–50 total expected). Within the "commodity
  cloud" allowance of the independence rule.
- Fallbacks, in order: (a) rented GPU + vLLM serving the same checkpoint (higher setup, same
  independence); (b) if 32B access is blocked entirely, the claim amends to the teacher we
  actually used — per the covenant, the resume follows the repo.
