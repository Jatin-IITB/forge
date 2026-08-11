# ADR 0010 — Teacher scale-up: Qwen3-8B → GPT-OSS-120B (Cerebras free tier)

**Status:** accepted
**Date:** 2026-08-11
**Context:** The project's headline claim is a large-teacher→1.5B distillation with ≥0.98×
teacher parity (gate G1). The development teacher is Qwen3-**8B** running locally via Ollama —
chosen for zero-cost iteration during Phases 1–3, never intended as the final bar.

Three forces make the 8B teacher untenable as the final teacher:

1. **Claim integrity.** The public claim names a much larger teacher. Parity measured against
   an 8B is a different (weaker) claim. The claim ledger (`NORTH_STAR.md`) requires the repo to
   make the claim true or amend it to measured truth.
2. **Teacher quality ceiling.** On a 10-record dev sample the 8B teacher managed 7/10
   schema-valid responses at 116 s average latency (local, 16 GB MPS). A teacher that fails
   schema 30% of the time caps distillation quality: the verification gate discards its noise,
   but cannot manufacture signal the teacher never produced — especially on India-specific
   types (AADHAAR, PAN, DRIVER_LICENSE) where run_001 analysis showed teacher weakness.
3. **Hardware reality.** The dev machine has 16 GB RAM; no 30B+ model can run locally. This is
   a constraint to route around, not to inherit into the claim.

## Decision

Adopt **OpenAI GPT-OSS-120B** (`gpt-oss-120b`, Apache-2.0 open weights; MoE, ~117B total /
~5.1B active parameters) as the teacher, served via **Cerebras inference free tier**
(OpenAI-compatible endpoint, key via `CEREBRAS_API_KEY` env var — never committed).

The headline claim **amends upward**: 8B→1.5B development story becomes **120B→1.5B**
(~78× total-parameter compression). Per the covenant, the resume follows the repo — and here
the repo's truth got stronger, not weaker.

### Alternatives considered (verified 2026-08-11)

| Option | Verdict |
|---|---|
| **Qwen3-32B via OpenRouter** ($0.08/M in, $0.28/M out, ~$10 total) | Viable and cheap; rejected in favor of a *larger* teacher at *zero* cost. Kept as fallback. |
| **Qwen3.6-27B via Groq free tier** | 200k tokens/day starves the data engine (a single round would take months); model is 27B, weakening the claim. Rejected as primary. |
| **Local 32B (Ollama)** | Impossible: 16 GB RAM vs ~20 GB Q4 weights. |
| **Keep 8B, amend claim down** | Last resort only; contradicts "make the claim true first" ordering. |

### Independence analysis (adr/0003 litmus)

- **Weights are Apache-2.0** and public (`openai/gpt-oss-120b` on Hugging Face). Distillation
  is unrestricted by the license.
- **Provider is fungible — provably.** The same checkpoint is served by at least Cerebras and
  Groq, and is self-hostable on a rented GPU with vLLM. A stranger cloning this repo can
  rebuild with any of the three. The pipeline touches the teacher only through
  `--base-url` + `--model` + an env-var key, which `run_inference.py` and the data engine
  already support.
- Cerebras ToS re: using outputs for training — re-verify at signup (open-weight hosts
  generally permit; the model license itself does).
- **Never** an employer-internal model or endpoint. Unchanged.

### Honest-numbers footnotes (bind the writeup)

- GPT-OSS-120B is a **mixture-of-experts**: ~117B total, ~5.1B active per token. The headline
  "120B→1.5B" uses the model's official name and total parameters; the honest assessment must
  state both ratios (≈78× total-param, ≈3.4× active-param) and not pretend the teacher is dense.
- GPT-OSS is a **reasoning model** (harmony format). The pipeline must (a) request low
  reasoning effort for annotation calls to control token burn, and (b) parse only the final
  content, ignoring any reasoning field — verify empirically before the baseline run
  (think-tag robustness from Phase 3 gives prior art).

### What changes

1. **Teacher baseline (Phase 1 redo, once):** score GPT-OSS-120B on the frozen 385-record test
   set under the unchanged harness → `teacher_score`, `teacher_p95`, `teacher_$/1k`. This
   becomes the G1/G3/G4 bar. The 8B numbers stay in reports as history, labelled
   "development teacher".
2. **Teacher-annotated data:** the 150-record tranche annotated by the 8B is regenerated with
   the 120B (k-sample self-consistency unchanged). Construction-verified synthetic data
   (ADR 0009) is teacher-independent and carries over untouched.
3. **Error-driven loop (Phase 4):** targeted generation spends against the 120B, concentrated
   on student-weak slices.

### Budget & throughput (free tier, measured limits)

Cerebras free tier for `gpt-oss-120b`: **5 req/min, 30k tokens/min, 1M tokens/day.**
- Test-set baseline (385 records ≈ 0.5–0.6M tokens): fits in one day, ~80 min wall-clock.
- Data rounds are sized to the 1M-token daily budget; the engine's resume support (ADR 0007)
  spreads larger rounds across days. Error-driven targeting keeps rounds small by design —
  spending teacher tokens only where the student is wrong is the project's stated principle.
- Dollar cost: **$0.** Hard cap unchanged in spirit: one daily token budget per loop iteration,
  logged per run.

## Consequences

- The parity denominator gets *harder* — an honest raising of our own bar. Expect the measured
  gap to widen before it closes.
- $/1k and p95 comparisons become meaningful and favorable-by-architecture: hosted 120B vs
  on-device 1.5B is exactly the deployment story the project sells. (For G3, the cost model
  will price the teacher at *paid-tier* rates with methodology stated — free tiers are a
  bootstrap subsidy, not an economics claim.)
- A second teacher family (OpenAI GPT-OSS) enters a Qwen student pipeline. Cross-family
  distillation via span-JSON supervision is format-agnostic; no tokenizer coupling exists in
  the pipeline (supervision is text, not logits).
- Fallbacks, in order: (a) Groq's free `gpt-oss-120b` (same checkpoint, different host);
  (b) OpenRouter Qwen3-32B paid (~$10); (c) rented GPU + vLLM. All preserve independence.
