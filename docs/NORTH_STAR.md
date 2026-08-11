# Forge — North Star

*The vision, the claim ledger, and the execution arc. Read after `DESIGN.md` and `SUCCESS.md`.*

---

## The promise

A hiring manager opens a laptop with Wi-Fi off. They paste a messy support ticket — names, an
Aadhaar number, a credit card, a password someone should never have typed — into a terminal.
One second later every span is found, typed, and redacted. Nothing left the machine. The model
doing it is 1.5B parameters, was manufactured — not hand-tuned — by a pipeline in this repo,
and carries a receipt: a frozen 574-record test set says it performs within 2% of the 32B
teacher it was distilled from, at a measured fraction of the cost and latency.

That receipt is the product. Anyone can fine-tune a small model. **Forge's claim is that model
manufacture can be *engineered* — eval-first, verification-gated, error-driven — so the result
is a measured artifact, not a lucky checkpoint.**

## The covenant

Every public claim about this project maps to a number this repo produces. Two rules, in order:

1. **The repo makes the claim true.** We build until the measured number meets the claim.
2. **If the repo lands elsewhere, the claim changes to the measured number.** Never the reverse.
   A resume bullet is a *summary of evidence*, not a target to decorate.

## The claim ledger

Every bullet on the resume, against the repo, as of 2026-08-11:

| # | Claim | Status | Evidence / gap |
|---|-------|--------|----------------|
| 1 | Distilled **32B→1.5B** | ❌ **gap** | Teacher is Qwen3-**8B** (local Ollama; 16 GB RAM cannot host 32B). Close via hosted open-weight Qwen3-32B endpoint → ADR 0010. |
| 2 | On-device PII specialist | 🟡 partial | Student is Qwen2.5-1.5B, trains + runs locally. "On-device" fully earned when GGUF artifact passes gates (Arc C). |
| 3 | ~80× lower cost | ❌ unmeasured | Needs economics harness + published cost model (Arc D). Claim becomes the measured multiple. |
| 4 | ~20× lower p95 | ❌ unmeasured | Teacher-API p95 vs student on-device p95, same harness (Arc D). Claim becomes the measured multiple. |
| 5 | ≥0.98× teacher F1 parity | ❌ **the fight** | Gate G1 pre-committed at 0.98×. run_001 F1 = 0.52. Teacher not yet scored on frozen test. Arc A/B. |
| 6 | 574-record frozen gold | ✅ true | 385 test + 189 dev = 574, frozen, protocol documented (`data/gold/PROTOCOL.md`). |
| 7 | 6 gates | ✅ true | G1–G6 pre-committed in `SUCCESS.md`. |
| 8 | 19 PII types | ✅ true | `forge/schema.py::PIIType`. |
| 9 | High-severity recall ≥0.99 on 9 critical | 🟡 gate set | `HIGH_SEVERITY` frozenset + per-type gate wired into eval. Achievement pending (Arc A/B). |
| 10 | k-sample self-consistency majority vote | ✅ true | `forge/verify.py::majority_vote_spans`, k-sample engine in Phase 2. |
| 11 | 3-layer dedup | ✅ true | `forge/dedup.py`: exact / near-dup (n-gram Jaccard) / gold-leakage. |
| 12 | Filtered unreliable teacher outputs | ✅ true | Verification gate with logged accept/reject (ADR 0002). |
| 13 | Fine-tuned with LoRA | ✅ true | run_001, run_002 (PEFT LoRA on MPS). |
| 14 | …& QLoRA | ❌ not run | One documented QLoRA run required (Arc C): MLX 4-bit local or brief rented GPU. |
| 15 | DPO when needed | 🟡 conditional | Decision gate after parity loop: if span-level FPs persist, build preference pairs and run DPO; either way, document the decision (ADR). |
| 16 | Packaged AWQ | ❌ not run | AWQ needs CUDA → one short rented-GPU session (Arc C). |
| 17 | Packaged GGUF | ❌ not run | llama.cpp convert + quantize, fully local (Arc C). Quantized artifact must re-pass all gates. |
| 18 | Offline private inference | ❌ pending | The airplane-mode demo: GGUF via llama.cpp/Ollama, Wi-Fi off (Arc C/E). |

**Ledger discipline:** this table is updated (with dates) whenever a row changes state. A row
flips to ✅ only on a committed, reproducible measurement — never on "it should work now."

## The execution arc

Four arcs close the ledger. Each ends in a checkable artifact.

### Arc A — Win on the current field *(running tonight)*
Prove the error-driven loop itself: run_002 (837 records, 5.6× data, targeted augmentation)
evaluates against the frozen test set. Iterate error-analysis → targeted generation → retrain
until the curve bends hard. This validates the *machinery* even before the teacher upgrade.
- **Artifact:** `reports/eval_run_002.md` + loop log with per-iteration deltas.

### Arc B — Raise the bar to 32B *(the claim-integrity arc)*
Stand up Qwen3-32B as teacher via a hosted open-weight endpoint (ADR 0010; independence
preserved — Apache-2.0 weights, fungible provider, litmus test passes). Score the teacher on
the frozen test → the real `teacher_score`, `teacher_p95`, `teacher_$/1k` bar. Regenerate the
teacher-annotated tranche with the 32B where the student is weak; re-distill; loop until
**G1 (≥0.98×)** and **high-severity recall ≥0.99** hold.
- **Budget:** hard cap per loop iteration on teacher tokens (order: single-digit $).
- **Artifact:** `reports/baseline_32b.md`, updated data card, gate table with CIs.

### Arc C — Make it an artifact *(packaging)*
One QLoRA run (claim 14). DPO decision documented (claim 15). Merge adapter → base; produce
**GGUF** (Q4_K_M + Q8_0, local) and **AWQ** (rented GPU, ~$1) artifacts; **re-run the full
gate suite on each quantized artifact** — quantization that breaks a gate doesn't ship.
- **Artifact:** `models/` manifests + `MODEL_CARD.md` + quantized gate table.

### Arc D — Put numbers on the economics *(the reason it exists)*
A measurement harness, not a vibe: teacher endpoint $/1k-records and p95 vs student on-device
$/1k (amortized, methodology published) and p95, same records, same harness. G3/G4 verdicts.
Claims 3 and 4 become whatever the harness prints.
- **Artifact:** `reports/economics.md` with the cost model spelled out.

### Arc E — Tell it honestly *(the writeup)*
README hero table (measured numbers only), the airplane-mode demo, model card, honest
assessment (novel vs solid-engineering, field comparison, known weaknesses), and `make forge`
rebuilding the artifact end-to-end from a clean clone.
- **Artifact:** a repo a stranger can clone, audit, and reproduce.

## Non-negotiables

- **Independence** (`adr/0003`): open weights, public data, commodity compute. The litmus test
  runs at every arc boundary.
- **Frozen test set**: `test.jsonl` is never trained on, never regenerated, never "fixed" to
  help a number. The Faker-version pin issue is resolved by pinning, not regenerating (or by a
  documented, versioned regeneration *before* any new claims — never after seeing results).
- **No post-hoc gates** (`SUCCESS.md`): moving a threshold after seeing results voids the run.
- **Honest numbers**: every published multiple (cost, latency, parity) links to the harness
  output that produced it.

## Definition of done

All 18 ledger rows ✅ (or amended to measured truth), all six gates passing on the frozen test
set with CIs, quantized artifacts re-verified, `make forge` green on a clean machine, and a
writeup that separates what's novel from what's solid engineering. Then — and only then — the
resume bullet is not a claim. It's a citation.
