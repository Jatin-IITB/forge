# Forge — Definition of Success

Criteria are scored against **pre-committed gates**, with confidence, severity-tagged risks, and an explicit verdict. Success is **not** "the model seems good." Success is "every gate below passed on a frozen test set, with confidence intervals, and the result is reproducible."

There are three layers of success. **Layer 1 is non-negotiable; Layer 2 makes it credible; Layer 3 makes it valuable to a reader.**

---

## Layer 1 — The product gates (must ALL pass on the frozen test set)

| # | Gate | Threshold (pre-committed in contract) | Why it matters |
|---|------|----------------------------------------|----------------|
| G1 | **Quality parity** | `student_score ≥ 0.98 × teacher_score`, reported with 95% CI | the core claim — the specialist actually matches the frontier model |
| G2 | **Reliability / format** | schema-validity ≥ 99.9% (structured tasks) | a small model that emits valid output every time, via constrained decoding |
| G3 | **Economics** | `cost_per_1k_req ≤ teacher_cost / 10` (target ≤ /100) | the entire reason the project exists — a real bill that drops |
| G4 | **Latency** | `p95 ≤ teacher_p95 / 5` | usable inline, not just batch |
| G5 | **Deployability** | runs within target hardware (e.g. single 24GB GPU; ideally CPU/laptop quantized) | private / on-prem / air-gapped — the fintech story |
| G6 | **Safety / OOD** | out-of-domain inputs handled per contract (refuse/flag), adversarial probe pass-rate ≥ threshold | the specialist doesn't confidently mishandle inputs outside its task |

**A run "succeeds" iff G1–G6 all pass on the frozen `test.jsonl`, measured by the Phase-1 harness unchanged.** Anything less is "promising," not "successful."

---

## Layer 2 — Engineering maturity (credibility of the result)

Scored /10. Target ≥ 7 on each before calling v1 done.

| Criterion | Target | What "good" looks like |
|-----------|--------|------------------------|
| Eval rigor | 9 | frozen gold set, human-curated, leakage-checked; CIs on every number; teacher measured under identical harness |
| Data integrity | 9 | verification gate with logged accept/reject; dedup + leakage = 0; data card published |
| Reproducibility | 8 | `make forge TASK=<contract>` rebuilds the asset on a clean machine; seeds fixed; versions pinned |
| Cost transparency | 8 | $/1k and per-loop teacher-token spend tracked and reported (tracked per loop iteration, not estimated) |
| Honesty of writeup | 9 | novel-vs-standard calibration; field comparison; weaknesses section — no hype |
| Robustness | 7 | OOD + adversarial probe sets exist and are reported, not hidden |
| Licensing/legal | 10 (pass/fail) | teacher ToS + base-model license cleared for distillation & release |

---

## Layer 3 — Portfolio value (why a reader cares)

The artifact must reduce to **one undeniable sentence + one table + one command**:

> "A 1–3B model I built that matches GPT-4-class quality on **\<task\>** at **~1% of the cost**, running on a laptop — here's the benchmark table, and `make forge` rebuilds it from scratch."

It succeeds on this layer iff it demonstrably proves the muscle an orchestration portfolio does **not**: **model manufacture** (SFT/distillation/DPO), **inference economics** (quantization/serving/$-per-1k), and **measurement rigor** (eval-first parity contract). If a reader finishes and still thinks "this is just prompting," Layer 3 failed regardless of Layer 1.

---

## What would make this a *failure* (stated plainly)
- **Moving the gate after seeing results** — if the parity target is renegotiated post-hoc, the result is void.
- **Parity on a leaky or trivial gold set** — a high number on a contaminated/easy test set is worse than no number.
- **Cost win without quality** (or vice versa) — both G1 and G3 must hold; a cheap-but-worse model is not the product.
- **Irreproducible** — a great model that only exists in one notebook on one machine is not an asset.
- **Another orchestration project in disguise** — if it ends up demonstrating LLM *orchestration* again instead of model *building*, it failed its entire reason for existing.

---

## Verdict template (filled at the end)
> Forge v1 produced a `<base-model>`-derived specialist for `<task>` scoring **`<x>%` of teacher quality (95% CI ±`<y>`)**, at **`<n>`× lower cost** and **`<m>`× lower p95 latency**, on **`<hardware>`**, passing G1–G6. Reproducible via `make forge`. Honest limitations: `<...>`. Net: the portfolio now proves model manufacture, not just orchestration.
