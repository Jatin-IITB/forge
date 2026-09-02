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

Every bullet on the resume, against the repo, as of **2026-09-03**.

> **Reading rule added 2026-09-03:** every quantity below now carries a 95% interval
> (`forge/ci.py`, `make audit`, ADR 0014). Ratio gates use a **paired** bootstrap, because
> student and teacher are scored on the same frozen records. Where a measurement has zero
> observed failures the bootstrap is degenerate and the exact binomial bound is quoted
> instead — a `[1.0000, 1.0000]` interval means the estimator had nothing to resample, not
> that the result is certain.

| # | Claim | Status | Evidence / gap |
|---|-------|--------|----------------|
| 1 | Distilled **120B→1.5B** *(amended up from 32B, 2026-08-11)* | 🟡 in progress | Final teacher: GPT-OSS-120B (Apache-2.0, MoE 117B/5.1B active) on Cerebras free tier → ADR 0010, contract v2. Wired and scoring the frozen test set (2026-08-15). Dev teacher was Qwen3-8B. Honest writeup states both compression ratios (≈78× total-param, ≈3.4× active). |
| 2 | On-device PII specialist | 🟡 partial | Student is Qwen2.5-1.5B, trains + runs locally. "On-device" fully earned when GGUF artifact passes gates (Arc C). |
| 3 | ~80× lower cost | ❌ **measured 1.20× — the student is *more expensive*** | $0.1910/1k vs teacher $0.1592/1k. Cause is entirely the serving stack: 4.93 s/record, unquantized fp16, batch 1, on MPS (≈21 tok/s) against a wafer-scale accelerator. Reaching G3 needs **0.411 s/record — a 12× throughput gain**. GGUF Q4_K_M gives 2–4×, continuous batching 4–8×; neither alone suffices. WP-1, in flight. The claim becomes whatever the harness prints. |
| 4 | ~20× lower p95 | ❌ **measured 1.08×, and the denominator is unsound** | Student p95 8.68 s vs teacher 8.02 s. But the teacher p95 is **not reproducible**: two runs of the identical config give 0.790 s (n=60) and 8.024 s (n=302) while their p50s agree within 14%. The 60-sample p95 CI is [0.689, 1.054], so 8.024 sits 7.6× above it — not sampling noise. Since G4's threshold is `teacher_p95 / 5`, the gate inherits the instability: a clean p95 near 0.8 s makes the target **≤ 0.16 s**, ~10× harder than assumed. Re-measurement with the pooling fix is in flight. |
| 5 | ≥0.98× teacher F1 parity | ❌ **the fight** | Gate G1 pre-committed at 0.98×. **Measured 2026-09-03 with a paired bootstrap: ratio 0.6064 [0.5624, 0.6485]** (student 0.5750 [0.5324, 0.6162] vs teacher 0.9482 [0.9305, 0.9641]). The whole interval sits 0.33 below the gate — this fails structurally, not statistically. No interval rescues it. WP-2 + WP-3. |
| 6 | 574-record frozen gold | ⚠️ **true with two disclosed defects** | 385 test + 189 dev = 574. **(a)** It was silently drifting — a clock-dependent generator made it reproducible only within a single day (ADR 0011); fixed, reproduces bit-for-bit, regression test verified to fail on the old code. **(b)** The test split holds **20 byte-identical duplicate records** (365 unique texts, not 385), so effective n is 365 and intervals are marginally optimistic. Impact measured and immaterial — teacher F1 ±0.0000, student +0.0050, G1 ratio +0.0054, all far inside ±0.042. **Freeze kept**: regenerating a test set after seeing results is what the contract forbids. Pinned at exactly 20 by a test. "Human-verified" remains **not** claimable (Protocol §5 never run). |
| 6a | dev usable as a validation split | ❌ **false — 79.4% contaminated** | **150 of 189 dev records appear verbatim in `train.jsonl`.** The engine was seeded from dev and `forge/dedup.py` was handed *test* to check leakage against, so `removed_leakage: 11` counted only what it was asked to look for. WP-3 was planned as a sweep "on dev only" and would have selected the most overfit config by scoring memorised text. Replaced by `data/gold/val.jsonl` (533 records, seed 4242, disjointness asserted on written bytes). **Test itself is clean: 0/385 leaked** — every published measurement stands. |
| 7 | 6 gates | ✅ true | G1–G6 pre-committed in `SUCCESS.md`. |
| 8 | 19 PII types | ✅ true | `forge/schema.py::PIIType`. |
| 9 | High-severity recall ≥0.99 on 9 critical | ✅ **true — demonstrated out-of-sample 2026-09-03** | The validator layer was *developed by inspecting test-set misses*, so its 1.0000 on test was a fitted score. Re-measured on `val` (seed 4242, built after the validators were frozen, verified disjoint): **1.0000 again, 339 instances, zero misses.** Pooled test+val = **571 instances, 0 misses, 95% lower bound 0.9948** — clears the 0.99 floor with evidence. Caveat published: 55 FPs, all landing on real PII (zero over-redaction of clean text); 41 are `+91`-prefixed phones read as cards, 13 usernames read as passwords, 1 truncates a TLD. Weakness is type disambiguation, not detection. `reports/measurement_integrity.md` §3. |
| 9a | *Per-type* ≥0.99 floor on the frozen test set | ❌ **not measurable at this sample size** | Demonstrating ≥0.99 at 95% confidence with zero misses requires **n ≥ 299 per type**. Test carries 15 (`DRIVER_LICENSE`) to 41 (`CREDIT_CARD`); pooled 232 falls 67 short at 0.987. The gate had been reading as passing off the point estimate. Disclosed rather than quietly claimed. |
| 10 | k-sample self-consistency majority vote | ✅ true | `forge/verify.py::majority_vote_spans`, k-sample engine in Phase 2. |
| 11 | 3-layer dedup | ⚠️ **implemented, but the leakage layer was mis-invoked** | `forge/dedup.py` has all three: exact / near-dup (n-gram Jaccard) / gold-leakage. The code is correct; the **call site passed only the test split**, so dev contamination went undetected and `removed_leakage: 11` under-counted by 150 (see row 6a). The lesson is that a layer which is never asked about a split cannot protect it — `scripts/audit_gold.py` now checks every split independently of the engine that wrote the data. |
| 12 | Filtered unreliable teacher outputs | ✅ true | Verification gate with logged accept/reject (ADR 0002). |
| 13 | Fine-tuned with LoRA | ✅ true | run_001, run_002 (PEFT LoRA on MPS). |
| 14 | …& QLoRA | ❌ **blocked on hardware** | bitsandbytes has no MPS backend; this Mac cannot run QLoRA. Needs a rented GPU (~$1) or the claim is dropped. Not "pending" — blocked, and the writeup must not imply otherwise. |
| 15 | DPO when needed | 🟡 conditional | Decision gate after parity loop: if span-level FPs persist, build preference pairs and run DPO; either way, document the decision (ADR). |
| 16 | Packaged AWQ | ❌ **blocked on hardware** | AWQ calibration needs CUDA kernels. `export_model.py awq` refuses with an explanation rather than silently skipping. Same rented-GPU session as row 14. |
| 17 | Packaged GGUF | 🟡 code ready | `export_model.py gguf` implemented (merge → f16 → Q4_K_M/Q8_0), runs fully on Mac. Blocked only on a trained adapter. Quantized artifact must re-pass all gates before this flips ✅. |
| 18 | Offline private inference | ❌ pending | The airplane-mode demo: GGUF via llama.cpp/Ollama, Wi-Fi off (Arc C/E). |

### Infrastructure claims (added 2026-08-15)

| Claim | Status | Evidence |
|---|---|---|
| One-command rebuild (`make forge`) | 🟡 wired, unproven end-to-end | Full chain in the Makefile, teacher-first ordering so parity can't be back-fitted. Not yet run start-to-finish on a clean clone. |
| Reproducible from fixed seeds | ✅ true *(as of the ADR 0011 fix)* | 143 tests green including a clock-shift regression guard. |
| Gates pre-committed, never moved | ✅ true | Teacher change forced contract **v2**; all six thresholds verified byte-identical to v1 rather than edited. Two later opportunities to move a line were declined: the 0.99 floor was found unmeasurable at n=15–41 and disclosed rather than lowered, and 20 duplicate test records were documented rather than regenerated away. |
| All gate numbers carry 95% CIs | ✅ true *(2026-09-03)* | `forge/ci.py`; `run_eval --ci`. Records are the resampling unit, ratio gates are paired, zero-failure cases use the exact binomial bound. Contract v2 required this from the start and nothing had implemented it. |
| Data defects caught by tooling, not luck | ✅ true *(2026-09-03)* | `make audit` reads the **committed bytes**. Both data defects were invisible to `make test`, because every test regenerated the data the same wrong way — the same blind spot that let the ADR 0011 clock bug survive a green suite for weeks. |

**Ledger discipline:** this table is updated (with dates) whenever a row changes state. A row
flips to ✅ only on a committed, reproducible measurement — never on "it should work now."

## The execution arc

Four arcs close the ledger. Each ends in a checkable artifact.

### Arc A — Win on the current field *(running tonight)*
Prove the error-driven loop itself: run_002 (837 records, 5.6× data, targeted augmentation)
evaluates against the frozen test set. Iterate error-analysis → targeted generation → retrain
until the curve bends hard. This validates the *machinery* even before the teacher upgrade.
- **Artifact:** `reports/eval_run_002.md` + loop log with per-iteration deltas.

### Arc B — Raise the bar to 120B *(the claim-integrity arc)*
Stand up GPT-OSS-120B as teacher via Cerebras free tier (ADR 0010; independence preserved —
Apache-2.0 weights served by ≥2 hosts + self-hostable, litmus test passes). Score the teacher
on the frozen test → the real `teacher_score`, `teacher_p95`, `teacher_$/1k` bar. Regenerate
the teacher-annotated tranche with the 120B where the student is weak; re-distill; loop until
**G1 (≥0.98×)** and **high-severity recall ≥0.99** hold.
- **Budget:** $0 (free tier: 5 req/min, 1M tokens/day — rounds sized to the daily budget,
  engine resume spreads bigger rounds across days).
- **Artifact:** `reports/baseline_120b.md`, updated data card, gate table with CIs.

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
