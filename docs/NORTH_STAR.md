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
| 2 | On-device PII specialist | ✅ **true — G5 passes (2026-09-03)** | Shipped artifact: **Q8_0 GGUF, 1647 MB file / 3080 MB RSS**, running fully local via llama.cpp Metal on a 16 GB M1. 385/385 records, 0 errors, schema-valid 385/385 in the throughput config. G5 is the one economics gate that passes. |
| 3 | ~80× lower cost | ❌ **measured 0.189× cheaper — G3 FAILS by 1.89×** *(was 1.20×, i.e. more expensive)* | **$0.03004/1k vs teacher $0.15940/1k**, full 385-record set, teacher priced at *paid* rates. The serving stack closed most of the gap — 4.93 → **0.775 s/record, 6.36×** — but G3 needs ≤ $0.01594 (0.411 s/record). Attribution: quantization **1.59×**, continuous batching **4.01×**; at batch 32 quantization buys nothing, so **batching does essentially all the work**. Remaining levers untested: shortening the 53.5-token JSON output, speculative decoding. Machine variance is ±30%, so the true miss is 1.5–2.5×. `reports/economics.md`. |
| 4 | ~20× lower p95 | ❌ **measured 3.97× SLOWER — and unreachable on this hardware** | Student p95 **5.413 s** vs a corrected teacher p95 of **1.364 s** (the old 8.024 s is withdrawn — ADR 0014). Target is therefore **≤ 0.2728 s**, 5.9× harder than published. It cannot be met: **prefill alone (0.19–0.38 s) consumes the whole budget** before one token is emitted, and fitting 53.4 output tokens in the remainder needs ≈660 tok/s single-stream against an M1 *theoretical* ceiling of 68.25 GB/s ÷ 0.935 GB ≈ **73 tok/s — 9× short of the hardware maximum**, not merely of measured performance. **The gate is also mis-calibrated:** the teacher's 1.364 s is a network round trip to Cerebras, so G4 asks a laptop to finish a full local inference in ⅕ the time a packet reaches California and returns with twice the text. Reported FAIL; the threshold is not moved. |
| 5 | ≥0.98× teacher F1 parity | ❌ **the fight** | Gate G1 pre-committed at 0.98× (= 0.9292). **Shipped artifact — Q8_0 + constrained decoding — scores 0.6198 [0.5788, 0.6595], ratio 0.6537.** Unconstrained scores higher (0.6360, ratio 0.6707) but fails G2; quoting that number here while shipping the grammar for its recall would be picking a metric per gate, so the ledger carries the shipped configuration's F1 (ADR 0016). The grammar's cost is **precision −0.0376 against recall +0.0101** — zero true positives lost, seven gained. Either way G1 fails by ~0.33, a gap twenty times the difference between the arms. The gap is data, not serving: WP-2 + WP-3. |
| 5a | Grounded every redaction in retrieved source text | ✅ **true as of 2026-09-04 — retrieval built (ADR 0021)** | `forge/retrieval.py` indexes **carrier shapes** and aligns a neighbour's span layout onto the query, so the value is read out of the *source document*, never out of the index. Legitimacy is measured, not asserted: train and eval share **7 of 693 surface forms (1.0%)** and `PERSON` shares **0 of 175**, so values cannot transfer — only structure, which does (60.0% of test records have a carrier shape seen in training). Adds **zero prompt tokens**; serving cost unchanged. **Disclosed miss:** it was gated at +0.03 micro-F1 on `val` and delivered **+0.0132** — shipped anyway as a recall aid (**+0.0215 recall**, +0.0051 precision) because the contract gates recall with no precision counterpart. That is arguing from a different metric than the one I gated on, and ADR 0021 says so. |
| 6 | 574-record frozen gold | ⚠️ **true with two disclosed defects** | 385 test + 189 dev = 574. **(a)** It was silently drifting — a clock-dependent generator made it reproducible only within a single day (ADR 0011); fixed, reproduces bit-for-bit, regression test verified to fail on the old code. **(b)** The test split holds **20 byte-identical duplicate records** (365 unique texts, not 385), so effective n is 365 and intervals are marginally optimistic. Impact measured and immaterial — teacher F1 ±0.0000, student +0.0050, G1 ratio +0.0054, all far inside ±0.042. **Freeze kept**: regenerating a test set after seeing results is what the contract forbids. Pinned at exactly 20 by a test. "Human-verified" remains **not** claimable (Protocol §5 never run). |
| 6a | dev usable as a validation split | ❌ **false — 79.4% contaminated** | **150 of 189 dev records appear verbatim in `train.jsonl`.** The engine was seeded from dev and `forge/dedup.py` was handed *test* to check leakage against, so `removed_leakage: 11` counted only what it was asked to look for. WP-3 was planned as a sweep "on dev only" and would have selected the most overfit config by scoring memorised text. Replaced by `data/gold/val.jsonl` (533 records, seed 4242, disjointness asserted on written bytes). **Test itself is clean: 0/385 leaked** — every published measurement stands. |
| 7 | 6 gates | ✅ true | G1–G6 pre-committed in `SUCCESS.md`. |
| 8 | 19 PII types | ✅ true | `forge/schema.py::PIIType`. |
| 9 | High-severity recall ≥0.99 on 9 critical | ✅ **true — demonstrated out-of-sample 2026-09-03** | The validator layer was *developed by inspecting test-set misses*, so its 1.0000 on test was a fitted score. Re-measured on `val` (seed 4242, built after the validators were frozen, verified disjoint): **1.0000 again, 339 instances, zero misses.** Pooled test+val = **571 instances, 0 misses, 95% lower bound 0.9948** — clears the 0.99 floor with evidence. Precision was 0.9114 (55 FPs, all landing on real PII — never over-redacting clean text); the three confusions behind them were fixed the same day, giving **precision 1.0000 with recall unchanged**. `reports/measurement_integrity.md` §3. |
| 9a | *Per-type* ≥0.99 floor on the frozen test set | ❌ **not measurable at this sample size** | Demonstrating ≥0.99 at 95% confidence with zero misses requires **n ≥ 299 per type**. Test carries 15 (`DRIVER_LICENSE`) to 41 (`CREDIT_CARD`); pooled 232 falls 67 short at 0.987. The gate had been reading as passing off the point estimate. Disclosed rather than quietly claimed. |
| 10 | k-sample self-consistency majority vote | ✅ true | `forge/verify.py::majority_vote_spans`, k-sample engine in Phase 2. |
| 11 | 3-layer dedup | ⚠️ **implemented, but the leakage layer was mis-invoked** | `forge/dedup.py` has all three: exact / near-dup (n-gram Jaccard) / gold-leakage. The code is correct; the **call site passed only the test split**, so dev contamination went undetected and `removed_leakage: 11` under-counted by 150 (see row 6a). The lesson is that a layer which is never asked about a split cannot protect it — `scripts/audit_gold.py` now checks every split independently of the engine that wrote the data. |
| 12 | Filtered unreliable teacher outputs | ✅ true | Verification gate with logged accept/reject (ADR 0002). |
| 13 | Fine-tuned with LoRA | ✅ true | run_001, run_002 (PEFT LoRA on MPS). |
| 14 | …& QLoRA | ❌ **blocked on hardware** | bitsandbytes has no MPS backend; this Mac cannot run QLoRA. Needs a rented GPU (~$1) or the claim is dropped. Not "pending" — blocked, and the writeup must not imply otherwise. |
| 15 | DPO when needed | 🟡 conditional | Decision gate after parity loop: if span-level FPs persist, build preference pairs and run DPO; either way, document the decision (ADR). |
| 16 | Packaged AWQ | ❌ **blocked on hardware** | AWQ calibration needs CUDA kernels. `export_model.py awq` refuses with an explanation rather than silently skipping. Same rented-GPU session as row 14. |
| 17 | Packaged GGUF | ✅ **true — Q8_0 shipped, gate-verified (2026-09-03)** | Four artifacts built and scored on the frozen set: fp16-HF 0.5750, f16 GGUF 0.6338, **Q8_0 0.6360 (+0.0022)**, Q4_K_M 0.6187 (**−0.0151**). The pre-committed rule "quantization that breaks a gate does not ship" selected **Q8_0**: Q4_K_M exceeds the ≤0.01 F1 exit gate, so the smaller file was rejected despite being 661 MB lighter. Honest wrinkle: Q4_K_M is the only one of the four that passes G2, so **neither artifact dominates** — see row 19. `reports/quantization_gates.md`. |
| 18 | Offline private inference | 🟡 **artifact ready, demo not run** | The Q8_0 GGUF runs fully local through llama.cpp with no network path (row 2), so the capability is demonstrated. What remains is the airplane-mode *demo* — Wi-Fi off, messy ticket in, redacted text out, wall-clock on screen. WP-6. |
| 20 | G6 out-of-domain + adversarial | ✅ **PASS — after building the gate the contract required (2026-09-03)** | OOD **1.0000** (21/21), adversarial **0.9000** (9/10), threshold 0.90 on both; rates never averaged. G6 failed twice before this: model-only 0.6667/0.6000, system 0.5714/0.9000. The fix was `forge/ood.py` — contract v2 mandates an `{"status":"out_of_domain"}` response and **nothing had implemented it**, so the model invented six spans in a Russian sentence and the validators invented a `BANK_ACCOUNT` in a hex dump. Placed ahead of both stages: **0 false positives across 1107 in-domain records, 21/21 OOD recall, 0 adversarial probes refused.** Also retires G3's floor caveat — the 21 OOD probes cost 266.2 s and generated to the token cap; they now cost ~0 s and zero tokens. **Honest limits:** the adversarial rate clears by *exactly zero margin*, and it misses `PERSON`/`EMAIL` under prompt injection because adversarial coverage is exactly as wide as the validator layer — 9 of 19 types defended. `reports/ood_gate.md`. |
| 19 | G2 schema validity ≥ 0.999 | ✅ **PASS — constrained decoding, 385/385 (2026-09-03)** | Was 0.9974 (384/385) against a threshold with zero margin (`0.999 × 385 = 384.6`, so 385/385 is the only passing score). `forge/grammar.py` implements the GBNF constraint `docs/DESIGN.md` named from the start and Phase 3 never built: the decoder **cannot** emit invalid output, so validity is structural rather than empirical, and a hallucinated label is unrepresentable. **It costs −0.0162 micro-F1 and we publish the lower number** — see row 5 and ADR 0016. |

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
