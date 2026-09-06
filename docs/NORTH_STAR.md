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

Every bullet on the resume, against the repo, as of **2026-09-07**.

> **Reading rule added 2026-09-03:** every quantity below now carries a 95% interval
> (`forge/ci.py`, `make audit`, ADR 0014). Ratio gates use a **paired** bootstrap, because
> student and teacher are scored on the same frozen records. Where a measurement has zero
> observed failures the bootstrap is degenerate and the exact binomial bound is quoted
> instead — a `[1.0000, 1.0000]` interval means the estimator had nothing to resample, not
> that the result is certain.

| # | Claim | Status | Evidence / gap |
|---|-------|--------|----------------|
| 1 | Distilled **120B→1.5B** *(amended up from 32B, 2026-08-11)* | 🟡 in progress | Final teacher: GPT-OSS-120B (Apache-2.0, MoE 117B/5.1B active) on Cerebras free tier → ADR 0010, contract v2. Wired and scoring the frozen test set (2026-08-15). Dev teacher was Qwen3-8B. Honest writeup states both compression ratios (≈78× total-param, ≈3.4× active). |
| 2 | On-device PII specialist | ⚠️ **true of the shipped artifact — which is no longer the best model (2026-09-07)** | Shipped artifact: **Q8_0 GGUF, 1647 MB file / 3080 MB RSS**, fully local via llama.cpp Metal on a 16 GB M1, 385/385 records, 0 errors. G5 passes on it. **Open tension:** that artifact is the *generative* student scoring 0.6198. The model that passes G1 (row 5) is the v4 **token classifier**, which has never been quantized, never run on the M1, and has no GGUF — llama.cpp serves causal LMs, and a 77-label tagging head is not one. **So the model that is deployable and the model that is accurate are currently two different models.** Closing that is the top open task; until it closes, no single artifact can claim both G1 and G5. |
| 3 | ~80× lower cost | 🟡 **still FAIL as measured (0.189×); the blocker that caused it is gone, un-remeasured on the Mac** | Standing published number is unchanged: **$0.03004/1k vs teacher $0.15940/1k**, G3 needs ≤ $0.01594. That figure is for the *generative* student, where autoregressive decoding of ~53 JSON tokens was **89% of serving time**. The v4 token classifier emits **0 output tokens** — one forward pass, decode phase deleted — and on an RTX 3050 turns in **66.3 rec/s**. **That CUDA number is not a G3 result and is not quoted as one:** `run_economics.py` carries the M1's purchase price and 22 W draw, so pricing a 3050 run against it would be exactly the substitution `PARALLEL_PLAN.md` names as "the easiest available way to fake this gate." G3 stays ❌ until the classifier is measured on the M1 under its own cost model. `reports/economics.md`, `reports/bench/tc_v4_cuda_raw.json`. |
| 4 | ~20× lower p95 | 🟡 **the "unreachable" verdict is withdrawn — the proof was architecture-specific; not yet re-measured on the Mac** | The 2026-09-03 analysis showed G4 **could not** be met: prefill alone (0.19–0.38 s) ate the 0.2728 s budget, and fitting 53.4 output tokens needed ≈660 tok/s against an M1 *theoretical* ceiling of ≈73 tok/s. **Every term in that argument is about the decode phase, which the token classifier does not have** — 0 output tokens, a single forward pass, so the 9× hardware shortfall it rested on no longer exists. On CUDA, p95 is **0.1556 s**. That is the right shape but the wrong machine, so it is **not** claimed against G4. What stands: the impossibility argument is retired, the gate is reopened, and the verdict waits on an M1 run. **The mis-calibration objection survives** — the teacher's 1.364 s is still a network round trip to Cerebras, so the gate still compares local compute against transit. Threshold not moved. |
| 5 | ≥0.98× teacher F1 parity | ✅ **PASS — 2026-09-07, ratio 1.0288** | **An architecture change closed it, not more data.** Replacing the vocabulary head with a 77-label BIOES tagging head (`forge/token_classifier.py`) and reconstructing spans in deterministic code moved model-only micro-F1 from **0.6198 → 0.9755 [0.9622, 0.9869]** against the 0.9292 gate. Ratio **1.0288** — the 1.5B student scores *above* the 120B teacher's 0.9482. The verdict is robust to both intervals at once: the student's CI floor (0.9622) clears 0.98 × the teacher's CI *ceiling* (0.9448) with 0.0174 to spare, so no pairing of the two samples fails the gate. **Disclosed gap:** the reading rule above asks for a *paired* bootstrap and it has **not** been run — `data/predictions_teacher_120b_relat.jsonl` is gitignored and no longer on disk, so the paired CI waits on regenerating teacher predictions. The two-interval bound quoted here is strictly more conservative than a paired estimate, so it cannot flatter the result. v4 = union of v2+v3 (2478 records, seed 42), QLoRA on an RTX 3050. `reports/bench/tc_v4_cuda_raw.json`. |
| 5a | Grounded every redaction in retrieved source text | ✅ **true as of 2026-09-04 — retrieval built (ADR 0021)** | `forge/retrieval.py` indexes **carrier shapes** and aligns a neighbour's span layout onto the query, so the value is read out of the *source document*, never out of the index. Legitimacy is measured, not asserted: train and eval share **7 of 693 surface forms (1.0%)** and `PERSON` shares **0 of 175**, so values cannot transfer — only structure, which does (60.0% of test records have a carrier shape seen in training). Adds **zero prompt tokens**; serving cost unchanged. **Disclosed miss:** it was gated at +0.03 micro-F1 on `val` and delivered **+0.0132** — shipped anyway as a recall aid (**+0.0215 recall**, +0.0051 precision) because the contract gates recall with no precision counterpart. That is arguing from a different metric than the one I gated on, and ADR 0021 says so. |
| 6 | 574-record frozen gold | ⚠️ **true with three disclosed defects — a new one found 2026-09-07** | 385 test + 189 dev = 574. **(a)** clock-dependent generator, fixed (ADR 0011). **(b)** 20 byte-identical duplicate test records; impact measured, immaterial, freeze kept. **(c) NEW — the set cannot be rebuilt by any available Faker.** `test_gold_builder.py::test_reproducibility` fails: a fresh build differs from the committed bytes in **116 of 385 records** across **10 labels**. **Diagnosis went three levels deep.** *Not* the ADR 0011 clock bug — the DOB window is 22646 days at both `DOB_EPOCH` and today, so the correction is exact and the RNG stream is intact. *Not* a stream cascade — **267 records remain byte-identical after the first difference**, so draw alignment is preserved. It is **Faker provider-data drift**, and `pyproject.toml` declares `faker>=24.0` — a floor, no ceiling, no lockfile. **But a pin alone does not fix it:** ~25 versions were bisected in an isolated install (15.0 → 40.38.0) and the diff count sits in three plateaus — **283** (≤27.1), **123** (27.2–39.x), **116** (all of 40.x) — **and never reaches zero**. No available Faker reproduces the committed bytes. **What still stands:** `test.jsonl` is unchanged and is the file every published number was scored against, so **no measurement is affected**. **What breaks:** the rebuild path — the CLAUDE.md litmus test. The set must now be treated as a **committed artifact, verifiable by hash but not regenerable**. Not fixed by regenerating; the contract forbids it, and doing so after seeing results is exactly what row 6 has refused twice before. |
| 6a | dev usable as a validation split | ❌ **false — 79.4% contaminated** | **150 of 189 dev records appear verbatim in `train.jsonl`.** The engine was seeded from dev and `forge/dedup.py` was handed *test* to check leakage against, so `removed_leakage: 11` counted only what it was asked to look for. WP-3 was planned as a sweep "on dev only" and would have selected the most overfit config by scoring memorised text. Replaced by `data/gold/val.jsonl` (533 records, seed 4242, disjointness asserted on written bytes). **Test itself is clean: 0/385 leaked** — every published measurement stands. |
| 7 | 6 gates | ✅ true | G1–G6 pre-committed in `SUCCESS.md`. |
| 8 | 19 PII types | ✅ true | `forge/schema.py::PIIType`. |
| 9 | High-severity recall ≥0.99 on 9 critical | ✅ **true — held again under v4 (2026-09-07)** | Previously demonstrated out-of-sample on `val` (1.0000, 339 instances) after the validators were frozen; pooled test+val = **571 instances, 0 misses, 95% lower bound 0.9948**. **v4 re-confirms it and shifts where it comes from:** the model now carries 8 of the 9 types *unaided* at 1.0000 (model-only), where the generative student needed the validator layer for all of them. The one exception is `API_KEY` at **0.9655 model-only** (1 miss in 29) — the validators take it to 1.0000, so the system floor is **1.0000 on all 9** and the shipping rule holds. The layer is now a backstop rather than the load-bearing element. `reports/measurement_integrity.md` §3. |
| 9a | *Per-type* ≥0.99 floor on the frozen test set | ❌ **not measurable at this sample size** | Demonstrating ≥0.99 at 95% confidence with zero misses requires **n ≥ 299 per type**. Test carries 15 (`DRIVER_LICENSE`) to 41 (`CREDIT_CARD`); pooled 232 falls 67 short at 0.987. The gate had been reading as passing off the point estimate. Disclosed rather than quietly claimed. |
| 10 | k-sample self-consistency majority vote | ✅ true | `forge/verify.py::majority_vote_spans`, k-sample engine in Phase 2. |
| 11 | 3-layer dedup | ⚠️ **implemented, but the leakage layer was mis-invoked** | `forge/dedup.py` has all three: exact / near-dup (n-gram Jaccard) / gold-leakage. The code is correct; the **call site passed only the test split**, so dev contamination went undetected and `removed_leakage: 11` under-counted by 150 (see row 6a). The lesson is that a layer which is never asked about a split cannot protect it — `scripts/audit_gold.py` now checks every split independently of the engine that wrote the data. |
| 12 | Filtered unreliable teacher outputs | ✅ true | Verification gate with logged accept/reject (ADR 0002). |
| 13 | Fine-tuned with LoRA | ✅ true | run_001, run_002 (PEFT LoRA on MPS). |
| 14 | …& QLoRA | ✅ **true as of 2026-09-07** | v4 trained with `--qlora` (4-bit base + LoRA) on an RTX 3050: 3 epochs, 465 steps, ~36.9 h wall clock. `checkpoints/token_classifier_v4/train_meta.json` records `qlora: true`, `seed: 42`, base model pinned at commit `989aa79`. The MPS blocker was real and simply does not apply on CUDA — the claim needed a GPU, not a redesign. `PARALLEL_PLAN.md` predicted this closes "on any CUDA device"; it did. |
| 15 | DPO when needed | 🟡 conditional | Decision gate after parity loop: if span-level FPs persist, build preference pairs and run DPO; either way, document the decision (ADR). |
| 16 | Packaged AWQ | ❌ **blocked on hardware** | AWQ calibration needs CUDA kernels. `export_model.py awq` refuses with an explanation rather than silently skipping. Same rented-GPU session as row 14. |
| 17 | Packaged GGUF | ✅ **true for the generative student; ⚠️ does not cover v4** | Four artifacts built and scored: fp16-HF 0.5750, f16 GGUF 0.6338, **Q8_0 0.6360**, Q4_K_M 0.6187 (**−0.0151**). The pre-committed rule "quantization that breaks a gate does not ship" selected **Q8_0**. `reports/quantization_gates.md`. **Scope limit added 2026-09-07:** all four are the generative architecture. The v4 token classifier is unquantized bf16 and its export path is unbuilt — GGUF is not the obvious target for a tagging head, so the packaging question is open, not merely unfinished. |
| 18 | Offline private inference | 🟡 **artifact ready, demo not run** | The Q8_0 GGUF runs fully local through llama.cpp with no network path (row 2), so the capability is demonstrated. What remains is the airplane-mode *demo* — Wi-Fi off, messy ticket in, redacted text out, wall-clock on screen. WP-6. |
| 20 | G6 out-of-domain + adversarial | ✅ **PASS — after building the gate the contract required (2026-09-03)** | OOD **1.0000** (21/21), adversarial **0.9000** (9/10), threshold 0.90 on both; rates never averaged. G6 failed twice before this: model-only 0.6667/0.6000, system 0.5714/0.9000. The fix was `forge/ood.py` — contract v2 mandates an `{"status":"out_of_domain"}` response and **nothing had implemented it**, so the model invented six spans in a Russian sentence and the validators invented a `BANK_ACCOUNT` in a hex dump. Placed ahead of both stages: **0 false positives across 1107 in-domain records, 21/21 OOD recall, 0 adversarial probes refused.** Also retires G3's floor caveat — the 21 OOD probes cost 266.2 s and generated to the token cap; they now cost ~0 s and zero tokens. **Honest limits:** the adversarial rate clears by *exactly zero margin*, and it misses `PERSON`/`EMAIL` under prompt injection because adversarial coverage is exactly as wide as the validator layer — 9 of 19 types defended. `reports/ood_gate.md`. |
| 19 | G2 schema validity ≥ 0.999 | ✅ **PASS — and now structural rather than constrained (2026-09-07)** | Was 0.9974 (384/385), then 385/385 via the GBNF constraint in `forge/grammar.py`. **The token classifier makes the gate vacuous in the strongest sense:** there is no decoder, so there is no token sequence that could be malformed — spans are reconstructed from BIOES tags by deterministic code, and the output is a data structure that was never text. v4 scores **385/385, schema-valid 385/385, 0 errors**. **This also refunds the grammar's cost:** row 5 previously carried a **−0.0162 micro-F1** tax for shipping constrained decoding to satisfy G2 (ADR 0016). That trade no longer exists — validity and F1 are no longer in tension, and the 0.9755 in row 5 is *not* paying it. |

| 21 | Span extraction as **token classification**, not generation | ✅ **built, gate-verified, recorded (2026-09-07)** | `forge/token_classifier.py` replaces the vocabulary head with **77 BIOES labels** and reconstructs spans in deterministic code. The single change that moved G1 from 0.6198 → **0.9755**, made G2 structural (row 19), unblocked QLoRA (14), and deleted the decode phase rows 3 and 4 were losing to. `PARALLEL_PLAN.md` called it "the lever that outranks the hardware" before it was benchmarked; the prediction held. **Mechanism, and it is falsifiable:** under-enumeration was a property of sequential generation under one output budget, not a knowledge gap — so span ratio should jump to ≈1.0 and stay there across data mixtures. It does: **0.984 / 1.027 / 0.994** on v2/v3/v4 against **0.46** for the generative arm. **→ ADR 0017**, which also closes ROADMAP WP-4's exit gate ("a written finding that the task formulation is the ceiling") and records what the change gives up: no rationales, no unseen labels, no GGUF path. |
### Infrastructure claims (added 2026-08-15)

| Claim | Status | Evidence |
|---|---|---|
| One-command rebuild (`make forge`) | 🟡 wired, unproven end-to-end | Full chain in the Makefile, teacher-first ordering so parity can't be back-fitted. Not yet run start-to-finish on a clean clone. |
| Reproducible from fixed seeds | ❌ **false as of 2026-09-07 — Faker is not pinned** | The ADR 0011 clock fix works and its regression guard still passes. But `faker>=24.0` has no ceiling and there is no lockfile, so the gold builder's output tracks whatever Faker resolves to: at 40.38.0 a fresh build differs from the committed set in 116/385 records (row 6c). Seeds are fixed; the *generator behind them* is not. The claim was true when written and silently became false through a dependency upgrade — which is the same failure mode as ADR 0011, one layer further out. |
| Gates pre-committed, never moved | ✅ true | Teacher change forced contract **v2**; all six thresholds verified byte-identical to v1 rather than edited. Two later opportunities to move a line were declined: the 0.99 floor was found unmeasurable at n=15–41 and disclosed rather than lowered, and 20 duplicate test records were documented rather than regenerated away. |
| All gate numbers carry 95% CIs | ✅ true *(2026-09-03)* | `forge/ci.py`; `run_eval --ci`. Records are the resampling unit, ratio gates are paired, zero-failure cases use the exact binomial bound. Contract v2 required this from the start and nothing had implemented it. |
| Data defects caught by tooling, not luck | ✅ true *(2026-09-03)* | `make audit` reads the **committed bytes**. Both data defects were invisible to `make test`, because every test regenerated the data the same wrong way — the same blind spot that let the ADR 0011 clock bug survive a green suite for weeks. |

**Ledger discipline:** this table is updated (with dates) whenever a row changes state. A row
flips to ✅ only on a committed, reproducible measurement — never on "it should work now."

### 2026-09-07 — G1 falls

The parity gate, open since the project began and labelled "the fight," **passes**: model-only
micro-F1 **0.9755** against a 0.9292 threshold, a ratio of **1.0288** — the 1.5B student scores
above the 120B teacher it was distilled from.

What moved it was not more data. v3 tripled the corpus and made the score *worse* (0.8845 →
0.8517). The lever was **architecture**: a tagging head instead of a decoder (row 21). Four rows
changed state on that one change — G1 passes (5), G2 became structural (19), QLoRA unblocked
(14), and the impossibility proof behind G4 was retired (4).

**What this update deliberately does not claim.** G3 and G4 are *not* marked passing. The v4
numbers come from an RTX 3050 and the cost model is the M1's; quoting them against those gates
would be the one substitution this project has spent months refusing. Both rows say what changed
and what is still owed. G1's paired bootstrap is likewise **not** run — the conservative
two-interval bound is quoted in its place, and labelled as the weaker instrument it is.

**The new top problem is not accuracy.** It is that the model which passes G1 and the model which
passes G5 are two different models (rows 2 and 17). The project now has to make the accurate one
deployable, rather than make the deployable one accurate.

**Found while verifying the above:** the frozen gold set no longer rebuilds (row 6c), so the
"reproducible from fixed seeds" infrastructure claim is now ❌. Faker was never actually pinned —
`>=24.0` is a floor — but bisecting ~25 versions from 15.0 to 40.38.0 found **no version that
reproduces the committed bytes**, so this is worse than a missing pin and a pin will not close it.
Published numbers are unaffected: the committed `test.jsonl` is the file they were scored against.
What is lost is the *regeneration* path, which the contract forbade exercising anyway. Recorded
rather than quietly repaired, and **not** fixed by regenerating the set.

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

All 21 ledger rows ✅ (or amended to measured truth), all six gates passing on the frozen test
set with CIs **on one artifact** — the current split, where G1 is held by the token classifier
and G5 by the generative GGUF, does not count — quantized artifacts re-verified, `make forge`
green on a clean machine, and a writeup that separates what's novel from what's solid
engineering. Then — and only then — the
resume bullet is not a claim. It's a citation.
