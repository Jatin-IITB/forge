# Forge

**A PII redaction system that catches 100% of breach-severity identifiers — after proving its own 120B teacher catches only 87%.**

Forge is a task-specialization distillation pipeline: give it one expensive, high-volume LLM task and an open teacher, and it manufactures a small specialist that runs fully offline. The flagship task is **on-device PII detection & redaction** — where the privacy argument is airtight, because you cannot send sensitive text to a frontier API *in order to find the sensitive text*.

This is not an agent. It proves model **manufacture**: distillation, verification-gated data, serving economics, and eval-first measurement rigor.

---

## Headline result

The project's most valuable output was a **negative finding about its own teacher**, and the engineering response to it.

The teacher (GPT-OSS-120B, Apache-2.0) scores a strong **micro-F1 0.9482** overall. But the per-type breakdown showed it **missing 6 of 9 breach-severity recall floors** — types where a single miss is a reportable disclosure:

| Breach-severity type | Teacher | **Forge system** |
|---|---|---|
| DRIVER_LICENSE | 0.5333 | **1.0000** |
| BANK_ACCOUNT | 0.7931 | **1.0000** |
| AADHAAR | 0.8276 | **1.0000** |
| PASSPORT | 0.9130 | **1.0000** |
| PAN | 0.9310 | **1.0000** |
| PASSWORD | 0.9474 | **1.0000** |
| CREDIT_CARD / SSN / API_KEY | 1.0000 | **1.0000** |
| **minimum across all 9** | **0.5333** | **1.0000** |

Because distillation transfers a teacher's blind spots, **no student trained on this teacher could ever clear those floors** — a student at *perfect* parity would still inherit 0.53 recall on driver's licences. Moving the gate was not an option; the project's own rules void a run whose threshold is renegotiated after seeing results.

The fix ([ADR 0012](docs/adr/0012-hybrid-validator-layer.md)) was to stop asking a language model to do arithmetic: **deterministic validators** (Verhoeff for Aadhaar, Luhn for cards, format + nearest-keyword context rules) carry the nine high-severity types, while the distilled model keeps the contextual ones. All nine floors reach **1.000 recall**, and system F1 rises **+0.158** over the model alone.

> This was *predicted before it was measured*. [`HONEST_ASSESSMENT.md`](docs/HONEST_ASSESSMENT.md), written earlier, stated: *"for well-formed identifiers, a well-written regex with a checksum is likely to beat a 1.5B model."* The data agreed.

---

## Measured results

Frozen 385-record test set, exact-match `(start, end, label)` micro-F1, same harness for every row.

| | micro-F1 | precision | recall | min high-sev recall |
|---|---|---|---|---|
| Teacher — GPT-OSS-120B | 0.9482 | 0.9615 | 0.9353 | 0.5333 |
| Student model only | 0.5750 | 0.6375 | 0.5237 | 0.0000 |
| **Forge system** (student + validators) | **0.7334** | **0.7818** | **0.6906** | **1.0000** |

Three numbers are always published together — model-only, validator-only, system — because quoting the system score as if it measured the distillation would be the exact conflation this repo's gate discipline exists to prevent.

### Gate status

| Gate | Threshold | Measured | |
|---|---|---|---|
| **High-severity recall** | ≥ 0.99 on 9 types | **1.0000 on all 9** | ✅ **PASS** |
| G2 schema validity | ≥ 99.9% | 99.74% (1 failure in 385) | ⚠️ marginal |
| G1 quality parity | ≥ 0.9292 | 0.5750 model-only | ❌ **open** |
| G3 cost / G4 latency | ≤ teacher/10, ≤ 1.60 s p95 | harness built, awaiting final run | ⏳ |
| G5 deployability | laptop / CPU | export path built, unquantized | ⏳ |
| G6 OOD / adversarial | ≥ 0.90 both axes | 31-probe set built | ⏳ |

**The parity gate is not met and the README will not claim otherwise.** `make eval` prints `GATE CHECK: 11 FAILED` on the current student, and that number is reproducible by anyone who clones this repo. The claim ledger in [`docs/NORTH_STAR.md`](docs/NORTH_STAR.md) tracks every claim against its evidence.

---

## What the distillation experiments established

Two training runs, each with **predictions registered before execution**, each rejecting its own hypothesis:

| Run | Change | Loss | F1 | Verdict |
|---|---|---|---|---|
| run_002 | 5.6× targeted data (150 → 837) | plateaus 1.17 | 0.5750 | **underfit** — data was not the constraint |
| run_003 | 17× adapter capacity (r=16→64, +MLP) | **0.22** | 0.5379 | **overfit** — recall collapsed, span ratio halved |

run_003's loss fell 5× while F1 got *worse*: precision +0.24, recall −0.13. [ADR 0013](docs/adr/0013-capacity-not-data-diagnosis.md) had written the interpretation in advance — *"fitting the data better without learning to enumerate entities"* — so the result was diagnostic rather than confusing.

**Together they show capacity and data are jointly binding**, and isolate the remaining lever: every one of the 837 training records is template-generated, and the 150 "teacher-labelled" ones came from an 8B model later replaced for being too weak. **No training record has ever been labelled by the 120B teacher.** That is the next experiment, and it is the one the project is named for.

> Had ADR 0013 asked only "does loss improve?", run_003 would have read as a clean success and the recall collapse would have shipped undetected. Requiring *both* predictions to hold is what made it informative.

---

## Engineering rigor

- **169 tests**, 12 ADRs, contract versioning with gate immutability proven programmatically.
- **A critical reproducibility bug, found and fixed** ([ADR 0011](docs/adr/0011-frozen-gold-set-clock-dependence.md)): the "frozen" gold set was silently drifting one day per day, because Faker's `date_of_birth()` derives its window from `datetime.now()`. It was reproducible *within* a day and different across days — measured as an exact +8-day skew 8 days after the set was built. The fix reproduces the committed data **bit-for-bit**, so no prior measurement was invalidated, and a clock-shifted regression test (verified to fail on the old code) prevents recurrence.
- **Gates are never moved.** When the teacher changed, contract **v2** superseded v1 rather than editing it, with all six thresholds verified byte-identical.
- **Honest instrumentation:** the AWQ export path refuses to run rather than silently skipping when CUDA is absent; the economics harness prices the teacher at *paid* rates despite development running on a free tier.

---

## How it works

```
contract  ──▶  frozen gold set  ──▶  teacher baseline (the bar)
                                            │
                              verification-gated data engine
                              (k-sample vote, 3-layer dedup)
                                            │
                                   LoRA SFT  ──▶  student
                                            │
                        ┌───────────────────┴───────────────────┐
              deterministic validators                  distilled model
         (9 breach-severity identifiers)          (contextual PII types)
                        └───────────────────┬───────────────────┘
                                            ▼
                              eval on frozen test ──▶ gates
```

The teacher is scored **before** the student trains, so the parity threshold cannot be back-fitted to whatever the student happened to achieve.

## Reproducing

```bash
git clone https://github.com/Jatin-IITB/forge && cd forge
make install
export CEREBRAS_API_KEY=...       # free tier: https://cloud.cerebras.ai
make forge
```

Every stage resumes — these steps take hours and laptops sleep. The teacher endpoint is **fungible**: the same Apache-2.0 checkpoint is served by Cerebras and Groq and self-hostable with vLLM, so `TEACHER_URL` can point anywhere.

## Independence

Open-weight teacher, permissive base, public/synthetic data, own compute, OSS only. **Litmus test:** if all internal access were cut tomorrow, a stranger could clone this repo and rebuild it end to end ([ADR 0003](docs/adr/0003-independence-and-public-data.md)).

## Read in this order

1. [`docs/DESIGN.md`](docs/DESIGN.md) — first-principles design, field comparison, novelty calibration
2. [`docs/SUCCESS.md`](docs/SUCCESS.md) — the six gates and what counts as failure
3. [`docs/NORTH_STAR.md`](docs/NORTH_STAR.md) — the claim ledger: every claim mapped to evidence
4. [`docs/HONEST_ASSESSMENT.md`](docs/HONEST_ASSESSMENT.md) — what's novel, what's standard, what's weak
5. [`docs/adr/`](docs/adr/) — 12 decision records, including the two rejected hypotheses
6. [`MODEL_CARD.md`](MODEL_CARD.md) — intended use, training data, limitations

## License

Apache-2.0. Teacher, base model, and generated data are all permissively licensed.
