# Forge

**A task-specialization distillation pipeline.** Give it one expensive, high-volume LLM task and an open teacher; it manufactures a verified small specialist that matches teacher quality within a committed tolerance, at a fraction of the cost and latency, runnable fully offline.

Not an agent. This is the deliberate counterpart to an LLM-*orchestration* portfolio: it proves model **manufacture** — distillation, SFT/LoRA, verification-gated data, serving economics, and eval-first measurement rigor.

---

## Flagship task: on-device PII detection & redaction

The privacy argument is airtight: **you cannot send sensitive text to a frontier API in order to find the sensitive text.** A local specialist is the only compliant option under GDPR and India's DPDP Act 2023. So the task is real, the metric is crisp (entity-level F1), and the demo works with Wi-Fi off.

| | |
|---|---|
| **Teacher** | `openai/gpt-oss-120b` — Apache-2.0, MoE ~117B total / ~5.1B active |
| **Student** | `Qwen/Qwen2.5-1.5B-Instruct` + LoRA — Apache-2.0 |
| **Entity types** | 19, of which 9 are high-severity (recall gated at 0.99) |
| **Frozen gold set** | 574 records — 385 test / 189 dev, never trained on |
| **Primary metric** | micro-F1 over exact `(start, end, label)` spans |

## Status — honest

**Phase 4 of 6: the error-driven loop is running. No gate has passed yet except schema validity.**

| Gate | Threshold | Measured | |
|---|---|---|---|
| G1 quality parity | student >= 0.98 x teacher | teacher bar being measured | ⏳ |
| G2 schema validity | >= 99.9% | **100%** (385/385, run_001) | ✅ |
| G3 cost per 1k | <= teacher / 10 | harness built, awaiting both runs | ⏳ |
| G4 p95 latency | <= teacher / 5 | harness built, awaiting both runs | ⏳ |
| G5 deployability | laptop / CPU | unquantized MPS only | ⏳ |
| G6 safety / OOD | >= 0.90 | not started | ⏳ |

Run history: **run_001** (150 training records) scored **F1 0.52** — error analysis traced it to severe per-type imbalance and multi-span blindness. **run_002** (837 records after targeted augmentation) is training now.

The full claim-by-claim ledger, including what is *not* yet true, lives in [`docs/NORTH_STAR.md`](docs/NORTH_STAR.md). Nothing in this README is quoted as a result until its gate says PASS.

## How it works

```
contract  ──▶  frozen gold set  ──▶  teacher baseline (the bar)
                                            │
                                            ▼
                              verification-gated data engine
                              (k-sample vote, 3-layer dedup)
                                            │
                                            ▼
                                   LoRA SFT  ──▶  student
                                            │
                                            ▼
                          eval on frozen test ──▶ gates pass?
                                            │
                                   no ──────┴────── yes
                                    │                 │
                            error analysis      quantize (GGUF/AWQ)
                            targeted data       re-run gates
                                    │                 │
                                    └──▶ retrain      ▼
                                                  ship + card
```

The ordering is the point: **the teacher is scored before the student is trained**, so the parity threshold cannot be back-fitted to whatever the student happened to achieve.

## Design commitments

- **Eval-first.** The frozen gold set and the six gates exist *before* the model. Moving a threshold after seeing results voids the run ([`SUCCESS.md`](docs/SUCCESS.md)).
- **Verification-gated data.** No unverified teacher output reaches training. Rejects are logged, not silently dropped.
- **Error-driven.** Teacher tokens are spent where the student is measurably wrong, not uniformly.
- **Economics are gates.** `$/1k` and p95 are pass/fail criteria with a published cost model — the teacher is priced at *paid* rates even though development ran on a free tier, because a subsidy is not an economics claim.
- **Independence.** Open-weight teacher, permissive base, public data, own compute, OSS only. **Litmus test:** if all internal access were cut tomorrow, a stranger could clone this repo and rebuild end to end ([`adr/0003`](docs/adr/0003-independence-and-public-data.md)).

## Reproducing

```bash
git clone https://github.com/Jatin-IITB/forge && cd forge
make install
export CEREBRAS_API_KEY=...       # free tier: https://cloud.cerebras.ai
make forge
```

`make forge` runs the whole chain: validate contract → build gold → score teacher → generate verified data → train → evaluate gates → measure economics. Every stage resumes, because these steps take hours and laptops sleep.

The teacher endpoint is **fungible** — the same Apache-2.0 checkpoint is served by Cerebras and Groq and self-hostable with vLLM. Point `TEACHER_URL` elsewhere and nothing else changes.

## Read in this order

1. [`docs/DESIGN.md`](docs/DESIGN.md) — first-principles design, field comparison, honest novelty calibration.
2. [`docs/SUCCESS.md`](docs/SUCCESS.md) — the six gates, the maturity rubric, and what counts as failure.
3. [`docs/NORTH_STAR.md`](docs/NORTH_STAR.md) — the claim ledger: every public claim mapped to repo evidence.
4. [`docs/ACTION_PLAN.md`](docs/ACTION_PLAN.md) — phased plan with exit gates, budget, risk register.
5. [`docs/adr/`](docs/adr/) — decision records, 0001 through 0010.
6. [`MODEL_CARD.md`](MODEL_CARD.md) — intended use, training data, limitations.

## License

Apache-2.0. Teacher, base model, and generated data are all permissively licensed and redistributable.
