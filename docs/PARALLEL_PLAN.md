# Parallel execution plan — two machines, four lanes

*Supersedes the sequencing in `CLAIM_CLOSURE_PLAN.md`; the triage there still stands.*

## The insight this plan is built on

Four resources this project needs are **mutually orthogonal**. Nothing in one lane slows
anything in another, so four agents can run at full speed simultaneously:

| lane | resource | contends with |
|---|---|---|
| **A** | Cerebras API, 5 req/min — network-bound, ~0% CPU | only itself |
| **B** | Laptop **CUDA** GPU — training, quantization | only itself |
| **C** | M1 **Metal** GPU — serving benchmarks | only itself |
| **D** | CPU + docs — audit, ledger, reports | nothing |

The teacher job in lane A ran at **0.0% CPU** for 77 minutes while lane C benchmarked — that
is measured, not assumed. The only true serialization is that lane B's *outputs* (a new
adapter) become lane C's *inputs*.

**Prerequisite: push.** 27 commits and ~60 files are local-only. The second machine cannot
participate at all until `origin/main` is current. This is no longer housekeeping — it is
the thing that unlocks lane B.

---

## What a normal laptop GPU actually changes

Measured against the gate arithmetic, not assumed:

### G4 — improves, still fails

Decode ceiling = memory bandwidth ÷ model bytes (Q8_0 = 0.935 GB). Requirement is 660 tok/s.

| machine | bandwidth | ceiling | short by |
|---|---|---|---|
| M1 (measured) | 68 GB/s | 73 tok/s | **9.0×** |
| RTX 3050 laptop | 192 GB/s | 205 tok/s | **3.2×** |
| RTX 4060 laptop | 272 GB/s | 291 tok/s | **2.3×** |

The ADR 0019 proof — "9× beyond the hardware" — is **specific to the M1** and must be
restated as such. On a mid laptop GPU it becomes 2.3×, which is a hard engineering gap rather
than a physical impossibility. **G4 stays FAIL, but the reason changes**, and the ledger
should say so.

### G3 — genuinely helped, and possibly free

The cost model charges for the machine actually used, so a faster machine only wins if the
throughput gain beats its hourly cost:

| machine | $/hour | throughput needed to break even |
|---|---|---|
| M1 — $1599, 22 W | $0.1395 | baseline |
| budget GPU laptop — $1200, 120 W | **$0.1171** | **0.84× — cheaper per hour** |
| mid GPU laptop — $1800, 150 W | $0.1721 | 1.23× |
| high GPU laptop — $3000, 200 W | $0.2808 | 2.01× |

A **cheaper** laptop that is also faster wins twice. **Measure the real purchase price and
wall-draw of the actual machine and put them in `run_economics.py`.** Quoting GPU throughput
against the M1's cost model is the easiest available way to fake this gate.

### Unblocked outright

Ledger rows **14 (QLoRA)** and **16 (AWQ)** are marked *blocked on hardware* — `bitsandbytes`
has no MPS backend and AWQ calibration needs CUDA kernels. Both close on any CUDA device.

---

## The lever that outranks the hardware

`forge/token_classifier.py` is untracked, unbenchmarked, and replaces the vocabulary head with
**77 BIOES labels**, reconstructing spans in deterministic code.

Autoregressive decoding of ~53 JSON tokens is **89% of serving time**. A tagging head replaces
53 sequential steps with **one forward pass**. No serving flag can do that — and every serving
flag has now been swept and failed.

If it holds accuracy it reshapes G3 and G4 together, and it does so on the M1, without new
hardware. **Benchmark it before spending a day on QLoRA.** It is lane C's first task.

It also changes the architecture story: generative span extraction becomes encoder tagging.
That is a different set of trade-offs to defend, and a real one.

---

## Lane assignments

### Lane A — data engine *(Cerebras, network-bound)*
Owns `data/train_v3.jsonl`, `scripts/generate_carriers.py`, `scripts/build_train_v3.py`.

1. **Filter the semantic defect.** 20/145 AGE and 88/325 DOB spans sit in contexts that
   contradict the label. Only 8 of 325 DOB spans have a birth anchor. Do not train on this.
2. **Whitelist, not blacklist**, in the generator: no DOB slot unless the carrier already
   contains a birth anchor. 231 bad spans have no cue at all and a blacklist misses them.
3. Resume to ≥4000 records, ≥40% Track B (currently 180 of 1774).
4. `reports/data_card_v3.md`, leakage 0 naming all four splits.

**Exit:** ≥1600 Track B, contradicted == 0, asserted by tests.
**Serves:** resume claim 1 — "distilled GPT-OSS-120B".

### Lane B — training *(CUDA laptop)*
Owns `checkpoints/`, `scripts/run_train.py`, `scripts/export_model.py awq`.

1. **QLoRA run** → closes ledger row 14.
2. **AWQ export + gate suite** → closes row 16.
3. **Capacity sweep** on the v3 corpus once lane A ships it, selecting on
   `data/gold/val.jsonl` only — dev is 79% contaminated.

Pre-register, and report against them even when they fail:
- r=64+MLP beats r=16 at ~4000 records, reversing run_003;
- **span ratio > 0.85** — load-bearing: loss improving without it means the task formulation
  is the ceiling, not the data;
- STREET_ADDRESS F1 > 0.40, from 0.09.

**Serves:** claim 1, gate G1, ledger rows 14 and 16.

### Lane C — serving *(M1 Metal)*
Owns `scripts/bench_serving.py`, `reports/bench/`, `reports/economics.md`.

1. **Token classifier benchmark — first, before anything else.** Accuracy on the frozen 385
   and throughput against the Q8_0 baseline, same session, fresh baseline.
2. **CPU-only run** (`-ngl 0`). Decides "GPU-free on a laptop" outright; never measured.
3. Re-measure economics under whichever artifact wins.

**Serves:** claims 2 and 7, gates G3 and G4.

### Lane D — audit and ledger *(no compute)*
Owns `docs/NORTH_STAR.md`, `MODEL_CARD.md`, ADRs, and the audit.

Every delegated number is re-derived from committed artifacts before it enters the ledger.
Checklist, ordered by what has actually caught something here:

1. measured on the **shipped artifact**, or a proxy?
2. **best-of-N** quoted as the result?
3. same **record subset** on both sides?
4. **fresh baseline** in the same session? (this machine drifted 8% in hours)
5. does the aggregate **hide a defect another rule is masking**?
6. is the threshold **measurable at this n**?
7. **precision and recall** published, not just F1?
8. new splits asserted **disjoint on the written bytes**?

---

## Claim → lane map

| resume claim | status | lane |
|---|---|---|
| ~99% recall on 9 high-risk types | ✅ **true** (1.0000, LB 0.9948) | — |
| ~99.7% valid JSON | ✅ **true** (now 1.0000 under GBNF) | — |
| majority-vote + 3-stage dedup, ~21% filtered | ✅ **true** (20.6%) | — |
| distilled GPT-OSS-120B → 1.5B | 🟡 180/1774 records | **A → B** |
| GPU-free on a laptop | 🟡 never measured | **C** |
| ~80× cheaper | ❌ measured 5.3× | **reword now** |
| grounded in retrieved source text | ❌ no retrieval exists | **reword now** |

Three claims are already true. Two reword today. Two are real work.

---

## Reproducibility defect to fix while pushing

`.gitignore:53` excludes **`data/ood_probe.jsonl`** — the G6 probe set. G6 is a contract gate
and now the one that passes, yet its evaluation asset cannot be reproduced from a clone.
`data/carriers_v3.jsonl` is excluded too, so `train_v3` cannot be rebuilt.

`data/gold/val.jsonl` is committed at 148 KB, so the precedent for small evaluation assets
already exists. Commit both, or the litmus test in `adr/0003` fails on the project's own terms.
