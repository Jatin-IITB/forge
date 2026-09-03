# Economics & latency — gates G3 and G4

**Date:** 2026-09-03 · **Decision records:** `docs/adr/0018-serving-stack.md`,
`docs/adr/0019-g3-closure-attempt.md`
**Harness:** `scripts/bench_serving.py`, decoding A/B `scripts/bench_grammar.py`
**Artifacts:** `reports/bench/FINAL_*.json`, `reports/bench/compact_abcd.json`,
`reports/bench/spec_*.json`
**Hardware:** Apple **M1** (8 CPU, 8-core GPU, 16 GB unified), llama.cpp `9cffdcc`, Metal

Every student number below is measured on the **full frozen 385-record test set**, 0 errors.
Earlier WP-1 sweeps used 48- or 128-record subsets under heavy contention; they are kept in
`reports/bench/sweep_*.json` as exploration only. **None of them is a gate number**, and a
per-record cost measured on one subset is not comparable to another subset's, because it
depends on how long those particular outputs happened to be.

## Verdict

| Gate | Requirement | Measured | Ratio | Threshold | Verdict |
|---|---|---|---|---|---|
| **G3** | `$/1k ≤ teacher/10` | **$0.03050** /1k | **0.1913×** | ≤ 0.10× | ❌ **FAIL — 1.91× over** |
| **G4** | `p95 ≤ teacher_p95/5` | **5.4131 s** | **3.969×** | ≤ 0.20× | ❌ **FAIL — 19.8× over** |

G3 is close enough to be an engineering problem. G4 is not: the student is **4× slower** than
the teacher where the gate asks it to be 5× faster, and §6 shows the target sits below this
machine's theoretical hardware ceiling.

---

## 1. The bar moved: the teacher baseline was re-measured

G3 and G4 are defined as multiples of a teacher measurement, so the teacher number *is* the
gate. The previously published teacher p95 of **8.024 s is withdrawn** (ADR 0014): it came
from a 302-record run predating the `latencies_s` field, so its percentiles were computed from
whichever resumed segment ran last rather than the pooled distribution, and it ran during
episodic free-tier congestion. The per-call vector was never stored, so it cannot be audited.

| teacher run | n | p50 | **p95** | per-call vector stored? |
|---|---|---|---|---|
| `predictions_teacher_120b_test` (contaminated) | 302 | 0.586 | **8.024** | no — unauditable |
| `teacher_token_sample` | 60 | 0.507 | 0.790 | — |
| **`predictions_teacher_120b_relat` (clean)** | **385** | **0.492** | **1.364** | **yes, 385 values** |

**This moved G4 against us by 5.9×:**

| | old (withdrawn) | **current** |
|---|---|---|
| teacher p95 | 8.024 s | **1.364 s** |
| **G4 target** | ≤ 1.605 s | **≤ 0.2728 s** |

G3's target is unchanged, derived from the same clean run's token counts:

```
teacher tokens/record   345.5 in, 105.8 out    (132,999 / 40,752 over 385 records)
Cerebras gpt-oss-120b   $0.25 / 1M in, $0.69 / 1M out
                        public paid-tier list price. We measured on the free tier and
                        PRICE at paid — the direction that makes our own gate harder.
teacher $/1k records  = 345,452 x 0.25/1e6 + 105,849 x 0.69/1e6 = $0.15940
G3 target (<= 0.10x)  = $0.01594 / 1k
```

---

## 2. The cost model, spelled out

The student runs on-device, so its bill is amortized hardware plus metered energy — there is
no per-token price to look up, which is precisely why the model must be published rather than
asserted. `bench_serving.py` duplicates `run_economics.py`'s pricing deliberately and asserts
parity at runtime; neither definitive artifact carries a drift warning.

```
student_hardware_usd        1599.00     laptop purchase price
student_life_years             4.0      amortization window
student_busy_hours_per_day     8.0      generous to the teacher: idle hours are not charged
student_watts                 22.0      sustained package power under load
student_usd_per_kwh            0.12

hardware $/hour = 1599 / (4 x 365 x 8)   = $0.136901
energy   $/hour = 22/1000 x 0.12         = $0.002640
total    $/hour                          = $0.139541

$/1k records = 0.139541 x (sustained_s_per_record x 1000 / 3600)
             = 0.038761 x sustained_s_per_record
```

Inverting the gate: **G3 requires ≤ 0.4112 machine-seconds per record.**

The input is *machine*-seconds — wall clock over the whole set divided by n, so overlapping
requests are counted once. It is not the per-request round trip. Under concurrency 32 those
two differ by 27× here (0.775 s vs 21.4 s), and using the wrong one is the easiest available
way to fake this gate.

---

## 3. Two named configs, published side by side

G3 wants throughput (large batches); G4 wants single-request latency (batch 1). They pull in
opposite directions, so no single config optimises both. **Neither column may be read across.**
The p95 in the latency column and the $/1k in the throughput column come from different
servers; quoting the flattering number from each as if one config produced both is the
specific dishonesty this harness exists to prevent.

| | **Throughput config** (G3) | **Latency config** (G4) |
|---|---|---|
| server flags | `-np 32 -c 32768 --mlock` | `-np 1 -c 2048 --mlock` |
| client concurrency | 32 | 1 |
| slots × context/slot | 32 × 1024 | 1 × 2048 |
| RSS after load | 2341 MB | 1480 MB |
| records | 385/385, 0 errors | 385/385, 0 errors |
| schema-valid | 385/385 = 1.0000 | 385/385 = 1.0000 |
| prompt / output tok per record | 292.4 / 53.5 | 292.4 / 53.4 |
| output tok/s aggregate | **68.97** | 17.18 |
| **sustained s/record** | **0.7751** | 3.1086 |
| **$/1k records** | **$0.03004** | $0.12049 |
| p50 request latency | 17.474 s | 2.6125 s |
| p90 / **p95** / p99 | 32.618 / **39.417** / 59.716 s | 4.556 / **5.4131** / 8.658 s |
| min / max request | 6.816 / 185.577 s | 1.076 / 24.174 s |
| loadavg start → end | 2.73 → 3.84 | 3.62 → 2.26 |
| passes (`--repeat`) | 2, best reported | 1 |

**G3** consumes only the throughput column: $0.03004 vs $0.01594 → **FAIL, 1.885× over**.
Cost ratio to teacher 0.1885 against a ≤ 0.10 gate.

**G4** consumes only the latency column: p95 5.4131 s vs 0.2728 s → **FAIL, 19.84× over**.
p95 ratio to teacher 3.969 against a ≤ 0.20 gate.

---

## 4. Per-lever attribution

All three legs on the full 385-record set, same records, same prompt, greedy decoding:

| # | Stage | s/record | $/1k | step gain | cumulative |
|---|---|---|---|---|---|
| 0 | fp16, `transformers`, MPS, batch 1 | 4.9283 | $0.19101 | — | 1.00× |
| 1 | Q4_K_M, llama.cpp Metal, batch 1 | 3.1086 | $0.12049 | **1.59×** | 1.59× |
| 2 | + continuous batching, `-np 32`, c=32 | **0.7751** | **$0.03004** | **4.01×** | **6.36×** |

Leg 0 is the historical `predictions_student_run_002.meta.json` figure. **It recorded no token
counts**, so until now nothing established that it was doing the same amount of work as the
legs it is divided by. A 48-record re-measurement with tokens instrumented
(`reports/bench/ATTRIB_fp16_mps_b1.json`) closes that gap: **50.7 output tokens/record at
5.479 s/record**, against Q4_K_M's 53.4 at batch 1. The output lengths are comparable, so the
ratio is a serving-stack ratio and not an artefact of one leg emitting less text.

### Sub-attributing leg 1: quantization is the smaller half

`llama-bench` isolates quantization exactly — same runtime, same machine, only the weights
differ (5 repetitions, pp320/tg64):

| weights | file size | prefill tok/s | **decode tok/s** |
|---|---|---|---|
| F16 | 2.88 GiB | 187.1 ± 9.0 | 9.99 ± 0.29 |
| Q8_0 | 1.53 GiB | 150.7 ± 20.2 | 12.18 ± 0.31 |
| **Q4_K_M** | **934.7 MiB** | 142.3 ± 10.7 | **14.44 ± 0.21** |

Quantization f16 → Q4_K_M is worth **1.45×** at batch 1, which leaves only ~1.10× for the
runtime change. Note that `transformers`-on-MPS fp16 measured 9.2 output tok/s aggregate
against llama.cpp f16's 9.99 tok/s decode — **the two runtimes are within noise of each other
at batch 1.** The roadmap's expectation that "GGUF Q4_K_M vs transformers-MPS fp16 = 2–4×" was
optimistic by roughly 2×.

### And at concurrency 32, quantization buys essentially nothing

Full 385 records, identical server flags, only the weights swapped:

| weights | file size | RSS | s/record | output tok/s | $/1k |
|---|---|---|---|---|---|
| f16 GGUF | 3094 MB | 4405 MB | 0.8379 | 61.6 | $0.03248 |
| **Q8_0** | 1647 MB | 3080 MB | **0.7218** | **71.2** | **$0.02798** |
| Q4_K_M | 986 MB | 2341 MB | 0.7751 | 68.97 | $0.03004 |

The spread is 0.72–0.84 s/record — **smaller than this machine's run-to-run variance** (§5), so
these three are not distinguishable. Q8_0 came out nominally fastest despite Q4_K_M being the
only one given `--repeat 2`. The mechanism is straightforward: at batch 1 decode is
memory-bandwidth bound and shrinking the weights helps; at batch 32 the weights are read once
per step and amortized across 32 sequences, so the run becomes compute-bound and cheaper
weights stop paying. **Batching does essentially all the work in the shipped config.**

---

## 5. Measurement conditions — read this before quoting anything

This is a shared 16 GB laptop running its owner's normal applications and a corporate MDM
agent, not a quiesced benchmark host. Every artifact carries `contention_at_start` and
`contention_at_end`.

- **The machine was never below loadavg ~2.** `pmd` (Palo Alto Networks Traps, corporate
  endpoint security) holds a full core continuously and cannot be killed. Two unrelated
  `uvicorn` servers from another project were also resident. The brief's "quiet window" was
  partly real — load fell from 5–13 during the earlier sweeps to 2.0–3.8 — but not idle.
- **Both definitive runs exceeded loadavg 2 throughout**: throughput 2.73 → 3.84, latency
  3.62 → 2.26. Stated here and in the artifacts, per the reporting rule.
- **Swap was 5.2–5.5 GB in use** the whole session. `--mlock` was adopted for the definitive
  configs specifically to stop the weights being paged or compressed.
- **Run-to-run variance is large and must not be sanded off.** Two runs of the *identical*
  config on the *identical* 128 records producing the *identical* 6798 completion tokens took
  130.5 s and 207.6 s — a **1.59× spread**. Within the `--repeat 2` throughput run the two
  passes were 0.894 and 1.290 rec/s, a 1.44× spread. Single-shot numbers on this machine carry
  roughly ±30%; that is why config comparisons here lean on `output tok/s aggregate` over
  repeated passes rather than on one wall clock.

**Consequently: the G3 miss of 1.885× is real but not precisely 1.885×.** It is somewhere
around 1.5–2.5× depending on machine state. The G4 miss of 19.8× is far outside any variance
this machine exhibits and is not in question.

---

## 6. What it would take to pass

### G3 — an engineering gap

Need ≤ 0.4112 s/record. At 53.5 output tokens that is **130 output tok/s aggregate**; measured
68.97. **A 1.89× throughput gain closes it.** Levers already tested and exhausted: concurrency
64 is past the knee (48.1 tok/s), larger `-ub` hurts (49.4), `--kv-unified` hurts, and
quantization below Q8_0 does nothing at batch ≥ 32. What remains untested is reducing the
output itself — 53.5 tokens of JSON per record — and speculative decoding with a 0.5B draft
model, which needs a download this session was not permitted to make. G3 is plausibly
reachable on this hardware; it is not reachable by more of what has already been tried.

### G4 — below the hardware floor

Need p95 ≤ 0.2728 s. The budget does not survive first contact with the decomposition. A
streaming measurement over 12 records gives:

| component | measured | share of latency |
|---|---|---|
| prefill (TTFT, steady state) | 0.19–0.38 s | 11.2% |
| decode, 53.4 tokens @ 17.3 tok/s | ~3.1 s | 88.8% |

**Prefill alone — 0.19 s at best, 0.38 s typical — consumes the entire 0.2728 s budget before
a single output token is produced.** Prefill is not the problem to solve either: it runs at
774 tok/s, which is healthy.

To fit 53.4 output tokens into the ~0.08 s that would remain requires **≈ 660 output tok/s
single-stream**. This machine's *theoretical* ceiling is memory bandwidth divided by model
size: 68.25 GB/s ÷ 0.935 GB = **~73 tok/s**. The target is therefore **~9× beyond the
theoretical maximum of the hardware**, not merely beyond its measured performance. Measured
decode is 17.3 tok/s, i.e. 24% of that ceiling, so even perfect kernels leave a 9× hole.

Nothing available closes it:

| change | best case | still short by |
|---|---|---|
| perfect kernels (100% of M1 bandwidth) | 73 tok/s | 9.0× |
| swap to a 0.5B model at Q4 (~350 MB) | ~195 tok/s theoretical | 3.4× — and F1 collapses |
| speculative decoding, 0.5B draft | ~2× on structured JSON | 4.5× |
| M4 Max (546 GB/s), same model | ~584 tok/s theoretical, ~350 realistic | ~2× |

**G4 is unreachable on Apple M1 for any model capable of this task.** Passing it would need
different hardware *and* a shorter output format, together.

There is also a structural point worth stating plainly, because it is not a matter of effort.
The teacher's p95 of 1.364 s is a **network round trip to Cerebras wafer-scale silicon** — at
105.8 output tokens and a 0.492 s median, the remote model's compute is a small fraction of
that number and the rest is transit. G4 asks a laptop to complete a full local inference in
one fifth of the time it takes a packet to reach California and come back with twice as much
text. The contract is not moved and the gate is reported FAIL, but the reason it fails is that
the ratio was calibrated against a denominator dominated by network latency, not against a
compute budget.

---

## 7. G3 closure attempt — output shortening and speculative decoding

ADR 0019 tested the two remaining WP-1 levers on the full 385 records. All headline
throughput rows are repeated best-of-N measurements; every artifact records load average and
swap. No subset result is used for cost or quality.

### Compact output: one apparent pass did not reproduce

The compact shape kept source text and shortened only the envelope:
`{"s":[{"l":"PERSON","t":"Jessica Holmes"}]}`. Offsets were still reconstructed from the
emitted substring; no character-counting task was introduced.

The four-arm same-server A/B reduced output by **36.2%**, from 51.4 to 32.8 tokens/record.
Grammar-forcing compact output under the original verbose prompt collapsed to empty lists
(F1 0.0029, recall 0.0014). Requesting compact output in the prompt worked; combining prompt
and grammar produced 385/385 valid responses and one best pass just below the cost bar:

| same-server arm | s/record | $/1k | out tok/s | loadavg 1m start → end | P | R | F1 | schema |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| fresh verbose baseline | 0.5291 | $0.02051 | 97.1 | 1.60 → 2.13 | 0.6171 | 0.6561 | 0.6360 | 384/385 |
| compact grammar only | 0.1674 | $0.00649 | 30.5 | 1.70 → 1.60 | 0.5000 | 0.0014 | 0.0029 | 385/385 |
| compact prompt only | **0.3677** | **$0.01425** | 90.1 | 2.09 → 1.70 | 0.7276 | 0.6187 | 0.6687 | 381/385 |
| compact prompt + grammar | **0.3987** | **$0.01545** | 82.3 | 3.96 → 2.09 | 0.7163 | 0.6432 | 0.6778 | 385/385 |

The apparent prompt+grammar pass had only **3.1% margin**. A definitive three-pass run through
`bench_serving.py` did not reproduce it:

| standard harness | pass rates (records/s) | best s/record | $/1k | out tok/s | loadavg 1m start → end |
|---|---|---:|---:|---:|---:|
| fresh verbose baseline | 1.728, 1.465, 1.399 | 0.5785 | $0.02243 | 88.8 | 2.66 → 3.90 |
| compact prompt + grammar | 1.569, 1.542, 1.626 | **0.6149** | **$0.02383** | 53.4 | 1.67 → 9.69 |

All three compact passes missed 0.4112 s/record. The best missed by **1.495×**. A single pass
inside a machine's known ±30% run-to-run spread is not a closed gate.

Quality also changed directionally. Against the fresh verbose baseline, compact
prompt+grammar moved F1 **+0.0418** and precision **+0.0993**, but recall **−0.0129**.
The full `run_eval.py --ci --validators` suite measured F1 0.6778
[0.6387, 0.7150], P 0.7163, R 0.6432, schema 1.0000, and system high-severity
recall 1.0000. For redaction, the recall loss is disclosed rather than hidden behind the F1
gain.

### Speculative decoding: high acceptance, lower throughput

The draft was public, ungated **Qwen2.5-0.5B-Instruct**, Apache-2.0, pinned at Hugging Face
revision `7ae557604adf67be50417f59c2c2f167def9a775` and converted locally to a 531 MB Q8_0
GGUF (sha256 `9803f5ede78984082c3fa5693368a313a87220ff7fc35d1cccb5c5a5bd826c05`).
This satisfies the project's independence rule.

| config, repeated twice | s/record | $/1k | out tok/s | relative throughput | loadavg 1m start → end |
|---|---:|---:|---:|---:|---:|
| fresh baseline | 0.7868 | $0.03050 | 65.3 | 1.00× | 4.72 → 9.77 |
| draft max 3 | 1.1857 | $0.04596 | 43.4 | **0.664×** | 8.88 → 3.55 |
| draft max 8 | 1.9281 | $0.07474 | 26.6 | **0.408×** | 4.04 → 15.75 |

The server logs show many accepted draft runs, so token prediction was not the failure.
At concurrency 32, running a second model and verifying larger token blocks cost more than
the accepted tokens saved. Increasing the block from 3 to 8 made that overhead worse.

Speculation preserved quality within noise: draft-3 measured F1/P/R
0.6360/0.6171/0.6561 (identical to baseline), while draft-8 measured
0.6350/0.6165/0.6547 (deltas −0.0010/−0.0006/−0.0014). Both remained 384/385
schema-valid and retained 1.0000 system high-severity recall.

### G3 verdict

**G3 remains FAIL.** The clean same-session speculative baseline is $0.03050/1k against
$0.01594/1k: **1.91× over the gate**. Output shortening can cross the line in a favourable
single pass, but not reproducibly; speculative decoding moves in the wrong direction.

Closing G3 on this model/hardware now requires one of:

1. retraining on compact targets, then demonstrating ≤0.4112 s/record while preserving
   recall and the full gate suite;
2. a serving implementation that sustains about 125 verbose output tok/s without a second
   model; or
3. faster hardware under the unchanged amortization and energy model.

The threshold and cost model are unchanged.
