# ADR 0018 — The serving stack: what ships, and the two gates it still misses

**Status:** ACCEPTED (2026-09-03) — with G3 and G4 recorded as FAIL
**Date:** 2026-09-03
**Depends on:** ADR 0014 (measurement integrity — supplies the corrected teacher p95);
ADR 0012 (validator layer); `docs/ROADMAP.md` §3 WP-1.
**Artifacts:** `reports/economics.md`, `reports/quantization_gates.md`,
`reports/bench/FINAL_*.json`, `models/pii-1.5b-gguf/manifest.json`

## Context

WP-1 exists because the student was **more expensive than the teacher it was meant to
replace** — $0.1910/1k against $0.1592/1k, a 1.20× cost *increase*. The whole deficit traced
to one number: 4.93 s/record, unquantized fp16, batch 1, through `transformers` on MPS.

The roadmap's plan was two levers: GGUF Q4_K_M for 2–4×, continuous batching for 4–8×, "8–32×
together, 12× is reachable." This ADR records what those levers actually paid, on an Apple
**M1** (8 CPU, 8-core GPU, 16 GB unified) — hardware the roadmap never named and which turns
out to be the binding constraint.

Two corrections arrived with the work and change how the targets read:

- **The teacher p95 of 8.024 s was withdrawn** (ADR 0014) and re-measured clean at **1.364 s**
  over 385 records with the per-call vector stored. G4's target moved from ≤ 1.605 s to
  **≤ 0.2728 s — 5.9× harder.**
- **Every prior WP-1 number was taken on a 48- or 128-record subset** at loadavg 5–13 with swap
  86% full. None of them was a gate number. All figures here are full-385.

## Decision

**Ship the Q8_0 GGUF, served by llama.cpp with Metal, under two separately published configs.**

```
throughput config (gate G3)   llama-server -ngl 99 -np 32 -c 32768 --mlock   client c=32
latency config    (gate G4)   llama-server -ngl 99 -np 1  -c 2048  --mlock   client c=1
```

Three sub-decisions, each with its reason:

1. **Q8_0, not Q4_K_M.** WP-1 pre-committed to "quantized F1 within 0.01 of fp16, else ship
   Q8_0 and report the larger artifact honestly." Measured against the **f16 GGUF** — the only
   baseline that isolates quantization — Q4_K_M costs **0.0151 F1** and Q8_0 costs
   **−0.0022**. The rule is applied as written. Cost of the fallback: 1647 MB instead of
   986 MB. It costs nothing in speed (see Surprise 2).
2. **`--mlock`.** With 5.2–5.5 GB of swap in use throughout, the weights were being paged and
   compressed. Locking them was the single largest stabiliser found.
3. **Two configs, never merged into one row.** G3 and G4 pull opposite ways; the throughput
   config's p95 is 39.4 s and the latency config's $/1k is $0.1205. Both are published in full
   so neither can be quoted beside the other's flattering half.

## Evidence

Full 385-record test set, 0 errors, greedy decoding, `scripts/bench_serving.py`.

| Gate | Requirement | Measured (Q4_K_M) | Ratio | Verdict |
|---|---|---|---|---|
| G3 cost | ≤ $0.01594 /1k | $0.03004 /1k | 0.1885× vs ≤0.10 | ❌ **FAIL — 1.89× over** |
| G4 p95 | ≤ 0.2728 s | 5.4131 s | 3.969× vs ≤0.20 | ❌ **FAIL — 19.8× over** |

Per-lever attribution, all three legs full-385, comparable output lengths verified:

| stage | s/record | $/1k | step | cumulative |
|---|---|---|---|---|
| fp16, `transformers`, MPS, batch 1 | 4.9283 | $0.19101 | — | 1.00× |
| Q4_K_M, llama.cpp Metal, batch 1 | 3.1086 | $0.12049 | 1.59× | 1.59× |
| + continuous batching, c=32 | 0.7751 | $0.03004 | **4.01×** | **6.36×** |

**6.36× total, against the 12× needed.** The roadmap's 8–32× estimate was optimistic by
roughly 2×, entirely in the quantization term.

## What surprised us

**1. `-c` is the *total* KV pool, not per-slot — the premise behind the whole tuning plan was
inverted.** The session began believing `-c 16384 -np 32` reserved "32 slots × 16k context",
making a `-c` reduction the highest-value change available. The server log settles it:
`-c 32768 -np 32` yields `n_ctx_slot = 1024`. **`-c` is divided by `-np`.** So there was no
large over-reservation to reclaim on the throughput config — it was already correctly sized —
and cutting `-c` to 16384 gives 512 tokens/slot, which **truncates**: that run produced 6595
completion tokens against 6798 for every other config, and a schema-invalid record with it.

The reclaimable memory was on the *latency* config, where nothing needs 32 slots:
`-np 1 -c 2048` drops RSS from 2341 MB to 1480 MB. KV cost is exactly
2 × 28 layers × 2 KV heads × 128 dim × 2 bytes = **28.0 KiB/token**; the measured RSS slope
across `-c` 16384/32768/65536 implies 28,907 B/token, agreeing to 0.8%. The rule is
`-c` = `-np` × (max prompt + max output), and for this task that is 1024/slot.

**2. Quantization buys nothing at concurrency 32 — batching does all the work.** Identical
server flags, only the weights swapped: f16 0.8379 s/rec, Q8_0 0.7218, Q4_K_M 0.7751. The
spread is smaller than the machine's own variance, and Q8_0 came out nominally fastest despite
Q4_K_M being the only one given `--repeat 2`. At batch 1, quantization is worth 1.45×
(`llama-bench`: 14.44 vs 9.99 tok/s); at batch 32 the weights are read once per step and
amortized across 32 sequences, so decode turns compute-bound and cheaper weights stop paying.
This is why the fallback to Q8_0 is free, and why the roadmap's "Q4_K_M = 2–4×" did not
materialise.

**3. The merge-and-runtime change moved quality four times more than quantization did — in the
opposite direction.** fp16-HF → f16-GGUF is **+0.0588 F1**; f16-GGUF → Q4_K_M is **−0.0151**.
Comparing Q4_K_M directly to the published fp16 baseline shows *+0.0437*, which is a true
number from a real comparison and completely wrong about what it appears to measure. Without
the f16-GGUF control, a lossy artifact would have shipped advertised as a quality improvement.
The gain is concentrated in PERSON (85 → 256 predicted spans, recall 0.3714 → 0.8343) — the
fp16 model's largest error source. **Measured, reproducible, and not diagnosed.**

**4. Run-to-run variance is 1.59× on byte-identical work.** The same config on the same 128
records producing the same 6798 completion tokens took 130.5 s and 207.6 s. Any single-shot
serving number on this machine carries roughly ±30%, which is larger than several of the
config differences the earlier sweep drew conclusions from.

**5. The two artifacts split the gates.** Q4_K_M is the only one of four to pass G2
(385/385 schema-valid); Q8_0, f16-GGUF and fp16-HF all sit at 384/385 = 0.9974 and fail. G2
has **zero margin at n=385** — one malformed response fails it — and the same model is only
96.9% span-identical to itself across serving configs, so one record is inside drift. Recorded
as a real tension rather than resolved by picking the artifact that looks better.

## What we could not reach

**G3 — an engineering gap, 1.89× wide.** Need 130 output tok/s aggregate; measured 68.97.
Exhausted this session: concurrency 64 is past the knee (48.1 tok/s), larger `-ub` hurts
(49.4), `--kv-unified` hurts, quantization below Q8_0 does nothing at batch ≥ 32. Untested and
still open: shortening the 53.5-token JSON output, and speculative decoding with a 0.5B draft
model — which needs a download this session was not permitted to make.

**G4 — below the hardware floor, and not by a little.** A streaming decomposition shows
prefill is *not* the problem: it runs at 774 tok/s and is 11.2% of latency. Decode is 88.8% at
17.3 tok/s. But **prefill alone (0.19–0.38 s) already consumes the entire 0.2728 s budget
before one output token exists.** Fitting 53.4 output tokens into the remainder needs
**≈660 tok/s single-stream**. This machine's theoretical ceiling is bandwidth ÷ model size =
68.25 GB/s ÷ 0.935 GB = **~73 tok/s**. The target is **~9× beyond the theoretical maximum of
the hardware**, so no amount of kernel tuning reaches it; a 0.5B model is still 3.4× short and
would destroy F1; an M4 Max is still ~2× short.

**G4 is unreachable on Apple M1 for any model capable of this task**, and the reason is worth
stating precisely: the teacher's 1.364 s p95 is a **network round trip to Cerebras wafer-scale
silicon**, and at a 0.492 s median the remote compute is a small fraction of it. G4 asks a
laptop to finish a complete local inference in one fifth of the time a packet needs to reach
California and return with twice as much text. The threshold is not moved — it is
pre-committed and the contract forbids moving it — but the record should show that the gate
was calibrated against a denominator dominated by transit, not by compute.

## Consequences

- `docs/NORTH_STAR.md` claims 3 and 4 need updating from "in flight" to measured FAIL, with
  6.36× recorded as the improvement actually delivered. *(Owned by another agent; not edited
  here.)*
- `MODEL_CARD.md` should carry the Q8_0 artifact, its sha256, and both configs. *(Not edited
  here — same reason.)*
- `models/pii-1.5b-gguf/manifest.json` still reads `"verified": false`. It stays false until
  someone decides whether an artifact that passes G5 and fails G1–G4 counts as verified; that
  is a claim-ledger decision, not a serving one.
- The Q4_K_M artifact is **kept, not deleted**. It is the better choice for a
  memory-constrained deployment and the only one that passes G2, and §4 of
  `reports/quantization_gates.md` records the trade honestly.

## Alternatives considered

| option | why not |
|---|---|
| Ship Q4_K_M anyway (smaller, passes G2) | Misses a pre-committed fidelity threshold. Revising the rule after seeing which side the number fell on is the specific failure this project is built to avoid. |
| Report the throughput config's $/1k beside the latency config's p95 | That single row would show $0.03004 and 5.41 s and pass neither gate honestly. The harness stamps config names into artifacts to make it impossible. |
| Re-baseline G4 against the 60-record teacher sample (p95 0.790 s) | Makes the gate *harder* (0.158 s), and the 385-record clean run is the better measurement regardless. Not chosen for either reason — the contract names the measurement, not the convenient one. |
| Drop `--mlock` for a smaller resident set | Costs the stability that made repeated measurement meaningful, on a machine already 5+ GB into swap. |
| Speculative decoding with a 0.5B draft | Needs a model download; this session was network-restricted. Left open — but §"could not reach" shows ~2× is not enough for G4 anyway. |
