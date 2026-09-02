# Measurement integrity audit

**Date:** 2026-09-03 · **Decision record:** `docs/adr/0014-measurement-integrity.md`
**Harnesses:** `forge/ci.py`, `scripts/audit_gold.py`, `scripts/build_validation.py`
**Reproduce:** `make audit` · `scripts/run_eval.py … --ci --teacher-preds …`

Contract v2 requires every gate to be "measured with 95% CIs". None were. This audit added
the intervals, and in doing so found four defects and one result that was stronger than the
project had been able to claim.

---

## 1. What the gates actually measure

All numbers on the frozen 385-record test set. Student is `run_002`.

| Quantity | Point | 95% CI | Threshold | Verdict |
|---|---|---|---|---|
| teacher micro-F1 | 0.9482 | [0.9305, 0.9641] | — | the bar |
| student micro-F1 (model-only) | 0.5750 | [0.5324, 0.6162] | — | — |
| **G1 parity ratio** (paired) | **0.6064** | **[0.5624, 0.6485]** | ≥ 0.98 | ❌ **FAIL** |

G1 fails structurally, not statistically — the entire interval sits 0.33 below the threshold.

**The ratio must be bootstrapped paired.** Student and teacher are scored on the *same* frozen
records, so a record that is hard for one is usually hard for the other. Resampling the two
sides independently and dividing their intervals discards that correlation and reports **28%
more uncertainty than the data holds**:

| estimator | interval | width |
|---|---|---|
| paired (correct) | [0.5624, 0.6485] | 0.0861 |
| independent (naive) | [0.5324/0.9641, 0.6162/0.9305] | 0.1101 |

Records are the resampling unit throughout — never spans. Spans inside one record share a
fuzzy boundary and a missed sentence; treating them as independent draws would narrow every
interval, which is the direction that flatters a gate.

---

## 2. The high-severity floor is not measurable on this test set

The contract sets a **≥ 0.99 recall floor** on each of the nine high-severity types. The
system hits **1.0000 on all nine**. The bootstrap reports `[1.0000, 1.0000]`.

That interval is a degenerate artifact, not evidence. With zero misses in the sample, no
resample has any either — the estimator has nothing to vary. Perfect measured recall on 15
instances is not evidence of perfect recall.

For zero-failure data the exact (Clopper-Pearson) one-sided bound is the honest statement:
with 0 misses in *n* trials, recall ≥ `0.05^(1/n)`. Inverting it gives the sample size the
floor actually requires:

| floor to demonstrate | n needed per type (zero misses) |
|---|---|
| ≥ 0.95 | 59 |
| ≥ 0.98 | 149 |
| **≥ 0.99** | **299** |

Against what the frozen test set holds:

| type | test n | bound at 1.0000 | short of 299 by |
|---|---|---|---|
| DRIVER_LICENSE | 15 | 0.819 | 284 |
| SSN | 18 | 0.847 | 281 |
| PASSWORD | 19 | 0.854 | 280 |
| PASSPORT | 23 | 0.878 | 276 |
| AADHAAR / API_KEY / BANK_ACCOUNT / PAN | 29 | 0.902 | 270 |
| CREDIT_CARD | 41 | 0.930 | 258 |
| **pooled** | **232** | **0.987** | **67** |

**No per-type 0.99 claim is demonstrable here, and the pooled figure falls 67 instances
short.** This is a property of the test set, not of the model — the gate as written cannot be
satisfied with evidence at this sample size. It is disclosed rather than quietly reported as
passing on the point estimate.

---

## 3. The validator layer is not overfit — and it clears the floor out-of-sample

A real risk needed testing. The validator layer (ADR 0012) was **developed by inspecting
test-set misses** — it went from 3/9 to 9/9 floors by measuring which spans it was missing on
`test.jsonl`. Its perfect test recall is therefore partly a fitted score, closer to training
accuracy than to held-out accuracy.

The new `val` split (seed 4242, generated after the validators were frozen, verified disjoint
from train/dev/test) provides the first genuinely out-of-sample measurement:

| | n | misses | recall | 95% lower bound |
|---|---|---|---|---|
| test (validators tuned on this) | 232 | 0 | 1.0000 | 0.9872 |
| **val (never seen)** | **339** | **0** | **1.0000** | **0.9912 — floor MET** |
| **pooled** | **571** | **0** | **1.0000** | **0.9948** |

**The perfect score reproduces exactly on data the validators have never seen.** It was not
overfitting. And because `val` carries 339 high-severity instances against test's 232, the
pooled bound of **0.9948** clears the 0.99 floor with evidence for the first time.

### False positives: 55, and none of them redact clean text

Raw precision is 0.911 (571 TP, 55 FP). But every one of the 55 overlaps a real gold PII span
— there is **zero over-redaction of non-PII text**. They are label and boundary errors:

| count | confusion | example | gold |
|---|---|---|---|
| 41 | `CREDIT_CARD` ← PHONE | `'91 99854 35346'` | `'+91 99854 35346'` |
| 13 | `PASSWORD` ← USERNAME | `'randy04'` | `'randy04'` (identical span) |
| 1 | `PASSWORD` ← EMAIL | `'adam65@example'` | `'adam65@example.net'` |

Under exact-match scoring each counts as a FP and a FN. Under the redaction lens that matters
for the product, the PII is still covered in all 55 cases — **except one**: truncating
`adam65@example.net` to `adam65@example` leaves `.net` unredacted. That is a genuine partial
leak and the only one in 571 instances.

The dominant failure is a **type-disambiguation weakness, not a detection weakness**: a
`+`-prefixed international phone number reads as a card, and a username reads as a password.
Both are cheap to fix in `_resolve_overlaps` and are the highest-value next change to
`forge/validators.py`.

---

## 4. The teacher p95 is not reproducible

Two runs of the **identical configuration** — same model, endpoint, `reasoning_effort`, 5 rpm
throttle:

| artifact | n | p50 | **p95** | mean | max |
|---|---|---|---|---|---|
| `teacher_token_sample.meta.json` | 60 | 0.507 | **0.790** | 0.546 | 1.367 |
| `predictions_teacher_120b_test.meta.json` | 302 | 0.586 | **8.024** | 2.608 | — |

The p50s agree within 14%; the p95s differ by 10×. Bootstrapping the 60-sample p95 gives
**[0.689, 1.054]** — the long run's 8.024 sits **7.6× above that upper bound**, so it is
categorically not sampling noise. Solving the long run's mean against its median implies
≈15% of its calls took ~14 s.

**A hypothesis published earlier the same day was wrong and is retracted.** `docs/ROADMAP.md`
attributed the tail to client-side rate-limit stalling folded into the timer. Reading
`scripts/run_inference.py` disproves it: `t0` is set *after* the throttle sleep (line 81),
and the retry backoff sits in the `except` branch so every attempt resets it. Only successful
attempts append to `latencies`. Throttle and backoff were already excluded, exactly as the
docstring claimed. The tail is real latency — episodic congestion on shared free-tier capacity
over a ~77-minute run, absent from a ~12-minute one.

A contributing cause: the 302-record run predates the `latencies_s` field, so its percentiles
came from whichever resumed segment ran last rather than the pooled distribution. The per-call
vector was never stored, so it cannot be audited retrospectively.

**Consequence for G4.** The threshold is `teacher_p95 / 5`. If the clean p95 is ~0.8 s the
target is **≤ 0.16 s**, not the ≤ 1.60 s previously assumed — about ten times harder,
requiring ~660 output tok/s single-stream from a 1.5B. *A re-measurement with the pooling fix
in place is in flight; this section is updated when it lands.*

---

## 5. Two data defects

### dev is 79.4% contaminated — and was about to be selected on

**150 of 189 dev records appear verbatim in `data/train.jsonl`.** The data engine was seeded
from dev (`SEEDS ?= data/gold/dev.jsonl`) and `forge/dedup.py` was handed the *test* split to
check leakage against, so `train.meta.json`'s `removed_leakage: 11` counted only what it was
asked to look for.

WP-3 was planned as a capacity sweep "on dev only". It would have selected a LoRA rank by
scoring memorised training text and would have reliably preferred the most overfit
configuration. Fixed by `scripts/build_validation.py` — 533 clean records, disjointness
asserted on the written bytes, 339 high-severity instances against test's 232.

### The frozen test set holds 20 duplicate records

385 records, 365 unique texts; the 20 extras are byte-identical in both text and spans.
Effective n is 365, so intervals are marginally optimistic and duplicated records carry double
weight. Measured impact is immaterial:

| | n=385 | n=365 | Δ |
|---|---|---|---|
| teacher micro-F1 | 0.9482 | 0.9482 | +0.0000 |
| student micro-F1 | 0.5750 | 0.5801 | +0.0050 |
| G1 ratio | 0.6064 | 0.6118 | +0.0054 |

All far inside the ±0.042 interval. **The freeze is kept** — regenerating a test set after
seeing results is exactly what the contract forbids, and the defect changes no conclusion. It
is pinned at exactly 20 by a test and disclosed in the model card.

---

## 6. The important negative

**The frozen test set has zero training leakage: 0 of 385.** Every published test measurement
stands. Had this gone the other way, every number in this repository would have been void.

---

## 7. What this does not fix

`PROTOCOL.md` §5 — human verification — remains unperformed. `audit_gold.py` checks what a
machine can check and its output says so explicitly; it is not a substitute.
`contracts/pii_redaction_v2.yaml:114` still describes the gold set as "a human-verified
subset", which is not yet true. Correcting a frozen contract requires a v3 bump, so it is
recorded as an open item rather than silently edited.

**Method note.** Both data defects were found by reading committed bytes, not by running the
test suite. The ADR 0011 clock bug survived a green suite for weeks because every test
regenerated the data the same wrong way. `make audit` is deliberately separate from
`make test` for that reason.
