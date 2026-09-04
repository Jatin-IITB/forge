# Quantization gate suite — the quantized artifact re-earns every gate

**Date:** 2026-09-03 · **Decision record:** `docs/adr/0018-serving-stack.md`
**Reproduce:** `scripts/run_eval.py data/gold/test.jsonl <preds> --ci --validators --teacher-preds data/predictions_teacher_120b_relat.jsonl`
**Set:** frozen 385-record test split · **Teacher bar:** 0.9482 micro-F1, parity target 0.9292
**Manifest:** `models/pii-1.5b-gguf/manifest.json` (sha256 per artifact)

WP-1's non-negotiable: *quantization that breaks a gate does not ship.* The exit gate is
**quantized F1 within 0.01 of fp16**, with a pre-committed fallback — "if Q4_K_M costs more
than 0.01 F1, ship Q8_0 and report the larger artifact honestly."

## Verdict: ship **Q8_0**, not Q4_K_M

| artifact | file | micro-F1 | 95% CI | ΔF1 vs f16 GGUF | exit gate (≤0.01) |
|---|---|---|---|---|---|
| f16 GGUF (reference) | 3094 MB | 0.6338 | [0.5940, 0.6721] | — | — |
| **Q8_0 — SHIPPED** | **1647 MB** | **0.6360** | [0.5958, 0.6747] | **+0.0022** | ✅ **pass** |
| Q4_K_M | 986 MB | 0.6187 | [0.5787, 0.6566] | **−0.0151** | ❌ **miss** |

Q4_K_M costs **1.51 F1 points**, over the 1.00-point budget. The rule was written before these
numbers existed and is applied as written. **The honest cost of that decision: the shipped
file is 1647 MB instead of 986 MB — 67% larger — and its resident set at concurrency 32 is
3080 MB instead of 2341 MB.** In exchange it costs nothing in speed (§5).

---

## 1. The comparison that matters is against the f16 **GGUF**, not the fp16 HF baseline

The published fp16 baseline (`predictions_student_run_002`) was produced by a *different
stack*: HuggingFace `transformers` on MPS with the LoRA adapter applied at runtime. The GGUF
artifacts come from `models/pii-1.5b-merged` (adapter merged) converted by llama.cpp. Comparing
Q4_K_M to that baseline changes three things at once — merge, runtime, and quantization — and
attributes the sum to quantization.

Running the **f16 GGUF** through the identical llama.cpp path isolates it:

| step | comparison | ΔF1 | what it actually measures |
|---|---|---|---|
| A | fp16-HF 0.5750 → f16-GGUF 0.6338 | **+0.0588** | adapter merge + runtime change |
| B | f16-GGUF 0.6338 → Q4_K_M 0.6187 | **−0.0151** | **quantization, isolated** |
| B' | f16-GGUF 0.6338 → Q8_0 0.6360 | **+0.0022** | **quantization, isolated** |

**Step A is the larger effect and it is not quantization.** Had the f16 GGUF control been
skipped, Q4_K_M would have been reported as *+0.0437 better than fp16* — a real number, drawn
from a real comparison, and completely wrong about the thing it appears to measure. It would
have shipped a lossy artifact under the banner of a quality improvement.

Chat-template parity was verified before drawing that conclusion: the HF tokenizer and
llama-server agree exactly on prompt length for the first five records (298/301/287/286/342
tokens), so step A is a genuine model-behaviour difference and not a prompting bug.

---

## 2. Full metric table

| metric | fp16-HF | f16-GGUF | **Q8_0** | Q4_K_M |
|---|---|---|---|---|
| micro-F1 | 0.5750 | 0.6338 | **0.6360** | 0.6187 |
| 95% CI | [0.5324, 0.6162] | [0.5940, 0.6721] | **[0.5958, 0.6747]** | [0.5787, 0.6566] |
| micro-precision | 0.6375 | 0.6118 | 0.6171 | 0.6011 |
| micro-recall | 0.5237 | 0.6576 | 0.6561 | 0.6374 |
| partial-overlap F1 | 0.7209 | 0.7498 | 0.7502 | 0.7078 |
| redaction leak rate | 0.2186 | 0.1123 | 0.1144 | 0.1132 |
| predicted spans (gold 695) | 571 | 747 | 739 | 737 |
| **schema validity** | 0.9974 | 0.9974 | **0.9974** | **1.0000** |
| **G1 ratio (paired)** | 0.6098 | 0.6722 | **0.6744** | 0.6561 |
| G1 95% CI | [0.5650, 0.6533] | [0.6305, 0.7122] | [0.6322, 0.7154] | [0.6142, 0.6961] |
| G1 vs 0.98 | ❌ FAIL | ❌ FAIL | ❌ **FAIL** | ❌ FAIL |
| system F1 (+ validators) | 0.7334 | — | **0.7862** | 0.7937 |
| high-severity floors | 9/9 | — | **9/9** | 9/9 |

**G1 fails for every artifact.** Quantization is not the reason and cannot be the fix: the
whole interval sits ~0.30 below the gate for all four. That is WP-2/WP-3's problem, restated
here only so the quantization decision is not mistaken for a parity result.

---

## 3. Per-type F1 — and why most of this table is noise

`!!!` marks the nine high-severity types. Final column is the isolated quantization effect.

| type | fp16-HF | f16-GGUF | Q8_0 | Q4_K_M | Q4_K_M − f16 GGUF |
|---|---|---|---|---|---|
| AADHAAR !!! | 0.1887 | 0.1786 | 0.1786 | 0.1429 | −0.0357 |
| AGE | 0.6667 | 0.7778 | 0.7778 | 0.6667 | −0.1111 |
| API_KEY !!! | 0.5778 | 0.5116 | 0.5455 | 0.5238 | +0.0122 |
| BANK_ACCOUNT !!! | 0.3889 | 0.3889 | 0.3889 | 0.3889 | +0.0000 |
| CREDIT_CARD !!! | 0.5977 | 0.5957 | 0.6250 | 0.7179 | +0.1222 |
| DATE_OF_BIRTH | 0.5070 | 0.6111 | 0.6027 | 0.6316 | +0.0205 |
| DRIVER_LICENSE !!! | 0.0000 | 0.1250 | 0.1250 | 0.0000 | −0.1250 |
| EMAIL | 0.9265 | 0.9130 | 0.9197 | 0.8435 | −0.0695 |
| IP_ADDRESS | 0.7931 | 0.8852 | 0.8667 | 0.9091 | +0.0238 |
| LOCATION | 0.7805 | 0.5143 | 0.5217 | 0.5333 | +0.0190 |
| PAN !!! | 0.6800 | 0.5778 | 0.5333 | 0.4651 | −0.1127 |
| PASSPORT !!! | 0.6400 | 0.7600 | 0.7755 | 0.6667 | −0.0933 |
| PASSWORD !!! | 0.2500 | 0.5143 | 0.4706 | 0.3111 | −0.2032 |
| PERSON | 0.5000 | 0.7095 | 0.7181 | 0.6775 | −0.0320 |
| PHONE | 0.7381 | 0.8095 | 0.8049 | 0.8293 | +0.0197 |
| SSN !!! | 0.6667 | 0.8571 | 0.8571 | 0.8387 | −0.0184 |
| STREET_ADDRESS | 0.0923 | 0.0000 | 0.0000 | 0.0727 | +0.0727 |
| URL | 1.0000 | 0.9756 | 0.9756 | 0.8293 | −0.1463 |
| USERNAME | 0.3881 | 0.0833 | 0.0833 | 0.3607 | +0.2773 |

**Do not read individual rows as findings.** These types carry 15–41 gold instances each, so
one or two spans move an F1 by 0.05–0.15 and the per-type column is mostly sampling noise.
USERNAME swinging +0.2773 *toward* Q4_K_M while PASSWORD swings −0.2032 *away* is the
signature of noise, not of a quantization mechanism that would explain both. The defensible
statement is the aggregate with its interval, and even there the CIs overlap heavily: the
0.0151 gap between Q4_K_M and f16-GGUF is **not statistically significant**.

The rule is still applied on the point estimate, deliberately. It was pre-committed, Q8_0 is
free on every other axis, and re-litigating a threshold after seeing which side the number
landed on is exactly the move this project exists to not make.

---

## 4. One gate points the other way, and it is disclosed

| | Q8_0 | Q4_K_M |
|---|---|---|
| schema validity | 384/385 = 0.9974 | **385/385 = 1.0000** |
| **G2** (≥ 0.999) | ❌ **FAIL** | ✅ **PASS** |

**Neither artifact dominates.** Q4_K_M is the only one of the four that passes G2.

Two things keep this from overturning the decision:

1. **G2 has zero margin at n=385.** 0.999 × 385 = 384.6, so a single malformed response fails
   the gate. 384/385 is not "slightly below" — it is one record.
2. **384/385 is the norm, not Q8_0's defect.** Three of four artifacts — fp16-HF, f16-GGUF,
   Q8_0 — sit at exactly 384/385. Q4_K_M's clean sweep is the outlier in a single sample. With
   continuous batching the same model is only **96.9% span-identical to itself** across
   serving configs (373/385 records; measured Q4_K_M throughput-config vs latency-config), so
   one record's validity is well inside run-to-run drift.

The honest reading is that **G2 is not currently measurable to the resolution its threshold
implies.** Demonstrating ≥ 0.999 needs a test set where one failure is not 0.26% of the sample.
This is the same class of defect ADR 0014 recorded for the high-severity floors.

---

## 5. Quantization bought no speed, so the fallback costs nothing

Full 385 records, throughput config (`-np 32 -c 32768 --mlock`, concurrency 32):

| weights | file | RSS | s/record | output tok/s | $/1k |
|---|---|---|---|---|---|
| f16 GGUF | 3094 MB | 4405 MB | 0.8379 | 61.6 | $0.03248 |
| **Q8_0** | **1647 MB** | **3080 MB** | **0.7218** | **71.2** | **$0.02798** |
| Q4_K_M | 986 MB | 2341 MB | 0.7751 | 68.97 | $0.03004 |

Q8_0 was nominally *fastest*, despite Q4_K_M being the only one granted `--repeat 2`. The
0.72–0.84 s/record spread is smaller than this machine's 1.59× run-to-run variance
(`reports/economics.md` §5), so the three are not distinguishable — but the direction rules out
"Q4_K_M is worth its F1 cost because it is faster." At batch 32 the weights are read once per
step and amortized across 32 sequences, so decode is compute-bound and smaller weights stop
paying. Quantization only pays at batch 1, where `llama-bench` measures Q4_K_M 14.44 tok/s vs
Q8_0 12.18 vs f16 9.99.

**The fallback to Q8_0 therefore costs 661 MB of disk and 739 MB of RAM, and nothing else.**

---

## 6. Gate suite on the shipped artifact (Q8_0)

| Gate | Requirement | Measured | Verdict |
|---|---|---|---|
| **G1** parity | model-only F1 ≥ 0.98 × 0.9482 = 0.9292 | 0.6360 [0.5958, 0.6747]; ratio 0.6744 | ❌ FAIL — structural, ~0.29 below |
| **G2** schema | ≥ 0.999 | 0.9974 (384/385) | ❌ FAIL by one record — see §4 |
| **G3** cost | ≤ $0.01594 /1k | see `reports/economics.md` | ❌ FAIL |
| **G4** p95 | ≤ 0.2728 s | see `reports/economics.md` | ❌ FAIL |
| **G5** deployability | runs on a laptop, quantized | 1647 MB file, 3080 MB RSS, 16 GB M1, fully local | ✅ **PASS** |
| **G6** OOD | per contract | not re-measured this session — see below | ⚠️ not run |
| **high-severity floors** | recall ≥ 0.99 × 9 types | system 9/9 at 1.0000, pooled bound 0.9872 | ⚠️ pass on point estimate; bound short of 0.99 (ADR 0014) |

**Quantization broke nothing.** Every gate that fails for Q8_0 fails identically for the f16
GGUF and for the fp16 baseline, at the same magnitude. The one gate quantization *does* move —
G2, by a single record — moves in Q4_K_M's favour and is disclosed in §4 rather than buried.

**G6 was not re-run.** The OOD probe is a separate harness owned by another work package this
session, and no OOD behaviour change is expected from quantization — but "expected" is not
"measured", so it is marked not-run rather than inherited from the fp16 result.

---

## 7. What this does not establish

- **That Q8_0 is better than Q4_K_M.** The CIs overlap heavily (§3). What is established is
  that Q4_K_M's point estimate misses a pre-committed threshold that Q8_0's clears, and that
  choosing Q8_0 costs nothing measurable except file size.
- **That the merge/runtime gain (+0.0588, step A) is a real quality improvement.** It is
  measured and reproducible, but its mechanism is not diagnosed. It is concentrated in PERSON
  (85 → 256 predicted spans, recall 0.3714 → 0.8343), which was the fp16 model's single
  largest error source at 110 false negatives. Worth a follow-up; not claimed as understood.
- **That the model-only numbers are the product.** They are not — the system number (0.7862)
  includes the ADR 0012 validator layer, and G1 is measured on model-only. Both are reported
  above so neither can be quoted as the other.
