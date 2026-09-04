# Claim closure plan

*Written 2026-09-04. Purpose: for each resume claim, either make the repo produce the
number, or replace the claim with what the repo measures. The covenant in
`NORTH_STAR.md` allows exactly those two moves and nothing else.*

## Triage

| # | Claim | Measured today | Verdict |
|---|---|---|---|
| 1 | Distilled GPT-OSS-120B into a 1.5B | teacher scored the bar; **180 of 1774** v3 records are 120B-labelled | 🟡 **closable** |
| 2 | ~99% recall on 9 highest-risk types | **1.0000**, 571 instances, 95% LB **0.9948** | ✅ **already true** |
| 3 | ~80× cheaper inference | **5.3×** cheaper ($0.03004 vs $0.15940) | ❌ **dead — reword** |
| 4 | ~99.7% valid JSON | 0.9974 → **1.0000** under GBNF | ✅ **already true** |
| 5 | Grounded in retrieved source text | **no retrieval exists** | ⚠️ **reword or build** |
| 6 | GPU-free on a laptop | runs on Metal; CPU-only **never measured** | 🟡 **one experiment away** |
| 7 | ≥0.98× teacher parity *(gate G1)* | **0.6744** | 🔴 **the real fight** |

Two are already true. Two are cheap. One is dead. Two are work.

---

## Why 80× is dead, stated once so it is not relitigated

The gate is 10×, not 80×, and even 10× is currently missed by 1.89×. Reaching **80×**
requires ~15× more throughput than measured. Every serving lever has been swept and
recorded:

| lever | result |
|---|---|
| `-np` 1→64 | knee at 32 |
| `-c`, `-ub`, `--mlock`, `--kv-unified` | swept; best kept |
| weight quant f16/Q8_0/Q4_K_M @ batch 32 | spread inside run-to-run variance |
| KV-cache quant q8_0 / q4_0 | **0.79× / 0.15× — slower** |
| flash attention explicit | already on via `auto` |
| speculative decoding, 0.5B draft, k=3/8 | **0.66× / 0.41× — slower** |
| compact output (prompt + grammar) | 2.03× → **1.495×**, still FAIL |

The remaining idea — retraining on compact targets — is worth doing for the gate, but it
cannot produce another 15×. **Claim 3 becomes the measured multiple.** The defensible
sentence is *"~5× cheaper per record than the 120B teacher priced at paid API rates, running
fully local."*

---

## Track A — make claim 1 literally true *(delegate: long-running)*

**State.** `data/train_v3.jsonl` holds 1774 records: 1594 Track A construction, **180 Track B
labelled by gpt-oss-120b with k=3 self-consistency.** The engine stalled below its 4500 target.

**Blocking defect.** `audit_semantics` finds **20/145 AGE and 88/325 DATE_OF_BIRTH spans in
train_v3 sit in contexts that contradict the label** — "Order #1234 confirmed on 25/02/1971"
tagged DATE_OF_BIRTH. Only 8 of 325 DOB spans carry a birth anchor. Training on this teaches
that any date is a birth date, in a corpus whose whole point is a precision/recall frontier.

**Work, in order:**
1. Filter train_v3 to zero contradicted labels. Assert it with `audit_semantics` in a test.
2. Fix the generator so a DOB slot requires a birth anchor in the carrier (whitelist, not
   blacklist — 231 of the bad spans have no cue at all, so a blacklist misses them).
3. Resume generation to ≥4000 records, ≥40% Track B.
4. Publish `reports/data_card_v3.md` with per-type coverage, accept/reject, leakage 0 naming
   all four splits.

**Exit gate.** ≥1600 Track B records, contradicted == 0, leakage == 0 vs train/dev/val/test.

---

## Track B — parity *(delegate, blocked on A)*

**Gap.** 0.6360 → 0.9292 is +0.29 absolute. This is the hard one and may not close.

**Pre-registered predictions** — record before running, report against them either way:
1. r=64+MLP beats r=16 on `val` at ~4000 records, reversing run_003 (data:param ratio moves
   from 11:1 to ~54:1).
2. **Span ratio rises above 0.85** from 0.46/0.84. *Load-bearing*: loss improving without span
   ratio improving means the task formulation is the ceiling, not the data.
3. STREET_ADDRESS F1 clears 0.40 from 0.09 — the direct test of whether Track B teaches fuzzy
   boundaries.

**Select on `data/gold/val.jsonl` only.** Dev is 79% contaminated. Touch the frozen test set
once, after the config is chosen.

**Kill criterion.** If prediction 2 fails, stop adding data and escalate to the recall-repair
ladder in ROADMAP WP-4. More data will not fix an enumeration failure.

---

## Track C — claim 6, one experiment *(me, ~1 hour)*

`llama-server -ngl 0` runs pure CPU. Measure the shipped Q8_0 on the full 385 at `-ngl 0`
versus `-ngl 99`, same harness, same session, contention recorded.

- If CPU-only p95 is usable, **claim 6 becomes true as written** and is a stronger claim than
  the GPU one, because it removes the accelerator from the deployment story entirely.
- If not, the honest wording is *"runs fully offline on a laptop GPU, no datacentre."*

Cheap, decisive, and nobody has run it.

---

## Track D — claim 5, decide then act *(me)*

Two honest paths. **Not both.**

**D1 — reword (30 minutes).** *"Verified every predicted span as an exact substring of the
source and constrained decoding to a JSON grammar — 100% schema-valid, zero hallucinated
spans."* Already true and already measured.

**D2 — build real retrieval (2–3 days).** Retrieve k nearest labelled examples from the
training corpus by embedding similarity, inject as few-shot context. This is genuine RAG and
would plausibly help exactly the types that fail — PERSON, LOCATION, STREET_ADDRESS — because
their labels are context-dependent rather than shape-determined.

**D2 only ships if it earns its place on a measured number.** It costs prompt tokens, which
hurts G3 directly. Gate: **+0.03 micro-F1 on `val`, or it is reverted and reported as a
negative.** Do not keep it for the vocabulary.

---

## Track E — adversarial audit *(me, continuous)*

Every delegated result gets checked before it enters the ledger. What I check, in order of how
often it has caught something in this project:

1. **Was the number measured on the shipped artifact, or on a proxy?** G6 had never been
   scored on the Q8_0 that ships.
2. **Best-of-N quoted as the result?** A favourable single pass at 3.1% margin against ±30%
   machine variance is not a pass. Caught once already.
3. **Same record subset on both sides?** A per-record cost from 48 records is not comparable
   to one from 385.
4. **Fresh baseline in the same session?** This machine drifted 8% in hours; a stale baseline
   invents or hides small effects.
5. **Does the aggregate hide a defect another rule is masking?** The grammar's email-truncation
   bug read as 0 FP at corpus level because the single-claimant rule happened to cover it.
6. **Is the threshold measurable at this n?** Three gates here had thresholds their own test
   set could not resolve.
7. **Precision and recall published, not just F1?** They are not comparable errors for a
   redactor, and F1 hides which one moved.
8. **Contamination:** any new split asserted disjoint from train/dev/val/test *on the written
   bytes*.

A finding is only "confirmed" when I have reproduced it from the committed artifacts myself.

---

## Sequencing

```
now ─┬─ Track C  CPU-only benchmark            (me, ~1h)      → claim 6
     ├─ Track D1 reword grounding              (me, 30m)      → claim 5
     └─ Track A  filter + resume v3 engine     (delegate, days) → claim 1
                        │
                        ▼
                  Track B  capacity sweep on val  (delegate)   → claim 7
                        │
                  Track E audit at every boundary (me)
```

Claims 2 and 4 need no work. Claim 3 gets reworded today.

## Definition of done

Every resume bullet maps to a committed artifact, and any bullet that cannot is deleted rather
than softened. The ledger in `NORTH_STAR.md` is the single source of truth; this document is
retired when its rows are all ✅ or amended.
