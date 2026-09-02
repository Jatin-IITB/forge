# Forge — Roadmap

*What remains, in dependency order, with exit gates and kill criteria.
Written 2026-09-03 against measured artifacts. Companion to `NORTH_STAR.md` (the claim
ledger) and `ACTION_PLAN.md` (the phase playbook).*

---

## 1. Standing, as measured

Every number below comes from a committed artifact, not an estimate.

| Gate | Definition | Teacher | System / student | Ratio | Threshold | Verdict |
|---|---|---|---|---|---|---|
| **G1** | micro-F1 parity | 0.9482 | **0.5750** (model-only) | 0.607× | ≥ 0.98× | ❌ |
| **G2** | schema validity | 0.7844 | **0.9974** | — | ≥ 0.999 | ❌ *(1 record)* |
| **G3** | $ / 1k records | $0.1592 | **$0.1910** | **1.20×** | ≤ 0.1× | ❌ |
| **G4** | p95 latency | 8.02 s | **8.68 s** | **1.08×** | ≤ 0.2× | ❌ |
| **G5** | deployability | — | fp16 on MPS, unquantized | — | quantized, CPU-capable | ❌ |
| **G6** | OOD / adversarial | — | probe set built, unscored on a passing model | — | per contract | ⏸ |

Alongside those failures, three results are solid and worth protecting:

- **Validator layer: 1.0000 recall on all 9 high-severity types** — the 120B teacher scores
  0.87 mean on the same nine and fails six of them outright (`DRIVER_LICENSE` 0.53,
  `BANK_ACCOUNT` 0.79, `AADHAAR` 0.83). The hybrid system beats the teacher where it
  matters most. This is the project's strongest genuine result.
- **Data engine discipline**: 189 seeds → 161 verified → 150 after dedup = **20.6% filtered**,
  leakage 0.
- **Reproducibility**: gold set reproduces bit-for-bit after the ADR 0011 clock-dependence fix;
  143 tests green including a clock-shift regression guard.

### The finding that shapes everything below

`run_002` underfit (r=16 attention-only, loss flat at 1.17). `run_003` added capacity
(r=64 + MLP, 73.9M params) on *identical data*: loss fell 5× to 0.22, precision rose to 0.86,
and **recall collapsed to 0.39** — span ratio halved to 0.46. Net F1 got slightly worse.

Capacity and data are **jointly binding**. Adding capacity alone moves along a frontier
instead of toward it. And the reason is not subtle:

> All 837 training records are template-generated. 687 by construction, 150 by the **8B
> development teacher that was later replaced for being too weak**. Not one training record
> has been labelled by the 120B teacher. **A project named for distillation has not yet
> distilled from its teacher.**

That is the single largest untouched lever, and WP-1 exists to pull it.

---

## 2. Two open problems, and why they are independent

| | **Parity (G1, G2, G6)** | **Economics (G3, G4, G5)** |
|---|---|---|
| Blocked on | training data diversity | the serving stack |
| Gap | 0.5750 → 0.9292 (**+0.354 absolute**) | 1.20× → 0.10× (**12× throughput**) |
| Difficulty | open-ended research | known engineering |
| Confidence | moderate | high |
| Cost | ~8 days of free-tier tokens + training time | ~2 sessions |

**They share no dependency.** The economics work needs *a* trained adapter, not a *good* one —
quantization and batching speed up a weak model exactly as much as a strong one.

**Therefore: do the economics first.** It converts two hard FAILs to likely PASSes in about
two sessions, and it de-risks the schedule before entering the open-ended parity fight. The
current instinct — "fix quality first, then optimise" — has the risk ordering backwards.

---

## 3. Work packages

### WP-0 — Measurement integrity *(do first; it changes the targets)*

Three numbers currently in the repo are not yet defensible, and two of them make the gates
**easier** than they should be. Fixing them before optimising against them is the only
honest order.

1. **Teacher p95 is not reproducible.** *(Mechanism corrected 2026-09-03 — see below.)*
   Two runs of the **identical configuration** — same model, endpoint, `reasoning_effort`,
   and 5 rpm throttle — disagree by 10× on p95:

   | Artifact | n | p50 | **p95** | mean | max |
   |---|---|---|---|---|---|
   | `teacher_token_sample.meta.json` | 60 | 0.507 | **0.790** | 0.546 | 1.367 |
   | `predictions_teacher_120b_test.meta.json` | 302 | 0.586 | **8.024** | 2.608 | — |

   The p50s agree to within 14%; the p95s differ by an order of magnitude. The 60-record
   sample contains no call slower than 1.37 s. Solving the longer run's mean against its
   median implies **≈15% of its calls took ~14 s** — episodic server-side congestion on
   shared free-tier capacity during a ~77-minute run, absent from the ~12-minute one.

   **The original hypothesis in this document was wrong.** It claimed the tail was
   client-side rate-limit stalling folded into the timer. Reading `run_inference.py`
   disproves that: `t0` is set *after* the throttle sleep and is reset on every retry
   attempt, and only successful attempts append to `latencies`. Throttle and backoff were
   already correctly excluded — the docstring's claim holds. The tail is real latency
   that really happened; it just is not a stable property of the teacher.

   A gate threshold defined as `teacher_p95 / 5` inherits that instability. **If the clean
   p95 is ~0.8 s, G4's target is ≤ 0.16 s — ten times harder than the ≤ 1.60 s this
   document originally assumed.** Re-measurement is running; the verdict follows the number.

   *Contributing cause:* the 302-record run predates the `latencies_s` field, so its
   percentiles were computed from whichever resumed segment ran last rather than the pooled
   distribution (`run_inference.py:310-312` was added to fix exactly this). Its p95 is
   therefore both unstable **and** computed from a truncated slice, and cannot be audited
   after the fact because the per-call vector was never stored.
2. **No confidence intervals.** The contract says "all measured with 95% CIs"; none of the
   published numbers carry one. Add bootstrap CIs (n=10,000) to F1, recall, and per-type
   scores. On 385 records a ±0.05 CI on F1 is expected, which matters for a 0.98× gate.
3. **"Human-verified" is claimed but never earned.** `PROTOCOL.md` §5 defines a verification
   pass that has not been run. Either do a 100-record two-pass audit with logged
   inter-annotator agreement, or strike the word from the ledger, README, and model card.

**Exit gate:** every published number carries a CI and a stated measurement condition; the
ledger's "human-verified" row reads either ✅-with-evidence or ❌-and-removed.
**Artifact:** `reports/measurement_integrity.md`, ADR 0014.

---

### WP-1 — The serving stack *(the economics fix — highest value per hour)*

The entire G3/G4 failure reduces to one number: **4.93 s/record**, unquantized fp16, batch 1,
via `transformers` on MPS. At ~106 output tokens that is ≈21 tok/s, which is far below what
this hardware can do.

**Required:** ≤ **0.411 s/record** sustained (≈258 output tok/s aggregate) to reach G3.

| Lever | Expected gain | Status |
|---|---|---|
| GGUF Q4_K_M via llama.cpp (vs transformers-MPS fp16) | 2–4× | `export_model.py gguf` written, never run |
| Continuous batching, 8–16 concurrent | 4–8× | not started |
| Drop rationale tokens at inference (keep for training) | 1.2–2× | needs an ablation |
| Speculative decoding w/ 0.5B draft | 1.5–2× | optional, only if still short |

First two levers alone give 8–32×. **12× is reachable, but not with either one alone.**

**The tension worth naming:** G3 wants throughput (large batches), G4 wants single-request
latency (batch 1). They pull in opposite directions. The honest report is **both
configurations measured and published side by side** — a throughput config for $/1k and a
latency config for p95 — never the flattering number from each.

**Non-negotiable:** the quantized artifact re-runs the **full** gate suite. Quantization that
breaks a gate does not ship. Expect Q4_K_M to cost 1–3 F1 points; if it costs more, ship Q8_0
and report the larger file.

**Exit gate:** G3 ≤ 0.1× and G4 ≤ 0.2× (against the WP-0-corrected teacher p95), both under a
named, reproducible serving config; quantized F1 within 0.01 of fp16.
**Artifact:** `models/` manifests, `reports/economics.md`, `reports/quantization_gates.md`.

---

### WP-2 — Real distillation: a teacher-labelled data engine

The design follows directly from two measurements.

**First**, given the validator layer already hits **1.000 recall on all nine structured
high-severity types**, the model does not need to learn them. Training capacity spent on
`AADHAAR` or `CREDIT_CARD` is capacity spent re-teaching a solved problem. **The model's job
is the complement of the validators.**

**Second**, the failing types are precisely the ones templates cannot teach:

| Type | Student F1 | Why templates fail it |
|---|---|---|
| `STREET_ADDRESS` | **0.0923** | multi-token, fuzzy boundaries |
| `PERSON` | 0.5000 | 110 FN — high-frequency, multi-entity |
| `LOCATION` | 0.7805 | context-dependent (teacher itself scores 0.41) |

Fuzzy boundaries cannot be learned from a template pool that always places the entity in the
same syntactic slot. So the data engine splits by type:

| Track | Types | Label source | Cost |
|---|---|---|---|
| **A — construction** | 9 structured high-severity + `EMAIL`/`URL`/`IP`/`PHONE` | injection (exact by construction) | free |
| **B — distillation** | `PERSON`, `LOCATION`, `STREET_ADDRESS`, `USERNAME`, `AGE` | **120B teacher**, k=3 self-consistency + verification gate | teacher tokens |

Track B is the genuine distillation, aimed exactly where the student is weak — and it is what
makes the project's name accurate.

**Volume target:** 4,000–5,000 records, ≥40% Track B, ≥300 distinct carrier shapes
(vs. today's template pool).
**Token budget:** ~600 tok/record for carrier generation + ~1,350 for k=3 labelling on Track B.
Against the 1M tok/day free tier that is **~6–8 days** of background generation. The engine
already supports `--resume`, so it spreads across days without babysitting.

**Carrier text licensing is a hard gate.** ADR 0003 and contract v2 already reject
`ai4privacy/pii-masking-200k` (academic-only). Any new public corpus must clear the same bar
**before** generation starts, with the clearance written into the data card. Teacher-generated
carrier text is the fallback that always clears.

**Exit gate:** data card with per-type coverage, carrier-shape diversity count, accept/reject
rates, leakage = 0, and an explicit license clearance line per source.
**Artifact:** `data/train_v3.jsonl`, `reports/data_card_v3.md`, ADR 0015.

---

### WP-3 — Capacity × data sweep

`run_003` proved r=64 overfits at 837 records. At ~4,500 records that conclusion does not
transfer, so the setting must be re-derived rather than assumed.

Sweep r ∈ {16, 32, 64} × MLP {on, off}, **on dev only**. The frozen test set is not touched
until one config is chosen.

**Pre-registered predictions** (recorded before the run, per ADR 0013 discipline — the
conjunction is what makes it falsifiable):

1. r=64+MLP now *outperforms* r=16 on dev, reversing run_003 — because the data:parameter
   ratio has moved from 11:1 to ~60:1.
2. **Span ratio rises above 0.85** (from 0.46/0.84). This is the load-bearing prediction: if
   loss improves without span ratio improving, the problem is the task formulation, not the
   data, and WP-4 becomes mandatory rather than conditional.
3. `STREET_ADDRESS` F1 clears 0.40 (from 0.09) — the direct test of whether Track B data
   teaches fuzzy boundaries.

**Exit gate:** one config selected on dev with the sweep table published including the losers;
a single scored run against the frozen test set.
**Artifact:** `reports/sweep_capacity_v3.md`, ADR 0016.

---

### WP-4 — Recall repair *(conditional — only if WP-3 prediction 2 fails)*

If span ratio stays below 0.85, the model is not enumerating entities and no amount of data
will fix it. Escalation ladder, cheapest first:

1. **Decode audit** — is under-detection a sampling artifact? Test greedy vs. beam vs. lower
   temperature. Costs one evening; occasionally solves it outright.
2. **Loss reweighting** — upweight FN on the fuzzy types.
3. **Two-pass decode** — detect, then re-prompt with detected spans masked to surface the rest.
   Directly targets multi-entity under-enumeration (`PERSON`, 110 FN).
4. **DPO** — preference pairs from teacher-correct vs. student-missed. This is claim 15's
   decision gate; document the call either way.

**Exit gate:** span ratio ≥ 0.9 on dev, or a written finding that the task formulation is the
ceiling.
**Artifact:** ADR 0017.

---

### WP-5 — QLoRA + AWQ *(hardware-blocked — a rented GPU, not a code problem)*

`bitsandbytes` has no MPS backend; AWQ calibration needs CUDA kernels. `export_model.py awq`
already refuses with an explanation rather than silently skipping — keep it that way.

One rented A10/L4 session (~$1–5) closes ledger rows 14 and 16 together: QLoRA run on the
final data, AWQ export, both gate-verified. **Batch this into a single session** with the
final dataset in hand; renting a GPU before WP-2 finishes wastes the run.

**Kill criterion:** if WP-1 already passes G3/G4/G5 via GGUF, AWQ adds nothing but a ledger
tick. Drop rows 14/16 to "not pursued — GGUF met the deployability gate" and say so plainly.
An honest "we didn't need it" beats a rented checkbox.

---

### WP-6 — G6 and the demo

Score the 31-probe OOD/adversarial set against the final artifact (rates reported separately,
never averaged — the harness already enforces this). Then the airplane-mode demo: Wi-Fi off,
GGUF via llama.cpp, a messy ticket in, redacted text out, wall-clock on screen.

**Exit gate:** G6 verdict; demo runs from a clean clone with no network.

---

### WP-7 — Close the ledger

`make forge` end-to-end on a clean clone (never yet run start-to-finish — and the last fresh-
clone bug broke *every* make target, so assume more). Then every ledger row reads ✅ or is
amended to the measured truth, README hero table carries only measured numbers, and
`HONEST_ASSESSMENT.md` separates novel from solid-engineering.

---

## 4. Critical path

```
WP-0 measurement integrity ──┬──► WP-1 serving stack ──────────────► G3 G4 G5
      (targets become real)  │      (2 sessions, high confidence)
                             │
                             └──► WP-2 data engine ──► WP-3 sweep ──► G1 G2
                                    (6–8 days bg)       (2 sessions)   │
                                                                       ▼
                                                         WP-4 recall repair (conditional)
                                                                       │
                                    WP-5 QLoRA/AWQ (optional) ─────────┤
                                                                       ▼
                                                          WP-6 G6 + demo ──► WP-7 ledger
```

WP-2's token generation runs unattended in the background, so **WP-1 executes inside WP-2's
wall-clock**. The schedule is gated by the data engine, not by engineer hours.

**Ordering rule:** WP-0 before WP-1, because optimising against a target you already know is
wrong is wasted work.

---

## 5. Risk register

| Risk | Likelihood | Impact | Mitigation / kill criterion |
|---|---|---|---|
| Corrected teacher p95 makes G4 unreachable | **high — now evidenced** | G4 permanently fails | A clean teacher p95 of ~0.8 s puts the target at **≤ 0.16 s**, requiring ~660 output tok/s single-stream from a 1.5B. Publish both conditions; if the clean target is unreachable, amend the claim to the measured multiple. The covenant already requires this. |
| Gate thresholds defined *relative to* an unstable teacher measurement | **high** | any of G1/G3/G4 silently mis-set | G1/G3/G4 are all `teacher_X × k`. The teacher side needs a CI and a stated measurement condition, or the gates inherit its noise. WP-0 fixes this for latency; check F1 and cost the same way. |
| 12× throughput not reached | medium | G3 fails | Ladder in WP-1 has 4 independent levers; measure after each. If stuck ≥0.3×, report the honest multiple. |
| Q4_K_M costs >0.01 F1 | medium | quantization can't ship | Fall back to Q8_0; report the larger artifact. |
| Parity still short after WP-2/3 | **high** | G1 fails | This is the real research risk. A 0.354 gap is large. Fallback: report the hybrid **system** score (validators + model) as the product number, with model-only reported separately and never conflated. |
| Free-tier quota throttles data gen | medium | schedule slips | `--resume` already spreads across days; size rounds to daily budget. |
| Public corpus fails license review | medium | Track B shrinks | Teacher-generated carrier text always clears ADR 0003. |
| Test-set overfitting via repeated evaluation | **high** | silent invalidation | **Hard rule: dev only during WP-3.** Every frozen-test scoring is logged with a date and a reason in the ledger. |

**The honest headline risk:** G1 may not be reachable on this hardware with this data budget.
The project's defence is that it will have *measured* that precisely, with a documented chain
of falsifiable experiments — which is a better portfolio artifact than an unfalsifiable claim
of success. `SUCCESS.md` already defines this as a legitimate outcome.

---

## 6. Decisions needing a human call

1. **Report model-only or system F1 as the headline?** The system number (0.7334 with
   validators) is the product's real behaviour; the model-only number (0.5750) is the
   distillation result. ADR 0012's three-number rule says publish both. Confirm the headline.
2. **Rent a GPU?** ~$1–5 closes two ledger rows. Only worthwhile if GGUF misses G5.
3. **Stop condition.** If WP-3 lands at, say, F1 0.80 (0.84× teacher), is that a stop-and-write-up
   or another round? Deciding *now*, before seeing the number, is what keeps the gate honest.

---

## 7. Resume-claim status

Tracked here because the covenant in `NORTH_STAR.md` §"The promise" requires it: a claim
changes to the measured number, never the reverse.

| Bullet | Status | Unblocked by |
|---|---|---|
| Distilled GPT-OSS-120B → 1.5B | ✅ true | — |
| 100% recall on 9 highest-risk types | ✅ **true and understated** (measured 1.0000 vs. teacher 0.87) | — |
| ~99.7% schema-valid JSON | ✅ true (384/385) | — |
| Majority-vote + 3-stage dedup, ~21% filtered | ✅ true (20.6%) | — |
| **~80× cheaper inference** | ❌ **measures 1.20× — student is *more* expensive** | **WP-1** |
| **~20× lower p95** | ❌ **measures 1.08×** | **WP-0 + WP-1** |
| **GPU-free on a laptop** | ❌ runs on Apple MPS, unquantized | **WP-1** (GGUF makes CPU inference real) |
| **Grounded in retrieved source text** | ❌ there is no retrieval in this system | *reword*: spans are verified as exact substrings of the source (0/116 predicted spans absent) |
| ≥0.98× teacher parity | ❌ 0.607× | WP-2 + WP-3 |

Four bullets hold today. Three become true if WP-1 succeeds. One needs rewording to describe
what the system actually does — substring verification, not retrieval.
