# ADR 0019 — G3 closure attempt: compact output and speculative decoding

**Status:** ACCEPTED (2026-09-03) — G3 remains FAIL  
**Date:** 2026-09-03  
**Depends on:** ADR 0018 (shipped Q8_0 serving stack); commit `f138210`
(KV-cache quantization and exact-spacing hypotheses refuted)  
**Artifacts:** `reports/bench/compact_abc.json`, `reports/bench/compact_abcd.json`,
`reports/bench/FINAL_compact_*.{json,txt}`, `reports/bench/spec_*.{json,txt}`

## Context

G3 requires student cost ≤ teacher cost / 10:

```
teacher cost                    $0.15940 / 1k records
G3 threshold                    $0.01594 / 1k records
machine cost                    $0.139541 / hour
equivalent throughput target    <= 0.4112 machine-seconds / record
```

The shipped Q8_0 stack remained about 2× over the gate after WP-1. Concurrency, context
size, unified KV, update batch, weight quantization, KV quantization, flash-attention
selection, and mlock had already been swept. Commit `f138210` specifically disproved two
hypotheses that are not reopened here:

1. quantizing K/V to q8_0 or q4_0 does not remove the bottleneck; dequantization makes the
   model slower, and q4_0 destroys termination;
2. the permissive grammar's behaviour is not caused by an alternative whitespace
   tokenization; exact spacing produced identical output and metrics.

The remaining levers were therefore shorter output and speculative decoding.

## Measurement rules

- Apple M1, 16 GB unified memory, llama.cpp `9cffdcc`, Metal.
- Shipped Q8_0 target, `-np 32 -c 32768 --mlock`, client concurrency 32.
- Full frozen 385-record test set only; no subset is a gate number.
- Repeated passes, best throughput retained exactly as `bench_serving.py` specifies.
- Fresh baseline in the same session for each comparison.
- Every artifact records load average and swap. `pmd` remained resident and unkillable.
- Every changed-output configuration was scored by `scripts/run_eval.py --ci` with
  validators and teacher predictions.

## Experiment 1 — compact output

The compact contract was:

```json
{"s":[{"l":"PERSON","t":"Jessica Holmes"}]}
```

The text field stays. `parse_response` maps `s/l/t` back to the ordinary record and
reconstructs offsets by locating the exact emitted substring. Offsets were not delegated to
the model.

Four arms isolated the interventions:

1. verbose prompt, unconstrained decode (fresh baseline);
2. verbose prompt, compact grammar;
3. compact-format prompt, unconstrained decode;
4. compact-format prompt, compact grammar.

The compact prompt preserves the original task instructions and changes only the response
field names and example. An earlier exploratory prompt also shortened the instructions; that
confounded format with task wording and is retained separately in `compact_abc.json`, not
used for the decision.

### Same-server A/B result

Best of two full-set passes:

| arm | tok/record | s/record | $/1k | out tok/s | loadavg 1m start → end | P | R | F1 | schema |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| verbose baseline | 51.4 | 0.5291 | $0.02051 | 97.1 | 1.60 → 2.13 | 0.6171 | 0.6561 | 0.6360 | 384/385 |
| compact grammar | 5.1 | 0.1674 | $0.00649 | 30.5 | 1.70 → 1.60 | 0.5000 | 0.0014 | 0.0029 | 385/385 |
| compact prompt | 33.1 | **0.3677** | **$0.01425** | 90.1 | 2.09 → 1.70 | 0.7276 | 0.6187 | 0.6687 | 381/385 |
| prompt + grammar | 32.8 | **0.3987** | **$0.01545** | 82.3 | 3.96 → 2.09 | 0.7163 | 0.6432 | 0.6778 | 385/385 |

Grammar alone made the fine-tuned model choose the shortest legal completion, `{"s":[]}`,
almost universally: two predicted spans over 695 gold spans. It is fast because it does no
task, not because it serves the task efficiently.

Prompt-only crossed G3 but failed G2 at 381/385. Prompt+grammar crossed both G2 and G3 in its
best pass, but the cost margin was only 3.1%, inside known machine variance.

### Reproduction through the standard harness

A separate three-pass full-set run was pre-registered as the definitive check:

| config | pass records/s | best s/record | $/1k | out tok/s | loadavg 1m start → end |
|---|---|---:|---:|---:|---:|
| verbose baseline | 1.728, 1.465, 1.399 | 0.5785 | $0.02243 | 88.8 | 2.66 → 3.90 |
| prompt + grammar | 1.569, 1.542, 1.626 | **0.6149** | **$0.02383** | 53.4 | 1.67 → 9.69 |

Every compact pass missed 0.4112 s/record. The best missed by **1.495×**. The isolated
single pass is therefore not accepted as closure.

### Quality decomposition

The definitive compact predictions scored:

| metric | verbose baseline | compact prompt + grammar | delta |
|---|---:|---:|---:|
| micro-F1 | 0.6360 | **0.6778** [0.6387, 0.7150] | **+0.0418** |
| micro-precision | 0.6171 | **0.7163** | **+0.0993** |
| micro-recall | **0.6561** | 0.6432 | **−0.0129** |
| schema validity | 384/385 | **385/385** | +1 record |
| system high-severity recall | 1.0000 | 1.0000 | 0 |

The F1 gain does not erase the recall loss. A missed entity is a leak; a false positive is
local over-redaction. The product contract has a high-severity recall floor and no precision
floor, so all three components are published.

## Experiment 2 — speculative decoding

The draft is `Qwen/Qwen2.5-0.5B-Instruct`: public, ungated, Apache-2.0, same tokenizer
family, revision `7ae557604adf67be50417f59c2c2f167def9a775`. It was converted locally
through `scripts/export_model.py` to Q8_0:

```
file      models/qwen2.5-0.5b-instruct-gguf/model-Q8_0.gguf
size      531,068,384 bytes
sha256    9803f5ede78984082c3fa5693368a313a87220ff7fc35d1cccb5c5a5bd826c05
```

This satisfies the INDEPENDENCE rule: no private model, data, credential, or code is needed.

Two current llama.cpp draft block sizes were tested with
`--spec-type draft-simple`: 3 and 8. Both were repeated twice.

| config | s/record | $/1k | out tok/s | throughput vs fresh | loadavg 1m start → end |
|---|---:|---:|---:|---:|---:|
| fresh baseline | 0.7868 | $0.03050 | 65.3 | 1.00× | 4.72 → 9.77 |
| draft max 3 | 1.1857 | $0.04596 | 43.4 | **0.664×** | 8.88 → 3.55 |
| draft max 8 | 1.9281 | $0.07474 | 26.6 | **0.408×** | 4.04 → 15.75 |

Server logs show high acceptance on many requests. Acceptance was not the binding failure:
at batch 32, target verification already amortizes target weights across 32 sequences, while
speculation adds a second 531 MB model and larger verification graphs. That overhead exceeds
the saved target steps. Increasing draft length magnifies the loss.

Quality stayed within noise:

| config | F1 | precision | recall | schema |
|---|---:|---:|---:|---:|
| baseline | 0.6360 | 0.6171 | 0.6561 | 384/385 |
| draft max 3 | 0.6360 | 0.6171 | 0.6561 | 384/385 |
| draft max 8 | 0.6350 | 0.6165 | 0.6547 | 384/385 |

All three retain 1.0000 high-severity recall after deterministic validators. Draft-8 deltas
are F1 −0.0010, precision −0.0006, recall −0.0014, consistent with the existing
Metal/batched decode drift.

## Decision

**Keep the shipped serving configuration unchanged and record G3 as FAIL.**

The clean same-session speculative baseline is $0.03050/1k versus a $0.01594/1k gate:
**1.91× over**. Output shortening produced a favourable one-off pass but failed the
pre-committed reproduction; speculation was 0.66× and 0.41× as fast as baseline.

No threshold or cost-model term changes.

## What would be required to close G3

At least one of:

1. retrain the student on compact targets, then re-run the full gate suite and demonstrate
   ≤0.4112 s/record without the observed recall loss;
2. a single-model serving path sustaining about 125 verbose output tok/s aggregate on this
   workload (measured 65.3 in the final fresh baseline); or
3. faster hardware evaluated under the same amortization and energy formula.

A task-trained draft is not the obvious next step: generic-draft acceptance was already
often high, yet second-model overhead dominated at batch 32. The next defensible software
experiment is compact-target retraining, not another draft-length sweep.

## Consequences

- G3 is closed as a measured negative, not passed.
- Q8_0 remains the shipped artifact and `-np 32 -c 32768 --mlock` remains its throughput
  configuration.
- Compact parsing and benchmark support remain so a future compact-target checkpoint can be
  evaluated without changing the measurement machinery after seeing its result.
- The 0.5B draft artifact is reproducible but is not part of the shipped serving stack.
