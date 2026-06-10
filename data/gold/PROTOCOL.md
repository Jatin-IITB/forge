# Gold-set labelling protocol — `pii_redaction_v1`

The gold set is the foundation of every parity claim Forge makes (DESIGN §7.2). A weak
or leaky gold set makes the whole result meaningless, so this protocol is deliberately
strict. It is the highest-leverage, highest-risk artifact in the project.

> **Status:** this directory currently holds only `sample.jsonl` — a tiny *illustrative*
> seed, **NOT** the frozen eval set. The real `dev.jsonl` / `test.jsonl` are produced by
> the synthetic builder below, human-verified, then frozen. `test.jsonl` is untouched
> until Phase 4 gates.

## 1. Record format

One JSON object per line, conforming to `forge.schema.PIIRecord`
(validated by `make validate`):

```json
{"id": "...", "text": "...", "spans": [{"start": 8, "end": 20, "label": "PERSON", "text": "Rajesh Kumar"}], "lang": "en", "source": "synthetic:faker", "split": "test"}
```

- `spans` are character offsets `[start, end)` into `text`; `text[start:end]` **must**
  equal the span's `text`. Spans **must not overlap**.
- The redacted string is *derived* (`PIIRecord.redacted`), never stored — it can never
  silently disagree with the spans.

## 2. Provenance (independence-clean — ADR 0003 / 0005)

**Primary source: self-generated synthetic.** [Faker](https://faker.readthedocs.io/)
(MIT) produces fake PII *values* (names, emails, cards, Aadhaar/PAN-shaped strings,
keys) which are injected at known offsets into carrier sentences drawn from templates
and public-domain text. Because we control insertion, **ground-truth spans are exact by
construction** — no manual offset labelling, no annotation drift on position.

- Fully reproducible: `make gold` regenerates identical data from a fixed seed.
- Fully redistributable: MIT + project-owned text. A stranger can rebuild it.

**Rejected: `ai4privacy/pii-masking-200k`.** Academic-use only; commercial use is gated
behind `licensing@ai4privacy.com`. That conflicts with the "a stranger can clone and
rebuild end-to-end" litmus test. It may be used **only** as an external academic
benchmark for context, **never** as the committed gold/training asset.

**Optional enrichment: public-domain real text** (e.g. the Enron email corpus, a US
FERC public release) for realism — only if a per-document licence check passes and PII
is re-labelled under this protocol.

## 3. Entity taxonomy

The 19 types in `forge.schema.PIIType`. Labelling rules for the ambiguous ones:

- **PERSON** — personal names (incl. partial). Public figures in a generic/news sense
  are *not* labelled (they are not the data subject); a name used as an identifier *is*.
- **LOCATION vs STREET_ADDRESS** — a full mailing address → `STREET_ADDRESS`; a bare
  city/region used as context → `LOCATION`. A generic place mention that does not
  identify a person (e.g. "the weather in Mumbai") is **left unlabelled** (see
  `sample-0005`, a negative example).
- **High-severity types** (`CREDIT_CARD, BANK_ACCOUNT, SSN, AADHAAR, PAN, PASSPORT,
  DRIVER_LICENSE, PASSWORD, API_KEY`) — a single miss is a reportable breach, so the
  contract sets a per-type **recall floor ≥ 0.99** on these. Label generously.

## 4. Splits, dedup, leakage

- **Splits:** `dev` (for the error loop) and `test` (frozen). No `train` records live
  here — training data lives outside `data/gold/` and is git-ignored.
- **Dedup:** near-duplicate detection within the set and between train ↔ {dev,test}.
- **Leakage policy:** any train carrier sentence overlapping a `test` carrier is
  dropped. The data card must report **leakage count = 0**. `test.jsonl` is never used
  for training, prompt-tuning, or model selection.

## 5. Human verification (before freezing)

Synthetic spans are exact by construction, but *realism and label-policy* still need a
human pass:

1. Sample ≥ 100 records across types; a second annotator independently checks
   label correctness + naturalness of the carrier text.
2. Record an inter-annotator agreement (Cohen's κ) spot-check in the data card.
3. Fix systematic issues in the builder (not by hand-editing records), regenerate,
   re-check. Only then `freeze` (`test.jsonl` becomes read-only for the project).

## 6. Size target

300–1,000 verified examples (ACTION_PLAN Phase 0), stratified so every entity type —
especially the high-severity ones — has enough support for a meaningful per-type CI.
