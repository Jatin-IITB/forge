#!/usr/bin/env python3
"""Build `data/train_v3.jsonl` — the two-track data engine (ADR 0015, stage 2 of 2).

Reads carrier shapes from `scripts/generate_carriers.py`, fills them with Faker
values so spans are exact by construction, then splits by type:

  Track A - CONSTRUCTION.  Labels come from the fill. Free.
  Track B - DISTILLATION.  The 120B teacher labels the filled text k=3 times;
            majority vote (forge/verify.py) plus a construction anchor
            (forge/carriers.py) decide whether the record is kept.

The construction anchor is the part that does not exist in `run_data_engine.py`,
and it exists because of a measurement. On the frozen test set the teacher's
per-type exact recall is 0.994 for PERSON and 0.844 for STREET_ADDRESS but only
**0.273 for LOCATION** (16 of 22 missed). A teacher false negative that lands in
*training* data teaches the student to skip that entity — and under-enumeration
is exactly the student's diagnosed failure (span ratio 0.46-0.84, `PERSON` 110
FN). So a Track B record is kept only when the teacher's consensus accounts for
every injected model-owned entity, at the boundary `data/gold/PROTOCOL.md` §3
specifies.

Resumability: teacher results are appended to a cache keyed by record text, so a
killed job re-reads instead of re-paying. Assembly from the cache is offline and
idempotent, so the data card can be rebuilt without an API call
(`--assemble-only`).

    scripts/nohup_run.sh logs/train_v3.log .venv/bin/python -u \
        scripts/build_train_v3.py --api-key-env CEREBRAS_API_KEY --resume
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from build_gold import PIIValueGenerator

from forge.carriers import (
    TRACK_B_FOCUS,
    VALIDATOR_OWNED,
    Carrier,
    CarrierError,
    anchor_against_construction,
    fill,
    known_given_names,
    merge_anchor,
    shape_of,
    validate_shape,
)
from forge.dedup import dedup_training_data
from forge.inference import build_messages, parse_response
from forge.schema import PIIRecord, PIISpan, PIIType
from forge.teacher_client import ThrottledTeacher
from forge.verify import verify_record

# Not 42 (frozen gold), not 1337 (ADR 0009 augmentation), not 4242 (val split).
# A mix-up should be visible rather than silent.
DATA_ENGINE_SEED = 7717

EVAL_SPLITS = [
    Path("data/gold/dev.jsonl"),
    Path("data/gold/val.jsonl"),
    Path("data/gold/test.jsonl"),
]
PRIOR_TRAIN = [Path("data/train.jsonl"), Path("data/train_v2.jsonl")]


# ---------------------------------------------------------------------------
# IO helpers
# ---------------------------------------------------------------------------
def load_records(path: Path) -> list[PIIRecord]:
    if not path.exists():
        return []
    return [
        PIIRecord.model_validate_json(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def read_cache(path: Path) -> tuple[dict[str, dict], int]:
    """Read the teacher cache, tolerating a torn final line.

    `--assemble-only` is meant to be usable *while* a labelling run is still
    appending — the corpus takes days at 5 rpm, and being unable to look at it
    until it finishes would mean not looking at it. A half-written final line is
    the expected state during a concurrent read, not a corruption.
    """
    if not path.exists():
        return {}, 0
    cached: dict[str, dict] = {}
    torn = 0
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            d = json.loads(line)
        except json.JSONDecodeError:
            torn += 1
            continue
        cached[d["text"]] = d
    return cached, torn


def load_carriers(path: Path) -> tuple[list[Carrier], Counter]:
    """Load carriers, re-validating each one on the way in.

    The file is trusted by nobody: it was written by a language model and may have
    been produced by an older, looser version of the screen. A carrier containing
    literal PII does not fail loudly downstream — it yields a record with a real
    entity and no span for it, which trains the model that the entity is not PII.
    Cheaper to re-check 456 strings here than to find it in a data card later.
    """
    out: list[Carrier] = []
    seen: set[str] = set()
    dropped: Counter = Counter()
    names = known_given_names()
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        d = json.loads(line)
        try:
            validate_shape(d["shape"], known_names=names)
        except CarrierError as exc:
            dropped[str(exc).split("(")[0].strip()] += 1
            continue
        c = Carrier(shape=d["shape"], source=d.get("source", "unknown"),
                    register=d.get("register", "unspecified"))
        if c.normalised() in seen:
            dropped["duplicate shape"] += 1
            continue
        seen.add(c.normalised())
        out.append(c)
    return out, dropped


def _spans_json(spans: list[PIISpan]) -> list[dict]:
    return [{"start": s.start, "end": s.end, "label": s.label.value, "text": s.text} for s in spans]


def _spans_from_json(raw: list[dict]) -> list[PIISpan]:
    return [PIISpan(start=d["start"], end=d["end"], label=PIIType(d["label"]), text=d["text"])
            for d in raw]


# ---------------------------------------------------------------------------
# Instantiation
# ---------------------------------------------------------------------------
def stable_bucket(shape: str, n: int) -> int:
    """Deterministic bucket from the shape text.

    Track assignment must not shift when the carrier file grows between resumed
    runs, so it is derived from content rather than from list position.
    """
    h = hashlib.sha256(shape.encode("utf-8")).digest()
    return int.from_bytes(h[:4], "big") % n


def instantiate(
    carriers: list[Carrier],
    total: int,
    track_b_fraction: float,
    seed: int,
    overplan: float = 1.0,
) -> list[tuple[PIIRecord, str]]:
    """Fill every carrier `repeats` times; label each instance A or B.

    Values come from the same `PIIValueGenerator` the frozen gold set uses, under a
    different seed. Holding the value distribution fixed is deliberate: the carrier
    text is the variable this ADR changes, and changing both at once would repeat
    run_002's attribution failure.

    `overplan` compensates for the Track B accept rate: Track A records are free and
    always survive, Track B records are not, so planning both at the target ratio
    guarantees undershooting it. The alternative — discarding good Track A records
    afterwards to make the ratio look right — would be gaming a number.
    """
    gen = PIIValueGenerator(seed)
    rng = random.Random(seed + 1)

    eligible = [c for c in carriers if c.track_b_eligible]
    repeats = max(1, round(total / max(1, len(carriers))))

    out: list[tuple[PIIRecord, str]] = []
    seq = 0
    # Fraction of an eligible carrier's instances that go to Track B, chosen so the
    # corpus-level Track B share lands on target given how many carriers qualify.
    b_share = 0.0
    if eligible:
        b_share = min(1.0, overplan * track_b_fraction * len(carriers) / len(eligible))

    for c in carriers:
        # A carrier with no placeholders fills to the same string every time, so
        # further instances are exact duplicates the dedup pass would drop anyway.
        n_reps = 1 if c.is_negative else repeats
        n_b = round(n_reps * b_share) if c.track_b_eligible else 0
        for i in range(n_reps):
            seq += 1
            track = "B" if i < n_b else "A"
            rec = fill(
                c,
                record_id=f"train-v3-{seq:05d}",
                value_for=gen.gen,
                split="train",
                source=f"{c.source}|carrier",
            )
            out.append((rec, track))
    rng.shuffle(out)
    return out


# ---------------------------------------------------------------------------
# Track B: teacher labelling
# ---------------------------------------------------------------------------
def query_teacher(teacher, text, model, max_tokens, temperature, timeout, reasoning_effort,
                  teacher_mode=False):
    messages = build_messages(text, teacher_mode=teacher_mode)
    resp, _ = teacher.complete(
        model=model, messages=messages, max_tokens=max_tokens,
        temperature=temperature, timeout=timeout,
        extra_body={"reasoning_effort": reasoning_effort} if reasoning_effort else None,
    )
    raw = resp.choices[0].message.content or ""
    rec, valid = parse_response("tmp", text, raw, split="train")
    return rec, valid


def label_track_b(
    pending: list[PIIRecord],
    cache_path: Path,
    args,
) -> None:
    """Query the teacher for uncached records, appending each result immediately."""
    from openai import OpenAI

    key = os.environ.get(args.api_key_env)
    if not key:
        print(f"Environment variable {args.api_key_env} is not set.", file=sys.stderr)
        sys.exit(1)

    teacher = ThrottledTeacher(
        # max_retries=0: ThrottledTeacher is the only retrier, so the SDK cannot
        # silently spend requests against an exhausted quota before we see it.
        OpenAI(base_url=args.base_url, api_key=key, max_retries=0), rpm=args.rpm,
        on_retry=lambda a, e, d: print(f"    {type(e).__name__}, retry {a} in {d:.0f}s",
                                       flush=True),
    )
    cache = cache_path.open("a", encoding="utf-8")
    t_start = time.monotonic()
    budget_hit = False

    for i, rec in enumerate(pending, 1):
        samples: list[PIIRecord] = []
        flags: list[bool] = []
        errors = 0
        for _ in range(args.k):
            st = teacher.stats
            if st.calls_ok >= args.max_api_calls or st.total_tokens >= args.token_budget:
                budget_hit = True
                break
            try:
                s, valid = query_teacher(
                    teacher, rec.text, args.model, args.max_tokens,
                    args.temperature, args.timeout, args.reasoning_effort,
                    teacher_mode=args.teacher_mode,
                )
                samples.append(s)
                flags.append(valid)
            except Exception as exc:  # noqa: BLE001
                errors += 1
                print(f"    gave up on a sample: {type(exc).__name__}", flush=True)

        # A record labelled with fewer than k samples has not passed the gate the
        # ADR specifies, so it is left out of the cache entirely rather than
        # written with weaker evidence than everything around it.
        if len(samples) < args.k:
            reason = "budget/rate limit" if budget_hit else "teacher errors"
            print(f"  stopping at {i - 1}/{len(pending)} ({reason}): "
                  f"calls={teacher.stats.calls_ok} tokens={teacher.stats.total_tokens}",
                  flush=True)
            break

        result = verify_record(
            record_id=rec.id, text=rec.text, samples=samples,
            schema_valid_flags=flags,
            consistency_threshold=args.consistency_threshold,
            min_agreement=args.min_agreement, split="train",
        )
        cache.write(json.dumps({
            "text": rec.text,
            "teacher_spans": _spans_json(result.record.spans),
            "gate_accepted": result.accepted,
            "agreement": round(result.agreement_ratio, 4),
            "reject_reasons": [r.value for r in result.reject_reasons],
            "api_errors": errors,
        }) + "\n")
        cache.flush()

        st = teacher.stats
        rate = st.calls_ok / max(1e-9, (time.monotonic() - t_start) / 60)
        print(f"  [{i}/{len(pending)}] {rec.id}: "
              f"{'gate-ok' if result.accepted else 'gate-reject'} "
              f"({len(result.record.spans)} spans, agr={result.agreement_ratio:.2f}) "
              f"calls={st.calls_ok} tok={st.total_tokens} rpm={rate:.1f}", flush=True)

    cache.close()
    print(f"\nteacher stage: {json.dumps(teacher.stats.summary())}", flush=True)
    return teacher.stats.summary()


# ---------------------------------------------------------------------------
# Assembly
# ---------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--carriers", type=Path, default=Path("data/carriers_v3.jsonl"))
    ap.add_argument("--output", type=Path, default=Path("data/train_v3.jsonl"))
    ap.add_argument("--cache", type=Path, default=Path("data/train_v3.teacher_cache.jsonl"))
    ap.add_argument("--card", type=Path, default=Path("reports/data_card_v3.md"))
    ap.add_argument("--total", type=int, default=4500)
    ap.add_argument("--track-b-fraction", type=float, default=0.40,
                    help="Target Track B share of the FINAL corpus")
    ap.add_argument("--track-b-overplan", type=float, default=1.3,
                    help="Plan this multiple of the target, since Track B records "
                         "can be rejected by the gate and Track A records cannot")
    ap.add_argument("--seed", type=int, default=DATA_ENGINE_SEED)

    ap.add_argument("--model", default="gpt-oss-120b")
    ap.add_argument("--base-url", default="https://api.cerebras.ai/v1")
    ap.add_argument("--api-key-env", default="CEREBRAS_API_KEY", metavar="VAR")
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--max-tokens", type=int, default=1024)
    ap.add_argument("--timeout", type=float, default=90.0)
    ap.add_argument("--reasoning-effort", default="low", choices=["low", "medium", "high"])
    ap.add_argument("--teacher-mode", action="store_true",
                    help="Use forge.inference.TEACHER_SYSTEM_PROMPT (thorough, emits "
                         "rationales) instead of the plain prompt. OFF by default: the "
                         "committed teacher baseline this ADR's design is derived from was "
                         "measured with the plain prompt (scripts/run_inference.py calls "
                         "build_messages without teacher_mode), and generating from a "
                         "different distribution than the one measured is how run_002 "
                         "became unattributable. See scripts/probe_teacher_prompt.py.")
    ap.add_argument("--rpm", type=float, default=5.0)
    ap.add_argument("--max-api-calls", type=int, default=10_000)
    ap.add_argument("--token-budget", type=int, default=900_000,
                    help="Stop before the 1M/day free-tier ceiling")

    ap.add_argument("--consistency-threshold", type=float, default=0.5)
    ap.add_argument("--min-agreement", type=float, default=0.6)
    ap.add_argument("--dedup-threshold", type=float, default=0.85)
    ap.add_argument("--assemble-only", action="store_true",
                    help="Skip the teacher entirely; build from the cache as it stands")
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    if not args.carriers.exists():
        print(f"no carrier file at {args.carriers} — run scripts/generate_carriers.py first",
              file=sys.stderr)
        return 2

    carriers, carrier_drops = load_carriers(args.carriers)
    if not carriers:
        print(f"no usable carriers in {args.carriers}", file=sys.stderr)
        return 2
    planned = instantiate(carriers, args.total, args.track_b_fraction, args.seed,
                          overplan=args.track_b_overplan)
    track_a = [r for r, t in planned if t == "A"]
    track_b = [r for r, t in planned if t == "B"]
    print(f"carriers: {len(carriers)} distinct shapes "
          f"({sum(1 for c in carriers if c.track_b_eligible)} Track-B eligible)")
    if carrier_drops:
        print(f"  dropped at load: {dict(carrier_drops)}")
    print(f"planned: {len(planned)} records = {len(track_a)} Track A + {len(track_b)} Track B")

    if not args.resume and args.cache.exists() and not args.assemble_only:
        print(f"refusing: {args.cache} exists and --resume was not given", file=sys.stderr)
        return 2

    cached, torn = read_cache(args.cache)
    if args.cache.exists():
        print(f"teacher cache: {len(cached)} records already labelled"
              + (f" ({torn} unreadable line(s) skipped)" if torn else ""))

    if not args.assemble_only:
        pending = [r for r in track_b if r.text not in cached]
        print(f"teacher work remaining: {len(pending)} records x k={args.k} "
              f"= {len(pending) * args.k} calls "
              f"(~{len(pending) * args.k / max(args.rpm, 1e-9) / 60:.1f} h at {args.rpm} rpm)\n")
        if pending:
            label_track_b(pending, args.cache, args)
            cached, _ = read_cache(args.cache)

    # --- Track B assembly: verification gate, then the construction anchor -----
    accepted_b: list[PIIRecord] = []
    b_reject: Counter[str] = Counter()
    anchor_stats = Counter()
    extra_by_type: Counter[str] = Counter()
    boundary_by_type: Counter[str] = Counter()
    missing_by_type: Counter[str] = Counter()
    repaired_by_type: Counter[str] = Counter()
    b_attempted = 0

    for rec in track_b:
        entry = cached.get(rec.text)
        if entry is None:
            continue
        b_attempted += 1
        if not entry["gate_accepted"]:
            for r in entry["reject_reasons"] or ["unspecified"]:
                b_reject[r] += 1
            continue
        teacher_spans = _spans_from_json(entry["teacher_spans"])
        anchor = anchor_against_construction(list(rec.spans), teacher_spans)
        for s in anchor.missing:
            missing_by_type[s.label.value] += 1
        for inj, _ in anchor.boundary:
            boundary_by_type[inj.label.value] += 1
        for s in anchor.repaired:
            repaired_by_type[s.label.value] += 1
        if not anchor.ok:
            b_reject[anchor.reason.split(":")[0]] += 1
            anchor_stats["rejected"] += 1
            continue
        anchor_stats["accepted"] += 1
        for s in anchor.extra:
            extra_by_type[s.label.value] += 1
        merged = merge_anchor(anchor, list(rec.spans))
        accepted_b.append(rec.model_copy(update={
            "spans": merged,
            "source": rec.source + "|track_b:teacher_verified",
        }))

    accepted_a = [
        r.model_copy(update={"source": r.source + "|track_a:construction"})
        for r in track_a
    ]

    # --- Dedup against every split, not just the one we remembered to pass -----
    gold_all: list[PIIRecord] = []
    for p in EVAL_SPLITS + PRIOR_TRAIN:
        gold_all.extend(load_records(p))

    combined = accepted_a + accepted_b
    combined.sort(key=lambda r: r.id)
    dedup = dedup_training_data(combined, gold=gold_all, near_threshold=args.dedup_threshold)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        for rec in dedup.kept:
            f.write(rec.model_dump_json() + "\n")

    # --- Disjointness on the WRITTEN BYTES, naming every split ----------------
    written = {r.text for r in load_records(args.output)}
    leak_report: dict[str, int] = {}
    fatal = False
    for p in EVAL_SPLITS + PRIOR_TRAIN:
        overlap = written & {r.text for r in load_records(p)}
        leak_report[str(p)] = len(overlap)
        if overlap and p in EVAL_SPLITS:
            fatal = True
    print("\n--- leakage check (on written bytes) ---")
    for name, n in leak_report.items():
        print(f"  {name:<28} {n}")
    if fatal:
        print("FATAL: written training records overlap an evaluation split", file=sys.stderr)
        return 1

    write_card(args, carriers, dedup, accepted_a, accepted_b, track_b, b_attempted,
               b_reject, anchor_stats, extra_by_type, boundary_by_type,
               missing_by_type, repaired_by_type, leak_report, cached)
    print(f"\nwrote {len(dedup.kept)} records -> {args.output}")
    print(f"data card -> {args.card}")
    return 0


def write_card(args, carriers, dedup, accepted_a, accepted_b, track_b, b_attempted,
               b_reject, anchor_stats, extra_by_type, boundary_by_type,
               missing_by_type, repaired_by_type, leak_report, cached) -> None:
    kept = dedup.kept
    per_type: Counter[str] = Counter()
    per_type_a: Counter[str] = Counter()
    per_type_b: Counter[str] = Counter()
    for r in kept:
        tgt = per_type_b if "track_b" in r.source else per_type_a
        for s in r.spans:
            per_type[s.label.value] += 1
            tgt[s.label.value] += 1

    kept_b = [r for r in kept if "track_b" in r.source]
    kept_a = [r for r in kept if "track_b" not in r.source]
    shapes = {shape_of(r) for r in kept}
    n_spans = sum(len(r.spans) for r in kept)
    hi_sev = sum(per_type[t.value] for t in VALIDATOR_OWNED)

    carrier_meta = {}
    cm = args.carriers.with_suffix(".meta.json")
    if cm.exists():
        carrier_meta = json.loads(cm.read_text(encoding="utf-8"))

    # Comparison baselines are recomputed from the committed files rather than
    # quoted from a report, so the card cannot drift away from the data.
    prior = {}
    for label, path in (("train_v2", Path("data/train_v2.jsonl")),
                        ("test", Path("data/gold/test.jsonl"))):
        rs = load_records(path)
        pt: Counter[str] = Counter()
        for r in rs:
            for s in r.spans:
                pt[s.label.value] += 1
        tot = sum(pt.values())
        prior[label] = {
            "per_type": pt,
            "shapes": len({shape_of(r) for r in rs}),
            "hi_sev_share": sum(pt[t.value] for t in VALIDATOR_OWNED) / max(1, tot),
        }

    L = []
    A = L.append
    A("# Data card — `data/train_v3.jsonl`\n")
    A("*Generated by `scripts/build_train_v3.py`. Every number below is counted from the "
      "written file, not from an in-memory intermediate.*\n")
    A("**Contract:** `contracts/pii_redaction_v2.yaml` · **Decision record:** "
      "`docs/adr/0015-teacher-labelled-data-engine.md`\n")

    # Completion state first. A reader who stops after one paragraph should not
    # come away thinking this is a finished corpus if it is not.
    b_share = len(kept_b) / max(1, len(kept))
    b_done = len(cached)
    b_planned = len(track_b)
    if b_share >= 0.40:
        A("\n> **Status: complete.** Track B labelling reached its target share; the "
          "numbers below describe a finished corpus.\n")
    else:
        A(f"\n> **Status: PARTIAL — do not train on this file expecting a distilled "
          f"corpus.** Track B is **{b_share:.1%}** of records against a ≥40% target: "
          f"{b_done} of {b_planned} planned Track B records have been labelled "
          f"({b_done / max(1, b_planned):.1%}). Track A is complete and is exact by "
          f"construction, so what exists is a large, carrier-diverse *construction* "
          f"corpus with a distilled fraction still accumulating.\n>\n"
          f"> The limit is **requests, not tokens**. The teacher tier answers an "
          f"exhausted hourly allowance with `retry-after: 3600`, and at k=3 this "
          f"engine spends three requests per Track B record — "
          f"{b_planned * 3} for the full set. Re-run `make train-v3` to continue; it "
          f"resumes from the cache and costs nothing for work already done, and "
          f"`make train-v3-card` regenerates this file at any point mid-run.\n")

    A("\n## 1. Size and composition\n")
    A("| | records | spans | spans/record |")
    A("|---|---|---|---|")
    for name, rs in (("Track A — construction", kept_a), ("Track B — distillation", kept_b),
                     ("**total**", kept)):
        s = sum(len(r.spans) for r in rs)
        A(f"| {name} | {len(rs)} | {s} | {s / len(rs):.2f} |" if rs
          else f"| {name} | 0 | 0 | — |")
    A("")
    A(f"- Track B share: **{len(kept_b) / len(kept):.1%}** of records "
      f"({len(kept_b)}/{len(kept)}); target was ≥40%.")
    A(f"- Distinct carrier shapes in the written file: **{len(shapes)}**; target was ≥300.")
    A(f"  For comparison: `data/gold/test.jsonl` uses {prior['test']['shapes']}, "
      f"`data/train_v2.jsonl` {prior['train_v2']['shapes']}.")
    A(f"- Records per shape: {len(kept) / max(1, len(shapes)):.1f}.")
    A(f"- High-severity (validator-owned) spans: {hi_sev}/{n_spans} = "
      f"**{hi_sev / max(1, n_spans):.1%}** of the corpus. In `train_v2.jsonl` the same "
      f"nine types were {prior['train_v2']['hi_sev_share']:.1%}. "
      f"They are down-weighted, not removed: `forge/validators.py` "
      f"already detects them at 1.0000 recall and 1.0000 precision (ADR 0012), but "
      f"removing them from the label "
      f"set would change what G1 measures, which is a contract decision this ADR does not make.")

    A("\n## 2. Per-type coverage\n")
    A("| type | owner | total | Track A | Track B | train_v2 | test |")
    A("|---|---|---|---|---|---|---|")
    v2 = prior["train_v2"]["per_type"]
    te = prior["test"]["per_type"]
    for t in sorted(PIIType, key=lambda x: -per_type[x.value]):
        owner = "validator" if t in VALIDATOR_OWNED else ("model ★" if t in TRACK_B_FOCUS else "model")
        A(f"| `{t.value}` | {owner} | {per_type[t.value]} | {per_type_a[t.value]} | "
          f"{per_type_b[t.value]} | {v2[t.value]} | {te[t.value]} |")
    A("\n★ = Track B focus type: model-owned and measurably failing.")

    A("\n## 3. Accept / reject, with reasons\n")
    A("### Track A — construction\n")
    A(f"Labels are exact by construction (offsets accumulated during the fill, "
      f"`forge/carriers.fill`), so there is no accept/reject decision to make: "
      f"{len(accepted_a)} instantiated, {len(kept_a)} survive dedup.\n")
    A("### Track B — distillation\n")
    A(f"{len(track_b)} records planned; {b_attempted} received k={args.k} teacher samples "
      f"({len(cached)} in cache).\n")
    total_b = max(1, b_attempted)
    A("| stage | outcome | n | rate |")
    A("|---|---|---|---|")
    A(f"| self-consistency + schema (`forge/verify.py`) | accepted | "
      f"{b_attempted - sum(v for k, v in b_reject.items() if not k.startswith('anchor'))} | "
      f"{(b_attempted - sum(v for k, v in b_reject.items() if not k.startswith('anchor'))) / total_b:.1%} |")
    for reason, n in sorted(b_reject.items(), key=lambda kv: -kv[1]):
        A(f"| {'construction anchor' if reason.startswith('anchor') else 'verification gate'} "
          f"| rejected — `{reason}` | {n} | {n / total_b:.1%} |")
    A(f"| **final** | **accepted** | **{anchor_stats['accepted']}** | "
      f"**{anchor_stats['accepted'] / total_b:.1%}** |")

    A("\n**What the construction anchor caught.** The teacher never sees the injected "
      "labels; these are disagreements between its independent k=3 consensus and ground "
      "truth we already hold.\n")
    A("| type | teacher missed it entirely | teacher chose a different boundary |")
    A("|---|---|---|")
    for t in sorted(set(missing_by_type) | set(boundary_by_type),
                    key=lambda x: -(missing_by_type[x] + boundary_by_type[x])):
        A(f"| `{t}` | {missing_by_type[t]} | {boundary_by_type[t]} |")
    if not (missing_by_type or boundary_by_type):
        A("| — | 0 | 0 |")
    A(f"\nValidator-owned spans repaired from construction rather than rejected "
      f"(ADR 0012 — the deterministic layer is the authority on these): "
      f"{sum(repaired_by_type.values())} spans, {dict(repaired_by_type)}.\n")
    A(f"Entities the teacher found in its own prose that were **not** injected — the "
      f"genuinely distilled signal, resting on the k=3 consensus alone: "
      f"{sum(extra_by_type.values())} spans, {dict(extra_by_type)}.")

    A("\n## 4. Deduplication and leakage\n")
    A(f"- removed exact duplicates: {dedup.removed_exact}")
    A(f"- removed near-duplicates (Jaccard ≥ {args.dedup_threshold} on char 5-grams): "
      f"{dedup.removed_near}")
    A(f"- removed for leakage: {dedup.removed_leakage}")
    A("\n**Leakage = 0, checked against every split by name.** `data/gold/dev.jsonl` is "
      "79.4% contaminated by `data/train.jsonl` because the previous engine was seeded "
      "from dev and `forge/dedup.py` was handed only the *test* split to check against "
      "(ADR 0014 finding 3). The check below is run on the bytes of the written file, "
      "after it is saved.\n")
    A("| split checked | overlapping records |")
    A("|---|---|")
    for name, n in leak_report.items():
        A(f"| `{name}` | **{n}** |")
    A("\nNo carrier text is seeded from any evaluation split: shapes are teacher-written "
      "and values are freshly generated, so disjointness is structural rather than "
      "filtered-for. Generated shapes colliding with an eval-split shape are dropped at "
      "generation time — stricter than the contract's carrier-sentence rule.")

    A("\n## 5. Licence clearance, per source\n")
    A("| source | licence | ADR 0003 litmus | cleared before generation |")
    A("|---|---|---|---|")
    A(f"| Carrier text — `{carrier_meta.get('model', args.model)}` (open-weight, Apache-2.0) "
      "| Apache-2.0; output is project-owned synthetic text | **passes** — a stranger can "
      "regenerate from any host serving the same open checkpoint, or self-host with vLLM "
      "| yes |")
    A("| PII values — Faker | MIT | **passes** — pip-installable, no account | yes |")
    A("| Labels — same open-weight teacher | Apache-2.0, distillation permitted | "
      "**passes** | yes |")
    A("| `ai4privacy/pii-masking-200k` | academic-use only | **fails** | "
      "**not used** — rejected by contract v2 and ADR 0003 |")
    A("\nNo public corpus was ingested. Every byte in this file is generated by code and "
      "models in this repository, which is why the clearance table has no open questions.")

    A("\n## 6. Reproduction\n")
    A("```bash")
    A("python scripts/generate_carriers.py --api-key-env CEREBRAS_API_KEY \\")
    A(f"    --model {args.model} --target {carrier_meta.get('target', '400')} --rpm {args.rpm}")
    A("python scripts/build_train_v3.py --api-key-env CEREBRAS_API_KEY \\")
    A(f"    --total {args.total} --track-b-fraction {args.track_b_fraction} --resume")
    A("```")
    A(f"\nSeeds: carriers {carrier_meta.get('seed', '—')}, fill/instantiation {args.seed}. "
      f"Teacher: `{args.model}` at temperature {args.temperature}, k={args.k}, "
      f"reasoning effort `{args.reasoning_effort}`.")
    tok = carrier_meta.get("total_tokens", 0)
    A(f"\nTeacher tokens: {tok} for carrier generation "
      f"({carrier_meta.get('api_calls', 0)} calls), plus k={args.k} labelling calls for "
      f"{len(cached)} Track B records.")

    args.card.parent.mkdir(parents=True, exist_ok=True)
    args.card.write_text("\n".join(L) + "\n", encoding="utf-8")

    meta = {
        "records": len(kept),
        "spans": n_spans,
        "track_a": len(kept_a),
        "track_b": len(kept_b),
        "track_b_fraction": round(len(kept_b) / max(1, len(kept)), 4),
        "distinct_carrier_shapes": len(shapes),
        "per_type": dict(per_type),
        "dedup": dedup.summary(),
        "leakage_by_split": leak_report,
        "track_b_attempted": b_attempted,
        "track_b_accepted": anchor_stats["accepted"],
        "track_b_reject_reasons": dict(b_reject),
        "anchor_missing_by_type": dict(missing_by_type),
        "anchor_boundary_by_type": dict(boundary_by_type),
        "anchor_repaired_by_type": dict(repaired_by_type),
        "teacher_extra_by_type": dict(extra_by_type),
        "seed": args.seed,
        "model": args.model,
        "k": args.k,
    }
    args.output.with_suffix(".meta.json").write_text(
        json.dumps(meta, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    sys.exit(main())
