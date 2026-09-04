#!/usr/bin/env python3
"""Make a training corpus representable in BIOES, or drop what is not.

`scripts/train_token_classifier.py` calls `assert_bioes_round_trip` on every
record and raises on the first failure. That is the right behaviour for a gate
-- a silently dropped gold span trains a model that scores badly for reasons you
would look for in the wrong place -- but it is fail-fast, so it reports one
record when there may be many. On `data/train_v3_clean.jsonl` it reported
`train-v3-00768`; there were five.

The five are two different problems, and only one of them is a labelling error.

## A. Spans padded with whitespace -- FIXED, because the label is wrong

    'Tristan Batta ... noted incident at 12/074, Ganguly Zila, Khora  (near East April)'
                          STREET_ADDRESS = '12/074, Ganguly Zila, Khora '
                                                                      ^ trailing space

Faker's Indian address values end in a space, and the template concatenated one
straight in. Whitespace is not PII, so the span is simply wrong at its right
edge and trimming it loses nothing. `decode_bioes` already strips whitespace
from every span it recovers (`forge/token_classifier.py:285`), so the trimmed
form is the one the decoder canonicalises to -- this aligns the label with the
representation rather than guessing at a boundary.

`data/train_v2.jsonl` carries the same defect (`aug-0398`, 'Kara Road, Khora .')
and survives it only by accident: there the trailing space is followed by '.',
Qwen merges ' .' into a single token, and the STREET_ADDRESS punctuation clip
happens to land back on the original offset. That is the coincidence the comment
at `token_classifier.py:290` records. In v3 the next character is another space,
no merge saves it, and the defect finally surfaces.

## B. A span boundary inside a merged token -- DROPPED, because gold is right

    'The conference agenda (see https://price.com/) includes a session on ...'
                          URL = 'https://price.com/'        gold is CORRECT

Qwen merges '/)' into one token, so no token boundary exists at the end of the
URL and BIOES cannot express the span. The decoder recovers 'https://price.com/)'
-- one character too long. Nothing about the label is wrong; the record is
unrepresentable at this tokenisation, which is exactly the condition the gate
exists to detect.

This is fixable, by adding ')' to the decoder's trailing clip set for URL. It is
deliberately NOT fixed here. `decode_bioes` also runs at inference, so changing
it changes the shipped model's predictions, and the run this unblocks holds every
variable fixed except the corpus in order to attribute the result to the data.
Changing the decoder in the same run would confound exactly that. A balanced-paren
clip (strip ')' only when unmatched, so 'wiki/Foo_(bar)' survives) is the right
follow-up, with the baseline re-scored so the numbers stay comparable.

So: trim what is mislabelled, drop what cannot be represented, count both.

Whole records are dropped rather than the offending span, for the reason
`filter_train_v3.py` gives: a record with one bad span is evidence of careless
construction, and its remaining spans are not independently trustworthy.

The frozen gold set is untouched by any of this -- `data/gold/test.jsonl` and
`data/gold/val.jsonl` both round-trip cleanly at 0 failures, so evaluation
integrity is not in question. This only conditions training input.

    python scripts/normalize_spans.py --in data/train_v3_clean.jsonl \
        --out data/train_v3_aligned.jsonl --max-length 128
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from collections import Counter
from pathlib import Path

from forge.schema import HIGH_SEVERITY, PIIRecord, PIISpan
from forge.token_classifier import assert_bioes_round_trip

BASE_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"


def load_offsets_fn(base_model: str, max_length: int):
    """Return `text -> offsets`, and the backend name, for provenance.

    Prefers `transformers`, which is what training actually uses. Falls back to
    the Rust `tokenizers` library reading the same `tokenizer.json`, so this can
    run on a machine with no torch install -- the eval-only venv on the laptop
    that produced the corpus, for instance. Both paths must produce identical
    offsets; the backend is printed so a disagreement is traceable rather than
    mysterious.
    """
    try:
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained(base_model)

        def offsets_of(text: str) -> list[tuple[int, int]]:
            enc = tok(
                text,
                add_special_tokens=False,
                truncation=True,
                max_length=max_length,
                return_offsets_mapping=True,
            )
            return [tuple(p) for p in enc["offset_mapping"]]

        return offsets_of, "transformers.AutoTokenizer"
    except ImportError:
        pass

    try:
        from tokenizers import Tokenizer
    except ImportError:
        print(
            "needs either `transformers` or `tokenizers`:\n"
            "  pip install -e '.[train]'      (full)\n"
            "  pip install tokenizers          (offsets only, no torch)",
            file=sys.stderr,
        )
        raise SystemExit(2)

    path = _find_tokenizer_json(base_model)
    tok = Tokenizer.from_file(str(path))
    tok.enable_truncation(max_length=max_length)

    def offsets_of(text: str) -> list[tuple[int, int]]:
        return [tuple(p) for p in tok.encode(text, add_special_tokens=False).offsets]

    return offsets_of, f"tokenizers.Tokenizer({path.parent.name[:12]})"


def _find_tokenizer_json(base_model: str) -> Path:
    """Resolve tokenizer.json from the hub, then from the local cache."""
    try:
        from huggingface_hub import hf_hub_download

        return Path(hf_hub_download(base_model, "tokenizer.json"))
    except Exception:  # noqa: BLE001 - offline or no hub; try the cache directly
        pass

    stem = "models--" + base_model.replace("/", "--")
    roots = [
        Path.home() / ".cache" / "huggingface" / "hub",
        Path.home() / "Library" / "Caches" / "huggingface" / "hub",
    ]
    import os

    if os.environ.get("HF_HOME"):
        roots.insert(0, Path(os.environ["HF_HOME"]) / "hub")
    for root in roots:
        found = sorted((root / stem).glob("snapshots/*/tokenizer.json")) if (root / stem).exists() else []
        if found:
            return found[0]
    raise SystemExit(f"could not locate tokenizer.json for {base_model}")


def trim_record(record: PIIRecord) -> tuple[PIIRecord, int]:
    """Strip whitespace from every span edge. Returns the record and n changed."""
    changed = 0
    spans: list[PIISpan] = []
    for span in record.spans:
        start, end = span.start, span.end
        while start < end and record.text[start].isspace():
            start += 1
        while end > start and record.text[end - 1].isspace():
            end -= 1
        if start >= end:  # the span was nothing but whitespace
            changed += 1
            continue
        if (start, end) != (span.start, span.end):
            changed += 1
            spans.append(
                PIISpan(start=start, end=end, label=span.label, text=record.text[start:end])
            )
        else:
            spans.append(span)
    if not changed:
        return record, 0
    return record.model_copy(update={"spans": spans}), changed


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--in", dest="src", type=Path, default=Path("data/train_v3_clean.jsonl"))
    ap.add_argument("--out", type=Path, default=Path("data/train_v3_aligned.jsonl"))
    ap.add_argument("--base-model", default=BASE_MODEL)
    ap.add_argument("--max-length", type=int, default=128)
    ap.add_argument(
        "--check-only", action="store_true",
        help="Report and exit non-zero on any defect; write nothing. Use on the "
             "frozen gold set, which must always be 0/0.",
    )
    args = ap.parse_args()

    if not args.src.exists():
        print(f"missing {args.src}", file=sys.stderr)
        return 1

    records = [
        PIIRecord.model_validate_json(line)
        for line in args.src.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    offsets_of, backend = load_offsets_fn(args.base_model, args.max_length)
    print(f"tokenizer: {args.base_model} via {backend}, max_length={args.max_length}")

    kept: list[PIIRecord] = []
    trimmed_spans = 0
    trimmed_records = 0
    dropped: list[tuple[str, str]] = []
    dropped_by_label: Counter[str] = Counter()

    for record in records:
        fixed, n_trim = trim_record(record)
        if n_trim:
            trimmed_spans += n_trim
            trimmed_records += 1

        offsets = offsets_of(fixed.text)
        if offsets and offsets[-1][1] < len(fixed.text):
            dropped.append((fixed.id, "truncation would discard labelled text"))
            for span in fixed.spans:
                dropped_by_label[span.label.value] += 1
            continue
        try:
            assert_bioes_round_trip(fixed, offsets)
        except ValueError as exc:
            detail = str(exc).split(";", 1)[-1].strip()
            dropped.append((fixed.id, f"unrepresentable in BIOES: {detail[:110]}"))
            for span in fixed.spans:
                dropped_by_label[span.label.value] += 1
            continue
        kept.append(fixed)

    n_in_spans = sum(len(r.spans) for r in records)
    n_out_spans = sum(len(r.spans) for r in kept)

    print(f"\nin   {len(records):>5} records, {n_in_spans:>5} spans")
    print(f"trim {trimmed_spans:>5} spans on {trimmed_records} records (whitespace at a span edge)")
    print(f"drop {len(dropped):>5} records that BIOES cannot represent")
    for rid, why in dropped:
        print(f"       {rid}  {why}")
    if dropped_by_label:
        print("     spans lost with them:")
        for label, n in dropped_by_label.most_common():
            print(f"       {label:<16}{n:>4}")
    print(f"out  {len(kept):>5} records, {n_out_spans:>5} spans   "
          f"({len(kept) / max(len(records), 1) * 100:.2f}% of records kept)")

    if args.check_only:
        ok = not trimmed_spans and not dropped
        print("\nCHECK ONLY: " + ("clean, nothing to fix" if ok else "DEFECTS PRESENT (see above)"))
        return 0 if ok else 1

    args.out.parent.mkdir(parents=True, exist_ok=True)
    # newline="\n" explicitly: Python's text mode rewrites "\n" to "\r\n" on
    # Windows, and this file is generated on the training box while its sha is
    # checked against one computed on a laptop. Without this the two are
    # byte-different for no semantic reason and the hash stops being evidence.
    with args.out.open("w", encoding="utf-8", newline="\n") as handle:
        for record in kept:
            handle.write(record.model_dump_json() + "\n")

    # Re-read the written bytes and re-verify from scratch, rather than trusting
    # the in-memory result. Same discipline as filter_train_v3.py and
    # build_validation.py: what gets trained on is the file, not the variable.
    reread = [
        PIIRecord.model_validate_json(line)
        for line in args.out.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    residual = 0
    for record in reread:
        try:
            assert_bioes_round_trip(record, offsets_of(record.text))
        except ValueError:
            residual += 1
    if residual:
        print(f"\nFATAL: {residual} records still fail the round trip", file=sys.stderr)
        return 1
    if len(reread) != len(kept):
        print(f"\nFATAL: wrote {len(kept)} records, read back {len(reread)}", file=sys.stderr)
        return 1

    hs = {t.value for t in HIGH_SEVERITY}
    per_type = Counter(s.label.value for r in reread for s in r.spans)
    sha = hashlib.sha256(args.out.read_bytes()).hexdigest()

    print("\nsurviving per-type coverage:")
    for label, n in per_type.most_common():
        print(f"  {label:<16}{n:>5}{'  *high-severity' if label in hs else ''}")
    missing = sorted(hs - set(per_type))
    if missing:
        print(f"\n  high-severity types ABSENT from training: {missing}")

    print(f"\nverified on the written bytes: {len(reread)} records, 0 round-trip failures")
    print(f"wrote  {args.out}")
    print(f"sha256 {sha}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
