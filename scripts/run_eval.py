#!/usr/bin/env python3
"""Evaluate model predictions against the gold set.

Reads a gold JSONL and a predictions JSONL (same format: one PIIRecord per
line, matched by ``id``), scores them with forge.eval, and prints the full
report table.  Optionally checks the measurable contract gates (G1 quality,
G2 schema, G6 high-severity recall) and exits non-zero if any fail.

G3 (cost), G4 (latency), and G5 (deployability) require runtime measurements
and are checked separately during serving benchmarks.

Usage:
    python scripts/run_eval.py data/gold/test.jsonl predictions.jsonl
    python scripts/run_eval.py data/gold/test.jsonl predictions.jsonl \\
        --check-gates --teacher-f1 0.92
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from forge.contracts import load_contract
from forge.eval import evaluate
from forge.schema import PIIRecord
from forge.validators import find_high_severity, merge_with_model


def load_records(path: Path) -> list[PIIRecord]:
    records = []
    for i, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = line.strip()
        if not line:
            continue
        try:
            records.append(PIIRecord.model_validate_json(line))
        except (ValueError, json.JSONDecodeError) as e:
            print(f"[FAIL] {path}:{i}: {e}", file=sys.stderr)
            sys.exit(1)
    return records


def _print_ci_block(gold, preds, system_report, args) -> None:
    """95% bootstrap intervals for the gate quantities (contract v2 requires them).

    Two things here are deliberate and easy to get wrong:

    * The G1 ratio is bootstrapped **paired** — one resample scores both models,
      because both are measured on the same frozen records. Dividing two
      independent intervals reports ~28% more uncertainty than the data holds.
    * A high-severity type with zero misses gets the Clopper-Pearson bound, not
      the bootstrap. With nothing to resample the bootstrap returns [1.0, 1.0],
      which reads as certainty when it only means the sample had no errors.
    """
    from forge.ci import micro_f1_ci, paired_ratio_ci, zero_failure_recall_bound

    n = args.ci_resamples
    print()
    print("=" * 62)
    print(f"  95% BOOTSTRAP CONFIDENCE INTERVALS  (n={n:,} resamples, record-level)")
    print("=" * 62)

    f1 = micro_f1_ci(gold, preds, n_resamples=n)
    print(f"  {'model-only micro-F1':<26}{f1}")

    if args.teacher_preds:
        teacher = load_records(args.teacher_preds)
        ratio = paired_ratio_ci(gold, preds, teacher, n_resamples=n)
        print(f"  {'G1 ratio (paired)':<26}{ratio}")
        verdict = "PASS" if ratio.lo >= 0.98 else "FAIL"
        qual = "" if (ratio.hi < 0.98 or ratio.lo >= 0.98) else "  (interval straddles the gate)"
        print(f"  {'G1 vs 0.98 threshold':<26}{verdict}{qual}")

    if system_report is not None:
        print()
        print("  High-severity recall — zero-miss types use the exact binomial")
        print("  lower bound; the bootstrap is degenerate when nothing was missed.")
        print(f"  {'type':<18}{'n':>5}{'recall':>9}{'95% lower bound':>18}")
        print(f"  {'-' * 50}")
        tot_n = tot_miss = 0
        for label in sorted(system_report.high_severity_recall()):
            m = system_report.per_type.get(label)
            if m is None:
                continue
            n_gold, miss = m.tp + m.fn, m.fn
            tot_n += n_gold
            tot_miss += miss
            lb = f"{zero_failure_recall_bound(n_gold):>18.4f}" if miss == 0 else f"{'(has misses)':>18}"
            print(f"  {label:<18}{n_gold:>5}{m.recall:>9.4f}{lb}")
        if tot_miss == 0 and tot_n:
            print(f"  {'-' * 50}")
            print(f"  {'POOLED':<18}{tot_n:>5}{1.0:>9.4f}{zero_failure_recall_bound(tot_n):>18.4f}")
            print()
            print(f"  Publishable claim: >={zero_failure_recall_bound(tot_n) * 100:.1f}% recall "
                  f"on the {len(system_report.high_severity_recall())} high-severity types.")
            print("  The point estimate is 1.0000; the sample supports the bound.")


def check_gates(report, contract, teacher_f1: float | None) -> list[str]:
    """Check measurable gates. Returns list of failure descriptions."""
    failures = []
    g = contract.gates

    if teacher_f1 is not None:
        threshold = g.parity_target * teacher_f1
        if report.micro_f1 < threshold:
            failures.append(
                f"G1 quality: micro-F1 {report.micro_f1:.4f} < "
                f"{g.parity_target} * teacher_F1 {teacher_f1:.4f} = {threshold:.4f}"
            )
    else:
        print("  [WARN] --teacher-f1 not provided; G1 parity gate skipped")

    if report.schema_validity < g.schema_validity_min:
        failures.append(
            f"G2 schema: validity {report.schema_validity:.4f} < {g.schema_validity_min}"
        )

    hs = report.high_severity_recall()
    for floor in g.high_severity_recall_floors:
        if floor.label not in hs:
            if report.n_gold_spans > 0:
                print(f"  [WARN] {floor.label} has 0 gold instances; recall floor skipped")
            continue
        actual = hs[floor.label]
        if actual < floor.min_recall:
            failures.append(
                f"G6 recall floor: {floor.label} recall {actual:.4f} < {floor.min_recall}"
            )

    return failures


def main() -> int:
    ap = argparse.ArgumentParser(description="Evaluate predictions against gold set.")
    ap.add_argument("gold", type=Path, help="Path to gold JSONL (e.g. data/gold/test.jsonl)")
    ap.add_argument("preds", type=Path, help="Path to predictions JSONL")
    ap.add_argument("--contract", type=Path, default=Path("contracts/pii_redaction_v1.yaml"))
    ap.add_argument(
        "--check-gates", action="store_true",
        help="Check measurable contract gates (G1, G2, G6) and exit non-zero on failure",
    )
    ap.add_argument(
        "--teacher-f1", type=float, default=None,
        help="Teacher micro-F1 on the same gold set (required for G1 parity gate)",
    )
    ap.add_argument("--json", action="store_true", help="Output report as JSON instead of table")
    ap.add_argument(
        "--inference-meta", type=Path, default=None,
        help="Path to inference .meta.json (provides schema_valid_count for G2 gate)",
    )
    ap.add_argument(
        "--validators", action="store_true",
        help=(
            "Apply the deterministic high-severity validators (ADR 0012) and report "
            "model-only, validator-only and system numbers together"
        ),
    )
    ap.add_argument(
        "--retrieval", type=Path, default=None, metavar="TRAIN_JSONL",
        help=(
            "Add retrieval-proposed spans from a carrier index built over TRAIN_JSONL "
            "(ADR 0020). Fills gaps only — the model wins every conflict. Adds no "
            "prompt tokens; retrieval runs beside the model, not in its context"
        ),
    )
    ap.add_argument(
        "--ci", action="store_true",
        help=(
            "Report 95%% bootstrap confidence intervals (contract v2 requires them). "
            "Resamples records, not spans; the G1 ratio uses a paired resample"
        ),
    )
    ap.add_argument(
        "--teacher-preds", type=Path, default=None,
        help=(
            "Teacher predictions on the same gold set. With --ci this enables the "
            "paired ratio CI for G1, which is tighter and more correct than "
            "dividing two independent intervals"
        ),
    )
    ap.add_argument(
        "--ci-resamples", type=int, default=10_000,
        help="Bootstrap resamples (default 10000)",
    )
    args = ap.parse_args()

    gold = load_records(args.gold)
    preds = load_records(args.preds)

    gold_ids = {r.id for r in gold}
    extra_preds = [r for r in preds if r.id not in gold_ids]
    if extra_preds:
        print(f"[WARN] {len(extra_preds)} prediction records have no matching gold record", file=sys.stderr)

    schema_valid_count = None
    if args.inference_meta:
        meta = json.loads(args.inference_meta.read_text(encoding="utf-8"))
        schema_valid_count = meta["schema_valid"]
    elif not args.inference_meta:
        auto_meta = args.preds.with_suffix(".meta.json")
        if auto_meta.exists():
            meta = json.loads(auto_meta.read_text(encoding="utf-8"))
            schema_valid_count = meta["schema_valid"]

    print(f"loaded {len(gold)} gold records, {len(preds)} predictions")

    report = evaluate(gold, preds, schema_valid_count=schema_valid_count)

    # ADR 0020 ships retrieval as a recall aid despite missing its own F1 gate,
    # so every layer is reported separately. A single "system" number would let
    # a rules-and-retrieval result read as a distillation result, which is the
    # conflation ADR 0012's three-number rule exists to prevent.
    retrieval_report = None
    if args.retrieval:
        from forge.retrieval import SpanRetriever
        from forge.retrieval import merge_with_model as merge_retrieved

        index = SpanRetriever(load_records(args.retrieval))
        print(f"retrieval index: {len(index)} carriers from {args.retrieval}")
        retrieved = [
            r.model_copy(update={"spans": merge_retrieved(r.spans, index.propose(r.text))})
            for r in preds
        ]
        retrieval_report = evaluate(gold, retrieved, schema_valid_count=schema_valid_count)
        preds = retrieved

    # ADR 0012 requires all three numbers to be published together. Reporting the
    # system score alone would let a rules-driven result read as a distillation
    # result, which is exactly the conflation the gate discipline exists to stop.
    system_report = None
    if args.validators:
        merged = [
            r.model_copy(
                update={
                    "spans": merge_with_model(r.spans, find_high_severity(r.text)),
                }
            )
            for r in preds
        ]
        system_report = evaluate(gold, merged, schema_valid_count=schema_valid_count)

    if args.json:
        out = {
            "micro_f1": report.micro_f1,
            "micro_precision": report.micro_precision,
            "micro_recall": report.micro_recall,
            "partial_overlap_f1": report.partial_overlap_f1,
            "leak_rate": report.leak_rate,
            "schema_validity": report.schema_validity,
            "n_records": report.n_records,
            "n_gold_spans": report.n_gold_spans,
            "n_pred_spans": report.n_pred_spans,
            "high_severity_recall": report.high_severity_recall(),
            "per_type": {
                k: {"p": v.precision, "r": v.recall, "f1": v.f1, "tp": v.tp, "fp": v.fp, "fn": v.fn}
                for k, v in report.per_type.items()
            },
        }
        if system_report is not None:
            out["system"] = {
                "micro_f1": system_report.micro_f1,
                "micro_precision": system_report.micro_precision,
                "micro_recall": system_report.micro_recall,
                "high_severity_recall": system_report.high_severity_recall(),
                "per_type": {
                    k: {"p": v.precision, "r": v.recall, "f1": v.f1}
                    for k, v in system_report.per_type.items()
                },
            }
        print(json.dumps(out, indent=2))
    else:
        print()
        print(report.format_table())
        if retrieval_report is not None:
            print()
            print("=" * 62)
            print("  MODEL-ONLY vs + RETRIEVAL  (ADR 0020)")
            print("  Shipped as a recall aid; it MISSED its own +0.03 F1 gate.")
            print("  Both numbers are published so the miss stays visible.")
            print("=" * 62)
            print(f"  {'':<20}{'model-only':>13}{'+retrieval':>13}{'delta':>10}")
            print(f"  {'-' * 54}")
            for name, a, b in (
                ("micro-F1", report.micro_f1, retrieval_report.micro_f1),
                ("micro-precision", report.micro_precision, retrieval_report.micro_precision),
                ("micro-recall", report.micro_recall, retrieval_report.micro_recall),
            ):
                sign = "+" if b - a >= 0 else ""
                print(f"  {name:<20}{a:>13.4f}{b:>13.4f}{sign}{b - a:>9.4f}")
            print()
            print("  Retrieval adds no prompt tokens, so serving cost is unchanged.")
        if system_report is not None:
            print()
            print("=" * 62)
            print("  MODEL-ONLY vs SYSTEM (model + deterministic validators)")
            print("  ADR 0012 — both are always reported; neither stands alone.")
            print("=" * 62)
            print(f"  {'':<24}{'model-only':>14}{'system':>12}")
            print(f"  {'-' * 50}")
            model_hs = report.high_severity_recall()
            sys_hs = system_report.high_severity_recall()
            rows = [
                ("micro-F1", report.micro_f1, system_report.micro_f1),
                ("micro-precision", report.micro_precision, system_report.micro_precision),
                ("micro-recall", report.micro_recall, system_report.micro_recall),
                (
                    "min high-sev recall",
                    min(model_hs.values(), default=0.0),
                    min(sys_hs.values(), default=0.0),
                ),
            ]
            for name, model_v, sys_v in rows:
                delta = sys_v - model_v
                sign = "+" if delta >= 0 else ""
                print(f"  {name:<24}{model_v:>14.4f}{sys_v:>12.4f}   ({sign}{delta:.4f})")

            print()
            print(f"  {'high-severity type':<24}{'model-only':>14}{'system':>12}")
            print(f"  {'-' * 50}")
            below = 0
            for label in sorted(sys_hs):
                m, s = model_hs.get(label, 0.0), sys_hs[label]
                flag = "" if s >= 0.99 else "  <- below 0.99 floor"
                if s < 0.99:
                    below += 1
                print(f"  {label:<24}{m:>14.4f}{s:>12.4f}{flag}")
            print()
            print(f"  {len(sys_hs) - below}/{len(sys_hs)} high-severity floors pass under the system.")
            print("  G1 parity is measured on model-only; the validator layer")
            print("  carries the high-severity floor.")

        if args.ci:
            _print_ci_block(gold, preds, system_report, args)

    if args.check_gates:
        contract = load_contract(args.contract)
        failures = check_gates(report, contract, args.teacher_f1)
        if failures:
            print(f"\n{'=' * 42}")
            print(f"  GATE CHECK: {len(failures)} FAILED")
            print(f"{'=' * 42}")
            for f in failures:
                print(f"  [FAIL] {f}")
            return 1
        else:
            print("\n[OK] All measurable gates pass.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
