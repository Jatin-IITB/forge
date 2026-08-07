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
        print(json.dumps(out, indent=2))
    else:
        print()
        print(report.format_table())

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
