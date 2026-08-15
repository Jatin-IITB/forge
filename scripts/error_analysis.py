#!/usr/bin/env python3
"""Phase 4 error analysis — cluster student failures and compute augmentation targets.

Reads gold + predictions + training data, produces a JSON report showing:
- Per-type recall/precision failures
- Training data distribution vs test set needs
- Concrete augmentation targets (how many more examples of each type)
- Record-level failure details for debugging

Usage:
    python scripts/error_analysis.py \
        data/gold/test.jsonl \
        data/predictions_student_run001.jsonl \
        --train-data data/train.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

from forge.eval import EvalReport, _span_key, evaluate
from forge.schema import HIGH_SEVERITY, PIIRecord, PIIType


def load_records(path: Path) -> list[PIIRecord]:
    records = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            records.append(PIIRecord.model_validate_json(line))
    return records


def analyze_failures(
    gold: list[PIIRecord],
    preds: list[PIIRecord],
    report: EvalReport,
) -> dict:
    pred_map = {r.id: r for r in preds}

    fn_by_type: dict[str, list[dict]] = {}
    fp_by_type: dict[str, list[dict]] = {}

    for g in gold:
        p = pred_map.get(g.id)
        gold_keys = {_span_key(s): s for s in g.spans}
        pred_keys = {_span_key(s): s for s in p.spans} if p else {}

        for key in set(gold_keys) - set(pred_keys):
            s = gold_keys[key]
            fn_by_type.setdefault(key.label, []).append({
                "record_id": g.id,
                "text_snippet": s.text,
                "start": s.start,
                "end": s.end,
            })

        for key in set(pred_keys) - set(gold_keys):
            s = pred_keys[key]
            fp_by_type.setdefault(key.label, []).append({
                "record_id": g.id,
                "text_snippet": s.text,
                "start": s.start,
                "end": s.end,
            })

    return {"false_negatives": fn_by_type, "false_positives": fp_by_type}


def compute_augmentation_targets(
    report: EvalReport,
    train_type_counts: Counter,
    test_type_counts: Counter,
    target_recall: float = 0.90,
    min_train_ratio: float = 0.8,
) -> dict[str, dict]:
    targets = {}
    for t in PIIType:
        label = t.value
        test_count = test_type_counts.get(label, 0)
        train_count = train_type_counts.get(label, 0)
        if test_count == 0:
            continue

        m = report.per_type.get(label)
        recall = m.recall if m else 0.0
        precision = m.precision if m else 0.0
        is_hs = t in HIGH_SEVERITY
        recall_target = 0.99 if is_hs else target_recall

        needed_ratio = max(0, min_train_ratio * test_count - train_count)
        recall_gap = max(0, recall_target - recall)

        if recall_gap > 0 or needed_ratio > 0:
            boost = max(int(needed_ratio), int(recall_gap * test_count * 2))
            boost = max(boost, 5)
            targets[label] = {
                "current_train": train_count,
                "test_count": test_count,
                "current_recall": round(recall, 4),
                "recall_target": recall_target,
                "recall_gap": round(recall_gap, 4),
                "current_precision": round(precision, 4),
                "augmentation_count": boost,
                "high_severity": is_hs,
                "priority": "critical" if (is_hs and recall < 0.5) else
                            "high" if recall_gap > 0.3 else "medium",
            }

    return dict(sorted(targets.items(), key=lambda x: -x[1]["recall_gap"]))


def main() -> int:
    ap = argparse.ArgumentParser(description="Phase 4 error analysis.")
    ap.add_argument("gold", type=Path, help="Gold JSONL")
    ap.add_argument("preds", type=Path, help="Predictions JSONL")
    ap.add_argument("--train-data", type=Path, default=None, help="Training JSONL for distribution analysis")
    ap.add_argument("--output", type=Path, default=None, help="Output JSON report")
    ap.add_argument(
        "--inference-meta", type=Path, default=None,
        help="Inference .meta.json (for schema_valid_count)",
    )
    args = ap.parse_args()

    gold = load_records(args.gold)
    preds = load_records(args.preds)

    schema_valid_count = None
    meta_path = args.inference_meta or args.preds.with_suffix(".meta.json")
    if meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        schema_valid_count = meta.get("schema_valid")

    report = evaluate(gold, preds, schema_valid_count=schema_valid_count)

    test_type_counts: Counter = Counter()
    for r in gold:
        for s in r.spans:
            test_type_counts[s.label.value] += 1

    train_type_counts: Counter = Counter()
    if args.train_data and args.train_data.exists():
        train_records = load_records(args.train_data)
        for r in train_records:
            for s in r.spans:
                train_type_counts[s.label.value] += 1

    failures = analyze_failures(gold, preds, report)
    targets = compute_augmentation_targets(report, train_type_counts, test_type_counts)

    # Multi-span analysis
    multi_span_fn = 0
    for g in gold:
        p = {r.id: r for r in preds}.get(g.id)
        if p and len(g.spans) > 1 and len(p.spans) < len(g.spans):
            multi_span_fn += 1

    result = {
        "summary": {
            "micro_f1": round(report.micro_f1, 4),
            "micro_precision": round(report.micro_precision, 4),
            "micro_recall": round(report.micro_recall, 4),
            "schema_validity": round(report.schema_validity, 4),
            "multi_span_under_predict": multi_span_fn,
        },
        "augmentation_targets": targets,
        "failure_counts": {
            label: len(items) for label, items in failures["false_negatives"].items()
        },
        "fp_counts": {
            label: len(items) for label, items in failures["false_positives"].items()
        },
    }

    print(json.dumps(result, indent=2))

    if args.output:
        args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        print(f"\nwrote report -> {args.output}", file=sys.stderr)

    print(f"\n{'='*60}", file=sys.stderr)
    print("  AUGMENTATION TARGETS (Phase 4)", file=sys.stderr)
    print(f"{'='*60}", file=sys.stderr)
    print(f"  {'Type':<20} {'Need':>5} {'Recall':>8} {'Target':>8} {'Priority':<10}", file=sys.stderr)
    print(f"  {'-'*55}", file=sys.stderr)
    total_aug = 0
    for label, t in targets.items():
        total_aug += t["augmentation_count"]
        print(
            f"  {label:<20} {t['augmentation_count']:>5} "
            f"{t['current_recall']:>8.2%} {t['recall_target']:>8.2%} "
            f"{t['priority']:<10}",
            file=sys.stderr,
        )
    print(f"\n  Total augmentation records needed: ~{total_aug}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    sys.exit(main())
