"""Evaluation harness — score predicted PII spans against gold-standard spans.

Computes every metric in the TaskContract (contracts/pii_redaction_v1.yaml):
- Micro-averaged F1 over (start, end, label) exact-match spans  [primary]
- Per-type precision / recall / F1
- Partial-overlap F1 (diagnostic)
- Redaction leak rate (fraction of gold PII characters left unmasked)
- Schema-validity rate

This module is pure scoring — no model inference, no I/O.  Plug any model's
predictions in as ``list[PIIRecord]`` and get back a structured report.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import NamedTuple

from forge.schema import HIGH_SEVERITY, PIIRecord, PIISpan, PIIType


class SpanKey(NamedTuple):
    start: int
    end: int
    label: str


@dataclass
class TypeMetrics:
    tp: int = 0
    fp: int = 0
    fn: int = 0

    @property
    def precision(self) -> float:
        return self.tp / (self.tp + self.fp) if (self.tp + self.fp) else 0.0

    @property
    def recall(self) -> float:
        return self.tp / (self.tp + self.fn) if (self.tp + self.fn) else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if (p + r) else 0.0


@dataclass
class EvalReport:
    """Full evaluation report matching the contract's metric specification."""

    micro_f1: float = 0.0
    micro_precision: float = 0.0
    micro_recall: float = 0.0

    per_type: dict[str, TypeMetrics] = field(default_factory=dict)

    partial_overlap_f1: float = 0.0

    leak_rate: float = 0.0

    schema_valid_count: int = 0
    schema_total_count: int = 0

    n_records: int = 0
    n_gold_spans: int = 0
    n_pred_spans: int = 0

    @property
    def schema_validity(self) -> float:
        return self.schema_valid_count / self.schema_total_count if self.schema_total_count else 1.0

    def high_severity_recall(self) -> dict[str, float]:
        return {
            t.value: self.per_type[t.value].recall
            for t in HIGH_SEVERITY
            if t.value in self.per_type
        }

    def format_table(self) -> str:
        lines = [
            f"{'Metric':<30} {'Value':>10}",
            "-" * 42,
            f"{'micro-F1 (primary)':<30} {self.micro_f1:>10.4f}",
            f"{'micro-precision':<30} {self.micro_precision:>10.4f}",
            f"{'micro-recall':<30} {self.micro_recall:>10.4f}",
            f"{'partial-overlap F1':<30} {self.partial_overlap_f1:>10.4f}",
            f"{'redaction leak rate':<30} {self.leak_rate:>10.4f}",
            f"{'schema validity':<30} {self.schema_validity:>10.4f}",
            "",
            f"{'records':<30} {self.n_records:>10}",
            f"{'gold spans':<30} {self.n_gold_spans:>10}",
            f"{'pred spans':<30} {self.n_pred_spans:>10}",
            "",
            f"{'Type':<20} {'P':>8} {'R':>8} {'F1':>8} {'TP':>5} {'FP':>5} {'FN':>5}  {'Sev':>3}",
            "-" * 70,
        ]
        for t in PIIType:
            if t.value not in self.per_type:
                continue
            m = self.per_type[t.value]
            sev = "!!!" if t in HIGH_SEVERITY else ""
            lines.append(
                f"{t.value:<20} {m.precision:>8.4f} {m.recall:>8.4f} {m.f1:>8.4f} "
                f"{m.tp:>5} {m.fp:>5} {m.fn:>5}  {sev}"
            )
        return "\n".join(lines)


def _span_key(s: PIISpan) -> SpanKey:
    return SpanKey(s.start, s.end, s.label.value)


def _spans_overlap(a: PIISpan, b: PIISpan) -> bool:
    return a.start < b.end and b.start < a.end and a.label == b.label


def _overlap_len(a: PIISpan, b: PIISpan) -> int:
    return max(0, min(a.end, b.end) - max(a.start, b.start))


def _compute_leak_rate(gold: list[PIIRecord], preds: list[PIIRecord]) -> float:
    """Fraction of gold PII character positions left unmasked by predictions."""
    total_gold_chars = 0
    leaked_chars = 0

    pred_map = {r.id: r for r in preds}

    for g in gold:
        p = pred_map.get(g.id)
        for gs in g.spans:
            span_len = gs.end - gs.start
            total_gold_chars += span_len
            if p is None:
                leaked_chars += span_len
                continue
            covered = set()
            for ps in p.spans:
                if ps.start < gs.end and gs.start < ps.end:
                    for i in range(max(gs.start, ps.start), min(gs.end, ps.end)):
                        covered.add(i)
            leaked_chars += span_len - len(covered)

    return leaked_chars / total_gold_chars if total_gold_chars else 0.0


def _compute_partial_overlap_f1(gold: list[PIIRecord], preds: list[PIIRecord]) -> float:
    """Character-level, label-aware overlap F1 — diagnostic, not primary.

    Each character position is tagged with its label. Overlap counts only
    when both position AND label match. Uses sets so overlapping spans of the
    same label within one record are naturally deduplicated.
    """
    pred_map = {r.id: r for r in preds}
    total_overlap = 0
    total_gold_chars = 0
    total_pred_chars = 0

    gold_ids = set()
    for g in gold:
        gold_ids.add(g.id)
        gold_positions: set[tuple[int, str]] = set()
        for gs in g.spans:
            for i in range(gs.start, gs.end):
                gold_positions.add((i, gs.label.value))
        total_gold_chars += len(gold_positions)

        p = pred_map.get(g.id)
        if p is None:
            continue

        pred_positions: set[tuple[int, str]] = set()
        for ps in p.spans:
            for i in range(ps.start, ps.end):
                pred_positions.add((i, ps.label.value))
        total_pred_chars += len(pred_positions)
        total_overlap += len(gold_positions & pred_positions)

    for p in preds:
        if p.id not in gold_ids:
            for ps in p.spans:
                total_pred_chars += ps.end - ps.start

    prec = total_overlap / total_pred_chars if total_pred_chars else 0.0
    rec = total_overlap / total_gold_chars if total_gold_chars else 0.0
    return 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0


def evaluate(gold: list[PIIRecord], preds: list[PIIRecord], schema_valid_count: int | None = None) -> EvalReport:
    """Score predictions against gold records.

    Args:
        gold: Gold-standard records with verified spans.
        preds: Model predictions — one PIIRecord per input, matched by ``id``.
        schema_valid_count: How many predictions had valid JSON schema output.
            If None, assumed all valid (e.g. when using constrained decoding).

    Returns:
        Full EvalReport with all contract-defined metrics.
    """
    pred_map = {r.id: r for r in preds}

    micro = TypeMetrics()
    per_type: dict[str, TypeMetrics] = {}

    n_gold_spans = 0
    n_pred_spans = 0

    for g in gold:
        p = pred_map.get(g.id)

        gold_keys = {_span_key(s) for s in g.spans}
        pred_keys = {_span_key(s) for s in p.spans} if p else set()

        n_gold_spans += len(gold_keys)
        n_pred_spans += len(pred_keys)

        tp_keys = gold_keys & pred_keys
        fp_keys = pred_keys - gold_keys
        fn_keys = gold_keys - pred_keys

        micro.tp += len(tp_keys)
        micro.fp += len(fp_keys)
        micro.fn += len(fn_keys)

        for key in tp_keys:
            per_type.setdefault(key.label, TypeMetrics()).tp += 1
        for key in fp_keys:
            per_type.setdefault(key.label, TypeMetrics()).fp += 1
        for key in fn_keys:
            per_type.setdefault(key.label, TypeMetrics()).fn += 1

    schema_total = len(gold)
    if schema_valid_count is None:
        schema_valid_count = schema_total

    return EvalReport(
        micro_f1=micro.f1,
        micro_precision=micro.precision,
        micro_recall=micro.recall,
        per_type=per_type,
        partial_overlap_f1=_compute_partial_overlap_f1(gold, preds),
        leak_rate=_compute_leak_rate(gold, preds),
        schema_valid_count=schema_valid_count,
        schema_total_count=schema_total,
        n_records=len(gold),
        n_gold_spans=n_gold_spans,
        n_pred_spans=n_pred_spans,
    )
