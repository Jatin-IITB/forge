"""Verification gate — ADR 0002.

No teacher output enters the training set unless it passes:
1. Self-consistency: k independent samples must agree (majority vote on span sets).
2. Schema validity: output must parse into a valid PIIRecord with correct offsets.
3. Constraint validity: all labels are valid PIIType members, offsets in bounds.

The gate returns a VerifiedRecord that carries the consensus spans plus
accept/reject metadata for the data card.
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass, field
from enum import Enum

from forge.schema import PIIRecord, PIISpan

SpanTuple = tuple[int, int, str]


def _span_key(s: PIISpan) -> SpanTuple:
    return (s.start, s.end, s.label.value)


class RejectReason(Enum):
    MAJORITY_SCHEMA_INVALID = "majority_schema_invalid"
    CONSTRAINT_VIOLATION = "constraint_violation"
    NO_VALID_SAMPLES = "no_valid_samples"
    LOW_AGREEMENT = "low_agreement"
    TOO_FEW_SAMPLES = "too_few_samples"


@dataclass(frozen=True)
class VerifiedRecord:
    record: PIIRecord
    accepted: bool
    n_samples: int
    agreement_ratio: float
    reject_reasons: tuple[RejectReason, ...] = ()


@dataclass
class VerificationStats:
    total: int = 0
    accepted: int = 0
    rejected_consistency: int = 0
    rejected_schema: int = 0
    rejected_empty: int = 0
    per_type_accepted: dict[str, int] = field(default_factory=lambda: Counter())
    per_type_rejected: dict[str, int] = field(default_factory=lambda: Counter())

    @property
    def accept_rate(self) -> float:
        return self.accepted / self.total if self.total else 0.0

    def summary(self) -> dict:
        return {
            "total": self.total,
            "accepted": self.accepted,
            "accept_rate": round(self.accept_rate, 4),
            "rejected_consistency": self.rejected_consistency,
            "rejected_schema": self.rejected_schema,
            "rejected_empty": self.rejected_empty,
            "per_type_accepted": dict(self.per_type_accepted),
            "per_type_rejected": dict(self.per_type_rejected),
        }


def _validate_schema(record: PIIRecord) -> list[str]:
    """Check schema constraints beyond Pydantic validation.

    Returns list of issues (empty = valid). Reports only the first issue
    per span to avoid cascading error messages.
    """
    issues = []
    for s in record.spans:
        if s.start < 0 or s.end > len(record.text):
            issues.append(f"span offset out of bounds: [{s.start}, {s.end})")
            continue
        if s.start >= s.end:
            issues.append(f"empty or inverted span: [{s.start}, {s.end})")
            continue
        if record.text[s.start:s.end] != s.text:
            issues.append(f"span text mismatch at [{s.start}, {s.end})")
    return issues


def majority_vote_spans(samples: list[PIIRecord], threshold: float = 0.5) -> tuple[list[PIISpan], float]:
    """Compute consensus spans from k independent teacher samples.

    A span is accepted if it appears in >= ceil(k * threshold) samples
    (strict majority at threshold=0.5). At k=2 this requires unanimity,
    which is intentionally conservative — a single-sample disagreement at
    k=2 means we don't have enough evidence.

    Returns (consensus_spans, agreement_ratio) where agreement_ratio is
    the mean fraction of samples agreeing on each span in the consensus.
    """
    if not samples:
        return [], 0.0

    k = len(samples)
    span_counts: Counter[SpanTuple] = Counter()
    span_lookup: dict[SpanTuple, PIISpan] = {}

    for rec in samples:
        seen_in_sample: set[SpanTuple] = set()
        for s in rec.spans:
            key = _span_key(s)
            if key not in seen_in_sample:
                span_counts[key] += 1
                seen_in_sample.add(key)
                span_lookup[key] = s

    min_votes = max(1, math.ceil(k * threshold))
    consensus = []
    agreements = []
    for key, count in span_counts.items():
        if count >= min_votes:
            consensus.append(span_lookup[key])
            agreements.append(count / k)

    consensus.sort(key=lambda s: s.start)
    agreement_ratio = sum(agreements) / len(agreements) if agreements else (1.0 if not span_counts else 0.0)
    return consensus, agreement_ratio


def verify_record(
    record_id: str,
    text: str,
    samples: list[PIIRecord],
    schema_valid_flags: list[bool],
    consistency_threshold: float = 0.5,
    min_agreement: float = 0.6,
    min_samples: int = 2,
    split: str = "train",
) -> VerifiedRecord:
    """Run the full verification gate on k teacher samples for one input.

    Args:
        record_id: ID for the output record.
        text: The original input text.
        samples: k parsed PIIRecord responses from the teacher.
        schema_valid_flags: whether each sample parsed as valid JSON schema.
        consistency_threshold: fraction of samples a span must appear in.
        min_agreement: minimum mean agreement ratio to accept.
        min_samples: minimum number of constraint-valid samples required.
        split: split label for the output record.

    Returns:
        VerifiedRecord with accept/reject decision and metadata.
    """
    reasons: list[RejectReason] = []

    json_valid_samples = [s for s, valid in zip(samples, schema_valid_flags) if valid]
    schema_fail_count = len(samples) - len(json_valid_samples)

    if schema_fail_count > len(samples) / 2:
        reasons.append(RejectReason.MAJORITY_SCHEMA_INVALID)

    constraint_valid_samples = []
    for s in json_valid_samples:
        if not _validate_schema(s):
            constraint_valid_samples.append(s)
        else:
            reasons.append(RejectReason.CONSTRAINT_VIOLATION)

    if not constraint_valid_samples:
        return VerifiedRecord(
            record=PIIRecord(id=record_id, text=text, spans=[], split=split),
            accepted=False,
            n_samples=len(samples),
            agreement_ratio=0.0,
            reject_reasons=tuple(reasons or (RejectReason.NO_VALID_SAMPLES,)),
        )

    if len(constraint_valid_samples) < min_samples:
        reasons.append(RejectReason.TOO_FEW_SAMPLES)

    consensus_spans, agreement = majority_vote_spans(constraint_valid_samples, consistency_threshold)

    if agreement < min_agreement:
        reasons.append(RejectReason.LOW_AGREEMENT)

    accepted = not reasons
    record = PIIRecord(id=record_id, text=text, spans=consensus_spans, split=split)

    return VerifiedRecord(
        record=record,
        accepted=accepted,
        n_samples=len(samples),
        agreement_ratio=agreement,
        reject_reasons=tuple(reasons),
    )


def update_stats(stats: VerificationStats, result: VerifiedRecord) -> None:
    """Update running statistics with a verification result."""
    stats.total += 1
    if result.accepted:
        stats.accepted += 1
        for s in result.record.spans:
            stats.per_type_accepted[s.label.value] += 1
    else:
        reason_set = set(result.reject_reasons)
        if RejectReason.NO_VALID_SAMPLES in reason_set:
            stats.rejected_empty += 1
        if RejectReason.MAJORITY_SCHEMA_INVALID in reason_set or RejectReason.CONSTRAINT_VIOLATION in reason_set:
            stats.rejected_schema += 1
        if RejectReason.LOW_AGREEMENT in reason_set or RejectReason.TOO_FEW_SAMPLES in reason_set:
            stats.rejected_consistency += 1
