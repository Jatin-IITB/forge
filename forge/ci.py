"""Bootstrap confidence intervals for the contract's metrics.

Contract v2 says every gate number is "measured with 95% CIs". Nothing in the
repo produced one until this module existed, so gates were being read as if
0.5750 and 0.9292 were exact. On 385 records they are not.

Three decisions, each load-bearing:

**Resample records, never spans.** A record contributes several spans whose
correctness is correlated — the same fuzzy address boundary, the same missed
sentence. Treating spans as independent draws would understate the interval,
which is the direction that flatters a gate. The record is the independent
sampling unit, so the record is what gets resampled.

**Pair the resample across models.** G1, G3 and G4 are all ratios of the form
``student_X / teacher_X``, and both sides are measured on the *same* frozen
records. Bootstrapping the two independently and dividing the intervals treats
that shared sample as two samples and inflates the result. Applying one
resample to both sides keeps the correlation, and the ratio interval comes out
correctly — usually much tighter, because a record that is hard for the student
is typically hard for the teacher too.

**Percentile method.** No normality assumption. F1 is a bounded ratio of
counts, and near the edges its sampling distribution is visibly skewed.

Resampling is expressed as multinomial weights rather than index gathers, so a
resample is a single matrix multiply against per-record count matrices instead
of a Python loop over 10,000 re-scorings.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from forge.eval import _span_key
from forge.schema import PIIRecord

DEFAULT_RESAMPLES = 10_000
DEFAULT_SEED = 42
DEFAULT_ALPHA = 0.05


@dataclass(frozen=True)
class Interval:
    """A point estimate with a percentile bootstrap interval around it."""

    point: float
    lo: float
    hi: float
    n_resamples: int
    alpha: float

    @property
    def half_width(self) -> float:
        return (self.hi - self.lo) / 2

    def __str__(self) -> str:
        pct = round((1 - self.alpha) * 100)
        return f"{self.point:.4f} [{self.lo:.4f}, {self.hi:.4f}] {pct}% CI"


def _count_matrices(
    gold: list[PIIRecord],
    preds: list[PIIRecord],
    labels: list[str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-record, per-label (tp, fp, fn) counts as (n_records, n_labels) arrays.

    Scoring semantics are exact-match on ``(start, end, label)`` — identical to
    ``forge.eval.evaluate``. This function only changes the *shape* of the
    result so a resample is a matrix product, never a re-scoring.
    """
    label_ix = {lab: i for i, lab in enumerate(labels)}
    n, k = len(gold), len(labels)
    tp = np.zeros((n, k), dtype=np.int32)
    fp = np.zeros((n, k), dtype=np.int32)
    fn = np.zeros((n, k), dtype=np.int32)

    pred_map = {r.id: r for r in preds}
    for i, g in enumerate(gold):
        p = pred_map.get(g.id)
        gold_keys = {_span_key(s) for s in g.spans}
        pred_keys = {_span_key(s) for s in p.spans} if p else set()

        for key in gold_keys & pred_keys:
            tp[i, label_ix[key.label]] += 1
        for key in pred_keys - gold_keys:
            fp[i, label_ix[key.label]] += 1
        for key in gold_keys - pred_keys:
            fn[i, label_ix[key.label]] += 1

    return tp, fp, fn


def _labels_of(*record_lists: list[PIIRecord]) -> list[str]:
    seen: set[str] = set()
    for records in record_lists:
        for r in records:
            for s in r.spans:
                seen.add(s.label.value)
    return sorted(seen)


def _weights(n_records: int, n_resamples: int, seed: int) -> np.ndarray:
    """Bootstrap resamples as multinomial multiplicities, shape (B, n_records).

    Row b holds how many times each record was drawn in resample b. Summing
    counts under a resample is then ``W @ counts`` — one matmul for all B
    resamples, instead of B gathers.
    """
    rng = np.random.default_rng(seed)
    return rng.multinomial(n_records, np.full(n_records, 1.0 / n_records), size=n_resamples)


def _f1_from(tp: np.ndarray, fp: np.ndarray, fn: np.ndarray) -> np.ndarray:
    denom = 2 * tp + fp + fn
    return np.divide(2 * tp, denom, out=np.zeros_like(denom, dtype=float), where=denom > 0)


def _recall_from(tp: np.ndarray, fn: np.ndarray) -> np.ndarray:
    denom = tp + fn
    return np.divide(tp, denom, out=np.zeros_like(denom, dtype=float), where=denom > 0)


def _pct_interval(samples: np.ndarray, point: float, alpha: float, n_resamples: int) -> Interval:
    lo, hi = np.quantile(samples, [alpha / 2, 1 - alpha / 2])
    return Interval(point=point, lo=float(lo), hi=float(hi), n_resamples=n_resamples, alpha=alpha)


def micro_f1_ci(
    gold: list[PIIRecord],
    preds: list[PIIRecord],
    *,
    n_resamples: int = DEFAULT_RESAMPLES,
    seed: int = DEFAULT_SEED,
    alpha: float = DEFAULT_ALPHA,
) -> Interval:
    """Percentile bootstrap CI for exact-match micro-F1, resampling records."""
    labels = _labels_of(gold, preds)
    tp, fp, fn = _count_matrices(gold, preds, labels)
    w = _weights(len(gold), n_resamples, seed)

    point = float(_f1_from(tp.sum(), fp.sum(), fn.sum()))
    samples = _f1_from(w @ tp.sum(1), w @ fp.sum(1), w @ fn.sum(1))
    return _pct_interval(samples, point, alpha, n_resamples)


def per_type_recall_ci(
    gold: list[PIIRecord],
    preds: list[PIIRecord],
    types: list[str] | None = None,
    *,
    n_resamples: int = DEFAULT_RESAMPLES,
    seed: int = DEFAULT_SEED,
    alpha: float = DEFAULT_ALPHA,
) -> dict[str, Interval]:
    """Per-type recall CIs — the form the high-severity floor is gated on.

    A floor of 0.99 on ~29 gold instances is a demanding claim, and the CI is
    what shows whether the sample can support it at all.
    """
    labels = _labels_of(gold, preds)
    tp, _fp, fn = _count_matrices(gold, preds, labels)
    w = _weights(len(gold), n_resamples, seed)

    wanted = types if types is not None else labels
    out: dict[str, Interval] = {}
    for lab in wanted:
        if lab not in labels:
            continue
        j = labels.index(lab)
        point = float(_recall_from(tp[:, j].sum(), fn[:, j].sum()))
        samples = _recall_from(w @ tp[:, j], w @ fn[:, j])
        out[lab] = _pct_interval(samples, point, alpha, n_resamples)
    return out


def paired_ratio_ci(
    gold: list[PIIRecord],
    student_preds: list[PIIRecord],
    teacher_preds: list[PIIRecord],
    *,
    n_resamples: int = DEFAULT_RESAMPLES,
    seed: int = DEFAULT_SEED,
    alpha: float = DEFAULT_ALPHA,
) -> Interval:
    """CI for ``student_f1 / teacher_f1`` — the quantity G1 actually gates on.

    Both models are scored on the same resampled records, so the shared
    difficulty of a record cancels instead of being counted twice as noise.
    Comparing two independently-computed intervals would overstate the
    uncertainty and could let a genuinely-failing gate read as ambiguous.
    """
    labels = _labels_of(gold, student_preds, teacher_preds)
    s_tp, s_fp, s_fn = _count_matrices(gold, student_preds, labels)
    t_tp, t_fp, t_fn = _count_matrices(gold, teacher_preds, labels)
    w = _weights(len(gold), n_resamples, seed)

    s_point = float(_f1_from(s_tp.sum(), s_fp.sum(), s_fn.sum()))
    t_point = float(_f1_from(t_tp.sum(), t_fp.sum(), t_fn.sum()))
    point = s_point / t_point if t_point else 0.0

    s = _f1_from(w @ s_tp.sum(1), w @ s_fp.sum(1), w @ s_fn.sum(1))
    t = _f1_from(w @ t_tp.sum(1), w @ t_fp.sum(1), w @ t_fn.sum(1))
    samples = np.divide(s, t, out=np.zeros_like(s), where=t > 0)
    return _pct_interval(samples, point, alpha, n_resamples)


def zero_failure_recall_bound(n_trials: int, alpha: float = DEFAULT_ALPHA) -> float:
    """Lower confidence bound on recall when *zero* misses were observed.

    The bootstrap is degenerate here and quietly misleading: with no misses in
    the sample, every resample also has no misses, so ``per_type_recall_ci``
    returns [1.0000, 1.0000] — an interval that looks like certainty but only
    reflects that the estimator has nothing to resample. Perfect measured
    recall on 29 instances is *not* evidence of perfect recall.

    The Clopper-Pearson one-sided bound is the right statement. For 0 failures
    in n trials the upper bound on the failure rate is ``1 - alpha**(1/n)``
    (the "rule of three", 3/n, is its familiar approximation), so recall is
    bounded below by ``alpha**(1/n)``.

    On this contract's frozen test set:

    - one type, n=29  -> recall >= 0.902
    - all nine pooled, n=232 -> recall >= 0.987

    So the defensible published claim on the nine high-severity types is
    "~99% recall", not "100%" — the point estimate is 1.0000, but the sample
    supports 0.987. Quoting the point estimate alone would overstate what 232
    instances can carry.
    """
    if n_trials <= 0:
        return 0.0
    return float(alpha ** (1.0 / n_trials))


def latency_ci(
    latencies: list[float],
    quantile: float = 0.95,
    *,
    n_resamples: int = DEFAULT_RESAMPLES,
    seed: int = DEFAULT_SEED,
    alpha: float = DEFAULT_ALPHA,
) -> Interval:
    """CI for a latency quantile — how unstable is a p95 at this sample size?

    Motivated by a measured failure: two runs of the identical teacher
    configuration reported p95 = 0.790 s (n=60) and p95 = 8.024 s (n=302). A
    tail quantile estimated from a few hundred samples carries far more
    uncertainty than a bare number suggests, and G4's threshold is defined as
    ``teacher_p95 / 5`` — so the gate inherits every bit of it.
    """
    arr = np.asarray(latencies, dtype=float)
    if arr.size == 0:
        return Interval(0.0, 0.0, 0.0, n_resamples, alpha)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, arr.size, size=(n_resamples, arr.size))
    samples = np.quantile(arr[idx], quantile, axis=1)
    return _pct_interval(samples, float(np.quantile(arr, quantile)), alpha, n_resamples)
