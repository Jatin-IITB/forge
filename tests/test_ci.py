"""Tests for bootstrap confidence intervals (forge/ci.py).

These lock in the three properties the gates depend on: that record-level
resampling is what happens, that the paired ratio estimator is genuinely
tighter than the naive one, and — most importantly — that the zero-failure
case is not read as certainty.
"""

from __future__ import annotations

import pytest

from forge import ci
from forge.schema import PIIRecord, PIISpan, PIIType


def _rec(rid: str, spans: list[tuple[int, int, PIIType]], text: str = "x" * 200) -> PIIRecord:
    return PIIRecord(
        id=rid,
        text=text,
        split="test",
        spans=[PIISpan(start=s, end=e, label=lab, text=text[s:e]) for s, e, lab in spans],
    )


def _pair(n: int, hit_every: int) -> tuple[list[PIIRecord], list[PIIRecord]]:
    """n records with one span each; the prediction is correct every `hit_every`."""
    gold, pred = [], []
    for i in range(n):
        gold.append(_rec(f"r{i}", [(0, 5, PIIType.EMAIL)]))
        spans = [(0, 5, PIIType.EMAIL)] if i % hit_every == 0 else [(10, 15, PIIType.EMAIL)]
        pred.append(_rec(f"r{i}", spans))
    return gold, pred


class TestMicroF1CI:
    def test_interval_brackets_the_point_estimate(self):
        gold, pred = _pair(80, 2)
        r = ci.micro_f1_ci(gold, pred, n_resamples=2000)
        assert r.lo <= r.point <= r.hi

    def test_perfect_predictions_give_a_degenerate_interval(self):
        gold, pred = _pair(50, 1)
        r = ci.micro_f1_ci(gold, pred, n_resamples=500)
        assert r.point == pytest.approx(1.0)
        assert r.lo == pytest.approx(1.0)

    def test_interval_narrows_as_records_are_added(self):
        small = ci.micro_f1_ci(*_pair(40, 2), n_resamples=4000)
        large = ci.micro_f1_ci(*_pair(400, 2), n_resamples=4000)
        assert large.half_width < small.half_width

    def test_same_seed_reproduces_exactly(self):
        gold, pred = _pair(60, 3)
        a = ci.micro_f1_ci(gold, pred, n_resamples=1000, seed=7)
        b = ci.micro_f1_ci(gold, pred, n_resamples=1000, seed=7)
        assert (a.lo, a.hi) == (b.lo, b.hi)

    def test_different_seeds_stay_close(self):
        gold, pred = _pair(200, 2)
        a = ci.micro_f1_ci(gold, pred, n_resamples=4000, seed=1)
        b = ci.micro_f1_ci(gold, pred, n_resamples=4000, seed=2)
        assert abs(a.lo - b.lo) < 0.02
        assert abs(a.hi - b.hi) < 0.02


class TestPairedRatio:
    def test_paired_estimator_is_tighter_than_independent(self):
        """The reason paired_ratio_ci exists.

        Student and teacher are scored on the SAME records, so a record that is
        hard for one is usually hard for the other. Bootstrapping the two sides
        separately and dividing the intervals throws that correlation away and
        reports more uncertainty than the data contains.
        """
        gold, student = _pair(150, 3)
        _, teacher = _pair(150, 2)

        paired = ci.paired_ratio_ci(gold, student, teacher, n_resamples=4000)
        s = ci.micro_f1_ci(gold, student, n_resamples=4000)
        t = ci.micro_f1_ci(gold, teacher, n_resamples=4000)
        naive_width = (s.hi / t.lo) - (s.lo / t.hi)

        assert paired.hi - paired.lo < naive_width

    def test_identical_models_give_a_ratio_of_one(self):
        gold, pred = _pair(100, 2)
        r = ci.paired_ratio_ci(gold, pred, pred, n_resamples=500)
        assert r.point == pytest.approx(1.0)
        assert r.lo == pytest.approx(1.0)


class TestZeroFailureBound:
    """Zero observed misses is the case where the bootstrap misleads."""

    def test_bootstrap_is_degenerate_when_nothing_was_missed(self):
        """Documents the trap rather than pretending it does not exist.

        A [1.0, 1.0] interval reads as certainty but only means the estimator
        had no errors to resample. This is why zero_failure_recall_bound is the
        function the high-severity claim must be published against.
        """
        gold, pred = _pair(29, 1)
        r = ci.per_type_recall_ci(gold, pred, ["EMAIL"], n_resamples=500)["EMAIL"]
        assert (r.lo, r.hi) == (1.0, 1.0)
        assert ci.zero_failure_recall_bound(29) < 0.95

    def test_matches_the_rule_of_three(self):
        """Exact Clopper-Pearson vs the 3/n approximation.

        The approximation is conservative and its error shrinks with n, so the
        tolerance is scaled rather than fixed — at n=20 the gap is ~0.011, at
        n=300 it is ~0.001. Pinning a single tight tolerance would only test
        the largest n.
        """
        for n in (20, 50, 100, 300):
            assert ci.zero_failure_recall_bound(n) == pytest.approx(1 - 3 / n, abs=0.25 / n)

    def test_bound_tightens_with_more_trials(self):
        bounds = [ci.zero_failure_recall_bound(n) for n in (15, 29, 100, 232)]
        assert bounds == sorted(bounds)

    def test_frozen_gold_high_severity_bound_supports_99_not_100(self):
        """The published claim is ~99%, not 100% — this pins why.

        232 high-severity gold instances with zero misses. The point estimate
        is 1.0000; the sample supports 0.987. A resume bullet saying "100%"
        would be quoting an estimate the test set cannot carry.
        """
        assert ci.zero_failure_recall_bound(232) == pytest.approx(0.9872, abs=1e-4)
        assert ci.zero_failure_recall_bound(15) == pytest.approx(0.8190, abs=1e-4)

    def test_degenerate_inputs(self):
        assert ci.zero_failure_recall_bound(0) == 0.0
        assert ci.latency_ci([]).point == 0.0


class TestLatencyCI:
    def test_tail_quantile_is_wider_than_the_median(self):
        lat = [0.5 + 0.01 * i for i in range(100)] + [9.0, 12.0, 14.0]
        p50 = ci.latency_ci(lat, 0.50, n_resamples=3000)
        p95 = ci.latency_ci(lat, 0.95, n_resamples=3000)
        assert p95.half_width > p50.half_width

    def test_small_sample_p95_cannot_rule_out_a_much_larger_tail(self):
        """Why the teacher p95 needed re-measuring.

        Two runs of one config reported p95 = 0.790s (n=60) and 8.024s (n=302).
        A 60-sample interval is wide, but not wide enough to absorb a 10x gap —
        so the discrepancy is a real difference in conditions, not noise.
        """
        tight = [0.4 + 0.005 * i for i in range(60)]
        r = ci.latency_ci(tight, 0.95, n_resamples=3000)
        assert r.hi < 1.5
