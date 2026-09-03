"""Throttle and retry behaviour for the hosted teacher.

A 5 rpm client with no retry measured a 50% drop rate against the Cerebras free
tier, and a dropped call is a missing sample rather than a visible error — so the
retry schedule and the placement of the latency timer are both load-bearing.
"""

from __future__ import annotations

import pytest

from forge.teacher_client import TeacherUnavailable, ThrottledTeacher


class Boom(ConnectionError):
    """Stands in for openai.RateLimitError, which is in the retryable tuple."""


class FakeUsage:
    def __init__(self, p, c):
        self.prompt_tokens = p
        self.completion_tokens = c


class FakeResponse:
    def __init__(self, p=10, c=20):
        self.usage = FakeUsage(p, c)


class FakeCompletions:
    def __init__(self, script):
        self.script = list(script)
        self.calls = 0
        self.kwargs = []

    def create(self, **kwargs):
        self.kwargs.append(kwargs)
        item = self.script[min(self.calls, len(self.script) - 1)]
        self.calls += 1
        if isinstance(item, BaseException):
            raise item
        return item


class FakeClient:
    def __init__(self, script):
        self.chat = type("chat", (), {"completions": FakeCompletions(script)})()


class Clock:
    """Monotonic clock that only advances when something sleeps or is told to."""

    def __init__(self):
        self.t = 0.0
        self.slept: list[float] = []

    def sleep(self, s):
        self.slept.append(s)
        self.t += s

    def now(self):
        return self.t

    def tick(self, s):
        self.t += s


def build(script, **kw):
    clock = Clock()
    client = FakeClient(script)
    teacher = ThrottledTeacher(client, sleep=clock.sleep, monotonic=clock.now, **kw)
    return teacher, client, clock


def test_successful_call_records_tokens_and_latency():
    teacher, client, _ = build([FakeResponse(11, 22)], rpm=None)
    resp, latency = teacher.complete(model="m", messages=[])
    assert resp.usage.prompt_tokens == 11
    assert teacher.stats.calls_ok == 1
    assert teacher.stats.total_tokens == 33
    assert latency == 0.0
    assert client.chat.completions.calls == 1


def test_backoff_schedule_matches_run_inference():
    teacher, _, _ = build([FakeResponse()], max_retries=6)
    assert [teacher.backoff_for(n) for n in range(1, 7)] == [15, 30, 60, 120, 120, 120]


def test_retries_then_succeeds():
    teacher, client, clock = build([Boom(), Boom(), FakeResponse()], rpm=None)
    teacher.complete(model="m", messages=[])
    assert client.chat.completions.calls == 3
    assert teacher.stats.calls_ok == 1
    assert teacher.stats.retries == 2
    assert clock.slept == [15.0, 30.0]


def test_gives_up_and_raises_after_max_retries():
    teacher, client, _ = build([Boom()] * 9, rpm=None, max_retries=3)
    with pytest.raises(TeacherUnavailable, match="gave up after 3"):
        teacher.complete(model="m", messages=[])
    assert client.chat.completions.calls == 3
    assert teacher.stats.calls_failed == 1
    assert teacher.stats.calls_ok == 0


def test_throttle_waits_between_calls_but_not_before_the_first():
    teacher, _, clock = build([FakeResponse()] * 3, rpm=5.0)
    teacher.complete(model="m", messages=[])
    assert clock.slept == []
    teacher.complete(model="m", messages=[])
    assert clock.slept == [12.0]


def test_throttle_sleep_is_excluded_from_latency():
    """ADR 0014 retracted a published claim on the strength of this property:
    the teacher's unstable p95 is real server latency, not client stalling folded
    into the timer. Anything reporting latency has to keep it true."""
    clock = Clock()

    def create(**_kwargs):
        clock.tick(0.4)  # the only real server time
        return FakeResponse()

    client = FakeClient([FakeResponse()])
    client.chat.completions.create = create
    teacher = ThrottledTeacher(client, rpm=5.0, sleep=clock.sleep, monotonic=clock.now)

    _, first = teacher.complete(model="m", messages=[])
    _, second = teacher.complete(model="m", messages=[])
    assert first == pytest.approx(0.4)
    assert second == pytest.approx(0.4), "a 12s throttle must not appear as latency"
    assert clock.slept == [pytest.approx(11.6)]


def test_backoff_sleep_is_excluded_from_latency():
    clock = Clock()
    state = {"n": 0}

    def create(**_kwargs):
        state["n"] += 1
        if state["n"] == 1:
            raise Boom()
        clock.tick(0.5)
        return FakeResponse()

    client = FakeClient([FakeResponse()])
    client.chat.completions.create = create
    teacher = ThrottledTeacher(client, rpm=None, sleep=clock.sleep, monotonic=clock.now)
    _, latency = teacher.complete(model="m", messages=[])
    assert latency == pytest.approx(0.5)
    assert clock.slept == [15.0]


class Throttled(ConnectionError):
    """A 429 carrying a Retry-After header, as the free tier actually sends it."""

    def __init__(self, retry_after):
        super().__init__("rate limited")
        self.response = type("resp", (), {"headers": {"retry-after": retry_after}})()


def test_retry_after_header_overrides_exponential_backoff():
    """The free tier answers an exhausted hourly allowance with retry-after: 3600.

    Exponential backoff tops out at 120s, so a client that ignores the header
    burns its entire retry budget in the first three minutes of a one-hour
    cooldown — and every one of those attempts is itself a request against the
    quota that is already exhausted.
    """
    teacher, _, clock = build([Throttled("3600"), FakeResponse()], rpm=None)
    teacher.complete(model="m", messages=[])
    assert clock.slept == [3600.0], "should wait the hour the server asked for"
    assert teacher.stats.wait_s == 3600.0


def test_retry_after_is_capped_so_a_bad_value_cannot_park_the_job():
    teacher, _, clock = build([Throttled("999999"), FakeResponse()], rpm=None,
                              retry_after_cap=3900.0)
    teacher.complete(model="m", messages=[])
    assert clock.slept == [3900.0]


def test_falls_back_to_backoff_when_header_is_absent_or_junk():
    teacher, _, clock = build([Boom(), Throttled("not-a-number"), FakeResponse()], rpm=None)
    teacher.complete(model="m", messages=[])
    assert clock.slept == [15.0, 30.0]


def test_on_retry_callback_reports_attempt_and_delay():
    seen = []
    teacher, _, _ = build(
        [Boom(), FakeResponse()], rpm=None,
        on_retry=lambda attempt, exc, delay: seen.append((attempt, type(exc).__name__, delay)),
    )
    teacher.complete(model="m", messages=[])
    assert seen == [(1, "Boom", 15.0)]


def test_stats_summary_is_json_shaped():
    teacher, _, _ = build([FakeResponse()], rpm=None)
    teacher.complete(model="m", messages=[])
    s = teacher.stats.summary()
    assert s["calls_ok"] == 1
    assert s["total_tokens"] == 30
    assert set(s) >= {"calls_ok", "calls_failed", "retries", "total_tokens"}
