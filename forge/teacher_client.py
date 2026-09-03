"""Throttled, retrying client for the hosted teacher.

`scripts/run_inference.py` already got this right — throttle before the timer,
retry with exponential backoff, record only successful attempts — but the logic
lived inline, so `run_data_engine.py`, `generate_carriers.py` and
`build_train_v3.py` each shipped without it. On the Cerebras free tier that is
not a cosmetic gap: a 5 rpm client with no retry measured a **50% drop rate**
against the documented 5 req/min limit, and every dropped call is a silently
missing sample rather than an error.

Two properties matter and are tested rather than asserted:

1. **The throttle sleep is outside the latency measurement.** ADR 0014 retracted a
   published hypothesis that the teacher's unstable p95 was client-side stalling
   folded into the timer; that turned out to be false precisely because
   `run_inference.py` set `t0` after the sleep. Anything that reports latency has
   to keep that property or the retraction stops being true.
2. **Backoff is bounded and deterministic.** `min(base * 2**(n-1), cap)`, the same
   schedule as `run_inference.py`, so behaviour under throttling is identical
   across every script that talks to the teacher.

The clock and sleep are injectable so the retry schedule can be unit-tested
without waiting two minutes for a backoff.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field


class TeacherUnavailable(RuntimeError):
    """Every retry was exhausted for one request."""


@dataclass
class TeacherStats:
    calls_ok: int = 0
    calls_failed: int = 0
    retries: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    wait_s: float = 0.0  # time spent in backoff/cooldown, not in server round-trips
    latencies_s: list[float] = field(default_factory=list)

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    def summary(self) -> dict:
        lat = sorted(self.latencies_s)
        return {
            "calls_ok": self.calls_ok,
            "calls_failed": self.calls_failed,
            "retries": self.retries,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "throttle_wait_s": round(self.wait_s, 1),
            "p50_latency_s": round(lat[len(lat) // 2], 3) if lat else None,
            "mean_latency_s": round(sum(lat) / len(lat), 3) if lat else None,
        }


def _retryable() -> tuple[type[BaseException], ...]:
    """Transport-level failures worth retrying, resolved lazily.

    Importing `openai` at module scope would make `forge.schema` consumers pay for
    it, and the package is an optional extra.
    """
    # Raw socket/timeout failures are retryable whether or not the openai package
    # wrapped them, so they are always in the set.
    base: tuple[type[BaseException], ...] = (ConnectionError, TimeoutError)
    try:
        import openai
    except ImportError:  # pragma: no cover - exercised only without the extra
        return base
    return base + (
        openai.RateLimitError,
        openai.APIConnectionError,
        openai.APITimeoutError,
        openai.InternalServerError,
    )


class ThrottledTeacher:
    """Rate-limited chat client with bounded retries.

    Usage::

        teacher = ThrottledTeacher(OpenAI(...), rpm=5)
        resp, latency = teacher.complete(model=..., messages=..., max_tokens=...)
    """

    def __init__(
        self,
        client,
        rpm: float | None = 5.0,
        max_retries: int = 5,
        backoff_base: float = 15.0,
        backoff_cap: float = 120.0,
        retry_after_cap: float = 3900.0,
        sleep: Callable[[float], None] = time.sleep,
        monotonic: Callable[[], float] = time.monotonic,
        on_retry: Callable[[int, BaseException, float], None] | None = None,
    ) -> None:
        self._client = client
        self._min_interval = 60.0 / rpm if rpm else 0.0
        self._max_retries = max_retries
        self._backoff_base = backoff_base
        self._backoff_cap = backoff_cap
        self._retry_after_cap = retry_after_cap
        self._sleep = sleep
        self._monotonic = monotonic
        self._on_retry = on_retry
        # None, not 0.0: with a clock that starts near zero, a 0.0 sentinel makes
        # the very first call wait a full interval for no reason.
        self._last_start: float | None = None
        self.stats = TeacherStats()

    def backoff_for(self, attempt: int) -> float:
        """Seconds to wait after failed `attempt` (1-based). Bounded, deterministic."""
        return min(self._backoff_base * (2 ** (attempt - 1)), self._backoff_cap)

    def delay_for(self, attempt: int, exc: BaseException) -> float:
        """Prefer the server's `Retry-After` over our guess.

        The Cerebras free tier does not only throttle per minute — once an hourly
        allowance is gone it answers 429 with `retry-after: 3600`. Exponential
        backoff is the wrong instrument for that: the schedule tops out at 120s, so
        a client that ignores the header burns its whole retry budget inside the
        first three minutes of a one-hour cooldown and reports the work as failed.
        Worse, each of those attempts is itself a request against the quota.

        The header is trusted up to `retry_after_cap`, so a pathological value
        cannot park a job for a week.
        """
        after = None
        response = getattr(exc, "response", None)
        headers = getattr(response, "headers", None)
        if headers:
            raw = headers.get("retry-after") or headers.get("Retry-After")
            if raw:
                try:
                    after = float(raw)
                except (TypeError, ValueError):
                    after = None
        if after is None:
            return self.backoff_for(attempt)
        return max(0.0, min(after, self._retry_after_cap))

    def _throttle(self) -> None:
        if not self._min_interval or self._last_start is None:
            return
        wait = self._min_interval - (self._monotonic() - self._last_start)
        if wait > 0:
            self._sleep(wait)

    def complete(self, **kwargs):
        """One chat completion, throttled and retried. Returns (response, latency_s)."""
        retryable = _retryable()
        last_exc: BaseException | None = None

        for attempt in range(1, self._max_retries + 1):
            self._throttle()
            self._last_start = self._monotonic()
            # t0 is set AFTER the throttle sleep and reset on every attempt, so a
            # reported latency is one server round-trip and nothing else.
            t0 = self._monotonic()
            try:
                resp = self._client.chat.completions.create(**kwargs)
            except retryable as exc:
                last_exc = exc
                self.stats.retries += 1
                if attempt == self._max_retries:
                    break
                delay = self.delay_for(attempt, exc)
                self.stats.wait_s += delay
                if self._on_retry:
                    self._on_retry(attempt, exc, delay)
                self._sleep(delay)
                continue

            latency = self._monotonic() - t0
            self.stats.calls_ok += 1
            self.stats.latencies_s.append(latency)
            usage = getattr(resp, "usage", None)
            if usage:
                self.stats.prompt_tokens += getattr(usage, "prompt_tokens", 0) or 0
                self.stats.completion_tokens += getattr(usage, "completion_tokens", 0) or 0
            return resp, latency

        self.stats.calls_failed += 1
        raise TeacherUnavailable(
            f"gave up after {self._max_retries} attempts: {type(last_exc).__name__}"
        ) from last_exc
