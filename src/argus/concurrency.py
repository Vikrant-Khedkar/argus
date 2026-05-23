"""Global concurrency control with AIMD backoff.

Replaces the nested-pool-bottleneck pattern with a single flat gate every
OpenRouter call acquires. With unrestricted outer ThreadPoolExecutors,
the gate is the actual concurrency cap, so real in-flight count matches
the configured limit.

AIMD ("Additive Increase, Multiplicative Decrease") — the classic TCP
congestion-control trick promptfoo uses for rate-limited APIs:
  - on every N successful calls, capacity += 1 (additive grow)
  - on 429 / rate-limit error, capacity //= 2  (multiplicative shrink)

Bounded between `min_concurrency` and `max_concurrency`. Reads
``ARGUS_MAX_CONCURRENT`` env var to override the default ceiling.
"""

from __future__ import annotations

import os
import threading
from contextlib import contextmanager


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


class AdaptiveSemaphore:
    """Counting semaphore with AIMD-adaptive capacity."""

    def __init__(
        self,
        initial: int = 24,
        min_concurrency: int = 1,
        max_concurrency: int = 64,
        adapt_threshold: int = 20,
    ):
        self._cond = threading.Condition()
        self._capacity = initial
        self._in_use = 0
        self._min = min_concurrency
        self._max = max_concurrency
        self._adapt_threshold = adapt_threshold
        self._wins = 0

    @property
    def capacity(self) -> int:
        with self._cond:
            return self._capacity

    @contextmanager
    def acquire(self):
        with self._cond:
            while self._in_use >= self._capacity:
                self._cond.wait()
            self._in_use += 1
        try:
            yield
        finally:
            with self._cond:
                self._in_use -= 1
                self._cond.notify()

    def signal_rate_limit(self) -> None:
        """Called when the API returned 429 / rate limit. Halve capacity."""
        with self._cond:
            self._capacity = max(self._min, self._capacity // 2)
            self._wins = 0
            # Wake one waiter — it may immediately re-block, but that's OK
            self._cond.notify()

    def signal_success(self) -> None:
        """Called after a successful API call. After N wins, grow."""
        with self._cond:
            self._wins += 1
            if (
                self._wins >= self._adapt_threshold
                and self._capacity < self._max
            ):
                self._capacity += 1
                self._wins = 0
                self._cond.notify()


# Shared semaphore for every OpenRouter-style call (model-under-test
# inference and scorer judges alike). Single instance so concurrency is
# bounded across the whole audit, regardless of where the call originates.
OPENROUTER_SEMAPHORE = AdaptiveSemaphore(
    initial=_env_int("ARGUS_MAX_CONCURRENT", 24),
    min_concurrency=2,
    max_concurrency=_env_int("ARGUS_MAX_CONCURRENT_CEIL", 64),
)


__all__ = ["AdaptiveSemaphore", "OPENROUTER_SEMAPHORE"]
