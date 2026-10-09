"""Shared pacing and a run-scoped circuit breaker for bounded search discovery.

One instance must be shared by all workers in a retrieval run. Admission times
are serialized, while sleeping and HTTP execution happen outside the lock.
Already admitted calls may finish after the circuit opens; waiting and future
calls are deferred. This class neither retries requests nor changes their results.
"""

import math
import re
import threading
import time
from collections.abc import Callable, Mapping
from typing import TypeVar

Result = TypeVar("Result")
_RATE_LIMIT = re.compile(r"too\s+many|(?<!\d)429(?!\d)|suspended", re.IGNORECASE)


class SearchDeferred(RuntimeError):
    """No callable was admitted because this run's rate-limit circuit is open."""

    def __init__(self, rate_limit_errors: int):
        self.code = "search_deferred_rate_limit"
        self.rate_limit_errors = rate_limit_errors
        super().__init__(self.code)


def _has_rate_limit(result: object) -> bool:
    if isinstance(result, Mapping):
        errors = result.get("unresponsive_engines")
    else:
        errors = getattr(result, "unresponsive_engines", None)
    pending = [errors]
    visited = set()
    while pending:
        item = pending.pop()
        if isinstance(item, str):
            if _RATE_LIMIT.search(item):
                return True
        elif isinstance(item, (int, float)) and item == 429:
            return True
        elif isinstance(item, Mapping) or isinstance(item, (list, tuple)):
            if id(item) in visited:
                continue
            visited.add(id(item))
            pending.extend(item.values() if isinstance(item, Mapping) else item)
    return False


class SearchGuard:
    """Pace callable admission and defer after cumulative rate-limit responses.

    ``max_rate_limit_errors`` counts responses, not failing engines. A response
    with several rate-limited engines counts once; intervening successes do not
    reset the counter. Exceptions with an explicit HTTP status/status_code 429
    count once and are re-raised unchanged. Other exceptions are not classified
    from their free text. Use a new instance for an explicitly authorized new run.

    ``clock`` and ``sleeper`` enable deterministic tests without real delays.
    Production defaults are a monotonic clock and ordinary sleep. The injected
    sleeper must allow time to advance; early wakeups are checked again.
    """

    def __init__(
        self,
        min_interval: float = 2.0,
        max_rate_limit_errors: int = 2,
        *,
        clock: Callable[[], float] | None = None,
        sleeper: Callable[[float], None] | None = None,
    ):
        if (
            isinstance(min_interval, bool)
            or not isinstance(min_interval, (int, float))
            or not math.isfinite(min_interval)
            or min_interval < 0
        ):
            raise ValueError("min_interval must be finite and nonnegative")
        if (
            isinstance(max_rate_limit_errors, bool)
            or not isinstance(max_rate_limit_errors, int)
            or max_rate_limit_errors < 1
        ):
            raise ValueError("max_rate_limit_errors must be a positive integer")
        self._min_interval = float(min_interval)
        self._max_rate_limit_errors = max_rate_limit_errors
        self._clock = time.monotonic if clock is None else clock
        self._sleeper = time.sleep if sleeper is None else sleeper
        self._lock = threading.Lock()
        self._next_start = None
        self._rate_limit_errors = 0

    @property
    def rate_limit_errors(self) -> int:
        """Number of completed responses or explicit exceptions indicating throttling."""
        with self._lock:
            return self._rate_limit_errors

    @property
    def circuit_open(self) -> bool:
        with self._lock:
            return self._rate_limit_errors >= self._max_rate_limit_errors

    def _record_rate_limit(self):
        with self._lock:
            self._rate_limit_errors += 1

    def _admit(self):
        while True:
            with self._lock:
                if self._rate_limit_errors >= self._max_rate_limit_errors:
                    raise SearchDeferred(self._rate_limit_errors)
                now = self._clock()
                if not math.isfinite(now):
                    raise ValueError("SearchGuard clock must return finite monotonic time")
                delay = 0 if self._next_start is None else self._next_start - now
                if delay <= 0:
                    self._next_start = now + self._min_interval
                    return
            # Do not reserve future slots: a waiter must recheck the circuit and
            # the common admission time after sleeping, including early wakeups.
            self._sleeper(delay)

    def run(self, call: Callable[[], Result]) -> Result:
        """Execute at most once, preserving the exact returned result or exception."""
        if not callable(call):
            raise TypeError("SearchGuard.run requires a zero-argument callable")
        self._admit()
        try:
            result = call()
        except Exception as error:
            if getattr(error, "status", None) == 429 or getattr(error, "status_code", None) == 429:
                self._record_rate_limit()
            raise
        if _has_rate_limit(result):
            self._record_rate_limit()
        return result
