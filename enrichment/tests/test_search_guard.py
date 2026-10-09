import threading
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest

from sber_enrichment.search_guard import SearchDeferred, SearchGuard


class FakeClock:
    def __init__(self):
        self.now = 0.0
        self.sleeps = []
        self.lock = threading.Lock()

    def __call__(self):
        with self.lock:
            return self.now

    def advance(self, duration):
        with self.lock:
            self.now += duration

    def sleep(self, duration):
        with self.lock:
            self.sleeps.append(duration)
            self.now += duration


def guard_with_clock(**kwargs):
    clock = FakeClock()
    guard = SearchGuard(clock=clock, sleeper=clock.sleep, **kwargs)
    return guard, clock


def limited(reason="Suspended: too many requests"):
    return {"results": [{"title": "still partial"}], "unresponsive_engines": [["google", reason]]}


def test_spaces_starts_instead_of_adding_fixed_delay_after_http():
    guard, clock = guard_with_clock()
    starts = []
    result = {"results": [], "unresponsive_engines": []}

    def call():
        starts.append(clock())
        clock.advance(0.5)
        return result

    assert all(guard.run(call) is result for _ in range(3))
    assert starts == [0.0, 2.0, 4.0]
    assert clock.sleeps == [1.5, 1.5]


def test_two_responses_open_circuit_once_per_response_without_success_reset():
    guard, clock = guard_with_clock()
    first = limited()
    first["unresponsive_engines"].append(["other", "HTTP 429"])
    assert guard.run(lambda: first) is first
    assert guard.rate_limit_errors == 1
    assert guard.run(lambda: {"results": []}) == {"results": []}
    assert guard.rate_limit_errors == 1
    second = limited("too many requests")
    assert guard.run(lambda: second) is second
    assert guard.circuit_open
    assert guard.rate_limit_errors == 2
    attempts = []
    for advance in (0, 10000):
        clock.advance(advance)
        with pytest.raises(SearchDeferred) as caught:
            guard.run(lambda: attempts.append("unexpected"))
        assert caught.value.code == "search_deferred_rate_limit"
        assert caught.value.rate_limit_errors == 2
    assert attempts == []
    fresh = SearchGuard(clock=clock, sleeper=clock.sleep)
    assert fresh.run(lambda: "new bounded run") == "new bounded run"
    assert not fresh.circuit_open


@pytest.mark.parametrize(
    "reason",
    [
        "Too Many Requests",
        "HTTP 429",
        "HTTP429",
        "Suspended",
        {"reason": "too many requests"},
        429,
    ],
)
def test_detects_reported_throttle_shapes(reason):
    guard, _clock = guard_with_clock(max_rate_limit_errors=1)
    result = SimpleNamespace(unresponsive_engines=[["engine", reason]])
    assert guard.run(lambda: result) is result
    assert guard.circuit_open


@pytest.mark.parametrize(
    "result",
    [
        {"results": [{"title": "429 too many requests discussed here"}]},
        {"unresponsive_engines": [["engine", "connection timeout"]]},
        {"unresponsive_engines": [["engine", "error code 14290"]]},
        {"unresponsive_engines": None},
    ],
)
def test_unrelated_errors_or_search_result_text_do_not_open_circuit(result):
    guard, _clock = guard_with_clock(max_rate_limit_errors=1)
    assert guard.run(lambda: result) is result
    assert guard.rate_limit_errors == 0


def test_exceptions_preserved_no_retries_and_only_explicit_429_counted():
    guard, _clock = guard_with_clock(max_rate_limit_errors=1)
    unrelated = RuntimeError("a free-text message mentioning 429")
    attempts = []

    def ordinary_error():
        attempts.append(1)
        raise unrelated

    with pytest.raises(RuntimeError) as caught:
        guard.run(ordinary_error)
    assert caught.value is unrelated
    assert guard.rate_limit_errors == 0
    http_error = RuntimeError("upstream body must not be exposed by the guard")
    http_error.status = 429

    def throttled():
        attempts.append(2)
        raise http_error

    with pytest.raises(RuntimeError) as caught:
        guard.run(throttled)
    assert caught.value is http_error
    assert attempts == [1, 2]
    assert guard.circuit_open


def test_http_does_not_hold_shared_lock_and_second_worker_is_paced():
    guard, clock = guard_with_clock()
    entered, release = threading.Event(), threading.Event()
    starts = []

    def slow_call():
        starts.append(clock())
        entered.set()
        assert release.wait(timeout=2)
        return {"results": []}

    def second_call():
        starts.append(clock())
        # A status read also acquires the guard lock and must not deadlock.
        assert guard.rate_limit_errors == 0
        return {"results": []}

    with ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(guard.run, slow_call)
        try:
            assert entered.wait(timeout=1)
            second = pool.submit(guard.run, second_call)
            assert second.result(timeout=1) == {"results": []}
        finally:
            release.set()
        assert first.result(timeout=1) == {"results": []}
    assert starts == [0.0, 2.0]


def test_concurrent_completions_account_for_both_responses():
    guard, _clock = guard_with_clock()
    barrier = threading.Barrier(2)

    def call():
        barrier.wait(timeout=2)
        return limited()

    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(guard.run, call) for _ in range(2)]
        assert all(future.result(timeout=2)["unresponsive_engines"] for future in futures)
    assert guard.rate_limit_errors == 2
    assert guard.circuit_open


def test_sleeping_worker_rechecks_open_circuit_before_executing_callable():
    clock = FakeClock()
    block_sleeps, waiting, wake = threading.Event(), threading.Event(), threading.Event()

    def sleeper(duration):
        if block_sleeps.is_set():
            waiting.set()
            assert wake.wait(timeout=2)
        clock.sleep(duration)

    guard = SearchGuard(clock=clock, sleeper=sleeper)
    guard.run(limited)  # First rate-limited response.
    in_flight, finish = threading.Event(), threading.Event()
    unexpected = []

    def second_call():
        in_flight.set()
        assert finish.wait(timeout=2)
        return limited()

    with ThreadPoolExecutor(max_workers=2) as pool:
        second = pool.submit(guard.run, second_call)
        try:
            assert in_flight.wait(timeout=1)
            block_sleeps.set()
            third = pool.submit(guard.run, lambda: unexpected.append("executed"))
            assert waiting.wait(timeout=1)
            finish.set()
            second.result(timeout=1)
            assert guard.circuit_open
            wake.set()
            with pytest.raises(SearchDeferred):
                third.result(timeout=1)
        finally:
            finish.set()
            wake.set()
    assert unexpected == []


def test_early_wakeup_rechecks_shared_pacing_deadline():
    clock = FakeClock()
    sleeps = []

    def sleeper(duration):
        sleeps.append(duration)
        clock.advance(min(duration, 1.0))

    guard = SearchGuard(clock=clock, sleeper=sleeper)
    guard.run(lambda: None)
    assert guard.run(clock) == 2.0
    assert sleeps == [2.0, 1.0]


@pytest.mark.parametrize(
    "kwargs",
    [
        {"min_interval": -1},
        {"min_interval": float("nan")},
        {"min_interval": float("inf")},
        {"max_rate_limit_errors": 0},
        {"max_rate_limit_errors": 1.5},
    ],
)
def test_rejects_invalid_limits(kwargs):
    with pytest.raises(ValueError):
        SearchGuard(**kwargs)
