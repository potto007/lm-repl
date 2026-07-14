"""Tests for the priority RequestScheduler and its OpenAIClient integration."""

import asyncio
import threading
import time
from unittest.mock import MagicMock, patch

import httpx
import openai as openai_sdk
import pytest

from prehend.clients.openai import (
    OpenAIClient,
    _ctx_overflow_budget,
    _is_context_contention,
)
from prehend.clients.scheduler import Priority, RequestScheduler, resolve_priority
from prehend.core.comms_utils import LMRequest
from prehend.core.lm_handler import LMHandler


def wait_until(cond, timeout=2.0, interval=0.005):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if cond():
            return True
        time.sleep(interval)
    return False


# =============================================================================
# resolve_priority
# =============================================================================


def test_resolve_priority_mappings():
    assert resolve_priority(None) == Priority.NORMAL
    assert resolve_priority("high") == Priority.HIGH
    assert resolve_priority("normal") == Priority.NORMAL
    assert resolve_priority("low") == Priority.LOW
    assert resolve_priority(1) == Priority.CONTENTION_RETRY
    assert resolve_priority(5) == Priority.BACKGROUND
    # Out-of-range / unknown values fall back to NORMAL
    assert resolve_priority(0) == Priority.NORMAL
    assert resolve_priority(99) == Priority.NORMAL
    assert resolve_priority("urgent") == Priority.NORMAL


# =============================================================================
# Scheduler core (sync)
# =============================================================================


def test_acquire_release_respects_max_concurrent():
    s = RequestScheduler(max_concurrent=2)
    s.acquire(Priority.NORMAL)
    s.acquire(Priority.NORMAL)
    assert s.active == 2

    third_admitted = threading.Event()

    def third():
        s.acquire(Priority.NORMAL)
        third_admitted.set()

    t = threading.Thread(target=third, daemon=True)
    t.start()
    assert wait_until(lambda: s.waiting == 1)
    assert not third_admitted.is_set()

    s.release(Priority.NORMAL)
    assert third_admitted.wait(2)
    assert s.active == 2

    s.release(Priority.NORMAL)
    s.release(Priority.NORMAL)
    assert s.active == 0


def test_p1_exclusivity_blocks_other_priorities():
    s = RequestScheduler(max_concurrent=4)
    s.acquire(Priority.CONTENTION_RETRY)

    normal_admitted = threading.Event()

    def normal():
        s.acquire(Priority.NORMAL)
        normal_admitted.set()

    t = threading.Thread(target=normal, daemon=True)
    t.start()
    assert wait_until(lambda: s.waiting == 1)
    # Capacity is available (1 of 4 used) but p1 is active, so NORMAL must wait.
    assert not normal_admitted.is_set()

    s.release(Priority.CONTENTION_RETRY)
    assert normal_admitted.wait(2)
    s.release(Priority.NORMAL)
    assert s.active == 0


def test_p1_waits_for_active_requests_to_drain():
    """A contention retry needs the FULL pool, so p1 must not start while any
    other request is in flight - blocking only NEW admissions is not enough."""
    s = RequestScheduler(max_concurrent=4)
    s.acquire(Priority.NORMAL)

    p1_admitted = threading.Event()

    def p1():
        s.acquire(Priority.CONTENTION_RETRY)
        p1_admitted.set()

    t = threading.Thread(target=p1, daemon=True)
    t.start()
    assert wait_until(lambda: s.waiting == 1)
    # Capacity is available (1 of 4 used) but a NORMAL is active: p1 must wait
    # for it to drain so the retry runs with the whole KV pool.
    assert not p1_admitted.is_set()

    s.release(Priority.NORMAL)
    assert p1_admitted.wait(2)
    assert s.active == 1
    s.release(Priority.CONTENTION_RETRY)
    assert s.active == 0


def test_p1_requests_serialize():
    """Concurrent p1 retries would re-exhaust the pool together (the unified-KV
    mass-kill fails ALL in-flight requests, which all escalate to p1), so p1
    requests must run one at a time."""
    s = RequestScheduler(max_concurrent=4)
    s.acquire(Priority.CONTENTION_RETRY)

    second_admitted = threading.Event()

    def second_p1():
        s.acquire(Priority.CONTENTION_RETRY)
        second_admitted.set()

    t = threading.Thread(target=second_p1, daemon=True)
    t.start()
    assert wait_until(lambda: s.waiting == 1)
    assert not second_admitted.is_set()

    s.release(Priority.CONTENTION_RETRY)
    assert second_admitted.wait(2)
    assert s.active == 1
    s.release(Priority.CONTENTION_RETRY)
    assert s.active == 0


def test_waiting_p1_blocks_new_lower_priority_admissions():
    """While a p1 waits for the pool to drain, new lower-priority arrivals must
    queue behind it - otherwise a steady request stream starves the retry
    forever (active never reaches 0)."""
    s = RequestScheduler(max_concurrent=4)
    s.acquire(Priority.NORMAL)

    order = []
    p1_admitted = threading.Event()
    late_admitted = threading.Event()

    def p1():
        s.acquire(Priority.CONTENTION_RETRY)
        order.append("p1")
        p1_admitted.set()

    t1 = threading.Thread(target=p1, daemon=True)
    t1.start()
    assert wait_until(lambda: s.waiting == 1)

    def late_normal():
        s.acquire(Priority.NORMAL)
        order.append("late-normal")
        late_admitted.set()

    t2 = threading.Thread(target=late_normal, daemon=True)
    t2.start()
    assert wait_until(lambda: s.waiting == 2)
    # Capacity exists (1 of 4) and no p1 is ACTIVE yet, but one is WAITING:
    # the late NORMAL must not barge past it.
    assert not late_admitted.is_set()
    assert not p1_admitted.is_set()

    s.release(Priority.NORMAL)
    assert p1_admitted.wait(2)
    assert not late_admitted.is_set()  # p1 runs solo

    s.release(Priority.CONTENTION_RETRY)
    assert late_admitted.wait(2)
    assert order == ["p1", "late-normal"]
    s.release(Priority.NORMAL)
    assert s.active == 0


def test_priority_ordering_high_before_normal():
    s = RequestScheduler(max_concurrent=1)
    s.acquire(Priority.NORMAL)  # occupy the only slot

    order = []

    def waiter(priority, tag):
        s.acquire(priority)
        order.append(tag)
        s.release(priority)

    # Enqueue LOW first so FIFO alone would dispatch it first.
    t_low = threading.Thread(target=waiter, args=(Priority.LOW, "low"), daemon=True)
    t_low.start()
    assert wait_until(lambda: s.waiting == 1)
    t_high = threading.Thread(target=waiter, args=(Priority.HIGH, "high"), daemon=True)
    t_high.start()
    assert wait_until(lambda: s.waiting == 2)

    s.release(Priority.NORMAL)
    t_low.join(2)
    t_high.join(2)
    assert order == ["high", "low"]
    assert s.active == 0


def test_fifo_within_same_priority():
    s = RequestScheduler(max_concurrent=1)
    s.acquire(Priority.NORMAL)

    order = []

    def waiter(tag):
        s.acquire(Priority.NORMAL)
        order.append(tag)
        s.release(Priority.NORMAL)

    threads = []
    for i in range(3):
        t = threading.Thread(target=waiter, args=(i,), daemon=True)
        t.start()
        assert wait_until(lambda n=i + 1: s.waiting == n)
        threads.append(t)

    s.release(Priority.NORMAL)
    for t in threads:
        t.join(2)
    assert order == [0, 1, 2]
    assert s.active == 0


# =============================================================================
# Aging / fairness
# =============================================================================


def test_aged_low_priority_overtakes_fresh_high():
    """One aging_interval of waiting is worth one priority level, so an old
    LOW (p4) outranks a HIGH (p2) that arrives more than 2 intervals later."""
    s = RequestScheduler(max_concurrent=1, aging_interval=0.05)
    s.acquire(Priority.NORMAL)  # occupy the only slot

    order = []

    def waiter(priority, tag):
        s.acquire(priority)
        order.append(tag)
        s.release(priority)

    t_low = threading.Thread(target=waiter, args=(Priority.LOW, "low"), daemon=True)
    t_low.start()
    assert wait_until(lambda: s.waiting == 1)

    time.sleep(0.15)  # LOW ages 3 intervals; parity needed only 2

    t_high = threading.Thread(target=waiter, args=(Priority.HIGH, "high"), daemon=True)
    t_high.start()
    assert wait_until(lambda: s.waiting == 2)

    s.release(Priority.NORMAL)
    t_low.join(2)
    t_high.join(2)
    assert order == ["low", "high"]
    assert s.active == 0


def test_aging_disabled_pure_priority():
    """aging_interval=None preserves strict priority order regardless of wait."""
    s = RequestScheduler(max_concurrent=1, aging_interval=None)
    s.acquire(Priority.NORMAL)

    order = []

    def waiter(priority, tag):
        s.acquire(priority)
        order.append(tag)
        s.release(priority)

    t_low = threading.Thread(target=waiter, args=(Priority.LOW, "low"), daemon=True)
    t_low.start()
    assert wait_until(lambda: s.waiting == 1)

    time.sleep(0.15)

    t_high = threading.Thread(target=waiter, args=(Priority.HIGH, "high"), daemon=True)
    t_high.start()
    assert wait_until(lambda: s.waiting == 2)

    s.release(Priority.NORMAL)
    t_low.join(2)
    t_high.join(2)
    assert order == ["high", "low"]
    assert s.active == 0


def test_p1_outranks_aged_waiters():
    """No amount of aging crosses the band boundary: a waiting p1 dispatches
    before any aged p2-p5 waiter."""
    s = RequestScheduler(max_concurrent=1, aging_interval=0.01)
    s.acquire(Priority.NORMAL)

    order = []

    def waiter(priority, tag):
        s.acquire(priority)
        order.append(tag)
        s.release(priority)

    t_low = threading.Thread(target=waiter, args=(Priority.LOW, "low"), daemon=True)
    t_low.start()
    assert wait_until(lambda: s.waiting == 1)

    time.sleep(0.1)  # LOW ages 10 intervals

    t_p1 = threading.Thread(
        target=waiter, args=(Priority.CONTENTION_RETRY, "p1"), daemon=True
    )
    t_p1.start()
    assert wait_until(lambda: s.waiting == 2)

    s.release(Priority.NORMAL)
    t_p1.join(2)
    t_low.join(2)
    assert order == ["p1", "low"]
    assert s.active == 0


def test_aged_admission_keeps_normal_semantics():
    """An aged waiter is admitted as a normal request: no p1 marker, no solo
    gating of other traffic."""
    s = RequestScheduler(max_concurrent=2, aging_interval=0.01)
    s.acquire(Priority.NORMAL)
    s.acquire(Priority.NORMAL)

    admitted = threading.Event()

    def aged_low():
        s.acquire(Priority.LOW)
        admitted.set()

    t = threading.Thread(target=aged_low, daemon=True)
    t.start()
    assert wait_until(lambda: s.waiting == 1)
    time.sleep(0.05)  # age well past HIGH parity

    s.release(Priority.NORMAL)
    assert admitted.wait(2)
    assert s._active_p1 == 0
    # Aged admission must not block other traffic the way p1 does.
    s.release(Priority.NORMAL)
    s.acquire(Priority.NORMAL)
    assert s.active == 2
    s.release(Priority.NORMAL)
    s.release(Priority.LOW)
    assert s.active == 0


# =============================================================================
# Scheduler core (async + mixed)
# =============================================================================


def test_async_acquire_release():
    async def main():
        s = RequestScheduler(max_concurrent=1)
        await s.aacquire(Priority.NORMAL)

        async def waiter():
            await s.aacquire(Priority.NORMAL)
            await s.arelease(Priority.NORMAL)
            return "done"

        task = asyncio.create_task(waiter())
        await asyncio.sleep(0.05)
        assert s.waiting == 1

        await s.arelease(Priority.NORMAL)
        assert await asyncio.wait_for(task, 2) == "done"
        assert s.active == 0

    asyncio.run(main())


def test_sync_release_wakes_async_waiter():
    async def main():
        s = RequestScheduler(max_concurrent=1)
        s.acquire(Priority.NORMAL)  # sync holder

        async def waiter():
            await s.aacquire(Priority.NORMAL)
            await s.arelease(Priority.NORMAL)
            return "done"

        task = asyncio.create_task(waiter())
        await asyncio.sleep(0.05)
        assert s.waiting == 1

        s.release(Priority.NORMAL)
        assert await asyncio.wait_for(task, 2) == "done"
        assert s.active == 0

    asyncio.run(main())


def test_async_priority_beats_sync_fifo():
    async def main():
        s = RequestScheduler(max_concurrent=1)
        s.acquire(Priority.NORMAL)

        order = []

        def sync_waiter():
            s.acquire(Priority.LOW)
            order.append("sync-low")
            s.release(Priority.LOW)

        t = threading.Thread(target=sync_waiter, daemon=True)
        t.start()
        assert wait_until(lambda: s.waiting == 1)

        async def async_waiter():
            await s.aacquire(Priority.HIGH)
            order.append("async-high")
            await s.arelease(Priority.HIGH)

        task = asyncio.create_task(async_waiter())
        await asyncio.sleep(0.05)
        assert s.waiting == 2

        s.release(Priority.NORMAL)
        await asyncio.wait_for(task, 2)
        await asyncio.get_running_loop().run_in_executor(None, t.join, 2)
        assert order == ["async-high", "sync-low"]
        assert s.active == 0

    asyncio.run(main())


# =============================================================================
# Cancellation safety (async)
# =============================================================================


def test_cancelled_async_waiter_does_not_leak_slot():
    async def main():
        s = RequestScheduler(max_concurrent=1)
        await s.aacquire(Priority.NORMAL)

        task = asyncio.create_task(s.aacquire(Priority.NORMAL))
        await asyncio.sleep(0.05)
        assert s.waiting == 1

        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

        await s.arelease(Priority.NORMAL)
        assert s.active == 0
        assert s.waiting == 0

        # The slot must be immediately reusable.
        await asyncio.wait_for(s.aacquire(Priority.NORMAL), 1)
        await s.arelease(Priority.NORMAL)

    asyncio.run(main())


def test_cancelled_p1_waiter_unblocks_lower_admissions():
    async def main():
        s = RequestScheduler(max_concurrent=2)
        await s.aacquire(Priority.NORMAL)  # keeps _active > 0 so p1 must queue

        p1_task = asyncio.create_task(s.aacquire(Priority.CONTENTION_RETRY))
        await asyncio.sleep(0.05)
        assert s.waiting == 1

        async def normal_waiter():
            await s.aacquire(Priority.NORMAL)  # blocked by the waiting p1
            await s.arelease(Priority.NORMAL)
            return "done"

        normal_task = asyncio.create_task(normal_waiter())
        await asyncio.sleep(0.05)
        assert s.waiting == 2

        p1_task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await p1_task

        # With the p1 gone, the queued NORMAL must dispatch without any release.
        assert await asyncio.wait_for(normal_task, 2) == "done"
        await s.arelease(Priority.NORMAL)
        assert s.active == 0

    asyncio.run(main())


def test_waiter_cancelled_after_dispatch_returns_slot():
    async def main():
        s = RequestScheduler(max_concurrent=1)
        await s.aacquire(Priority.NORMAL)

        task = asyncio.create_task(s.aacquire(Priority.NORMAL))
        await asyncio.sleep(0.05)
        assert s.waiting == 1

        # arelease dispatches the waiter (slot admitted) but the task has not
        # resumed yet; cancelling now hits the dispatched-but-cancelled window.
        await s.arelease(Priority.NORMAL)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

        assert s.active == 0
        await asyncio.wait_for(s.aacquire(Priority.NORMAL), 1)
        await s.arelease(Priority.NORMAL)

    asyncio.run(main())


def test_dead_loop_waiter_skipped_on_dispatch():
    s = RequestScheduler(max_concurrent=1)
    s.acquire(Priority.NORMAL)

    # A waiter whose event loop is torn down without the task being cancelled:
    # dispatch must skip it instead of raising on the closed loop.
    loop = asyncio.new_event_loop()
    task = loop.create_task(s.aacquire(Priority.NORMAL))
    task._log_destroy_pending = False  # abandoning it is the point of this test
    loop.run_until_complete(asyncio.sleep(0.01))
    assert s.waiting == 1
    loop.close()

    s.release(Priority.NORMAL)  # must not raise "Event loop is closed"
    assert s.active == 0
    assert s.waiting == 0

    s.acquire(Priority.NORMAL)  # slot must be available again
    s.release(Priority.NORMAL)
    del task


# =============================================================================
# Context-contention detection
# =============================================================================


def _contention_error() -> openai_sdk.BadRequestError:
    request = httpx.Request("POST", "http://localhost:8080/v1/chat/completions")
    response = httpx.Response(400, request=request)
    body = {
        "error": {
            "code": 400,
            "message": (
                "request (40069 tokens) exceeds the available context size "
                "(22016 tokens), try increasing it"
            ),
            "type": "exceed_context_size_error",
        }
    }
    return openai_sdk.BadRequestError(
        body["error"]["message"], response=response, body=body
    )


def _other_400_error() -> openai_sdk.BadRequestError:
    request = httpx.Request("POST", "http://localhost:8080/v1/chat/completions")
    response = httpx.Response(400, request=request)
    body = {"error": {"code": 400, "message": "invalid request", "type": "invalid_request_error"}}
    return openai_sdk.BadRequestError("invalid request", response=response, body=body)


def _pool_exhaustion_error() -> openai_sdk.InternalServerError:
    """The unified-KV contention failure observed live (llama-server
    server-context.cpp): when llama_decode can't find KV space at n_batch=1,
    the server 500s EVERY processing slot with this body and clears the
    entire context."""
    request = httpx.Request("POST", "http://localhost:8080/v1/chat/completions")
    response = httpx.Response(500, request=request)
    body = {
        "error": {
            "code": 500,
            "message": "Context size has been exceeded.",
            "type": "server_error",
        }
    }
    return openai_sdk.InternalServerError(
        body["error"]["message"], response=response, body=body
    )


def _memory_slot_error() -> openai_sdk.InternalServerError:
    """KV-pool exhaustion under concurrent load (rlm-trainer eval finding #3):
    llama-server's ``decode: failed to find a memory slot`` surfaced as a 500.
    The p1 drain-and-retry-solo recovery applies just like the context-size 500."""
    request = httpx.Request("POST", "http://localhost:8080/v1/chat/completions")
    response = httpx.Response(500, request=request)
    body = {
        "error": {
            "code": 500,
            "message": "decode: failed to find a memory slot for batch of size 1",
            "type": "server_error",
        }
    }
    return openai_sdk.InternalServerError(
        body["error"]["message"], response=response, body=body
    )


def _unrelated_500_error() -> openai_sdk.InternalServerError:
    request = httpx.Request("POST", "http://localhost:8080/v1/chat/completions")
    response = httpx.Response(500, request=request)
    body = {"error": {"code": 500, "message": "Compute error.", "type": "server_error"}}
    return openai_sdk.InternalServerError("Compute error.", response=response, body=body)


def test_is_context_contention():
    assert _is_context_contention(_contention_error())
    assert not _is_context_contention(_other_400_error())


def test_is_context_contention_matches_pool_exhaustion_500():
    assert _is_context_contention(_pool_exhaustion_error())


def test_is_context_contention_matches_memory_slot_500():
    assert _is_context_contention(_memory_slot_error())


def test_is_context_contention_rejects_unrelated_500():
    assert not _is_context_contention(_unrelated_500_error())


# =============================================================================
# OpenAIClient integration
# =============================================================================


def _ok_response(content="ok"):
    response = MagicMock()
    response.choices = [MagicMock(message=MagicMock(content=content))]
    response.usage = MagicMock(
        prompt_tokens=5, completion_tokens=5, total_tokens=10, cost=None, model_extra=None
    )
    return response


def _make_client(scheduler=None):
    with patch("prehend.clients.openai.openai.OpenAI"):
        with patch("prehend.clients.openai.openai.AsyncOpenAI"):
            return OpenAIClient(api_key="test", model_name="test-model", scheduler=scheduler)


def test_completion_contention_retries_at_p1():
    scheduler = RequestScheduler(max_concurrent=2)
    client = _make_client(scheduler)
    client.client.chat.completions.create = MagicMock(
        side_effect=[_contention_error(), _ok_response()]
    )

    with patch.object(scheduler, "acquire", wraps=scheduler.acquire) as acquire_spy:
        result = client.completion("hello")

    assert result == "ok"
    assert client.client.chat.completions.create.call_count == 2
    priorities = [call.args[0] for call in acquire_spy.call_args_list]
    assert priorities == [Priority.NORMAL, Priority.CONTENTION_RETRY]
    # No leaked slots or p1 markers after the retry path (double-release regression).
    assert scheduler.active == 0
    assert scheduler._active_p1 == 0
    assert scheduler.waiting == 0


def test_completion_contention_retry_fails_propagates():
    scheduler = RequestScheduler(max_concurrent=2)
    client = _make_client(scheduler)
    client.client.chat.completions.create = MagicMock(
        side_effect=[_contention_error(), _contention_error()]
    )

    with pytest.raises(openai_sdk.BadRequestError):
        client.completion("hello")

    # Retried exactly once: a p1 failure means the request is genuinely too large.
    assert client.client.chat.completions.create.call_count == 2
    assert scheduler.active == 0
    assert scheduler._active_p1 == 0


def test_completion_pool_exhaustion_500_retries_at_p1():
    scheduler = RequestScheduler(max_concurrent=2)
    client = _make_client(scheduler)
    client.client.chat.completions.create = MagicMock(
        side_effect=[_pool_exhaustion_error(), _ok_response()]
    )

    with patch.object(scheduler, "acquire", wraps=scheduler.acquire) as acquire_spy:
        result = client.completion("hello")

    assert result == "ok"
    assert client.client.chat.completions.create.call_count == 2
    priorities = [call.args[0] for call in acquire_spy.call_args_list]
    assert priorities == [Priority.NORMAL, Priority.CONTENTION_RETRY]
    assert scheduler.active == 0
    assert scheduler._active_p1 == 0
    assert scheduler.waiting == 0


def test_completion_pool_exhaustion_500_retry_fails_propagates():
    scheduler = RequestScheduler(max_concurrent=2)
    client = _make_client(scheduler)
    client.client.chat.completions.create = MagicMock(
        side_effect=[_pool_exhaustion_error(), _pool_exhaustion_error()]
    )

    with pytest.raises(openai_sdk.InternalServerError):
        client.completion("hello")

    assert client.client.chat.completions.create.call_count == 2
    assert scheduler.active == 0
    assert scheduler._active_p1 == 0


def test_completion_unrelated_500_no_retry():
    scheduler = RequestScheduler(max_concurrent=2)
    client = _make_client(scheduler)
    client.client.chat.completions.create = MagicMock(side_effect=_unrelated_500_error())

    with pytest.raises(openai_sdk.InternalServerError):
        client.completion("hello")

    assert client.client.chat.completions.create.call_count == 1
    assert scheduler.active == 0


def test_completion_pool_exhaustion_500_without_scheduler_no_retry():
    client = _make_client(scheduler=None)
    client.client.chat.completions.create = MagicMock(side_effect=_pool_exhaustion_error())

    with pytest.raises(openai_sdk.InternalServerError):
        client.completion("hello")

    assert client.client.chat.completions.create.call_count == 1


def test_acompletion_pool_exhaustion_500_retries_at_p1():
    async def main():
        scheduler = RequestScheduler(max_concurrent=2)
        client = _make_client(scheduler)

        calls = []

        async def fake_create(**kwargs):
            calls.append(kwargs)
            if len(calls) == 1:
                raise _pool_exhaustion_error()
            return _ok_response()

        client.async_client.chat.completions.create = fake_create

        acquired = []
        real_aacquire = scheduler.aacquire

        async def spy_aacquire(priority):
            acquired.append(priority)
            await real_aacquire(priority)

        scheduler.aacquire = spy_aacquire

        result = await client.acompletion("hello")
        assert result == "ok"
        assert len(calls) == 2
        assert acquired == [Priority.NORMAL, Priority.CONTENTION_RETRY]
        assert scheduler.active == 0
        assert scheduler._active_p1 == 0

    asyncio.run(main())


def test_completion_non_contention_400_no_retry():
    scheduler = RequestScheduler(max_concurrent=2)
    client = _make_client(scheduler)
    client.client.chat.completions.create = MagicMock(side_effect=_other_400_error())

    with pytest.raises(openai_sdk.BadRequestError):
        client.completion("hello")

    assert client.client.chat.completions.create.call_count == 1
    assert scheduler.active == 0


def test_completion_without_scheduler_unchanged():
    client = _make_client(scheduler=None)
    client.client.chat.completions.create = MagicMock(return_value=_ok_response())
    assert client.completion("hello") == "ok"

    # Without a scheduler there is no p1 exclusivity, so contention is not retried.
    client.client.chat.completions.create = MagicMock(side_effect=_contention_error())
    with pytest.raises(openai_sdk.BadRequestError):
        client.completion("hello")
    assert client.client.chat.completions.create.call_count == 1


def test_completion_priority_strings_are_resolved():
    scheduler = RequestScheduler(max_concurrent=2)
    client = _make_client(scheduler)
    client.client.chat.completions.create = MagicMock(return_value=_ok_response())

    with patch.object(scheduler, "acquire", wraps=scheduler.acquire) as acquire_spy:
        client.completion("hello", priority="high")
        client.completion("hello", priority="low")
        client.completion("hello")

    priorities = [call.args[0] for call in acquire_spy.call_args_list]
    assert priorities == [Priority.HIGH, Priority.LOW, Priority.NORMAL]
    assert scheduler.active == 0


def test_acompletion_contention_retries_at_p1():
    async def main():
        scheduler = RequestScheduler(max_concurrent=2)
        client = _make_client(scheduler)

        calls = []

        async def fake_create(**kwargs):
            calls.append(kwargs)
            if len(calls) == 1:
                raise _contention_error()
            return _ok_response()

        client.async_client.chat.completions.create = fake_create

        acquired = []
        real_aacquire = scheduler.aacquire

        async def spy_aacquire(priority):
            acquired.append(priority)
            await real_aacquire(priority)

        scheduler.aacquire = spy_aacquire

        result = await client.acompletion("hello")
        assert result == "ok"
        assert len(calls) == 2
        assert acquired == [Priority.NORMAL, Priority.CONTENTION_RETRY]
        assert scheduler.active == 0
        assert scheduler._active_p1 == 0

    asyncio.run(main())


def test_acompletion_without_scheduler_unchanged():
    async def main():
        client = _make_client(scheduler=None)

        async def fake_create(**kwargs):
            return _ok_response("async-ok")

        client.async_client.chat.completions.create = fake_create
        assert await client.acompletion("hello") == "async-ok"

    asyncio.run(main())


# =============================================================================
# LMRequest protocol round-trip
# =============================================================================


def test_lm_request_priority_round_trip():
    request = LMRequest(prompt="hi", priority="high", depth=0)
    data = request.to_dict()
    assert data["priority"] == "high"
    assert LMRequest.from_dict(data).priority == "high"

    # Omitted when unset, and absent keys round-trip to None.
    data = LMRequest(prompt="hi", depth=0).to_dict()
    assert "priority" not in data
    assert LMRequest.from_dict(data).priority is None


# =============================================================================
# LMHandler scheduler creation
# =============================================================================


def test_lm_handler_creates_and_attaches_scheduler():
    default = _make_client()
    other = _make_client()
    handler = LMHandler(
        default, other_backend_client=other, scheduler_max_concurrent=4
    )
    assert isinstance(handler.scheduler, RequestScheduler)
    # Both clients share the one scheduler so the priority queue spans all traffic.
    assert default.scheduler is handler.scheduler
    assert other.scheduler is handler.scheduler


def test_lm_handler_scheduler_aging_interval_passthrough():
    client = _make_client()
    handler = LMHandler(client, scheduler_max_concurrent=4, scheduler_aging_interval=7.5)
    assert handler.scheduler._aging_interval == 7.5

    # Default is 30s aging.
    handler = LMHandler(_make_client(), scheduler_max_concurrent=4)
    assert handler.scheduler._aging_interval == 30.0


def test_lm_handler_no_scheduler_by_default():
    client = _make_client()
    handler = LMHandler(client)
    assert handler.scheduler is None
    assert client.scheduler is None


# =============================================================================
# Cross-process gate integration (stub gate; real flock tested in
# tests/test_coordination.py)
# =============================================================================


class _StubGate:
    """Duck-typed CrossProcessGate: records calls, optionally fails enter."""

    def __init__(self, fail_enter=False):
        self.calls = []
        self.fail_enter = fail_enter
        self.aenter_started = threading.Event()
        self.block_aenter = False

    def enter(self, priority):
        self.calls.append(("enter", priority))
        if self.fail_enter:
            raise RuntimeError("gate boom")

    async def aenter(self, priority):
        self.calls.append(("aenter", priority))
        self.aenter_started.set()
        if self.block_aenter:
            await asyncio.Event().wait()  # parks forever until cancelled
        if self.fail_enter:
            raise RuntimeError("gate boom")

    def exit(self, priority):
        self.calls.append(("exit", priority))


def test_gate_enter_after_admission_exit_before_release():
    g = _StubGate()
    s = RequestScheduler(max_concurrent=2, gate=g)
    s.acquire(Priority.NORMAL)
    assert s.active == 1
    s.release(Priority.NORMAL)
    assert s.active == 0
    assert g.calls == [("enter", Priority.NORMAL), ("exit", Priority.NORMAL)]


def test_gate_failure_rolls_back_local_slot():
    g = _StubGate(fail_enter=True)
    s = RequestScheduler(max_concurrent=2, gate=g)
    with pytest.raises(RuntimeError, match="gate boom"):
        s.acquire(Priority.NORMAL)
    assert s.active == 0
    # Scheduler stays usable once the gate recovers
    g.fail_enter = False
    s.acquire(Priority.NORMAL)
    s.release(Priority.NORMAL)
    assert s.active == 0


def test_gate_failure_rollback_unblocks_waiters():
    g = _StubGate(fail_enter=True)
    s = RequestScheduler(max_concurrent=1, gate=g)
    with pytest.raises(RuntimeError):
        s.acquire(Priority.NORMAL)
    # The failed acquire must not leave a phantom active slot
    g.fail_enter = False
    admitted = threading.Event()

    def second():
        s.acquire(Priority.NORMAL)
        admitted.set()

    threading.Thread(target=second, daemon=True).start()
    assert admitted.wait(2)
    s.release(Priority.NORMAL)
    assert s.active == 0


def test_async_gate_failure_rolls_back_local_slot():
    async def main():
        g = _StubGate(fail_enter=True)
        s = RequestScheduler(max_concurrent=2, gate=g)
        with pytest.raises(RuntimeError, match="gate boom"):
            await s.aacquire(Priority.NORMAL)
        assert s.active == 0

    asyncio.run(main())


def test_cancel_during_gate_wait_rolls_back_local_slot():
    async def main():
        g = _StubGate()
        g.block_aenter = True
        s = RequestScheduler(max_concurrent=2, gate=g)
        task = asyncio.create_task(s.aacquire(Priority.NORMAL))
        await asyncio.sleep(0.05)
        assert g.aenter_started.is_set()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert s.active == 0

    asyncio.run(main())


def test_no_gate_means_no_gate_calls():
    s = RequestScheduler(max_concurrent=2)
    s.acquire(Priority.NORMAL)
    s.release(Priority.NORMAL)
    assert s.active == 0


# =============================================================================
# Coordination plumbing (LMHandler / RLM)
# =============================================================================


def _make_url_client(base_url="http://127.0.0.1:8080/v1"):
    with patch("prehend.clients.openai.openai.OpenAI"):
        with patch("prehend.clients.openai.openai.AsyncOpenAI"):
            return OpenAIClient(api_key="test", model_name="test-model", base_url=base_url)


def test_lmhandler_builds_gate_keyed_by_base_url(tmp_path):
    import hashlib

    h = LMHandler(
        _make_url_client(),
        scheduler_max_concurrent=4,
        scheduler_coordination_dir=tmp_path,
    )
    assert h.scheduler is not None
    assert h.scheduler._gate is not None
    key = hashlib.sha256(b"http://127.0.0.1:8080/v1").hexdigest()[:16]
    assert (tmp_path / f"{key}.gate").exists()
    assert (tmp_path / f"{key}.pool").exists()


def test_lmhandler_no_dir_no_gate():
    h = LMHandler(_make_url_client(), scheduler_max_concurrent=4)
    assert h.scheduler is not None
    assert h.scheduler._gate is None


def test_coordination_dir_requires_scheduler(tmp_path):
    with pytest.raises(ValueError, match="scheduler_max_concurrent"):
        LMHandler(_make_url_client(), scheduler_coordination_dir=tmp_path)


def test_openai_client_stores_base_url():
    client = _make_url_client()
    assert client.base_url == "http://127.0.0.1:8080/v1"


def test_rlm_stores_coordination_dir(tmp_path):
    from prehend.core.rlm import RLM

    rlm = RLM(
        backend_kwargs={"model_name": "m", "api_key": "x"},
        scheduler_max_concurrent=4,
        scheduler_coordination_dir=tmp_path,
    )
    assert rlm.scheduler_coordination_dir == tmp_path


# =============================================================================
# vLLM context-overflow clamp
# =============================================================================


def _ctx_overflow_error(ctx=81920, output=8192, prompt=74153) -> openai_sdk.BadRequestError:
    """vLLM's admission rejection, verbatim from vllm/renderers/params.py. The live
    shape behind the librarian's 500s (2026-07-14): the root RLM transcript grew to
    ~74k tokens and the 8192-token output reservation pushed it past max-model-len."""
    request = httpx.Request("POST", "http://localhost:8080/v1/chat/completions")
    response = httpx.Response(400, request=request)
    message = (
        f"This model's maximum context length is {ctx} tokens. However, you "
        f"requested {output} output tokens and your prompt contains {prompt} "
        f"input tokens, for a total of {prompt + output} tokens. Please reduce "
        f"the length of the input prompt or the number of requested output tokens."
    )
    body = {"error": {"code": 400, "message": message, "type": "BadRequestError"}}
    return openai_sdk.BadRequestError(message, response=response, body=body)


def test_ctx_overflow_budget_is_the_servers_own_arithmetic():
    # 81920 ctx - 74153 prompt - 256 margin
    assert _ctx_overflow_budget(_ctx_overflow_error()) == 7511


def test_ctx_overflow_budget_ignores_llama_server_contention():
    # llama-server's KV-pressure errors are a DIFFERENT failure with a different
    # fix (retry at p1). Clamping them would silently shrink a request that would
    # have fit given the pool to itself.
    assert _ctx_overflow_budget(_contention_error()) is None
    assert _ctx_overflow_budget(_pool_exhaustion_error()) is None
    assert _ctx_overflow_budget(_other_400_error()) is None


def test_ctx_overflow_budget_none_when_prompt_leaves_no_room():
    # Prompt alone all but fills the window: no useful answer fits, so the caller
    # should see the overflow rather than get a stub back.
    assert _ctx_overflow_budget(_ctx_overflow_error(prompt=81900)) is None


def test_completion_clamps_max_tokens_on_ctx_overflow():
    client = _make_client()  # no scheduler: the clamp is arithmetic, not contention
    client.client.chat.completions.create = MagicMock(
        side_effect=[_ctx_overflow_error(), _ok_response()]
    )

    assert client.completion("hello", max_tokens=8192) == "ok"
    assert client.client.chat.completions.create.call_count == 2
    retry_kwargs = client.client.chat.completions.create.call_args_list[1].kwargs
    assert retry_kwargs["max_tokens"] == 7511


def test_completion_ctx_overflow_terminates_and_propagates():
    # A server that keeps reporting the same overflow must not spin: the clamp only
    # retries while it strictly LOWERS max_tokens, so the second 400 propagates.
    client = _make_client()
    client.client.chat.completions.create = MagicMock(
        side_effect=[_ctx_overflow_error(), _ctx_overflow_error(output=7511)]
    )

    with pytest.raises(openai_sdk.BadRequestError):
        client.completion("hello", max_tokens=8192)
    assert client.client.chat.completions.create.call_count == 2
