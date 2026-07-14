"""Tests for the optional Prometheus instrumentation module."""

from __future__ import annotations

import pytest

pytest.importorskip("prometheus_client")

from prehend import metrics
from prehend.core.types import (
    CodeBlock,
    ModelUsageSummary,
    REPLResult,
    RLMChatCompletion,
    RLMIteration,
    RLMMetadata,
    UsageSummary,
)


def _sample_repl_result(input_tokens: int = 10, output_tokens: int = 20) -> REPLResult:
    usage = UsageSummary(
        model_usage_summaries={
            "test-model": ModelUsageSummary(
                total_calls=1,
                total_input_tokens=input_tokens,
                total_output_tokens=output_tokens,
            ),
        },
    )
    call = RLMChatCompletion(
        root_model="test-model",
        prompt="child prompt",
        response="child response",
        usage_summary=usage,
        execution_time=0.1,
    )
    return REPLResult(
        stdout="ok",
        stderr="",
        locals={},
        execution_time=0.1,
        rlm_calls=[call],
    )


def _read(counter, **labels) -> float:
    """Read the current value of a counter/gauge (labeled or unlabeled)."""
    if labels:
        return counter.labels(**labels)._value.get()
    return counter._value.get()


class TestClassifyProgram:
    def test_rlm_query_batched(self):
        assert metrics._classify_program("xs = rlm_query_batched(['a'])") == "rlm_query_batched"

    def test_rlm_query(self):
        assert metrics._classify_program("rlm_query('a')") == "rlm_query"

    def test_llm_query(self):
        assert metrics._classify_program("llm_query('a')") == "llm_query"

    def test_slice_when_unknown(self):
        assert metrics._classify_program("docs['001'][:1000]") == "slice"

    def test_other_when_empty(self):
        assert metrics._classify_program("") == "other"
        assert metrics._classify_program("   \n") == "other"


class TestPrometheusLogger:
    def test_log_metadata_picks_up_model(self):
        logger = metrics.PrometheusLogger()
        meta = RLMMetadata(
            root_model="kb-model",
            max_depth=3,
            max_iterations=10,
            backend="openai",
            backend_kwargs={},
            environment_type="local",
            environment_kwargs={},
        )
        logger.log_metadata(meta)
        assert logger.model_label == "kb-model"

    def test_log_iteration_records_program_and_tokens(self):
        logger = metrics.PrometheusLogger(model_label="kb-model")
        before = _read(metrics.iterations_total, program="rlm_query_batched", depth="0")
        tokens_before = _read(metrics.tokens_total, role="worker", kind="child", direction="prompt")
        iteration = RLMIteration(
            prompt="root prompt",
            response="root response",
            code_blocks=[
                CodeBlock(code="rlm_query_batched(['x','y'])", result=_sample_repl_result(7, 13)),
            ],
            iteration_time=0.5,
        )
        logger.log(iteration)
        assert _read(metrics.iterations_total, program="rlm_query_batched", depth="0") == before + 1
        assert (
            _read(metrics.tokens_total, role="worker", kind="child", direction="prompt")
            == tokens_before + 7
        )

    def test_log_iteration_swallows_internal_errors(self):
        logger = metrics.PrometheusLogger()
        before = _read(metrics.callback_failures_total)
        # An iteration with a code_blocks attribute that's not a list won't iterate.
        bad_iteration = RLMIteration(
            prompt="x", response="y", code_blocks=None  # type: ignore[arg-type]
        )
        logger.log(bad_iteration)  # must not raise
        # code_blocks=None gracefully iterates to no-op; failures counter shouldn't move.
        # Force a real error: monkey-patch attribute to trigger TypeError.
        class Boom:
            def __iter__(self):
                raise RuntimeError("boom")

        bad2 = RLMIteration(prompt="x", response="y", code_blocks=Boom())  # type: ignore[arg-type]
        logger.log(bad2)
        assert _read(metrics.callback_failures_total) >= before + 1


class TestBind:
    def test_bind_attaches_all_four_callbacks(self):
        class FakeRLM:
            on_subcall_start = None
            on_subcall_complete = None
            on_iteration_start = None
            on_iteration_complete = None
            logger = None
            backend_kwargs = {"model_name": "kb-model"}

        rlm = FakeRLM()
        metrics.bind(rlm, model_label="kb-model")
        assert callable(rlm.on_subcall_start)
        assert callable(rlm.on_subcall_complete)
        assert callable(rlm.on_iteration_start)
        assert callable(rlm.on_iteration_complete)
        assert isinstance(rlm.logger, metrics.PrometheusLogger)

    def test_bind_does_not_overwrite_user_logger(self):
        class FakeLogger:
            pass

        class FakeRLM:
            on_subcall_start = None
            on_subcall_complete = None
            on_iteration_start = None
            on_iteration_complete = None
            logger = FakeLogger()
            backend_kwargs = None

        rlm = FakeRLM()
        original_logger = rlm.logger
        metrics.bind(rlm)
        assert rlm.logger is original_logger


class TestConcurrencyTracker:
    def test_subcall_start_complete_pair(self):
        before_in_flight = metrics.concurrent_children._value.get()
        before_total = _read(
            metrics.calls_total, kind="child", model="kb-model", outcome="success"
        )
        metrics._tracker.on_subcall_start(2, "kb-model", "prompt preview")
        assert metrics.concurrent_children._value.get() == before_in_flight + 1
        metrics._tracker.on_subcall_complete(2, "kb-model", 0.42, None)
        assert metrics.concurrent_children._value.get() == before_in_flight
        assert (
            _read(metrics.calls_total, kind="child", model="kb-model", outcome="success")
            == before_total + 1
        )

    def test_subcall_complete_with_timeout(self):
        before = _read(metrics.timeouts_total, kind="child")
        metrics._tracker.on_subcall_start(1, "kb-model", "")
        metrics._tracker.on_subcall_complete(
            1, "kb-model", 1.0, "TimeoutExceededError: ran past deadline"
        )
        assert _read(metrics.timeouts_total, kind="child") >= before + 1


class TestSRLMHooks:
    """The SRLM module's optional metric emitters - no-op-safe and labelled."""

    def test_emit_route_records_repl_and_direct(self):
        from prehend.core.srlm import _emit_route

        before_repl = _read(metrics.srlm_route_total, route="repl")
        before_direct = _read(metrics.srlm_route_total, route="direct")
        _emit_route("rlm")
        _emit_route("direct")
        assert _read(metrics.srlm_route_total, route="repl") == before_repl + 1
        assert _read(metrics.srlm_route_total, route="direct") == before_direct + 1

    def test_emit_candidates_in_flight_sets_gauge(self):
        from prehend.core.srlm import _emit_candidates_in_flight

        _emit_candidates_in_flight(4)
        assert metrics.srlm_candidates_in_flight._value.get() == 4
        _emit_candidates_in_flight(0)
        assert metrics.srlm_candidates_in_flight._value.get() == 0

    def test_emit_candidate_outcome_counts_each(self):
        from prehend.core.srlm import _emit_candidate_outcome

        before_s = _read(metrics.srlm_candidates_used_total, outcome="success")
        before_e = _read(metrics.srlm_candidates_used_total, outcome="error")
        _emit_candidate_outcome("success")
        _emit_candidate_outcome("error")
        _emit_candidate_outcome("success")
        assert _read(metrics.srlm_candidates_used_total, outcome="success") == before_s + 2
        assert _read(metrics.srlm_candidates_used_total, outcome="error") == before_e + 1

    def test_emit_selection_seconds_observes_histogram(self):
        from prehend.core.srlm import _emit_selection_seconds

        before = metrics.srlm_selection_seconds._sum.get()
        _emit_selection_seconds(0.42)
        assert metrics.srlm_selection_seconds._sum.get() == pytest.approx(before + 0.42)


class TestCallScope:
    def test_records_duration_and_success(self):
        class FakeRLM:
            backend_kwargs = {"model_name": "kb-model"}

        before_total = _read(
            metrics.calls_total, kind="root", model="kb-model", outcome="success"
        )
        with metrics.CallScope(FakeRLM(), prompt="x" * 200):
            pass
        assert (
            _read(metrics.calls_total, kind="root", model="kb-model", outcome="success")
            == before_total + 1
        )

    def test_records_error_outcome(self):
        class FakeRLM:
            backend_kwargs = {"model_name": "kb-model"}

        before = _read(
            metrics.calls_total, kind="root", model="kb-model", outcome="error"
        )
        with pytest.raises(RuntimeError):
            with metrics.CallScope(FakeRLM(), prompt="x"):
                raise RuntimeError("boom")
        assert (
            _read(metrics.calls_total, kind="root", model="kb-model", outcome="error")
            >= before + 1
        )

    def test_calls_in_flight_returns_to_zero(self):
        class FakeRLM:
            backend_kwargs = {"model_name": "kb-model"}

        before = _read(metrics.calls_in_flight, kind="root")
        with metrics.CallScope(FakeRLM()):
            assert _read(metrics.calls_in_flight, kind="root") == before + 1
        assert _read(metrics.calls_in_flight, kind="root") == before


class TestPerScopeAttribution:
    """Each subcall attributes to ITS OWN root scope, at any concurrency.

    Subcalls fire on_subcall_start from ThreadPoolExecutor worker threads
    (local_repl batched queries, srlm candidates) and from broker/handler
    threads (isolated envs). The wrapper installed by CallScope.__enter__
    travels into children by reference, so firing the callback read from the
    rlm AFTER scope entry - from a foreign thread - is exactly the production
    path.
    """

    class _FakeRLM:
        backend_kwargs = {"model_name": "kb-model"}
        on_subcall_start = None
        on_subcall_complete = None
        on_iteration_start = None
        on_iteration_complete = None
        logger = None

    def test_concurrent_scopes_attribute_only_their_own_subcalls(self):
        from concurrent.futures import ThreadPoolExecutor, wait

        rlm_a, rlm_b = self._FakeRLM(), self._FakeRLM()
        metrics.bind(rlm_a)
        metrics.bind(rlm_b)
        children_before = metrics.concurrent_children._value.get()

        with metrics.CallScope(rlm_a) as scope_a, metrics.CallScope(rlm_b) as scope_b:
            # a: 5 subcalls, max depth 3; b: 2 subcalls, max depth 1.
            plan = [(rlm_a, d) for d in (1, 1, 2, 3, 1)] + [(rlm_b, d) for d in (1, 1)]
            with ThreadPoolExecutor(max_workers=4) as pool:
                futures = [
                    pool.submit(rlm.on_subcall_start, depth, "kb-model", "p")
                    for rlm, depth in plan
                ]
                wait(futures)
            assert scope_a._fanout == 5
            assert scope_a._max_depth_seen == 3
            assert scope_b._fanout == 2
            assert scope_b._max_depth_seen == 1
            # Global gauge stays global: all 7 subcalls counted once each.
            assert metrics.concurrent_children._value.get() == children_before + 7

        for rlm, depth in plan:
            metrics._tracker.on_subcall_complete(depth, "kb-model", 0.01, None)

    def test_bystander_scope_sees_no_foreign_subcalls(self):
        rlm_active, rlm_idle = self._FakeRLM(), self._FakeRLM()
        metrics.bind(rlm_active)
        metrics.bind(rlm_idle)
        with metrics.CallScope(rlm_active) as active, metrics.CallScope(rlm_idle) as idle:
            rlm_active.on_subcall_start(1, "kb-model", "p")
            assert active._fanout == 1
            assert idle._fanout == 0
            assert idle._max_depth_seen == 0
        metrics._tracker.on_subcall_complete(1, "kb-model", 0.01, None)

    def test_child_rlm_inherits_scope_wrapper_through_subcall(self):
        """The real spawn path: RLM._subcall constructs the child with
        on_subcall_start=self.on_subcall_start, so the scope wrapper follows
        the recursion regardless of which thread runs the subcall."""
        from concurrent.futures import ThreadPoolExecutor
        from unittest.mock import patch

        import prehend.core.rlm as rlm_module
        from prehend import RLM
        from tests.test_subcall import create_mock_lm, final

        captured: dict = {}
        original_rlm = rlm_module.RLM

        class CapturingRLM(original_rlm):
            def __init__(self, *args, **kwargs):
                captured.update(kwargs)
                super().__init__(*args, **kwargs)

        with patch.object(rlm_module, "get_client") as mock_get_client:
            mock_get_client.return_value = create_mock_lm([final("child answer")])
            parent = RLM(
                backend="openai",
                backend_kwargs={"model_name": "kb-model"},
                max_depth=3,
            )
            metrics.bind(parent)
            with metrics.CallScope(parent) as scope:
                with patch.object(rlm_module, "RLM", CapturingRLM):
                    with ThreadPoolExecutor(max_workers=1) as pool:
                        pool.submit(parent._subcall, "test prompt").result()
                assert captured["on_subcall_start"] is parent.on_subcall_start
                assert scope._fanout == 1
                assert scope._max_depth_seen == 1

    def test_scope_restores_previous_callback_on_exit(self):
        rlm = self._FakeRLM()
        metrics.bind(rlm)
        bound = rlm.on_subcall_start
        with metrics.CallScope(rlm):
            assert rlm.on_subcall_start is not bound
        assert rlm.on_subcall_start is bound


class TestPrometheusMemoryObserver:
    """MemoryHarness telemetry -> localai_prehend_memory_* series."""

    def test_on_retrieve_hit_records_all(self):
        obs = metrics.PrometheusMemoryObserver()
        before = _read(metrics.memory_retrieval_total, outcome="hit")
        ssum = metrics.memory_retrieve_seconds._sum.get()
        score_sum = metrics.memory_top_score._sum.get()
        obs.on_retrieve(entries=2, top_score=0.8, block_chars=120, seconds=0.01, error=False)
        assert _read(metrics.memory_retrieval_total, outcome="hit") == before + 1
        assert metrics.memory_retrieve_seconds._sum.get() == pytest.approx(ssum + 0.01)
        assert metrics.memory_top_score._sum.get() == pytest.approx(score_sum + 0.8)

    def test_on_retrieve_miss_and_error_outcomes(self):
        obs = metrics.PrometheusMemoryObserver()
        bm = _read(metrics.memory_retrieval_total, outcome="miss")
        be = _read(metrics.memory_retrieval_total, outcome="error")
        score_sum = metrics.memory_top_score._sum.get()
        obs.on_retrieve(entries=0, top_score=None, block_chars=0, seconds=0.0, error=False)
        obs.on_retrieve(entries=0, top_score=None, block_chars=0, seconds=0.0, error=True)
        assert _read(metrics.memory_retrieval_total, outcome="miss") == bm + 1
        assert _read(metrics.memory_retrieval_total, outcome="error") == be + 1
        # top_score is never observed without a hit.
        assert metrics.memory_top_score._sum.get() == score_sum

    def test_on_collect_written_sets_bank_gauge(self):
        obs = metrics.PrometheusMemoryObserver()
        bw = _read(metrics.memory_collect_total, outcome="written")
        ssum = metrics.memory_collect_seconds._sum.get()
        obs.on_collect(outcome="written", seconds=0.02, bank_size=7)
        assert _read(metrics.memory_collect_total, outcome="written") == bw + 1
        assert metrics.memory_collect_seconds._sum.get() == pytest.approx(ssum + 0.02)
        assert metrics.memory_bank_entries._value.get() == 7

    def test_on_collect_deferred_skips_latency_observe(self):
        obs = metrics.PrometheusMemoryObserver()
        bd = _read(metrics.memory_collect_total, outcome="deferred")
        ssum = metrics.memory_collect_seconds._sum.get()
        obs.on_collect(outcome="deferred", seconds=5.0, bank_size=None)
        assert _read(metrics.memory_collect_total, outcome="deferred") == bd + 1
        assert metrics.memory_collect_seconds._sum.get() == ssum  # not observed

    def test_handlers_swallow_internal_errors(self):
        obs = metrics.PrometheusMemoryObserver()
        before = _read(metrics.callback_failures_total)
        # A non-numeric bank_size makes Gauge.set() raise; must be swallowed.
        obs.on_collect(outcome="written", seconds=0.0, bank_size=object())  # type: ignore[arg-type]
        assert _read(metrics.callback_failures_total) >= before + 1
