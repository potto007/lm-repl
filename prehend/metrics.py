"""Prometheus instrumentation for prehend.

Optional module. Importing it requires `prometheus_client` to be installed.
Wire it from a host process via:

    import prometheus_client
    from prehend import SRLM
    from prehend.metrics import bind, CallScope, start_http_server

    start_http_server(9843)
    srlm = SRLM(...)
    bind(srlm, model_label="my-model")
    with CallScope(srlm, prompt=long_text):
        result = srlm.completion(long_text, question)

The module owns module-level Prometheus objects (single source of truth).
All callback handlers swallow exceptions and bump callback_failures_total
so a metrics bug never breaks the hot path.
"""

from __future__ import annotations

import re
import threading
import time
from contextlib import AbstractContextManager

try:
    from prometheus_client import (
        REGISTRY,
        Counter,
        Gauge,
        Histogram,
    )
    from prometheus_client import (
        start_http_server as _start_http_server,
    )
except ImportError as e:
    raise ImportError(
        "prehend.metrics requires prometheus_client. Install with "
        "`pip install prometheus_client`."
    ) from e

from prehend.core.types import RLMIteration, RLMMetadata

_PREFIX = "localai_prehend_"

_CALL_BUCKETS = (0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, 30, 60, 120, 300, 600)
_ITER_BUCKETS = (0.1, 0.25, 0.5, 1, 2.5, 5, 10, 30, 60, 120, 300)
_DEPTH_BUCKETS = (0, 1, 2, 3, 4, 5, 6, 8)
_FANOUT_BUCKETS = (1, 2, 4, 8, 16, 32, 64)
_CONTEXT_BUCKETS = (1_000, 5_000, 10_000, 30_000, 60_000, 120_000, 250_000, 500_000, 1_000_000)


def _gauge(name, doc, labels=()):
    return Gauge(_PREFIX + name, doc, labels)


def _counter(name, doc, labels=()):
    return Counter(_PREFIX + name, doc, labels)


def _histogram(name, doc, buckets, labels=()):
    return Histogram(_PREFIX + name, doc, labels, buckets=buckets)


# Concurrency
calls_in_flight = _gauge("calls_in_flight", "Live RLM calls by kind", ["kind"])
calls_total = _counter(
    "calls_total", "Completed RLM calls by kind/model/outcome", ["kind", "model", "outcome"]
)
subcall_depth = _histogram(
    "subcall_depth", "Depth at which a child RLM was spawned", _DEPTH_BUCKETS
)
root_max_depth = _histogram(
    "root_max_depth", "Max depth observed during a root RLM call", _DEPTH_BUCKETS
)
root_fanout = _histogram(
    "root_fanout", "Children spawned by a root RLM call", _FANOUT_BUCKETS
)
concurrent_children = _gauge(
    "concurrent_children", "Currently running child RLM calls across all roots"
)
srlm_candidates_in_flight = _gauge(
    "srlm_candidates_in_flight", "Live SRLM candidate trajectories"
)

# Timing
call_duration_seconds = _histogram(
    "call_duration_seconds",
    "Wall-clock duration of an RLM call",
    _CALL_BUCKETS,
    ["kind", "model", "outcome"],
)
iteration_duration_seconds = _histogram(
    "iteration_duration_seconds",
    "Wall-clock duration of one RLM iteration",
    _ITER_BUCKETS,
    ["depth"],
)
srlm_selection_seconds = _histogram(
    "srlm_selection_seconds",
    "Time to cluster and select among K SRLM candidates",
    _ITER_BUCKETS,
)

# Composition
iterations_total = _counter(
    "iterations_total", "Iterations executed, tagged by emitted program type", ["program", "depth"]
)
tokens_total = _counter(
    "tokens_total",
    "Tokens consumed during RLM calls",
    ["role", "kind", "direction"],
)
context_chars = _histogram(
    "context_chars", "Prompt context size in characters at root call entry", _CONTEXT_BUCKETS
)
srlm_route_total = _counter(
    "srlm_route_total", "SRLM routing decisions (direct vs repl)", ["route"]
)
srlm_candidates_used_total = _counter(
    "srlm_candidates_used_total", "Per-candidate outcomes for SRLM runs", ["outcome"]
)

# Health
errors_total = _counter(
    "errors_total", "Errors raised inside RLM calls", ["kind", "error_type"]
)
timeouts_total = _counter(
    "timeouts_total", "Subcall timeout hits", ["kind"]
)
callback_failures_total = _counter(
    "callback_failures_total", "Metrics callback handlers that raised internally"
)

# Memory harness (prehend.memory.harness.MemoryHarness): retrieve -> inject ->
# solve -> collect/distill. Bounded cardinality; single outcome label each.
_SCORE_BUCKETS = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)
_ENTRIES_BUCKETS = (0, 1, 2, 3, 4, 5, 8)
_BLOCK_BUCKETS = (100, 500, 1_000, 2_000, 5_000, 10_000, 20_000, 50_000)

memory_retrieve_seconds = _histogram(
    "memory_retrieve_seconds",
    "Latency of MemoryHarness retrieval (embed + search)",
    _ITER_BUCKETS,
)
memory_retrieval_total = _counter(
    "memory_retrieval_total",
    "MemoryHarness retrievals by outcome",
    ["outcome"],  # hit | miss | error
)
memory_retrieved_entries = _histogram(
    "memory_retrieved_entries",
    "Experiences injected into the root prompt per answer",
    _ENTRIES_BUCKETS,
)
memory_top_score = _histogram(
    "memory_top_score",
    "Top cosine score of the retrieved set (observed on hits only)",
    _SCORE_BUCKETS,
)
memory_block_chars = _histogram(
    "memory_block_chars",
    "Injected <Memory_Block> size in characters",
    _BLOCK_BUCKETS,
)
memory_collect_seconds = _histogram(
    "memory_collect_seconds",
    "Latency of MemoryHarness distillation/collect",
    _ITER_BUCKETS,
)
memory_collect_total = _counter(
    "memory_collect_total",
    "MemoryHarness collect outcomes",
    ["outcome"],  # written | empty | duplicate | deferred | dropped | error
)
memory_bank_entries = _gauge(
    "memory_bank_entries", "Current experience count in the memory bank"
)


class PrometheusMemoryObserver:
    """Updates the localai_prehend_memory_* metrics from MemoryHarness events.

    Duck-types ``prehend.memory.harness.MemoryObserver``. Every handler swallows
    exceptions into ``callback_failures_total`` so a metrics bug never breaks a
    solve. Attach a process-wide instance with :func:`memory_observer`.
    """

    def on_retrieve(
        self, *, entries: int, top_score: float | None, block_chars: int,
        seconds: float, error: bool,
    ) -> None:
        try:
            outcome = "error" if error else ("hit" if entries else "miss")
            memory_retrieval_total.labels(outcome=outcome).inc()
            memory_retrieve_seconds.observe(seconds)
            memory_retrieved_entries.observe(entries)
            memory_block_chars.observe(block_chars)
            if top_score is not None:
                memory_top_score.observe(top_score)
        except Exception:
            callback_failures_total.inc()

    def on_collect(
        self, *, outcome: str, seconds: float, bank_size: int | None,
    ) -> None:
        try:
            memory_collect_total.labels(outcome=outcome).inc()
            # deferred/dropped are zero-duration bookkeeping, not real distills.
            if outcome not in ("deferred", "dropped"):
                memory_collect_seconds.observe(seconds)
            if bank_size is not None:
                memory_bank_entries.set(bank_size)
        except Exception:
            callback_failures_total.inc()


_memory_observer = PrometheusMemoryObserver()


def memory_observer() -> PrometheusMemoryObserver:
    """Process-wide singleton observer to pass to a MemoryHarness/factory."""
    return _memory_observer


# Program classification: closed set of strings, low cardinality.
_PROGRAM_PATTERNS = (
    ("rlm_query_batched", re.compile(r"\brlm_query_batched\s*\(")),
    ("rlm_query", re.compile(r"\brlm_query\s*\(")),
    ("llm_query_batched", re.compile(r"\bllm_query_batched\s*\(")),
    ("llm_query", re.compile(r"\bllm_query\s*\(")),
)


def _classify_program(code: str) -> str:
    for name, pat in _PROGRAM_PATTERNS:
        if pat.search(code):
            return name
    # If there's any code at all, call it slicing/inspection.
    if code and code.strip():
        return "slice"
    return "other"


def _safe(fn):
    """Wrap a callback so a bug here never propagates into prehend."""

    def wrapped(*args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except Exception:
            callback_failures_total.inc()

    return wrapped


class PrometheusLogger:
    """prehend logger that updates Prometheus counters per iteration.

    Duck-types the surface prehend's core uses on a logger object:
    log_metadata / log / clear_iterations / get_trajectory / log_dir. Captures
    tokens and program-type composition. Wraps every update so the metrics
    layer cannot raise into the RLM loop.
    """

    log_dir: str | None = None  # prehend reads this on SRLM candidate spawn.

    def __init__(self, model_label: str | None = None) -> None:
        self.model_label = model_label or "unknown"

    def log_metadata(self, metadata: RLMMetadata) -> None:
        if not self.model_label or self.model_label == "unknown":
            self.model_label = metadata.root_model or "unknown"

    def log(self, iteration: RLMIteration) -> None:
        try:
            self._record(iteration)
        except Exception:
            callback_failures_total.inc()

    def clear_iterations(self) -> None:  # no-op: counters span all calls
        pass

    def get_trajectory(self) -> dict | None:  # no in-memory capture
        return None

    def _record(self, iteration: RLMIteration) -> None:
        # Token usage from this iteration's LLM calls.
        for block in iteration.code_blocks or []:
            program = _classify_program(block.code or "")
            iterations_total.labels(program=program, depth="0").inc()
            for call in (block.result.rlm_calls if hasattr(block.result, "rlm_calls") else []) or []:
                usage = call.usage_summary
                if usage is None:
                    continue
                for _model_name, summary in usage.model_usage_summaries.items():
                    tokens_total.labels(
                        role="worker", kind="child", direction="prompt"
                    ).inc(summary.total_input_tokens or 0)
                    tokens_total.labels(
                        role="worker", kind="child", direction="completion"
                    ).inc(summary.total_output_tokens or 0)
        # Root-level response counts as orchestrator output. Token counts at
        # the iteration level aren't directly exposed; we approximate via the
        # final aggregated UsageSummary which the caller can attribute via
        # CallScope.record_usage if desired.


class _ConcurrencyTracker:
    """Per-process running totals of child RLMs in flight.

    Process-global gauges only. Per-root attribution (root_fanout,
    root_max_depth) lives in CallScope, which hooks its root RLM's
    on_subcall_start so every subcall lands on its own scope at any
    concurrency (see CallScope.__enter__).
    """

    def on_subcall_start(self, depth: int, model: str, prompt_preview: str) -> None:
        concurrent_children.inc()
        calls_in_flight.labels(kind="child").inc()
        subcall_depth.observe(depth)

    def on_subcall_complete(
        self, depth: int, model: str, duration: float, error_msg: str | None
    ) -> None:
        concurrent_children.dec()
        calls_in_flight.labels(kind="child").dec()
        outcome = _outcome_from_error(error_msg)
        calls_total.labels(kind="child", model=model or "unknown", outcome=outcome).inc()
        call_duration_seconds.labels(
            kind="child", model=model or "unknown", outcome=outcome
        ).observe(duration)
        if outcome == "error":
            errors_total.labels(kind="child", error_type=_error_class(error_msg)).inc()
        elif outcome == "timeout":
            timeouts_total.labels(kind="child").inc()


def _outcome_from_error(error_msg: str | None) -> str:
    if not error_msg:
        return "success"
    lowered = error_msg.lower()
    if "timeout" in lowered or "timeoutexceeded" in lowered:
        return "timeout"
    return "error"


def _error_class(error_msg: str | None) -> str:
    if not error_msg:
        return "none"
    # The Exception class name is usually the first token before ": "; we
    # truncate aggressively to bound cardinality.
    head = error_msg.split(":", 1)[0].strip()
    head = re.sub(r"[^A-Za-z0-9_]", "", head)[:48]
    return head or "Exception"


_tracker = _ConcurrencyTracker()


def bind(rlm, *, model_label: str | None = None) -> None:
    """Attach Prometheus callbacks to an RLM/SRLM instance.

    Idempotent in the sense that re-binding overwrites prior callbacks - so
    bind BEFORE entering a CallScope, or the scope's attribution wrapper is
    clobbered. The
    binding installs a PrometheusLogger if `rlm.logger` is None; if the caller
    has already configured a logger, it is left alone (root iteration metrics
    are still produced via the on_iteration_complete callback).
    """
    rlm.on_subcall_start = _safe(_tracker.on_subcall_start)
    rlm.on_subcall_complete = _safe(_tracker.on_subcall_complete)
    rlm.on_iteration_start = _safe(_on_iteration_start)
    rlm.on_iteration_complete = _safe(_on_iteration_complete)
    if getattr(rlm, "logger", None) is None:
        rlm.logger = PrometheusLogger(model_label=model_label)


def _on_iteration_start(depth: int, iteration_num: int) -> None:
    pass


def _on_iteration_complete(depth: int, iteration_num: int, duration: float) -> None:
    iteration_duration_seconds.labels(depth=str(depth)).observe(duration)


class CallScope(AbstractContextManager):
    """Wrap a root RLM call to capture per-call metrics.

    Tracks calls_in_flight at the root kind, call_duration_seconds, context
    size, root_max_depth and root_fanout. Per-root max-depth/fanout are exact
    at any concurrency: __enter__ replaces the root RLM's on_subcall_start
    with a scope-bound wrapper (chaining to the bind()-installed global
    tracker), and every descendant RLM inherits that wrapper BY REFERENCE at
    construction (rlm.py _subcall, srlm.py _spawn_candidate_rlm). Attribution
    therefore rides the object graph, not thread context - it survives the
    LocalREPL/SRLM thread pools and the isolated-env broker/handler threads,
    which a contextvars approach cannot reach (those threads outlive and
    predate any scope). Enter the scope AFTER bind(), one scope per RLM
    instance at a time (an RLM cannot run two root completions concurrently
    anyway). The model label is read from `rlm.backend_kwargs["model_name"]`
    if not explicitly provided.
    """

    def __init__(self, rlm, *, prompt: str | None = None, model_label: str | None = None) -> None:
        self._rlm = rlm
        if model_label is None:
            kwargs = getattr(rlm, "backend_kwargs", None) or {}
            model_label = kwargs.get("model_name") or "unknown"
        self._model = model_label
        self._prompt = prompt
        self._start: float = 0.0
        self._outcome = "success"
        self._error_class = "none"
        self._max_depth_seen = 0
        self._fanout = 0
        self._attr_lock = threading.Lock()
        self._prev_on_subcall_start = None

    def __enter__(self) -> CallScope:
        calls_in_flight.labels(kind="root").inc()
        if self._prompt is not None:
            try:
                context_chars.observe(len(self._prompt))
            except Exception:
                callback_failures_total.inc()
        prev = getattr(self._rlm, "on_subcall_start", None)
        self._prev_on_subcall_start = prev

        def _on_subcall_start(depth: int, model: str, prompt_preview: str) -> None:
            try:
                with self._attr_lock:
                    if depth > self._max_depth_seen:
                        self._max_depth_seen = depth
                    self._fanout += 1
            except Exception:
                callback_failures_total.inc()
            if prev is not None:
                prev(depth, model, prompt_preview)

        self._rlm.on_subcall_start = _on_subcall_start
        self._start = time.perf_counter()
        return self

    def record_route(self, route: str) -> None:
        if route in ("direct", "repl"):
            srlm_route_total.labels(route=route).inc()

    def record_srlm_candidates(self, *, started: int) -> None:
        srlm_candidates_in_flight.set(started)

    def __exit__(self, exc_type, exc, tb) -> None:
        duration = time.perf_counter() - self._start
        if exc is not None:
            self._outcome = (
                "timeout" if exc_type and "Timeout" in exc_type.__name__ else "error"
            )
            self._error_class = (exc_type.__name__ if exc_type else "Exception")[:48]
        self._rlm.on_subcall_start = self._prev_on_subcall_start
        calls_in_flight.labels(kind="root").dec()
        srlm_candidates_in_flight.set(0)
        calls_total.labels(kind="root", model=self._model, outcome=self._outcome).inc()
        call_duration_seconds.labels(
            kind="root", model=self._model, outcome=self._outcome
        ).observe(duration)
        root_max_depth.observe(self._max_depth_seen)
        root_fanout.observe(self._fanout)
        if self._outcome == "error":
            errors_total.labels(kind="root", error_type=self._error_class).inc()
        elif self._outcome == "timeout":
            timeouts_total.labels(kind="root").inc()
        return None  # don't suppress


def start_http_server(port: int = 9843, addr: str = "127.0.0.1") -> None:
    """Expose `/metrics` on the given port. Wraps prometheus_client."""
    _start_http_server(port, addr=addr)


__all__ = [
    "CallScope",
    "PrometheusLogger",
    "bind",
    "start_http_server",
    "calls_in_flight",
    "calls_total",
    "subcall_depth",
    "root_max_depth",
    "root_fanout",
    "concurrent_children",
    "srlm_candidates_in_flight",
    "call_duration_seconds",
    "iteration_duration_seconds",
    "srlm_selection_seconds",
    "iterations_total",
    "tokens_total",
    "context_chars",
    "srlm_route_total",
    "srlm_candidates_used_total",
    "errors_total",
    "timeouts_total",
    "callback_failures_total",
    "PrometheusMemoryObserver",
    "memory_observer",
    "memory_retrieve_seconds",
    "memory_retrieval_total",
    "memory_retrieved_entries",
    "memory_top_score",
    "memory_block_chars",
    "memory_collect_seconds",
    "memory_collect_total",
    "memory_bank_entries",
    "REGISTRY",
]
