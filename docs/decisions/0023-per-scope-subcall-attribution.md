---
status: "accepted"
date: "2026-07-14"
deciders: "potto"
consulted: "full-concurrency refactor of the librarian ask path"
---

# Per-root subcall attribution via a scope-bound callback, not broadcast

## Context and Problem Statement

`prehend.metrics` attributes subcalls to root asks so `root_fanout` and
`root_max_depth` describe what ONE ask did. The original implementation was a
broadcast: `_ConcurrencyTracker.on_subcall_start` walked a process-global set
of every open `CallScope` and bumped ALL of them. Its own docstring conceded
the bias: with concurrent asks each scope saw every subcall on the process,
and correctness rested on kb-librarian throttling `max_concurrent_asks` to 2.
The telemetry shortcut had become the only justification for an artificial
concurrency cap.

The shortcut existed because the honest fix looked like it required thread
context propagation. Subcalls fire `on_subcall_start` from
`ThreadPoolExecutor` workers (LocalREPL batched queries, SRLM candidate
pools) and, for isolated environments, from long-lived HTTP handler and
broker poller threads. A `contextvars.ContextVar` holding the current scope
does NOT propagate into any of those: pool submissions run in an empty
context unless every submit site wraps work in `copy_context().run(...)`, and
the handler/poller threads are created once at environment startup - before
any scope exists - so there is no submit moment to capture a context at.
A contextvars design cannot cleanly cover those paths at all.

## Decision

Attribution rides the object graph instead of thread context.
`CallScope.__enter__` replaces the root RLM's `on_subcall_start` with a
scope-bound wrapper (chaining to the previous, bind()-installed global
tracker callback) and `__exit__` restores it. Every descendant RLM inherits
the callback BY REFERENCE at construction - `RLM._subcall` and
`SRLM._spawn_candidate_rlm` both forward `on_subcall_start=self.on_subcall_start`
into the child - so whichever thread eventually fires it, the event lands on
exactly the root scope whose tree spawned it. No spawn site changes; the
mechanism is immune to pools, handler threads, and pollers because a Python
function reference does not care what thread calls it.

Contract: bind() before entering a CallScope (rebinding overwrites the
wrapper), and one open CallScope per RLM instance at a time. The latter was
already implied - an RLM instance holds per-call mutable state
(`_cumulative_cost`, `_completion_start_time`) and cannot run two root
completions concurrently; kb-librarian builds a fresh SRLM per ask.

The genuinely process-global series (`concurrent_children`,
`calls_in_flight`, `subcall_depth`) stay on the global tracker unchanged;
they were never wrong.

## Consequences

- `root_fanout` / `root_max_depth` are exact at any concurrency.
  `tests/test_metrics.py::TestPerScopeAttribution` proves it: concurrent
  scopes with different fanouts and depths, events fired from a shared
  thread pool, each scope sees only its own; a bystander scope sees zero.
- kb-librarian's `max_concurrent_asks=2` loses its only real dependency.
  It is raised to 6 and re-documented as a pure backpressure knob: each ask's
  scheduler admits up to `scheduler_max_concurrent=8` in-flight LM calls, so
  6 asks bound the worst case at 48 sequences against vLLM's
  `--max-num-seqs 64` - saturating throughput while leaving ~25% headroom for
  prefill bursts and ad-hoc clients on the same server, with ask 7+ queueing
  at the semaphore instead of pushing vLLM into KV-cache preemption.
- Unresolved: 6 is a capacity argument, not a measurement. Nobody has yet
  load-tested concurrent asks above 2 against the live vLLM config; the
  right value may move with `--max-num-seqs`, context sizes, or
  `scheduler_max_concurrent`, and nothing links them mechanically. Revisit
  after the first real concurrency soak.
