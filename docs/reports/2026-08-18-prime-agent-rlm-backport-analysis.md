# Prime Agent RLM comparison and Prehend performance backport analysis

**Date:** 2026-08-18  
**Prehend revision:** `b7d1821`  
**Prime Agent revision:** `e7b8cae97`  
**Scope:** Compare the RLM implementations in `prehend` and `prime-agent`, then identify performance improvements that can be backported without breaking Prehend's synchronous recursive-inference API.

## Executive summary

Prehend and Prime Agent use the term **RLM** for related but materially different execution models:

- **Prehend** implements synchronous recursive inference. `rlm_query()` returns an answer, and map/reduce waits for its workers.
- **Prime Agent** implements asynchronous, durable delegation. `rlm()` returns a child-admission handle immediately; children report results later through messages or files.

A wholesale port would therefore be inappropriate. The useful backports are narrower ideas around resource ownership, concurrency, kernel startup, lifecycle, and context-window accounting.

The highest-confidence immediate wins are:

1. Remove a verified **~500 ms `LMHandler` teardown delay** paid by every root completion and every full recursive child.
2. Restore true concurrency in `llm_query_batched(..., context=...)` and `rlm_query_batched(..., context=...)`, which currently execute prompt pipelines sequentially.
3. Reuse the active handler and provider clients for recursive leaf calls instead of constructing a fresh client and HTTP pool per leaf.
4. Make scheduling and concurrency limits run-scoped across the full recursion tree.
5. Use the served model's actual context window and provider-reported input usage to trigger compaction, with one compact-and-retry attempt after overflow.

Prime Agent's detached child admission is valuable, but it should become an optional Prehend API rather than replacing the blocking `rlm_query()` contract.

## Performance framing

Prehend's optimization ledger reports that representative production asks spend approximately **94–97%** of server time decoding and only about **0.7%** in host-side orchestration. This constrains the value of most runtime micro-optimizations:

- Transport and Python bookkeeping changes alone will rarely create large end-to-end average gains.
- Changes that remove serialized model calls, avoid retries, bound runaway decoding, prevent failed near-window runs, or overlap independent inference have substantially higher upside.
- Kernel startup improvements matter only for subprocess-IPython workloads; they do not affect the default `local` environment.

The verified 500 ms handler teardown delay is an exception: it is a fixed tax paid once per root completion and once per full recursive child, regardless of inference duration.

## Architectural comparison

| Concern | Prehend | Prime Agent |
| --- | --- | --- |
| Recursive result contract | Blocking answer string | Immediate admission handle |
| Child lifetime | Fresh `RLM`, then `child.close()` | Retained `AgentSession`; optionally passivated and rehydrated |
| Concurrency | Explicit batches using threads and `asyncio.gather` | Independent detached child sessions |
| Provider ownership | Clients and `LMHandler` created per completion/child | Provider execution and lifecycle are host-owned |
| REPL | Local or IPython; fresh by default | Persistent IPython kernel by default |
| Child context | Explicit child prompt, not root transcript | `[task from parent]` plus explicit task, not parent transcript |
| Persistence | `persistent=True` preserves the environment only | Session, kernel, transcript, child registry, and artifacts persist |
| Kernel startup | Fresh subprocess kernel per subprocess `IPythonREPL` | Lazy/prewarmed startup, global boot gate, optional forkserver |
| Context handling | Heuristic token counting; compaction optional | Resolved context window, actual usage, threshold and overflow compaction |
| Lifecycle | Completion-scoped teardown | Registry, cancellation, retention, passivation, rehydration |

### Prime Agent execution seam

`packages/coding-agent/src/core/rlm-runtime.ts` is primarily a typed host interface. The actual child lifecycle is implemented in `agent-session.ts` and daemon ownership in `modes/daemon/daemon-mode.ts`.

Prime Agent deliberately separates admission from execution:

- The child is registered and its work launched in a detached asynchronous closure at `agent-session.ts:9581-9587`.
- The public call immediately returns `{rlm_child_id, name, session_dir, model}` at `agent-session.ts:9798-9803`.
- The child runs a normal, independent `AgentSession` and reports through agent messaging or files.

Prehend instead constructs a child, blocks in `child.completion()`, returns its response, and closes it at `prehend/core/rlm.py:1280-1360`.

## Findings and proposed backports

### P0: remove the fixed `LMHandler` teardown delay

#### Current behavior

Every root completion and full recursive child creates and stops an `LMHandler`:

- creation and startup: `prehend/core/rlm.py:445-480`
- shutdown: `prehend/core/rlm.py:527-535`
- server implementation: `prehend/core/lm_handler.py:363-400`

`LMHandler.start()` calls `serve_forever()` without a `poll_interval`, so Python's `socketserver.BaseServer` uses its default 0.5-second polling interval. `shutdown()` waits for that polling loop to observe the shutdown flag.

#### Native measurement

A no-op `BaseLM` and four independent `LMHandler` start/stop cycles produced:

| Cycle | Startup | Shutdown |
| ---: | ---: | ---: |
| 1 | 0.66 ms | 500.63 ms |
| 2 | 0.43 ms | 500.72 ms |
| 3 | 0.38 ms | 500.76 ms |
| 4 | 0.42 ms | 500.75 ms |

A direct `ThreadingLMServer` probe showed:

| `serve_forever` poll interval | Shutdown time |
| ---: | ---: |
| 0.5 s | 500.67 ms |
| 0.01 s | 10.15 ms |
| 0.001 s | 1.16 ms |

#### Proposed change

Start the server thread with a small explicit poll interval:

```python
self._thread = Thread(
    target=self._server.serve_forever,
    kwargs={"poll_interval": 0.01},
    daemon=True,
)
```

During `stop()`:

1. retain local references to the server and thread;
2. call `shutdown()`;
3. call `server_close()`;
4. join the thread with a bounded timeout;
5. clear object references.

#### Expected result

Approximately **490 ms saved per root completion and per full recursive child**. This is deterministic and independent of model speed. Explicit `server_close()` also avoids relying on garbage collection to release the listening socket.

A 10 ms interval is a reasonable first setting. It wakes idle handlers more frequently, so CPU use should be measured under large recursive fan-out before choosing a lower value.

### P0/P1: preserve batching when `context=` is supplied

#### Current behavior

The normal no-context paths are concurrent, but both context-bearing batch methods use sequential list comprehensions:

- `LocalREPL._llm_query_batched`: `prehend/environments/local_repl.py:530-551`
- `LocalREPL._rlm_query_batched`: `prehend/environments/local_repl.py:572-594`

By contrast, no-context recursive batching reaches `_rlm_send_batched`, whose `ThreadPoolExecutor` is bounded and order-preserving at `local_repl.py:795-856`.

Consequently, `N` small context-bearing prompts turn one concurrent batch into `N` sequential network or child-RLM calls. Large contexts can be worse because each prompt may independently run an entire map/reduce pipeline before the next prompt starts.

#### Proposed change

Implement two paths:

1. **Fast path for composed prompts that fit:** compose every prompt with the shared context, then send them through one `_send_batched` or `_rlm_send_batched` call.
2. **Oversized/map-reduce path:** execute independent `_dispatch_with_context()` pipelines concurrently under a bounded outer executor.

Preserve input ordering and per-item error isolation.

#### Required concurrency correction

When a client has no `RequestScheduler`, `LMHandler._handle_batched()` currently creates an `asyncio.Semaphore` inside each batch request at `prehend/core/lm_handler.py:171`. Multiple simultaneous batches therefore each receive the full concurrency allowance.

Move the semaphore to handler/run scope, or require the shared scheduler path, before enabling concurrent outer context pipelines. Otherwise nested batching can multiply concurrency and oversubscribe the inference server.

#### Expected result

For `N` independent prompt pipelines, wall time should move from approximately the sum of their durations toward the critical path, bounded by real server slots and decode throughput. This is the largest direct wall-time opportunity found for map/reduce-heavy workloads.

### P1: reuse the active handler for recursive leaf calls

#### Current behavior

At maximum depth, `_subcall()` creates a new provider client:

- max-depth branch: `prehend/core/rlm.py:1178-1223`
- fresh `get_client()` calls: lines 1182 and 1184

The parent completion already owns a configured and running `LMHandler` at `rlm.py:445-480`. Under batched recursive leaves, the current path can create one new sync/async SDK client and HTTP pool per leaf.

#### Proposed change

- When `_spawn_completion_context()` installs `subcall_fn` at `rlm.py:504-506`, bind the active handler in a closure or partial.
- Add an optional active-handler argument to `_subcall()`.
- Add `LMHandler.complete_subcall(...)`, used by both the socket request handler and max-depth recursion.
- Retain the fresh-client fallback for internal `_subcall()` invocations that occur outside an active completion.

#### Expected result

- one configured client per backend per run instead of per leaf;
- HTTP keepalive and pool reuse;
- fewer remote TLS handshakes;
- fewer connection failures and retries;
- lower setup latency for short leaf calls.

This is likely modest against a local vLLM endpoint, but can be material for remote providers or workloads with many short leaves.

#### Usage-accounting risk

Clients expose mutable `last_usage`. Concurrent reuse can misattribute usage. This race is already visible in batched handling, where per-item completions receive an approximate/shared usage value.

The robust fix is for client completion calls to return content and per-call usage atomically. Until then, batch aggregate usage should be exact and per-item usage explicitly documented as approximate.

### P1: create run-scoped shared services

#### Current behavior

Every full recursive child creates its own:

- provider clients;
- `LMHandler` and TCP server;
- asyncio loop thread;
- `RequestScheduler`.

This means the scheduler is shared across clients inside one handler, but not across the recursion tree. Multiple child handlers can collectively exceed the inference server's intended slot count.

#### Proposed design

Introduce a run-scoped `RLMServices` object owned by the root completion. It should own:

- provider clients;
- the async event loop;
- global scheduling/concurrency permits;
- cancellation state;
- run deadline;
- aggregate usage collection.

Child RLMs retain independent environments, message histories, verifier roots, and completion state, but borrow these services.

#### Expected result

- correct global concurrency across the recursive tree;
- connection reuse across children;
- less thread and event-loop churn;
- centralized deadline and cancellation behavior;
- a clean foundation for optional detached children.

This should follow the smaller active-handler leaf reuse rather than begin as a broad redesign.

### P1/P2: use exact context-window accounting and overflow recovery

#### Current behavior

Prehend compaction computes its trigger using a model-name lookup and token estimate:

```python
max_tokens = get_context_limit(model_name)
current_tokens = count_tokens(message_history, model_name)
```

See `prehend/core/rlm.py:972-980`.

This can disagree with the served model window. The optimization ledger already records a Qwen configuration where the inferred limit was 128,000 while the served limit was 98,304, placing the compaction threshold beyond the real failure boundary. Compaction is also not enabled by the high-level `Harness` defaults.

Prime Agent instead uses the resolved model context window, provider usage from completed turns, a reserve-token budget, and a single overflow compact-and-retry path.

#### Proposed change

- Add explicit `context_window` and `compaction_reserve_tokens` inputs to `RLM`.
- Have `Harness` pass `Runtime.ctx` when runtime probing resolves it.
- Retain the actual input-token count from each root provider response.
- Before the next turn, compact when the most recent input usage approaches `context_window - reserve_tokens`.
- On a context-overflow error:
  1. remove the failed error turn from active history;
  2. compact;
  3. retry once;
  4. surface a typed failure if the retry also overflows.

#### Expected result

This is primarily a reliability and p95 improvement:

- fewer hard context-window failures;
- fewer runs that spend most of their budget before failing;
- less conservative behavior than character-based estimates;
- predictable behavior near the served limit.

Do not port Prime Agent's full compaction subsystem. Prehend needs its accounting and recovery discipline, not its entire session machinery.

### P2: make `persistent=True` persist the LM transport

#### Current behavior

`persistent=True` preserves `_persistent_env`, but each `completion()` still recreates provider clients, the socket server, and the batch event-loop thread at `rlm.py:437-537`.

#### Proposed change

Add `_persistent_lm_handler`:

- create and start it once;
- retain a stable port;
- reset verifier root, deadline and cancellation state for every completion;
- calculate per-call usage from before/after snapshots;
- stop it in `RLM.close()`.

Concurrent `completion()` calls on one persistent instance should either be serialized or explicitly rejected because completion start time, verifier root, deadline, cancellation, and usage fields are currently mutable instance state.

#### Expected result

This helps chat-like and repeated-completion workflows, especially against remote providers. It has little value for one-shot `Harness.completion()` calls.

### P2: avoid disk serialization in `LocalREPL.add_context()`

#### Current behavior

The in-process `LocalREPL` writes context to a temporary text or JSON file, executes code to read and decode it, then executes another cell to create the `context` alias at `local_repl.py:861-908`.

Native setup measurements found approximately:

- 10 MB string: **22 ms**;
- list of 1,000,000 integers: **200 ms**.

#### Proposed change

Because `LocalREPL` is in-process, assign the payload directly:

```python
self.locals[var_name] = context_payload
self.locals["context"] = self.locals["context_0"]
```

Retain file-based loading only for subprocess or isolated environments where process boundaries require serialization.

#### Expected result

A clean setup-time win for very large contexts and repeated persistent additions. It is smaller than model latency and should follow the handler and batching fixes.

### P2/P3: subprocess-IPython startup improvements

Prime Agent contains three useful mechanisms:

1. lazy, memoized kernel startup and optional prewarming;
2. a process-wide kernel boot semaphore;
3. a Linux forkserver that forks a pristine, pre-imported template before threads, loops, or ZeroMQ sockets exist.

Prehend synchronously creates a fresh subprocess kernel in each subprocess `IPythonREPL`, and recursive child fan-out can start several simultaneously.

#### Recommended order

1. Add a process-wide semaphore around subprocess spawn and port resolution.
2. Start kernel provisioning concurrently with the first root model request and await it only before executing the first code block.
3. Add an optional Linux forkserver only after profiling shows kernel startup is material.

The forkserver must retain a direct-spawn fallback and avoid inheriting threads, event loops, locks, or ZMQ state.

These changes are irrelevant to Prehend's default `local` environment and should not precede inference-path fixes.

### P3: detached child admission as an opt-in API

Prime Agent's most visible latency feature is semantic:

```python
handle = await rlm("independent task")
```

returns after admission while the child continues in the background.

Prehend could add an experimental API such as:

```python
handle = rlm_submit(prompt)
rlm_status(handle)
result = rlm_result(handle, timeout=...)
rlm_cancel(handle)
```

This could overlap child inference with local parent work, permit straggler cancellation, and support partial fan-in. It must not replace `rlm_query()`, whose blocking answer-returning behavior is central to existing map/reduce code.

A safe implementation needs explicit ownership of:

- parent teardown and descendant cancellation;
- result retention;
- child registry state;
- handler and environment lifetime;
- exact usage attribution;
- cleanup after failed or cancelled startup.

A naive `Future` stored in the REPL is not sufficient.

## Features already present in Prehend

The following Prime-style ideas are already implemented and should not be re-ported:

- persistent asyncio loop for batched calls: `lm_handler.py:179-194`;
- bounded batch concurrency and per-item failure isolation: `lm_handler.py:143-177`;
- bounded, order-preserving recursive no-context batching: `local_repl.py:795-856`;
- subcall budgets and context-size guards;
- automatic large-context map/reduce and cached MAP partials;
- distinct child system prompts;
- child model selection;
- child thinking-mode controls;
- context isolation: children receive explicit prompts rather than the parent transcript;
- aggressive REPL stdout and generation caps.

## Backports that are not justified by performance

### Replace TCP with Jupyter comms

Not recommended. Prehend's measured host-side orchestration is already about 0.7% of ask time. Prime Agent uses Jupyter comms because it owns a persistent kernel/host bridge, not because that transport is intrinsically faster for Prehend's inference loop.

### Port the durable daemon registry and passivation system

Not recommended unless Prehend intentionally becomes a long-running agent platform. Passivation reduces resident memory but adds wake latency and significant lifecycle complexity.

### Port full session snapshots to the default LocalREPL

Not recommended. Direct in-process Python state is already persistent for the lifetime of the environment.

### Change `rlm_query()` to return a handle

Not recommended. This would break Prehend's synchronous recursive-inference contract and existing map/reduce composition.

## Implementation sequence

### PR 1: handler teardown

- set a small explicit `serve_forever` poll interval;
- call `server_close()` during teardown;
- boundedly join the server thread;
- add a regression benchmark asserting shutdown stays below an agreed threshold;
- run the full test suite and an FD/thread leak test.

### PR 2: context-bearing batch concurrency

- add the fitting composed-prompt fast path;
- add bounded parallelism for oversized independent pipelines;
- move fallback concurrency permits to handler/run scope;
- test ordering, error isolation, cancellation and maximum observed concurrency;
- benchmark with a delayed deterministic mock model.

### PR 3: active-handler leaf reuse

- add `LMHandler.complete_subcall()`;
- bind the handler into `_subcall()`;
- test client constructor counts for 1/4/16 leaf calls;
- test model routing, token caps, deadlines and error strings;
- make aggregate usage exact under concurrent calls.

### PR 4: exact context-window compaction

- thread `Runtime.ctx` into `RLM`;
- use actual root input usage and a reserve budget;
- add one overflow compact-and-retry path;
- measure overflow rate and p95 rather than only median latency.

### PR 5: run-scoped services and true persistence

- share clients, scheduler, cancellation and loop across the recursive tree;
- make `persistent=True` preserve the handler;
- add lifecycle tests for repeated completions, close, cancellation and failure during startup.

### PR 6: optional subprocess-kernel improvements

Only after profiling confirms subprocess IPython startup matters.

### Experimental follow-up: detached child API

Treat this as a new orchestration mode with its own registry and lifecycle contract, not as a transparent optimization.

## Benchmark plan

Report p50, p95, wall time, RSS, open file descriptors, thread count, TCP accepts and success rate.

### Handler lifecycle

- 100 handler start/stop cycles;
- compare default, 50 ms, 10 ms and 1 ms polling;
- measure idle CPU under 1/4/16/64 simultaneous handlers;
- verify sockets and threads are released.

### Leaf client reuse

Use a mock OpenAI-compatible server with configurable handshake and response delays:

- 1/4/16 recursive leaf calls;
- count client constructions and TCP accepts;
- measure local and simulated-remote conditions;
- validate aggregate token and cost accounting.

### Context-bearing batching

Use deterministic delayed replies:

- small fitting contexts: expect one batched send;
- oversized contexts: expect bounded parallel pipelines;
- validate wall time approaches `ceil(N / concurrency) * delay` rather than `N * delay`;
- verify the server's real concurrency never exceeds the configured global limit.

### Context overflow

- construct prompts just below, at and above the served boundary;
- validate proactive compaction before the boundary;
- validate exactly one compact-and-retry after a true overflow;
- ensure a second overflow becomes a typed failure rather than a loop.

### Persistent completion

- 100 sequential calls on one persistent instance;
- measure handler/client constructions, TCP accepts, threads and wall time;
- confirm per-call usage isolation and deadline/cancellation reset.

### Subprocess IPython

- cold and warm startup for N=1/4/16/64;
- direct spawn versus boot gate versus prewarm versus forkserver;
- measure readiness, failure rate, orphan processes and per-child namespace isolation.

## Final recommendation

Implement the changes in this order:

1. **Remove the 500 ms handler shutdown delay.**
2. **Make context-bearing batches genuinely concurrent under a global limiter.**
3. **Reuse active handler clients for recursive leaves.**
4. **Adopt exact context-window accounting and one overflow recovery.**
5. **Then consider run-scoped services, persistent handlers, and kernel startup work.**

This sequence preserves Prehend's current API, targets verified hot paths, and defers Prime Agent's much larger durable-agent lifecycle until there is a product requirement for it.
