# Prehend optimization runbook

Living ledger of optimization experiments on the prehend harness and the vLLM server it
drives. One row per experiment. An experiment that was run and **rolled back** is as
valuable as one that was kept, and is recorded with the same care - the point of this file
is that nobody re-runs a dead end.

Serving-side decisions live in local-ai `docs/decisions/` (ADR-0010, ADR-0011) and in
`local-ai/scripts/localai-vllm.service`. Harness decisions live in `docs/decisions/`.

## Ground rules

1. **State the instrument before the experiment.** Pick the measurement that responds to
   the mechanism you are claiming. Most failed reasoning here comes from measuring the
   wrong quantity.
2. **Record the rollback.** `kept` / `rolled-back` / `inconclusive` is a required field.
3. **One change at a time**, or accept that you have measured the bundle, not the parts.
4. Never point an inference server's output at only a private log. See local-ai CLAUDE.md.

## Instruments, and what each can actually resolve

| Instrument | Spread | Resolves | Do not use for |
| --- | --- | --- | --- |
| End-to-end ask latency (`kb_ask_eval.py`) | **heavy-tailed**: 2026-07-10 baseline over 18 rows was min 10s / median 22s / mean 50s / **max 421s** | Gross regressions (>2x on the median) | Anything under ~30%. The tail is intrinsic - a subset of asks fall into runaway REPL loops. A mean computed over 18 rows is a measurement of the tail, not of the change. |
| `ground_cited` rate | binomial, n=3/ask is nothing | Large quality shifts | Small shifts. 12/14 vs 17/18 is not a signal. |
| vLLM `vllm:prefix_cache_hits_total` / `vllm:prefix_cache_queries_total` (`/metrics`; also logged per 10s as "Prefix cache hit rate") | low | **Any claim about prefix reuse** - this is the direct observable | Decode speed |
| Fixed prompt, `ignore_eos`, exactly 512 decoded tokens | 0.3-0.9% | Kernel / decode-path A/B | Realistic mixed load |
| `vllm bench serve --dataset-name random` | ~10% | Realistic mixed-load throughput | Comparing kernels |

The last two rows and their noise figures are established in local-ai ADR-0011 ("Two noise
regimes"). Do not import one harness's noise floor into another.

Both counters and the endpoints below were confirmed live on 2026-07-10:

```bash
# prefix reuse, cumulative since server start (68.2% at 02:20)
curl -s http://127.0.0.1:8080/metrics | grep -E '^vllm:prefix_cache_(hits|queries)_total'
# exact prompt token count, from the server's own tokenizer - no local tokenizer needed
curl -s -X POST http://127.0.0.1:8080/tokenize \
  -H 'Content-Type: application/json' -d '{"model":"qwen3.6-35b-a3b","prompt":"..."}'
```

## Known defects in the harness (found, not yet fixed)

- **D1 - the root REPL transcript is unbounded and collides with `max_model_len`.**
  The root call requests a fixed output budget (observed: 8192 tokens; cf.
  `prehend/core/lm_handler.py:31` `DEFAULT_MAX_DECODE_TOKENS = 8192`) regardless of how
  large the accumulated transcript has grown. On 2026-07-10 a 14-message root transcript
  reached 90,113 input tokens; `90113 + 8192 = 98305 > 98304` (`max_model_len`) and vLLM
  returned a hard 400, which the librarian surfaced as a 500 (`infra_fail`).
  This is **ceiling-independent**: the identical failure appears in `/tmp/kb-librarian.log`
  from the era when `max_model_len` was 65536 (`... at least 57345 input tokens`). Raising
  the ceiling from 65536 to 98304 moved the wall; it did not remove it. Fix is to clamp the
  requested output budget to `max_model_len - prompt_tokens - margin`, and/or to bound
  transcript growth.

  **The server's contract, MEASURED directly on 2026-07-10** by feeding exact token-id
  prompts to `/v1/completions`: `prompt_tokens + max_tokens <= max_model_len` is accepted;
  exactly one token over is rejected. There is no rounding and no grace.

  | prompt tokens | max_tokens | sum | result |
  | --- | --- | --- | --- |
  | 97,000 | 1,304 | 98,304 | HTTP 200, decoded all 1,304 |
  | 97,000 | 1,305 | 98,305 | HTTP 400 |

  So the acceptance test for any D1 fix is: with a prompt near the ceiling, the harness must
  return a short answer, never a 400. Reproduce the boundary with `/tokenize` to build an
  exact-length prompt, then `/v1/completions` with `prompt` as the token-id array - this is
  deterministic and takes ~15s, versus replaying an ask that only fails 1 rep in 3.
- **D2 - runaway REPL loops correlate with uncited answers.** The pathological reps are
  slow *and* wrong: baseline `ask2` = `[56s, 26s, 421s]` with `ground_cited=[T, T, F]` -
  the 421s rep is the uncited one. Same shape post-change (455s rep, uncited). Whatever
  makes the loop run long also makes it stop grounding. Worth attacking directly; it is
  probably the single largest quality lever in the harness.
- **D3 - the two system prompts contradict each other on how to pass documents.**
  Prehend's REPL system prompt says *"do NOT paste text into the `prompt` string -- just
  pass `context=` and let it map-reduce"*. The librarian's orchestrator prompt
  (`knowledge-base/librarian/ask.py`) says *"You MUST PASTE THE DOCUMENT TEXT INTO EACH
  PROMPT"*. Both are live in the same request. Unresolved; see EXP-000.

## Where the time actually goes (MEASURED 2026-07-10)

Per-ask `/metrics` deltas against the live server. This is the budget every optimization
must be judged against:

| | ask A (91.6s wall) | ask B (198.5s wall) |
| --- | --- | --- |
| vLLM requests issued | 23 | 44 |
| prompt tokens | 240,340 | - |
| **generated tokens** | **18,188** | **55,654** |
| prefix cache hit rate | 66.8% | 83.2% |
| server prefill time | 4.95s (**5.4%**) | 11.26s (**3.3%**) |
| server decode time | 85.91s (**94.4%**) | 327.98s (**96.7%**) |
| host-side orchestration | 0.61s (0.7%) | - |

(Summed request time exceeds wall on ask B because sub-calls run concurrently,
`max_concurrent_subcalls=4`. The prefill:decode *ratio* is what matters and it is stable.)

**Decode dominates. Token generation volume is the cost.** Three consequences:

1. Prefill-side serving flags attack ~5% of the budget. `--max-num-batched-tokens 8192` cut
   fresh prefill by 30%, which is at most ~1.6% end-to-end. That is why the tuned probe in
   EXP-000 showed no end-to-end win, and it is the correct, expected result - not a defect.
2. **local-ai ADR-0011's first Decision Driver is wrong.** It states "Fresh prefill of long
   prompts, not decode throughput, dominates ask latency here." Measured end-to-end, prefill
   is 3-5% and decode is 94-97%. The ADR's *decisions* mostly survive (they were argued on
   microbenchmarks that are still valid measurements of what they measured), but its
   **FlashInfer rejection rests on the false premise** and must be re-derived. Applying the
   ADR's own per-kernel figures (FA2 decode 175.9 tok/s, fresh 56k prefill 4.26s; FlashInfer
   184.5 tok/s, 6.24s) to ask A's real budget:

   - prefill: `4.95s x (6.24/4.26) = 7.25s`, i.e. **+2.30s**
   - decode: `85.91s x (175.9/184.5) = 81.91s`, i.e. **-4.00s**
   - net: **-1.70s on 90.86s = -1.9%**, a *win* for FlashInfer, opposite in sign to ADR-0011.

   This is an extrapolation, not a measurement, and it is fragile in one specific way: it
   assumes FlashInfer's +46% prefill penalty scales proportionally. It was measured on a
   single fresh 56k prompt, whereas ask A prefills across 23 requests at 66.8% cache hit,
   so its fresh chunks are ~3.5k tokens each. The penalty at 3.5k is unmeasured and could
   be larger or smaller in relative terms. **Do not act on this without an end-to-end A/B**,
   and note that 1.9% sits near the end-to-end noise floor, so the A/B needs the fixed-prompt
   harness or many reps. ADRs are immutable: this needs a superseding ADR, not an edit.
3. The biggest lever available is **generating fewer tokens**, and after that, decoding them
   faster (MTP gave +18% output tok/s at concurrency 1, currently blocked on vllm#47861).

## Ledger

| ID | Date | Hypothesis | Change | Instrument | Result | Verdict |
| --- | --- | --- | --- | --- | --- | --- |
| EXP-000 | 2026-07-10 | Putting `{docs[id]}` first in the Map sub-call prompt lets successive sub-calls share a cached prefix | Reorder Map prompt template in `knowledge-base/librarian/ask.py` so doc text precedes the instruction | End-to-end 6-ask x 3-rep probe (**wrong instrument**) | Wash-to-worse, tail-dominated: median 22.0s -> 25.6s, mean 50.1s -> 69.8s, max 421s -> 455s, `ground_cited` 17/18 -> 16/18, plus 1 `infra_fail` traced to **D1**, not to the change. ask3/ask4 improved, ask1/ask2 got tail-hit. Prefix-cache hit rate - the quantity the hypothesis is actually about - **was never measured**. | **rolled-back** |

### EXP-000 notes

Confound worth naming: this probe ran with the reorder **and** a batch of new serving flags
(`--max-num-batched-tokens 8192`, `--prefix-caching-hash-algo xxhash`, `--stream-interval 8`,
`VLLM_USE_FASTOKENS=1`) live at once. Those four cannot change sampled text - they are
logit-identical - so the reorder is the only behavioural variable. But it also means this
probe is **not** evidence that the serving flags improved end-to-end ask latency. They did
not, measurably. Their win (-30% on a fresh 56,612-token prefill) is a microbenchmark
result and is recorded as such in local-ai ADR-0011. Do not let the two claims merge.

The reorder is theoretically sound and remains untested. Its mechanism is prefix reuse, so
it must be measured with the prefix-cache hit rate, not with end-to-end latency, which this
probe showed is dominated by a heavy tail the change does not touch. Reverted to restore a
known-good prod prompt before starting a campaign of further changes - an unproven diff in
the working tree is how confounds get baked into everything downstream.

Re-test design: hold the corpus and ask fixed, issue the same ask twice, and read
`vllm:prefix_cache_hits_total / vllm:prefix_cache_queries_total` from `/metrics` before and
after. Note the reuse quantum is **1056 tokens** (the derived attention block size - see
local-ai `docs/qwen36-vllm-0.24-reference.md`), so a prefix shorter than 1056 tokens yields
exactly zero reuse. The old template's shared prefix was ~15 tokens, i.e. structurally
incapable of any reuse at all. That, not the latency, is the argument for the reorder - and
it is checkable in one command.
