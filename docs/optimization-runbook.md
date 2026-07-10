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

**What `ground_cited` actually means**, because the lead misread it for most of 2026-07-10.
`rlm-trainer/scripts/kb_ask_eval.py:105`:

```python
ground_cited = completed and len(claimed) > 0 and len(from_training) == 0
```

It is **not** a gold-document match. `gold_ids` are not consulted. It means: the run completed,
the answer cited at least one id, and **none** of the cited ids were ungrounded - `from_training`
being ids the model claimed without ever retrieving or opening them. An answer citing four
documents, none of them the gold one, scores `True` provided all four were actually retrieved.
It measures *"did the model invent a citation"*, not *"did the model find the right document"*.
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

## Discipline note: ask the gold question, verbatim

On 2026-07-10 the lead ran a variance probe with the question *"Within how many calendar
days must a Return be filed?"* The eval set's actual question is *"Within how many calendar
days must a Return Edit be resolved"* (`gold_ids: ["007"]`, `expected_answer: "5 calendar
days"`). The paraphrase is ambiguous against this corpus, whose relevant passage concerns
resolving Return *Edits*. Two reps of the paraphrase cited different documents (`[007]` and
`[140]`), and **neither could be scored**, because a citation can only be called wrong
relative to a gold id for the question that was actually asked.

Copy questions verbatim from `rlm-trainer/logs/eval_asks_*.json`. A paraphrased probe
measures latency and variance fine, but it cannot measure grounding.

## Known defects in the harness (found, not yet fixed)

- **D1 - the root REPL transcript is unbounded and collides with `max_model_len`.**
  The root call requests a fixed output budget of 8192 tokens regardless of how large the
  accumulated transcript has grown. This is **deliberately configured**, not a leaked
  default: `knowledge-base/librarian/config.py:127` sets `root_max_tokens: int = 8192` and
  `ask.py` passes it explicitly to `SRLM(...)`. (Sub-calls get 2048, `config.py:73`. Note
  that is the *inverse* of what `prehend/core/lm_handler.py:23-27` claims in its comment -
  "sub-calls ... keep the 8192 headroom". One of the two is stale; the librarian's config
  is what runs.) On 2026-07-10 a 14-message root transcript reached 90,113 input tokens;
  `90113 + 8192 = 98305 > 98304` (`max_model_len`) and vLLM returned a hard 400, which the
  librarian surfaced as a 500 (`infra_fail`).
  This is **ceiling-independent**: the identical failure appears in `/tmp/kb-librarian.log`
  from the era when `max_model_len` was 65536 (`... at least 57345 input tokens`). Raising
  the ceiling from 65536 to 98304 moved the wall; it did not remove it. Fix is to clamp the
  requested output budget to `max_model_len - prompt_tokens - margin`, and/or to bound
  transcript growth.

  **Status: diagnosed, patch written and tested, NOT applied.** A ready diff lives at
  `scratchpad/d1-clamp.patch` (933 passed / 8 skipped, vs 926/8 baseline; 7 new tests). Held
  back because it adds a blocking `/tokenize` probe to every LM call (~15k calls in the log
  window, 2-11 ms each), and because the 400 has not recurred since the repeat-guard was
  restored (0 infra failures across EXP-002 and EXP-003). Land it deliberately, with a
  profile, not at 5am.

  Key facts for whoever lands it, all MEASURED:

  - **There is no choke point before the client.** Seven call sites reach it directly. But
    `max_tokens` enters the request body in exactly two places: `clients/openai.py:301`
    (`completion`) and `:450` (`acompletion`). Clamping there covers root and sub-call alike.
  - **`prompt_tokens` is not cheaply available.** `tiktoken` is deliberately bypassed for
    `qwen` (`token_utils.py:38`) because cl100k under-counts denser tokenizers; the
    `CONSERVATIVE_CHARS_PER_TOKEN = 2.0` fallback over-counts by ~85% (a transcript vLLM
    tokenizes at 90,113 estimates at 166,709). Clamping on that estimate would truncate a
    genuine 50k-token prompt to ~3.7k output tokens with 48k free.
  - **`/tokenize` answers both questions in one call.** VERIFIED live:
    `POST /tokenize {"model":..., "messages":[...]}` -> `{"count": 12, "max_model_len": 98304}`.
    No hardcoded 98304, no regex on the 400 message. Nothing in `prehend/` currently reads
    `max_model_len` from anywhere; the string `98304` appears nowhere in the package.
  - The clamp does **not** stop transcript growth, and `prompt_tokens >= max_model_len` still
    fails - as a typed `TokenLimitExceededError` rather than an opaque vendor 400. `ask.py`
    only catches `TimeoutExceededError`, so it would still surface as a librarian 500.

- **D6 - compaction's context limit is wrong for this model, and points past the cliff.**
  VERIFIED: `get_context_limit("qwen3.6-35b-a3b")` returns **128,000**, because the `"qwen3"`
  key matches as a substring. The served window is **98,304**. `rlm.py`'s compaction threshold
  of `0.85 x limit = 108,800` therefore sits *beyond* the wall it exists to avoid, so it could
  never fire in time even if `compaction=True` were passed - and the librarian does not pass it.

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
- **D2 - SOLVED (diagnosis). It is a verbatim repetition collapse, not exploration, and
  the uncited answer has a separate deterministic cause.** Established by the loop-forensics
  agent from the transcripts, with the key code sites re-verified by the lead.

  *The stall.* A root turn emits **zero executable code blocks** and instead rambles,
  repeating itself verbatim. Distinct-8gram ratio of the trailing assistant message collapses
  from 1.000 to 0.319-0.490 on slow runs, versus 0.924-0.936 on fast ones. One line appeared
  **343 times verbatim**. Because `prehend/utils/parsing.py:15` matches only ` ```repl `
  fences, a turn whose code sits in a ` ```python ` fence executes nothing - and
  `format_iteration` emits one user message *per code block*, so zero code blocks means zero
  user messages, the history keeps ending on `assistant`, and the next turn **merges into the
  same message**, growing it in place to 120k-192k chars. That growth is what eventually
  trips D1's 400.

  *This is not "more exploration."* The 421s run made **2** sub-calls; the fast controls made
  **4**. Slow runs explore *less*. Root generation dominates: ~57,321 root tokens on the 421s
  run against ~2,199 on a 16.4s control, a **26x** ratio, in a handful of enormous turns that
  each run to the 8192 cap.

  *The uncited answer.* `librarian/config.py:56` sets `max_iterations: int = 20`. On
  exhaustion `prehend/core/rlm.py:808` calls `_default_answer`, which appends a forcing
  sentence, generates once, and returns - **without ever calling `answer_verifier`**. The
  in-loop citation guard (`rlm.py:726-738`) is structurally bypassed on precisely the path a
  runaway loop leads to. Cross-tab over 51 completed runs: of 3 ungrounded answers, **3 of 3**
  came through `_default_answer`; of the 47 runs that never reached it, **0** were ungrounded.
  A 375.0s run that recovered before iteration 20 returned grounded with three citations.

  **So "slow implies uncited" is false. "Exhausting the iteration budget implies uncited" is
  true, and it is deterministic.**

  Two aggravators: `_default_answer` (`rlm.py:1057`) appends its forcing sentence with
  `role: "assistant"`, so during a stall it merges into the model's own ramble and the model
  is asked to continue its ramble with the instruction attributed to itself. And
  `_check_iteration_limits` only increments `_consecutive_errors` when a code block wrote to
  stderr; a turn with **zero** code blocks takes the `else` branch and **resets the counter**,
  so `max_errors` can never fire on a stall.

- **D5 - FIXED. The repeat-guard was silently killed by the migration to vLLM.**
  `prehend/clients/openai.py:380` read `if self._repeat_guard_threshold and not parts:`. The
  guard therefore only ran while the stream carried `reasoning_content` and no `content`.

  The librarian enables it (`.env.librarian`: `KB_REPEAT_GUARD_THRESHOLD=0.35`,
  `KB_REPEAT_GUARD_ABORT_LIMIT=4`), and it used to work. MEASURED, dates of the 103
  `repeat-guard: aborting` lines in `/tmp/kb-librarian.log`:

  | date | aborts |
  | --- | --- |
  | 2026-07-05 | 12 |
  | 2026-07-06 | 20 |
  | 2026-07-07 | 3 |
  | 2026-07-08 | **68** |
  | 2026-07-09, 07-10 (vLLM era) | **0** |

  The vLLM server runs with `reasoning_parser=''` (confirmed in `/tmp/vllm-server.log`), so it
  never emits `reasoning_content` - MEASURED: the string appears **0** times in the entire
  librarian log. `parts` is non-empty from the first token, `not parts` is never true, and the
  guard has been dead since the migration. It was firing 68 times on 2026-07-08 alone.

  **This is why the rambles run unchecked now and did not the week before.** The previous
  server (llama.cpp `--jinja`) routed thought tokens to `reasoning_content`; vLLM without a
  reasoning parser routes everything to `content`. Nothing about the model or the prompts
  changed - the guard's input stream did.

  Fixed in `a40bfa9`: the guard now watches whichever stream the generation is producing, and
  judges `content` only past `_REPEAT_GUARD_CONTENT_MIN_CHARS = 4000` (the largest healthy root
  turn observed was 2,575 chars). Four regression tests in `tests/test_repeat_guard_content.py`,
  including one asserting the original reasoning-only behaviour still works.

  **Lesson for the ledger:** a guard with no positive signal is indistinguishable from a guard
  that is working. This one went dark for two days and nothing alerted. Any future guard should
  emit a heartbeat, or a metric, or something that goes to zero loudly.
- **D3 - CORRECTED. The two prompts contradict each other but never meet.** The original
  framing (that both are live in the same request, and one overrides the other) is **wrong**.
  prehend's REPL prompt is the system message. The librarian's briefing is *not a message at
  all*: it is the string passed as the REPL's `context` variable
  (`prehend/environments/local_repl.py:880-883`), and it reaches the model only if the model
  chooses to `print(context)` - where REPL stdout is truncated at `max_output_chars=2000`
  (`knowledge-base/librarian/ask.py:422`). The briefing is 5,546 chars. See **P1** below,
  which is the real defect and supersedes this one.

## Prompt-delivery defects (found by the prompt-audit agent, 2026-07-10)

Numbered `P*` to avoid colliding with `D*` above. P1 and P2 were independently re-verified by
the lead against the full `/tmp/kb-librarian.log`; the rest are as reported.

**P1 - the briefing is delivered to the wrong model.** MEASURED by the lead over the whole
log, splitting requests by whether they carry a system message (root) or not (sub-call):

| | root requests | sub-call requests |
| --- | --- | --- |
| total | 8,459 | 6,472 |
| contain `CITATION RULE` | **20 (0.24%)** | **2,793 (43.2%)** |
| contain `MUST PASTE THE DOCUMENT TEXT` | 64 (0.76%) | - |

The citation rule reaches the orchestrator - the only model that emits citations - 0.24% of
the time. It reaches sub-calls 43.2% of the time: tool-less leaf LLMs that cannot cite. The
model ships it there itself by calling `llm_query(..., context=context)`, which forwards the
entire briefing downstream (1,194 such calls; see P2).

`ask.py` carries ~80 lines of post-hoc repair machinery (`scrub_unknown_markers`,
`_CITE_FEEDBACK`, `_GROUND_FEEDBACK`, `_build_citation_verifier`, the ungrounded guard at
`ask.py:515-543`) cleaning up fabricated citations. The rule that would prevent them has
effectively never been shown to the model that needs it.

**Fix:** deliver the briefing in the system prompt, via `custom_system_prompt` at the `SRLM(...)`
construction (`ask.py:~397`). Note `prompts.py:175` `.format()`s the template, so the four
brace pairs in the briefing must be escaped. This is also cache-positive: it grows the
constant root prefix from ~3,639 to ~5,138 tokens, i.e. 3 cached blocks to 4.

**P2 - `reject_self_context_delegation` exists for this exact bug and is off.**
`prehend/utils/subcall_guard.py:103-114` names the incident it was written for
("corpus-NIAH, 2026-06-27"). MEASURED: `"Sub-call guard rejected"` appears **0** times in the
log. 1,194 `context=context` calls ship ~22.5k tokens of briefing+catalog into a sub-LLM that
has no tools and cannot use it. Fix: pass `reject_self_context_delegation=True` to `SRLM(...)`.

**P3 - `llm_query_batched` is misdocumented, and its `context=` path is sequential.**
VERIFIED by the lead: `local_repl.py:518-526` accepts `context=` and `reduce=`, but
`prompts.py:13` advertises only `llm_query_batched(prompts, model=None)` and claims it "runs
multiple `llm_query` calls concurrently ... Much faster than sequential." When `context=` is
passed, `local_repl.py:530-539` is a plain list comprehension - N **sequential** calls. Only
the `context=None` path reaches `_send_batched`.

**P4** - a 77,691-char catalog is duplicated into `context` *and* into the `manifest` custom
tool (`ask.py:164-171`). **P5** - `rlm_query` is documented in the system prompt with two
worked examples, but the librarian sets `max_depth=1`, so all 185 `rlm_query` calls silently
degrade to `llm_query`. **P6** - unclosed ` ```repl ` fence at `prompts.py:103` renders
"Submitting your final answer:" inside a code block. **P7** - prehend's own worked examples
(`prompts.py:47-87`) do the very things its rules forbid (`context[:10000]`, for-loops over
chunks, pasting); the model copies the examples, not the rules - 1,925 bare-paste blocks
observed. **P8** - `prompts.py:68` says "when the context isn't that long (e.g. >100M
characters)"; the inequality is backwards.

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

## Changes applied (pending EXP-002 measurement)

Both are sampling-independent, both re-verified against the code, both committed separately so
either can be reverted alone. `root_max_tokens 8192 -> 2048` was **considered and skipped**:
`librarian/config.py:127`'s own comment records that "real final answers run 2-4K tokens",
so capping at 2048 would truncate healthy answers to buy an effect the repeat-guard already
delivers without that cost.

| commit | change | file |
| --- | --- | --- |
| `a40bfa9` | repeat-guard watches `content`, not only `reasoning_content` (D5) | `prehend/clients/openai.py` |
| `4f8dcdc` | `_default_answer` runs `answer_verifier` with one revision, and its forcing sentence is `role: "user"` (D2) | `prehend/core/rlm.py` |

Full prehend suite green after each: 898 passed, 9 skipped.

## Ledger

| ID | Date | Hypothesis | Change | Instrument | Result | Verdict |
| --- | --- | --- | --- | --- | --- | --- |
| EXP-005 | 2026-07-10 | ADR-0011 rejected FlashInfer on a "+46% prefill penalty"; under the corrected decode-dominant budget it should be a win | `--attention-backend FLASHINFER` (local-ai `621ccad`) | Fixed-prompt micro A/B (<0.5% spread) **weighted by the measured budget**, then a 6 gold x 3 gate for regression only | Prefill **-15%** @3.5k / **-18%** @56k, decode **+3.05%** @56k -> **-3.6% weighted**. The "+46% penalty" was **first-touch JIT** (1.515s -> 0.162s), caused by an upstream `all()`/`any()` warmup gate that skips hybrid models. Gate: **18/18 cited, 0 infra**, median 21.8 -> 19.1s, p90 85.7 -> 48.9s. KV pool -9.9%. | **kept** (see local-ai ADR-0013) |
| EXP-004 | 2026-07-10 | `llm_query(context=context)` hands the briefing to a tool-less sub-LLM which can only echo tool syntax; the guard prehend wrote for this will stop the echo reaching the user | `8d6ae45` - `reject_self_context_delegation=True`; `63918bb` - log when it fires | 6 gold questions x 3 reps vs EXP-003 (guard is the only variable) | **18/18 cited** (first clean sweep), **max 112.6s** (lowest of any arm), median 21.8s. Guard fired **10x**. Post-guard, **0 of 54 sub-calls** carry the briefing (36.7% under P1 alone). BUT `ask1` regressed on all 3 reps: `[21,28,29]` -> `[58,86,59]`; mean 33.9 -> 39.1s, p90 53.3 -> 85.7s. | **kept, flagged** (correctness + worst case; a real per-question latency regression) |
| EXP-003 | 2026-07-10 | The orchestrator never reads the CITATION RULE (0.24% of root requests); delivering the briefing in the system prompt will improve grounding | `cf21a21` - `custom_system_prompt = RLM_SYSTEM_PROMPT + briefing` (braces doubled) | 6 gold questions x 3 reps, vs EXP-002 (same guard + verifier), so briefing delivery is the only variable | Mean **50.1 -> 33.9s** (-32% vs baseline), p90 **154.3 -> 53.3s**, max 188.4s. `ask2` went **3/3 grounded for the first time**. But `ground_cited` **17/18, unchanged in all three arms** - the failure relocated to `ask4.r2`. Rule now reaches **all** root turns (verified: 6/6 in a smoke ask). | **kept** (latency + principle; no measurable quality win at n=18) |
| EXP-002 | 2026-07-10 | Restoring the dead repeat-guard (D5) and verifying the forced-final answer (D2) will cut the tail and the uncited answers | `a40bfa9` + `4f8dcdc` | 6 **gold** questions x 3 reps, `ground_cited` + guard-abort delta | Guard fired **4x**. Max **421.2s -> 171.5s**, mean 50.1 -> 44.6. But median flat (22.0 -> 23.2) and **p90 worse** (55.6 -> 154.3). `ground_cited` **17/18 in both**. 0 infra fails both. | **kept** (mechanism verified, tail bounded, no regression; aggregate effect within noise at n=18) |
| EXP-001 | 2026-07-10 | Prod samples at T=1.0 because no sampling params are sent; greedy will cut the token blowup and stabilise grounding | `--override-generation-config '{"temperature":0.0,...}'` on the vLLM unit | 1 fixed question x 4 reps, `/metrics` deltas (**underpowered**: the attractor fires ~1 in 3-6) | Arm B rep1 generated **93,153** tokens (2x arm A's worst) and 500'd. The failure is verbatim repetition collapse - the classic *greedy* pathology - and `librarian/config.py:67` already documented it. | **rolled-back** |
| EXP-000 | 2026-07-10 | Putting `{docs[id]}` first in the Map sub-call prompt lets successive sub-calls share a cached prefix | Reorder Map prompt template in `knowledge-base/librarian/ask.py` so doc text precedes the instruction | End-to-end 6-ask x 3-rep probe (**wrong instrument**) | Wash-to-worse, tail-dominated: median 22.0s -> 25.6s, mean 50.1s -> 69.8s, max 421s -> 455s, `ground_cited` 17/18 -> 16/18, plus 1 `infra_fail` traced to **D1**, not to the change. ask3/ask4 improved, ask1/ask2 got tail-hit. Prefix-cache hit rate - the quantity the hypothesis is actually about - **was never measured**. | **rolled-back** |

### EXP-004 notes: the self-delegation guard, and a second dead guard found

```
        baseline            EXP-002             EXP-003             EXP-004
ask0: [ 17C  90C  12C]  [ 16C  10C   8C]  [ 31C  10C   6C]  [ 21C  17C  21C]
ask1: [ 11C  14C  49C]  [154C  18C  11C]  [ 21C  28C  29C]  [ 58C  86C  59C]  <- REGRESSED, 3/3 reps
ask2: [ 56C  26C 421u]  [ 16C 170C 172u]  [188C  53C  59C]  [113C  50C  48C]
ask3: [ 23C  10C  17C]  [ 29C   9C  10C]  [ 16C  17C  11C]  [ 19C   9C  10C]
ask4: [ 40C  35C  22C]  [ 46C  32C  28C]  [ 26C  45C  18u]  [ 14C  96C  22C]  <- narration answer gone
ask5: [ 20C  22C  20C]  [  8C  36C  33C]  [ 13C  11C  26C]  [ 11C  39C  10C]
```

`subcall_guard.self_context_rejection`'s own docstring predicted the exact failure EXP-003
surfaced: *"hands that briefing to a sub-LLM that has no REPL and no tools, so the sub-LLM can
only echo tool syntax or refuse - and the orchestrator ships that echo as the final answer."*
EXP-003's `ask4.r2` was that echo, verbatim ("Let me read document 012 about...", 0 citations).
The guard was written for this incident (corpus-NIAH, 2026-06-27) and had never been enabled.

**Effect, MEASURED.** It fired 10 times. Sub-calls carrying the briefing went from 36.7% (P1
alone) to **0 of 54**. `ground_cited` reached 18/18 and the worst-case ask fell to 112.6s.

**Cost, MEASURED and not noise.** `ask1` regressed on all three reps, roughly 2-3x. A rejected
sub-call costs a REPL iteration and forces a re-plan, so mean and p90 moved the wrong way
(33.9 -> 39.1s, 53.3 -> 85.7s) even as median and max improved.

**Honest scoring.** 18/18 vs 17/18 is one discordant row at n=18. That is a directionally
correct result with a mechanism behind it, **not** a significant quality win. A 50-ask gate is
needed before anyone claims the citation-repair machinery in `ask.py` can be retired.

**And a second dead guard, found the same way as the first.** `local_repl.py` *returns* the
rejection string rather than sending the sub-call, and logged **nothing**. The string only
reaches a transcript if the model prints its return value. So the guard's activity was invisible:
the lead first measured 0 firings and nearly concluded it was inert, when in fact it had already
suppressed every self-delegation. Fixed in `63918bb` - it now emits an INFO line when it fires.

This is the same lesson as D5, twice in one night: **a guard with no positive signal cannot be
distinguished from a guard that is dead.** Every guard in this codebase should log, or export a
counter, when it acts.

### EXP-003 notes: the briefing lands, latency improves, grounding does not

```
        baseline            EXP-002             EXP-003
ask0: [ 17C  90C  12C]  [ 16C  10C   8C]  [ 31C  10C   6C]
ask1: [ 11C  14C  49C]  [154C  18C  11C]  [ 21C  28C  29C]
ask2: [ 56C  26C 421u]  [ 16C 170C 172u]  [188C  53C  59C]   <- first 3/3 ever
ask3: [ 23C  10C  17C]  [ 29C   9C  10C]  [ 16C  17C  11C]
ask4: [ 40C  35C  22C]  [ 46C  32C  28C]  [ 26C  45C  18u]   <- failure relocated here
ask5: [ 20C  22C  20C]  [  8C  36C  33C]  [ 13C  11C  26C]
```

**Delivery is confirmed.** A smoke ask's root turns went from carrying the CITATION RULE in
0.24% of requests to 6 of 6. The `.format()` hazard is real and handled: `prompts.py:175` formats
the whole template, and the briefing carries four literal brace groups (`{docs[id]}` etc), so
they are doubled at the call site.

**Latency clearly improves**: mean 50.1 -> 33.9s, and p90 returns to 53.3s after EXP-002 had
pushed it to 154.3s. Plausible mechanism: a model that has read the give-up rule and the Map
instruction stops improvising a search strategy. Not proven - only consistent.

**Grounding does not.** `ground_cited` is 17/18 in *all three* arms. `ask2`, the historical
offender, went 3/3 grounded for the first time. Simultaneously `ask4.r2` failed. Net zero. At
n=18 with one failure per arm, no quality claim is defensible in either direction.

**What the surviving failure teaches**, and it is worth more than the aggregate: `ask4.r2`
completed in 18s with `citations=[]`, `from_training=[]`, and this as its final answer:

> "Let me read document 012 about Medicare Opioid Treatment Programs Policy Manual, which
> mentions that no copayment is required for OTP services..."

That is the model's REPL **narration** shipped as an answer. It is not a fabrication and not a
runaway loop - it is fast, and it has no citations at all. `_build_citation_verifier` would
reject it (`claims` empty, no no-coverage phrase). It shipped anyway, which is **D7**: the
in-loop verifier had already spent its two retries and was therefore never called.

So the two levers that remain, both already written and both switched off:

1. **D7** - do not accept an answer that failed verification once retries are exhausted.
2. **`reject_code_shaped_answers`** (`harness.py:89`, added in `08021f2` for exactly this) -
   bounce a narration/tool-syntax answer back for revision instead of terminating on it.

### EXP-002 notes: what the fixes do and do not buy

Per-ask, `C` = `ground_cited`, `u` = not:

```
ask0: base [17C  90C  12C]   exp002 [ 16C  10C   8C]
ask1: base [11C  14C  49C]   exp002 [154C  18C  11C]
ask2: base [56C  26C 421u]   exp002 [ 16C 170C 172u]
ask3: base [23C  10C  17C]   exp002 [ 29C   9C  10C]
ask4: base [40C  35C  22C]   exp002 [ 46C  32C  28C]
ask5: base [20C  22C  20C]   exp002 [  8C  36C  33C]
```

**The guard works and bounds the worst case.** 421.2s -> 171.5s, and no transcript reached the
120k-192k chars that previously tipped requests into D1's 400. Zero infra failures.

**It does not prevent stalls, it truncates them.** Aborting the stream returns partial content,
which usually carries no ` ```repl ` fence, so the turn is still a no-op and the model can ramble
again next turn (until `repeat_guard_abort_limit=4` forces wrap-up). Hence p90 *rose*: one
catastrophic 421s stall became three bounded ~170s stalls. Total time spent in long runs is
essentially unchanged (511s baseline vs 496s). **At n=18 the aggregate latency effect is noise.**
Kept because it restores a regressed safety net and removes the catastrophic tail, not because
it demonstrably made the median ask faster. It did not.

**Correction to the D2 fix's advertised scope, and to the lead's first reading of it.** The
surviving uncited answer is `ask2.r2`, which cited `['091.1']`. The lead initially called this
"cited the wrong document". That was wrong: `ground_cited` never looks at `gold_ids` (see the
instrument note above). `091.1` scored `False` because it was **claimed without having been
retrieved** - an invented citation, `from_training`.

More importantly, **`4f8dcdc` was not even on that code path**, which is why it logged 0
revise prompts. There are **two** verifier bypasses, and the fix closed only one:

1. `_default_answer` never called `answer_verifier` at all. **Closed by `4f8dcdc`.**
2. **D7 (open):** the in-loop verifier at `prehend/core/rlm.py:726` is gated on
   `self._answer_retries < self.max_answer_retries`. Once the two citation retries are spent,
   the verifier is **not called at all** and the ungrounded answer is accepted and returned.
   The guard silently stops guarding exactly when the model has proven it needs guarding.

Fixing D7 means deciding what a run that cannot produce a grounded citation *should* return.
Almost certainly the mandated no-coverage sentence (a scorable refusal), not an invented
citation. That is a behaviour change with a refusal-rate cost, so it wants the user's call and
a real gate, not a 5am patch.

The forced-final answer text was also a first-person monologue ("The user wants the phone
number... I previously found:"), i.e. the model continuing its own narration.
`reject_code_shaped_answers` / the answer-shape guard remain **off**.

### EXP-001: greedy decoding. REJECTED, and it made things worse.

**Hypothesis (the lead's):** prod sends no sampling params, so vLLM applies the checkpoint's
`generation_config.json` (`temperature=1.0, top_k=20, top_p=0.95`). prehend's own default is
`rlm_temperature=0.0` (`harness.py:45`). High-entropy sampling in a code-writing REPL was
assumed to cause the wandering, the token blowup, and the uncited answers.

**Method:** one fixed question, 4 reps per arm, `/metrics` deltas per ask. Arm A = prod
(T=1.0). Arm B = server-wide greedy via `--override-generation-config
'{"temperature":0.0,"top_k":-1,"top_p":1.0}'` on the vLLM unit (a zero-code, reversible
experiment; under `temperature=0` vLLM takes argmax, so the `top_k`/`top_p` values are no-ops).

| arm | wall min / med / max | gen tokens min / med / max | outcome |
| --- | --- | --- | --- |
| A - T=1.0, prod | 38.8 / 145.0 / 221.2 s (**5.7x**) | 9,007 / 29,747 / 45,132 (**5.0x**) | 4/4 grounded |
| B - greedy | 35.9 s, then **484.7 s** | 6,788, then **93,153** | rep0 cited `[007]`; rep1 **HTTP 500** |

**Result: falsified, and dangerously so.** Arm B rep0 looked great. Arm B rep1 produced the
worst collapse of the entire night - 93,153 generated tokens, more than **double** arm A's
worst rep - and drove the transcript past `max_model_len` into D1's 400. Arm B was stopped
after 2 reps and the override reverted.

**Why the hypothesis was wrong.** D2's collapse is *verbatim repetition*, which is the
classic pathology of **greedy** decoding, not of high-entropy sampling. Temperature 1.0 would
predict divergent, non-repeating text; the transcripts show one line emitted 343 times. The
evidence was already in this repo: `knowledge-base/librarian/config.py:67` records that "a
greedy (temp 0) repetition loop once generated 35K+ tokens server-side after its ask had
already timed out." The lead did not look for prior art before changing prod.

**What survives.** The 5.7x wall / 5.0x token spread under T=1.0 is real, and the wiring gap
is real: the librarian constructs `SRLM(...)` directly (`ask.py:397`, the only site; `grep
Harness librarian/` returns zero hits), so nothing ever sets sampling params and **`seed` never
reaches the wire either**. Until a `seed` is sent, no A/B on this path is reproducible. Closing
the gap is still worth doing - but to a *measured* temperature, not an assumed 0.0, and it
needs >=10 reps because the repetition attractor fires roughly 1 rep in 3-6. n=4 cannot see it;
n=2 got lucky and caught it.

*(Correction: the lead repeatedly referred to `build_harness_from_config` and `HarnessConfig`.
Neither symbol exists. The dataclass is `Defaults`, `harness.py:25`, and the code in question is
`Harness.__init__`. The conclusion - that the librarian bypasses all of it - is unaffected.)*

**The seed seam, VERIFIED by the harness-map agent with a real client + mocked `create`:** the
route to the wire is `default_extra_body`, placed **inside `backend_kwargs`** on `ask.py`'s
`SRLM(...)` call - not as an `SRLM` kwarg. `OpenAIClient` merges it into every request body
(`openai.py:291`, `:442`), so it reaches root turns, socket sub-calls, child RLMs and the leaf
fallback alike (child clients inherit via `rlm.py:1132`). `subcall_extra_body` is not the seam:
`body = dict(self.default_extra_body)` then `body.update(extra_body)`, so a sub-call's
`chat_template_kwargs` replaces only that key.

```python
# inside backend_kwargs of the SRLM(...) call in librarian/ask.py
"default_extra_body": {"seed": 1234},
```

Note the ordering trap: at today's `temperature=1.0` a seed **does** control the sampler and
buys reproducibility. Pin `temperature: 0` in the same dict and decoding goes greedy, the seed
becomes inert - and greedy is what EXP-001 rejected. Send the seed; leave temperature alone.

**Verdict: rolled back.** The server unit is restored to HEAD. Sampling is a second-order
knob here. Fix the containment (D5) and the forced-final path (D2) first; both are
sampling-independent.

### EXP-000: final verdict, and why it could never have worked

Settled after the prompt audit. Two independent reasons, both VERIFIED:

1. **The instruction it edited is not delivered.** The Map paste instruction lives in the
   librarian briefing, which reaches the orchestrator in 64 of 8,459 root requests (0.76%).
   Editing text the model does not read is a no-op, which is exactly the noise the probe
   returned.
2. **The path the model actually takes is already data-first.** The model overwhelmingly
   calls `llm_query(..., context=...)` rather than pasting (1,997 vs 145 `llm_query_batched`
   calls). That path composes its sub-call prompts through
   `prehend/utils/mapreduce.py:_compose`, whose docstring records the decision: *"Data-first
   layout (ADR-0017): the large, stable data leads and the varying instruction trails ... The
   old instruction-first layout diverged at token 0 and re-prefilled the whole chunk every
   query (~6.4x re-prefill measured on the multihop bench)."*

So EXP-000 was re-deriving, in dead text, a decision prehend had already made and measured.
The right change is to **delete** the paste instruction (P3/D3), not reorder it.

A third reason it could never have paid, even if delivered: two *different* chunks share only
the `"Text:\n"` header, ~2 tokens. The reuse quantum is 1056 tokens. Distinct chunks are
distinct at token 0 by construction, so no ordering creates reuse between them. All real
reuse comes from re-reading the *same* doc, which the data-first layout already captures.

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
