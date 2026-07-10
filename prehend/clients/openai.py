import logging
import os
import threading
import time
from collections import defaultdict
from types import SimpleNamespace
from typing import Any

import openai
from dotenv import load_dotenv

from prehend.clients.base_lm import BaseLM
from prehend.clients.scheduler import Priority, RequestScheduler, resolve_priority
from prehend.core.types import ModelUsageSummary, UsageSummary
from prehend.utils.exceptions import CancellationError, TimeoutExceededError

log = logging.getLogger(__name__)

load_dotenv()

# Load API keys from environment variables
DEFAULT_OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
DEFAULT_OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
DEFAULT_VERCEL_API_KEY = os.getenv("AI_GATEWAY_API_KEY")
DEFAULT_PRIME_API_KEY = os.getenv("PRIME_API_KEY")
DEFAULT_PRIME_INTELLECT_BASE_URL = "https://api.pinference.ai/api/v1/"


def _is_context_contention(e: openai.APIStatusError) -> bool:
    """True when llama-server failed the request over KV/context pressure.

    Two distinct shapes, verified against llama-server source and live behavior:

    400 ``exceed_context_size_error`` - the prompt fails the server's STATIC per-slot
    check (``n_tokens >= slot.n_ctx``, server-context.cpp). Under ``--kv-unified``
    slot.n_ctx is the full pool, so this means the request can never fit; the p1 retry
    fails too and the error propagates after one attempt. Kept for the non-unified
    config and other OpenAI-compatible backends. Example body:
    ``{"error":{"code":400,"message":"request (40069 tokens) exceeds the available
    context size (22016 tokens), try increasing it","type":"exceed_context_size_error"}}``

    500 ``Context size has been exceeded.`` - the REAL unified-KV contention failure:
    when ``llama_decode`` cannot find KV space at n_batch=1, the server fails EVERY
    processing slot with this 500 and clears the whole context (server-context.cpp,
    "Context size has been exceeded"). These requests would fit with the pool to
    themselves, so the p1 retry (drain, then run solo) is exactly the right recovery.
    Example body:
    ``{"error":{"code":500,"message":"Context size has been exceeded.","type":"server_error"}}``

    500 ``failed to find a memory slot`` - the KV-pool exhaustion seen in the
    rlm-trainer 2026-06-16 eval (finding #3): under concurrent multi-doc asks the
    unified KV pool runs out of slots (``decode: failed to find a memory slot``).
    On builds that surface this as a 500 (rather than a hard ggml_abort that kills
    the process), the same recovery applies - drain the pool and retry the request
    solo at p1, where the slots it needs are free. A genuine process crash instead
    yields a connection error that the p1 retry cannot help and that propagates.
    """
    body = getattr(e, "body", None)
    if isinstance(body, dict):
        err = body.get("error", body)
        if isinstance(err, dict):
            if err.get("type") == "exceed_context_size_error":
                return True
            msg = str(err.get("message", "")).lower()
            if "context size has been exceeded" in msg or "find a memory slot" in msg:
                return True
    text = str(e).lower()
    return (
        "exceed_context_size_error" in text
        or "exceeds the available context size" in text
        or "context size has been exceeded" in text
        or "find a memory slot" in text
    )


_REASONING_TAIL_CHARS = 1500

# Reasoning-loop repeat-guard (rlm-trainer #6 part 3). A single leaf completion can
# spin in the thought-channel, repeating tokens/phrases with no content, until
# max_tokens or the run deadline. Detect it as a high word-level k-gram repeat rate
# over a trailing window: on the 2026-06-18 eval, legit reasoning measured <=0.006
# while every degeneration measured >=0.42 (exact-repeat AND no-progress-with-variation),
# so a ~0.35 threshold separates them with wide margin.
_REPEAT_GUARD_K = 4                 # word n-gram size
_REPEAT_GUARD_WINDOW_CHARS = 1500   # trailing reasoning examined
_REPEAT_GUARD_MIN_WORDS = 60        # don't judge until this much reasoning has accrued
_REPEAT_GUARD_CHECK_EVERY = 300     # chars of new reasoning between (cheap) re-checks
# Content is guarded too (see the stream loop), but only past this floor. A legit root
# turn peaked at 2,575 chars over the 2026-07-10 sample; degenerate turns reached
# 120k-192k. 4,000 leaves ~55% headroom over the largest healthy turn observed.
_REPEAT_GUARD_CONTENT_MIN_CHARS = 4000


def _reasoning_is_looping(
    text: str,
    threshold: float,
    k: int = _REPEAT_GUARD_K,
    window_chars: int = _REPEAT_GUARD_WINDOW_CHARS,
    min_words: int = _REPEAT_GUARD_MIN_WORDS,
) -> bool:
    """True when the trailing reasoning window is dominated by repeated k-grams.

    repeat_rate = fraction of word k-grams in the window that are NOT the first
    occurrence; high rate = the model is cycling (exact repetition or no-progress
    paraphrase) rather than making progress. Below ``min_words`` we cannot judge
    yet and return False, so the guard never fires on a short repetitive preamble.
    """
    words = text[-window_chars:].split()
    if len(words) < max(min_words, k + 1):
        return False
    grams = [tuple(words[i : i + k]) for i in range(len(words) - k + 1)]
    repeat_rate = 1.0 - len(set(grams)) / len(grams)
    return repeat_rate >= threshold


def _resolve_content(content: str | None, reasoning: str | None) -> str:
    """Final content for a completion, never a silent empty string.

    Reasoning-parsing servers (llama.cpp --jinja) route thought-channel tokens
    to reasoning_content; a generation that never exits the channel arrives as
    content="" (2026-06-12: the kb gemma fine-tune ruminated to the token cap
    on triage prompts, and the orchestrator retried the resulting empty answer
    for 20 iterations). Surfacing a tagged reasoning tail gives the caller
    signal to adapt instead of retrying blind.
    """
    if content:
        return content
    if reasoning:
        return (
            "[reasoning-only response - the model produced no answer, its "
            "reasoning ended with]\n" + reasoning[-_REASONING_TAIL_CHARS:]
        )
    return ""


def _merge_consecutive_roles(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Merge adjacent messages that share a role, concatenating their content.

    The RLM REPL can emit consecutive same-role turns - e.g. two assistant iterations with no
    code block between them produce no intervening REPL `user` message, leaving `[..., assistant,
    assistant]`. Strict OpenAI-compatible servers (llama.cpp `/chat/completions` applying a gemma
    jinja template) reject 2+ trailing assistant messages ("Cannot have 2 or more assistant
    messages at the end of the list"). Merging is semantically identical - the model sees the same
    text - and yields the alternating sequence those templates require. Lenient backends (e.g. the
    retired LM Studio endpoint) are unaffected. Only string content is merged; dict/list content
    (multimodal parts) is left as its own message so structured payloads are never corrupted.
    """
    merged: list[dict[str, Any]] = []
    for msg in messages:
        if (
            merged
            and merged[-1].get("role") == msg.get("role")
            and isinstance(merged[-1].get("content"), str)
            and isinstance(msg.get("content"), str)
        ):
            merged[-1] = {**merged[-1], "content": merged[-1]["content"] + "\n\n" + msg["content"]}
        else:
            merged.append(msg)
    return merged


class OpenAIClient(BaseLM):
    """
    LM Client for running models with the OpenAI API. Works with vLLM as well.

    Any additional keyword arguments (e.g. default_headers, default_query, max_retries)
    are passed through to the underlying openai.OpenAI and openai.AsyncOpenAI constructors.
    Only model_name is excluded, since it is not a client constructor argument.
    """

    def __init__(
        self,
        api_key: str | None = None,
        model_name: str | None = None,
        base_url: str | None = None,
        default_extra_body: dict[str, Any] | None = None,
        scheduler: RequestScheduler | None = None,
        stream: bool = False,
        repeat_guard_threshold: float | None = None,
        **kwargs,
    ):
        super().__init__(model_name=model_name, **kwargs)
        # Streamed completions (opt-in): content is assembled from chunks and the
        # call checks cancel_event/deadline between chunks, so an abandoned or
        # over-deadline generation is abandoned client-side AND the server sees
        # the disconnect and frees its slot (non-streaming servers only notice a
        # dead client when the full response is written). Recommended for local
        # single-server backends (llama-server).
        self.stream = stream
        # Reasoning-loop repeat-guard threshold (streamed calls only). None/<=0 = off,
        # preserving the prior behavior; a value in (0,1) aborts a no-progress reasoning
        # spin early and returns the tagged tail. See _reasoning_is_looping.
        self._repeat_guard_threshold = (
            repeat_guard_threshold if (repeat_guard_threshold or 0) > 0 else None
        )
        # Count of repeat-guard aborts in the CURRENT run (reset by set_deadline when a
        # run is armed). The orchestrator reads the per-run total via the handler and
        # escalates to a forced wrap-up once it crosses repeat_guard_abort_limit.
        self.repeat_guard_aborts = 0
        # Cooperative cancellation: the owning run (RLM) sets this to abort every
        # in-flight and queued call of this client; checked between stream chunks.
        self.cancel_event = threading.Event()
        # Optional wall-clock deadline for the whole run (set_deadline); enforced
        # between stream chunks so even a mid-generation call stops on time.
        self._deadline: float | None = None
        self._run_started: float | None = None
        self._max_timeout: float | None = None
        # Per-request body fields merged into every call (e.g. {"chat_template_kwargs":
        # {"enable_thinking": False}} to disable gemma CoT on the recursive sub-call backend
        # while the root orchestrator keeps thinking). Not an OpenAI client ctor arg, so it is a
        # named param here and never leaks into client_kwargs below.
        self.default_extra_body = default_extra_body or {}
        self.scheduler = scheduler

        if api_key is None:
            if base_url == "https://api.openai.com/v1" or base_url is None:
                api_key = DEFAULT_OPENAI_API_KEY
            elif base_url == "https://openrouter.ai/api/v1":
                api_key = DEFAULT_OPENROUTER_API_KEY
            elif base_url == "https://ai-gateway.vercel.sh/v1":
                api_key = DEFAULT_VERCEL_API_KEY
            elif base_url == DEFAULT_PRIME_INTELLECT_BASE_URL:
                api_key = DEFAULT_PRIME_API_KEY

        # Pass through arbitrary kwargs to the OpenAI client (e.g. default_headers, default_query, max_retries).
        # Exclude model_name since it is not an OpenAI client constructor argument.
        client_kwargs = {
            "api_key": api_key,
            "base_url": base_url,
            "timeout": self.timeout,
            **{k: v for k, v in self.kwargs.items() if k != "model_name"},
        }
        self.client = openai.OpenAI(**client_kwargs)
        self.async_client = openai.AsyncOpenAI(**client_kwargs)
        self.model_name = model_name
        self.base_url = base_url  # Track for cost extraction

        # Per-model usage tracking
        self.model_call_counts: dict[str, int] = defaultdict(int)
        self.model_input_tokens: dict[str, int] = defaultdict(int)
        self.model_output_tokens: dict[str, int] = defaultdict(int)
        self.model_total_tokens: dict[str, int] = defaultdict(int)
        self.model_costs: dict[str, float] = defaultdict(float)  # Cost in USD

    def set_deadline(self, max_timeout: float | None) -> None:
        """Arm (or clear) a wall-clock deadline for every call on this client.

        Set by the owning RLM run from its max_timeout; enforced between stream
        chunks (stream=True), so a mid-generation call aborts on time instead of
        blocking the run past its budget."""
        if max_timeout is None:
            self._deadline = self._run_started = self._max_timeout = None
            return
        # Arming a fresh run: reset the per-run repeat-guard abort count.
        self.repeat_guard_aborts = 0
        self._run_started = time.monotonic()
        self._deadline = self._run_started + max_timeout
        self._max_timeout = max_timeout

    def _check_abort(self) -> None:
        """Raise if the run was cancelled or its deadline passed."""
        if self.cancel_event.is_set():
            raise CancellationError(message="LM call aborted: run cancelled")
        if self._deadline is not None and time.monotonic() > self._deadline:
            raise TimeoutExceededError(
                elapsed=time.monotonic() - self._run_started,
                timeout=self._max_timeout,
                message=f"LM call aborted at run deadline ({self._max_timeout:.1f}s)",
            )

    def completion(
        self,
        prompt: str | list[dict[str, Any]],
        model: str | None = None,
        response_format: dict[str, Any] | None = None,
        priority: str | int | None = None,
        max_tokens: int | None = None,
        extra_body: dict[str, Any] | None = None,
    ) -> str:
        if isinstance(prompt, str):
            messages = [{"role": "user", "content": prompt}]
        elif isinstance(prompt, list) and all(isinstance(item, dict) for item in prompt):
            messages = _merge_consecutive_roles(prompt)
        else:
            raise ValueError(f"Invalid prompt type: {type(prompt)}")

        model = model or self.model_name
        if not model:
            raise ValueError("Model name is required for OpenAI client.")

        body = dict(self.default_extra_body)
        if self.client.base_url == DEFAULT_PRIME_INTELLECT_BASE_URL:
            body["usage"] = {"include": True}
        if extra_body:
            body.update(extra_body)

        create_kwargs: dict[str, Any] = {}
        if response_format is not None:
            create_kwargs["response_format"] = response_format
        if max_tokens is not None:
            create_kwargs["max_tokens"] = max_tokens

        p = resolve_priority(priority)
        return self._do_completion(model, messages, body, create_kwargs, p)

    def _do_completion(
        self,
        model: str,
        messages: list[dict[str, Any]],
        extra_body: dict[str, Any],
        create_kwargs: dict[str, Any],
        priority: int,
    ) -> str:
        while True:
            # The finally must release at the priority we acquired with, not the one the
            # retry branch escalates to for the next iteration.
            acquired = priority
            if self.scheduler:
                self.scheduler.acquire(acquired)
            try:
                # A call that was queued (scheduler) past cancellation/deadline
                # must not start a fresh generation.
                self._check_abort()
                if self.stream:
                    return self._stream_completion(model, messages, extra_body, create_kwargs)
                response = self.client.chat.completions.create(
                    model=model, messages=messages, extra_body=extra_body, **create_kwargs
                )
                self._track_cost(response, model)
                message = response.choices[0].message
                return _resolve_content(
                    message.content, getattr(message, "reasoning_content", None)
                )
            except (openai.BadRequestError, openai.InternalServerError) as e:
                if (
                    self.scheduler
                    and priority != Priority.CONTENTION_RETRY
                    and _is_context_contention(e)
                ):
                    log.info("context contention (%s), retrying at p1", str(e)[:80])
                    priority = Priority.CONTENTION_RETRY
                    continue
                raise
            finally:
                if self.scheduler:
                    self.scheduler.release(acquired)

    def _stream_completion(
        self,
        model: str,
        messages: list[dict[str, Any]],
        extra_body: dict[str, Any],
        create_kwargs: dict[str, Any],
    ) -> str:
        stream = self.client.chat.completions.create(
            model=model,
            messages=messages,
            extra_body=extra_body,
            stream=True,
            stream_options={"include_usage": True},
            **create_kwargs,
        )
        parts: list[str] = []
        reasoning_parts: list[str] = []
        usage = None
        reasoning_chars_at_check = 0
        content_chars_at_check = 0
        try:
            for chunk in stream:
                self._check_abort()
                if chunk.choices:
                    delta = chunk.choices[0].delta
                    if delta is not None and delta.content:
                        parts.append(delta.content)
                    reasoning = getattr(delta, "reasoning_content", None) if delta else None
                    if reasoning:
                        reasoning_parts.append(reasoning)
                if getattr(chunk, "usage", None) is not None:
                    usage = chunk.usage
                # Repeat-guard, re-checked every _REPEAT_GUARD_CHECK_EVERY chars of new
                # text so the scan stays cheap. On a no-progress loop, stop early and let
                # the finally close the stream (server frees the slot) - _resolve_content
                # then returns the tagged tail, the same output the spin would reach far later.
                #
                # It guards whichever stream this generation is producing. It used to guard
                # `reasoning_parts` only, behind `not parts`, which made it dead on the
                # orchestrator: that path runs with thinking OFF, so content arrives from the
                # first token and the guard never engaged. The prod failure is a CONTENT
                # ramble - a root turn emitting no ```repl fence and repeating one line up to
                # 343x, growing the merged assistant message to 192k chars (2026-07-10).
                # Content is only judged past _REPEAT_GUARD_CONTENT_MIN_CHARS so a healthy
                # turn can never trip it.
                if self._repeat_guard_threshold:
                    on_content = bool(parts)
                    watched = parts if on_content else reasoning_parts
                    at_check = content_chars_at_check if on_content else reasoning_chars_at_check
                    floor = _REPEAT_GUARD_CONTENT_MIN_CHARS if on_content else 0
                    total = sum(len(r) for r in watched)
                    if total >= floor and total - at_check >= _REPEAT_GUARD_CHECK_EVERY:
                        if on_content:
                            content_chars_at_check = total
                        else:
                            reasoning_chars_at_check = total
                        if _reasoning_is_looping(
                            "".join(watched), self._repeat_guard_threshold
                        ):
                            log.info(
                                "repeat-guard: aborting looping %s at %d chars",
                                "content" if on_content else "reasoning", total,
                            )
                            self.repeat_guard_aborts += 1
                            break
        finally:
            # On abort this closes the HTTP stream; llama-server notices the
            # disconnect and frees the slot instead of generating to completion.
            stream.close()
        # A repeat-guard abort breaks before the trailing usage chunk arrives; there
        # is no usage to track for an abandoned generation, so skip (best-effort).
        if usage is not None:
            self._track_cost(SimpleNamespace(usage=usage), model)
        return _resolve_content("".join(parts), "".join(reasoning_parts))

    async def acompletion(
        self,
        prompt: str | list[dict[str, Any]],
        model: str | None = None,
        priority: str | int | None = None,
        max_tokens: int | None = None,
        extra_body: dict[str, Any] | None = None,
    ) -> str:
        if isinstance(prompt, str):
            messages = [{"role": "user", "content": prompt}]
        elif isinstance(prompt, list) and all(isinstance(item, dict) for item in prompt):
            messages = _merge_consecutive_roles(prompt)
        else:
            raise ValueError(f"Invalid prompt type: {type(prompt)}")

        model = model or self.model_name
        if not model:
            raise ValueError("Model name is required for OpenAI client.")

        body = dict(self.default_extra_body)
        if self.client.base_url == DEFAULT_PRIME_INTELLECT_BASE_URL:
            body["usage"] = {"include": True}
        if extra_body:
            body.update(extra_body)

        create_kwargs: dict[str, Any] = {}
        if max_tokens is not None:
            create_kwargs["max_tokens"] = max_tokens

        p = resolve_priority(priority)
        return await self._ado_completion(model, messages, body, create_kwargs, p)

    async def _ado_completion(
        self,
        model: str,
        messages: list[dict[str, Any]],
        extra_body: dict[str, Any],
        create_kwargs: dict[str, Any],
        priority: int,
    ) -> str:
        while True:
            acquired = priority
            if self.scheduler:
                await self.scheduler.aacquire(acquired)
            try:
                self._check_abort()
                if self.stream:
                    return await self._astream_completion(
                        model, messages, extra_body, create_kwargs
                    )
                response = await self.async_client.chat.completions.create(
                    model=model, messages=messages, extra_body=extra_body, **create_kwargs
                )
                self._track_cost(response, model)
                message = response.choices[0].message
                return _resolve_content(
                    message.content, getattr(message, "reasoning_content", None)
                )
            except (openai.BadRequestError, openai.InternalServerError) as e:
                if (
                    self.scheduler
                    and priority != Priority.CONTENTION_RETRY
                    and _is_context_contention(e)
                ):
                    log.info("context contention (%s), retrying at p1", str(e)[:80])
                    priority = Priority.CONTENTION_RETRY
                    continue
                raise
            finally:
                if self.scheduler:
                    await self.scheduler.arelease(acquired)

    async def _astream_completion(
        self,
        model: str,
        messages: list[dict[str, Any]],
        extra_body: dict[str, Any],
        create_kwargs: dict[str, Any],
    ) -> str:
        stream = await self.async_client.chat.completions.create(
            model=model,
            messages=messages,
            extra_body=extra_body,
            stream=True,
            stream_options={"include_usage": True},
            **create_kwargs,
        )
        parts: list[str] = []
        reasoning_parts: list[str] = []
        usage = None
        try:
            async for chunk in stream:
                self._check_abort()
                if chunk.choices:
                    delta = chunk.choices[0].delta
                    if delta is not None and delta.content:
                        parts.append(delta.content)
                    reasoning = getattr(delta, "reasoning_content", None) if delta else None
                    if reasoning:
                        reasoning_parts.append(reasoning)
                if getattr(chunk, "usage", None) is not None:
                    usage = chunk.usage
        finally:
            await stream.close()
        self._track_cost(SimpleNamespace(usage=usage), model)
        return _resolve_content("".join(parts), "".join(reasoning_parts))

    def _track_cost(self, response: openai.ChatCompletion, model: str):
        self.model_call_counts[model] += 1

        usage = getattr(response, "usage", None)
        if usage is None:
            raise ValueError("No usage data received. Tracking tokens not possible.")

        self.model_input_tokens[model] += usage.prompt_tokens
        self.model_output_tokens[model] += usage.completion_tokens
        self.model_total_tokens[model] += usage.total_tokens

        # Track last call for handler to read
        self.last_prompt_tokens = usage.prompt_tokens
        self.last_completion_tokens = usage.completion_tokens

        # Extract cost from OpenRouter responses (cost is in USD)
        # OpenRouter returns cost in usage.model_extra for pydantic models
        self.last_cost: float | None = None
        cost = None

        # Try direct attribute first
        if hasattr(usage, "cost") and usage.cost:
            cost = usage.cost
        # Then try model_extra (OpenRouter uses this)
        elif hasattr(usage, "model_extra") and usage.model_extra:
            extra = usage.model_extra
            # Primary cost field (may be 0 for BYOK)
            if extra.get("cost"):
                cost = extra["cost"]
            # Fallback to upstream cost details
            elif extra.get("cost_details", {}).get("upstream_inference_cost"):
                cost = extra["cost_details"]["upstream_inference_cost"]

        if cost is not None and cost > 0:
            self.last_cost = float(cost)
            self.model_costs[model] += self.last_cost

    def get_usage_summary(self) -> UsageSummary:
        model_summaries = {}
        for model in self.model_call_counts:
            cost = self.model_costs.get(model)
            model_summaries[model] = ModelUsageSummary(
                total_calls=self.model_call_counts[model],
                total_input_tokens=self.model_input_tokens[model],
                total_output_tokens=self.model_output_tokens[model],
                total_cost=cost if cost else None,
            )
        return UsageSummary(model_usage_summaries=model_summaries)

    def get_last_usage(self) -> ModelUsageSummary:
        return ModelUsageSummary(
            total_calls=1,
            total_input_tokens=self.last_prompt_tokens,
            total_output_tokens=self.last_completion_tokens,
            total_cost=getattr(self, "last_cost", None),
        )
