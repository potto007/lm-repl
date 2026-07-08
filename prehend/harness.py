"""High-level Harness API: owns orchestration strategy, runtime detection, and
memory composition so clients do not hand-assemble SRLM. See
docs/superpowers/specs/2026-06-21-prehend-harness-api-design.md and ADR-0008."""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
import json
import logging
import urllib.request
from typing import TYPE_CHECKING

from prehend.core.srlm import SRLM
from prehend.core.types import RLMChatCompletion
from prehend.utils.prompts import RLM_SYSTEM_PROMPT
from prehend.utils.token_utils import per_call_subcall_budget, resolve_subcall_limit

if TYPE_CHECKING:
    from prehend.memory.harness import MemoryObserver

_log = logging.getLogger("prehend.harness")


@dataclass(frozen=True)
class Defaults:
    """Vetted Tier-A strategy/reliability defaults the Harness applies to SRLM."""
    max_output_chars: int = 500
    max_iterations: int = 10
    max_depth: int = 2
    max_errors: int = 3
    # >0 so the OpenAI SDK retries openai.APIConnectionError on a FRESH connection.
    # sglang's uvicorn closes idle keepalives after 5s; the map-reduce sub-call
    # batch path (AsyncOpenAI pool + asyncio.run-per-batch loop teardown, conc 16)
    # churns connections, so reuse races surfaced as un-retried "Error: Connection
    # error." (5-63/task) that map_reduce dropped -> wrong answers. A/B at conc 16:
    # 0 retries -> 5.6% conn errors; 2 -> 0%. Connection errors fail fast, so the
    # retry cost is ~0; the run deadline still bounds total wall-clock.
    max_retries: int = 2
    stream: bool = False
    # The RLM solve path (orchestrator, recursive RLMs, and map-reduce sub-calls)
    # runs deterministic: temperature 0.0 on every request. Rides in
    # default_extra_body, the same seam SRLM.candidate_temperature uses, which the
    # openai client merges into the request body. (The distiller samples at 1.0 -
    # that lives in the memory layer, MemoryConfig.reflect_temperature.)
    rlm_temperature: float = 0.0
    # Sampling seed sent on EVERY solve-path request (orchestrator, recursive
    # RLMs, and map-reduce sub-calls) via default_extra_body, so a run is
    # reproducible across the whole inference fan-out. None omits the field
    # entirely (server-side default RNG), keeping prior behavior byte-identical.
    seed: int | None = None
    subcall_enable_thinking: bool = False
    # Thinking toggle for the top-level orchestrator RLM call (distinct from the
    # sub-call one above). Gemma has no thinking mode so this was implicit; a
    # thinking-capable base (Qwen3 et al.) defaults thinking ON in its chat
    # template, which would wrap the orchestrator's REPL actions in <think> and
    # break parsing / burn the iteration budget. Default False keeps the
    # orchestrator deterministic and non-thinking, matching the Gemma path.
    # Rides via chat_template_kwargs, ignored by templates that don't reference
    # enable_thinking (e.g. Gemma), so the existing served-inference path is
    # byte-identical.
    rlm_enable_thinking: bool = False
    max_concurrent_subcalls: int = 4
    soft_timeout_pct: float | None = None
    # Sub-model context window (tokens) for the input-size guard. None lets the
    # Harness resolve it from runtime.ctx, then get_context_limit(model). A
    # Harness(subcall_context_limit=...) param overrides this field. Tier-A.
    subcall_context_limit: int | None = None
    # Dynamic-KV-pool engine (sglang, ADR-0015): when True the per-slot sub-call
    # division (per_call_subcall_budget) is bypassed - sglang's paged radix pool
    # LRU-evicts under contention instead of 500ing the way llama.cpp --kv-unified
    # did (ADR-0012), so each sub-call is budgeted against the FULL resolved pool
    # (the per-request context-length cap). Default False keeps the llama.cpp path.
    dynamic_kv_pool: bool = False
    # Generation cap for the ROOT orchestrator turn (tokens). Unset, the root rides
    # DEFAULT_MAX_DECODE_TOKENS=8192 (lm_handler), and a degenerate repeat/stutter
    # iteration runs to that ceiling (~8192 tok ~= 105s, 0 code blocks, no progress)
    # before the iteration-boundary timeout can fire - the dominant slow-task cost
    # in the gnosis-0.2 triage sweep (RCA 2026-06-28). 2048 is ~6-10x a legit root
    # turn (corpus/icd answers ~200-330 tok) so it ONLY bounds degenerate stutters;
    # sub-calls are untouched via root_max_tokens (NOT max_decode_tokens), so
    # extraction-map sub-calls keep the 8192 headroom (ADR-0018). ~105s -> ~26s.
    root_max_tokens: int = 2048
    # Corpus-NIAH failure-mode guards (2026-07-08 RCA; both default OFF so
    # teacher/trajectory-generation harnesses see unmodified behavior - the
    # leaf_prose_guard convention. Eval/serving callers opt in explicitly.)
    # reject_code_shaped_answers: bounce a code/tool-syntax final answer (e.g. a
    # sub-LLM's `find("...")` echo shipped via answer['content']) back for an
    # in-loop revision instead of terminating on it.
    reject_code_shaped_answers: bool = False
    # reject_self_context_delegation: when custom tools hold the data and a
    # sub-call's context= is the task's own briefing, reject deterministically
    # with a pivot-to-tools hint instead of paying for a futile send.
    reject_self_context_delegation: bool = False


VETTED = Defaults()


@dataclass(frozen=True)
class Runtime:
    """Resolved server facts (Tier B). slots drives map-reduce fan-out."""
    slots: int
    ctx: int | None = None


@dataclass(frozen=True)
class MemoryConfig:
    """ADR-0005 memory wiring, mapped to build_memory_harness_from_config."""
    bank_dir: str
    embed_model: str
    reflect_model: str
    embed_url: str | None = None
    embed_api_key: str | None = None
    # Distill endpoint: when set, reflect runs on its own server (e.g. a small
    # neutral model like Gemma 4 e4b on :8082) instead of the inference server router,
    # which must not swap. Defaults to the inference server base_url.
    reflect_url: str | None = None
    reflect_api_key: str | None = None
    k_max: int | None = None
    min_cosine: float | None = None
    # Trace distillation is mechanical JSON extraction: thinking OFF + bounded
    # output by default, so a reasoning reflect_model can't emit a giant CoT per
    # solve (the dominant memory overhead / GPU-contention source).
    reflect_enable_thinking: bool = False
    reflect_max_tokens: int | None = 512
    # The distiller samples at temperature 1.0 (diverse lesson phrasings), distinct
    # from the deterministic RLM solve path (Defaults.rlm_temperature=0.0).
    reflect_temperature: float = 1.0
    # Defer distillation until Harness.record_outcome(correct), so a scoring
    # caller learns only from correct solves (avoids poisoning the bank with
    # give-up lessons distilled from failed tasks).
    defer_collect: bool = False
    # Contrastive failure channel (ADR-0010): also learn NEGATIVE guard rules from
    # WRONG solves (requires defer_collect + a scoring caller). Default off
    # preserves correct-only. max_inject_negatives caps negative guard entries per
    # injected block so failure lessons cannot crowd out positive recipes.
    learn_from_failure: bool = False
    max_inject_negatives: int = 2
    # Document-signature retrieval gate: when True, each experience is stamped
    # with a ctx_sig tag from its source document and retrieval excludes entries
    # whose document conflicts with the one being solved (the bare-question
    # embedding alone cannot tell two same-question, different-document tasks
    # apart). Default off keeps the embedding-only path byte-identical.
    context_signature: bool = False
    # Write-only cold baseline: when True, the memory layer distills and writes
    # experiences as usual but injects NONE (retrieval short-circuits to empty),
    # so every solve is a true first-exposure. The cold/warm eval sets this for
    # its cold phase so a later cold task can't read memories written by earlier
    # cold tasks, while the bank still ends populated for the warm phase. Default
    # off keeps the standard read+write path.
    freeze_retrieval: bool = False
    # Telemetry sink for retrieve/collect events. Pass
    # prehend.metrics.memory_observer() to emit the localai_prehend_memory_*
    # Prometheus series; None -> MemoryHarness installs a no-op NullObserver, so
    # the no-metrics path is byte-identical. This is the only seam the top-level
    # Harness(memory=...) API exposes onto build_memory_harness_from_config's
    # observer= parameter.
    observer: MemoryObserver | None = None


def _default_probe(base_url: str, api_key: str) -> Runtime | None:
    """Best-effort llama-server probe. Returns None if facts are unavailable."""
    root = base_url.rstrip("/")
    if root.endswith("/v1"):
        root = root[: -len("/v1")]
    try:
        with urllib.request.urlopen(f"{root}/props", timeout=5) as r:
            props = json.loads(r.read())
        gen = props.get("default_generation_settings", {}) or {}
        ctx = gen.get("n_ctx") or None
        slots = props.get("total_slots") or gen.get("n_parallel") or 0
        if not slots or slots <= 0:
            return None
        return Runtime(slots=int(slots), ctx=int(ctx) if ctx else None)
    except Exception:
        return None


def detect_runtime(
    base_url: str,
    *,
    api_key: str = "not-needed",
    probe: Callable[[str, str], Runtime | None] | None = None,
) -> Runtime | None:
    """Hybrid Tier-B detection. None means 'ambiguous, caller should fall back'."""
    p = probe or _default_probe
    try:
        rt = p(base_url, api_key)
    except Exception:
        return None
    if rt is None or rt.slots <= 0:
        return None
    return rt


class Harness:
    """High-level entry point that assembles SRLM from vetted defaults + resolved runtime.

    Sub-calls may target a separate weight-shared worker via ``subcall_base_url``
    (a second llama-server sharing the master's weights over CUDA IPC but with
    its own private KV pool); budget and fan-out then come from that worker's
    runtime. ``subcall_base_url=None`` keeps the single-server path. See ADR-0013.
    """

    def __init__(
        self,
        model: str,
        base_url: str,
        *,
        subcall_base_url: str | None = None,
        subcall_runtime: "Runtime | str | None" = None,
        api_key: str = "not-needed",
        timeout: float | None = None,
        runtime: "Runtime | str" = "auto",
        defaults: Defaults | None = None,
        system_addendum: str | None = None,
        subcall_context_limit: int | None = None,
        dynamic_kv_pool: bool | None = None,
        subcall_verifier=None,
        answer_verifier=None,
        max_answer_retries: int | None = None,
        custom_tools: dict | None = None,
        observability: Callable[[object], None] | None = None,
        logger=None,
        memory=None,            # behavior added in Task 4
        direct_threshold: int | None = None,
        n_candidates: int | None = None,
        candidate_temperature: float | None = None,
        candidate_parallel: int | None = None,
        confidence_elicitation: bool | None = None,
        scheduler_max_concurrent: int | None = None,
        scheduler_coordination_dir: str | None = None,
    ):
        d = defaults or VETTED
        self.runtime = self._resolve_runtime(runtime, base_url, api_key, d)

        # Dual-instance weight-shared inference server (ADR-0013): sub-calls may target a
        # separate llama-server worker that shares the master's weights via CUDA
        # IPC but holds its OWN private KV pool. When subcall_base_url is set the
        # sub-call backend routes there and its budget/fan-out come from the
        # WORKER's runtime (its dedicated pool, its slots) - not the
        # orchestrator's. When None, the worker collapses onto the orchestrator
        # endpoint and behavior is byte-identical to the single-server path.
        eff_subcall_url = subcall_base_url or base_url
        if subcall_runtime is not None:
            # An explicit worker runtime always wins (mirrors how `runtime` is
            # honored for the orchestrator), whether or not a worker URL is set.
            self.subcall_runtime = self._resolve_runtime(
                subcall_runtime, eff_subcall_url, api_key, d
            )
        elif subcall_base_url is None:
            self.subcall_runtime = self.runtime
        else:
            self.subcall_runtime = self._resolve_runtime("auto", eff_subcall_url, api_key, d)
        self.subcall_base_url = eff_subcall_url

        # Resolve the effective sub-call context limit once (param > Defaults
        # field > runtime.ctx > get_context_limit(model)). Threaded into the
        # SRLM/RLM (guard + prompt) and the LocalREPL (llm_query guard). No env
        # var in core - Tier-B is explicit args (harness-api-design "no env hack").
        explicit_limit = (
            subcall_context_limit
            if subcall_context_limit is not None
            else d.subcall_context_limit
        )
        # The resolved limit is the server's SHARED context pool (n_ctx). Under
        # --kv-unified that pool is split across the up-to-`slots` concurrent
        # sub-calls a single map-reduce fans out, so the per-call guard budget
        # is pool // slots - else their sum exhausts the shared KV cache and the
        # server 500s every concurrent sub-call ("Context size has been
        # exceeded"). See token_utils.per_call_subcall_budget.
        shared_pool = resolve_subcall_limit(
            model, explicit=explicit_limit, runtime_ctx=self.subcall_runtime.ctx
        )
        # Dynamic-pool engines (sglang) skip the per-slot division: their paged KV
        # pool evicts under contention rather than 500ing, so each sub-call gets
        # the full resolved pool (the per-request context-length cap). The
        # llama.cpp --kv-unified path keeps pool // slots (ADR-0012). Param >
        # Defaults field.
        use_dynamic = dynamic_kv_pool if dynamic_kv_pool is not None else d.dynamic_kv_pool
        eff_subcall_limit = (
            shared_pool
            if use_dynamic
            else per_call_subcall_budget(shared_pool, self.subcall_runtime.slots)
        )

        # Solve-path sampling body shared by the orchestrator and sub-call
        # backends. seed rides here (same seam as temperature) so it lands on
        # every chat/completion the OpenAI client makes; None leaves it out.
        solve_extra_body = {"temperature": d.rlm_temperature}
        if d.seed is not None:
            solve_extra_body["seed"] = d.seed
        backend_kwargs = {
            "model_name": model, "base_url": base_url, "api_key": api_key,
            "max_retries": d.max_retries, "stream": d.stream,
            "default_extra_body": {
                **solve_extra_body,
                "chat_template_kwargs": {"enable_thinking": d.rlm_enable_thinking},
            },
        }
        subcall_kwargs = dict(backend_kwargs)
        subcall_kwargs["base_url"] = eff_subcall_url
        subcall_kwargs["default_extra_body"] = {
            **solve_extra_body,
            "chat_template_kwargs": {"enable_thinking": d.subcall_enable_thinking},
        }
        srlm_kwargs = dict(
            backend="openai",
            backend_kwargs=backend_kwargs,
            other_backends=["openai"],
            other_backend_kwargs=[subcall_kwargs],
            environment="local",
            environment_kwargs={
                "max_output_chars": d.max_output_chars,
                "subcall_context_limit": eff_subcall_limit,
                "model_name": model,
            },
            subcall_context_limit=eff_subcall_limit,
            max_iterations=d.max_iterations,
            max_depth=d.max_depth,
            max_errors=d.max_errors,
            max_timeout=timeout,
            max_concurrent_subcalls=self.subcall_runtime.slots,
            soft_timeout_pct=d.soft_timeout_pct,
            root_max_tokens=d.root_max_tokens,
            reject_code_shaped_answers=d.reject_code_shaped_answers,
            reject_self_context_delegation=d.reject_self_context_delegation,
            logger=logger,
            verbose=False,
        )
        if system_addendum is not None:
            srlm_kwargs["custom_system_prompt"] = RLM_SYSTEM_PROMPT + "\n\n" + system_addendum
        if subcall_verifier is not None:
            srlm_kwargs["subcall_verifier"] = subcall_verifier
        if answer_verifier is not None:
            srlm_kwargs["answer_verifier"] = answer_verifier
        if max_answer_retries is not None:
            srlm_kwargs["max_answer_retries"] = max_answer_retries
        if custom_tools is not None:
            srlm_kwargs["custom_tools"] = custom_tools

        for _name, _val in (
            ("direct_threshold", direct_threshold),
            ("n_candidates", n_candidates),
            ("candidate_temperature", candidate_temperature),
            ("candidate_parallel", candidate_parallel),
            ("confidence_elicitation", confidence_elicitation),
            ("scheduler_max_concurrent", scheduler_max_concurrent),
            ("scheduler_coordination_dir", scheduler_coordination_dir),
        ):
            if _val is not None:
                srlm_kwargs[_name] = _val

        self.srlm = SRLM(**srlm_kwargs)
        if observability is not None:
            observability(self.srlm)
        self.inference_client = self.srlm
        if memory is not None:
            from prehend.memory.factory import build_memory_harness_from_config
            tight = {k: v for k, v in (("k_max", memory.k_max),
                                       ("min_cosine", memory.min_cosine)) if v is not None}
            self.inference_client = build_memory_harness_from_config(
                self.srlm,
                bank_dir=memory.bank_dir,
                base_url=base_url,
                embed_model=memory.embed_model,
                reflect_model=memory.reflect_model,
                api_key=api_key,
                embed_base_url=memory.embed_url,
                embed_api_key=memory.embed_api_key,
                reflect_base_url=memory.reflect_url,
                reflect_api_key=memory.reflect_api_key,
                reflect_enable_thinking=memory.reflect_enable_thinking,
                reflect_max_tokens=memory.reflect_max_tokens,
                reflect_temperature=memory.reflect_temperature,
                defer_collect=memory.defer_collect,
                learn_from_failure=memory.learn_from_failure,
                max_inject_negatives=memory.max_inject_negatives,
                context_signature=memory.context_signature,
                freeze_retrieval=memory.freeze_retrieval,
                observer=memory.observer,
                **tight,
            )

    def _resolve_runtime(self, runtime, base_url, api_key, d: Defaults) -> Runtime:
        if isinstance(runtime, Runtime):
            return runtime
        detected = detect_runtime(base_url, api_key=api_key)
        if detected is not None:
            return detected
        _log.info("harness: runtime probe ambiguous; falling back to slots=%d",
                  d.max_concurrent_subcalls)
        return Runtime(slots=d.max_concurrent_subcalls)

    def completion(self, context: str, query: str) -> "RLMChatCompletion":
        return self.inference_client.completion(context, query)

    def record_outcome(self, correct: bool | None = True) -> None:
        """Distill the last solve when memory uses deferred collection.

        Call after scoring the answer: ``True``/``None`` distills a positive
        experience. ``correct is False`` drops the pending solve, UNLESS
        ``MemoryConfig.learn_from_failure`` is set, in which case it distills a
        negative guard rule (the contrastive failure channel, ADR-0010). No-op
        when memory is off or not deferring. Best-effort: never raises.
        """
        collect = getattr(self.inference_client, "collect_pending", None)
        if collect is not None:
            try:
                collect(correct)
            except Exception:
                pass
