import time
from collections.abc import Callable
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from prehend.clients import BaseLM, get_client
from prehend.core.lm_handler import DEFAULT_MAX_DECODE_TOKENS, LMHandler
from prehend.core.types import (
    ClientBackend,
    CodeBlock,
    EnvironmentType,
    REPLResult,
    RLMChatCompletion,
    RLMIteration,
    RLMMetadata,
    UsageSummary,
)
from prehend.core.verifier import REJECTION_PREFIX, SubcallReview, SubcallVerifier
from prehend.environments import BaseEnv, SupportsPersistence, get_environment
from prehend.environments.leaf_prose_guard import looks_like_code_reply
from prehend.logger import RLMLogger, VerbosePrinter
from prehend.utils.exceptions import (
    BudgetExceededError,
    CancellationError,
    ErrorThresholdExceededError,
    TimeoutExceededError,
    TokenLimitExceededError,
)
from prehend.utils.parsing import (
    find_code_blocks,
    format_iteration,
)
from prehend.utils.prompts import (
    RLM_SYSTEM_PROMPT,
    QueryMetadata,
    build_rlm_system_prompt,
    build_user_prompt,
)
from prehend.utils.rlm_utils import filter_sensitive_keys
from prehend.utils.subcall_guard import oversize_rejection, recommended_chunk_chars
from prehend.utils.token_utils import count_tokens, get_context_limit

# Injected (as an assistant turn) to force a final answer once iterations are
# exhausted. Because it lands as an assistant message, the model frequently
# CONTINUES it - echoing the sentence verbatim before the real answer. We strip
# that leading echo from the returned answer (see _strip_forcing_echo).
_FORCE_FINAL_MSG = "Please provide a final answer to the user's question based on the information provided."
_FORCE_FINAL_REVISE_MSG = "Your final answer did not pass verification; revise it."


def _strip_forcing_echo(response: str) -> str:
    """Drop a leading verbatim echo of the forced-final prompt from a response.

    No-op when the response does not start with the echo, or when the echo is the
    ONLY content (a degenerate generation - keep it as-is rather than manufacture
    an empty answer that downstream would read as a refusal)."""
    if not response:
        return response
    stripped = response.lstrip()
    if stripped[: len(_FORCE_FINAL_MSG)].lower() == _FORCE_FINAL_MSG.lower():
        remainder = stripped[len(_FORCE_FINAL_MSG):].lstrip(" \n\r\t")
        if remainder:
            return remainder
    return response


# Soft-budget wrap-up (rlm-trainer eval finding #4, 2026-06-16). On uncovered or
# hard tasks the model over-searches until the HARD deadline (max_timeout) elapses
# - the run then 504s with no answer, or its reasoning degenerates into a token
# loop. Once a fraction (soft_timeout_pct) of max_timeout has passed, inject this
# ONE-TIME message so the model wraps up from what it already has, or refuses,
# while budget remains. The librarian overrides this with its exact no-coverage
# sentence so a wrap-up on an uncovered ask becomes a clean, scorable refusal.
_SOFT_BUDGET_MSG = (
    "You are running low on time. Stop searching and reading more documents now. "
    "Using ONLY the information you have already gathered this run, write your "
    "single best final answer and set answer['ready'] = True this turn. If what "
    "you have read does not actually answer the question, do not guess - say "
    "plainly that the answer was not found."
)


# Rejection feedback for a code-shaped final answer (reject_code_shaped_answers).
# Observed corpus-NIAH failure (2026-06-27, rlm-trainer): a sub-LLM echoes tool
# syntax (`find("587X5749B")`) instead of an answer, and the orchestrator ships
# that echo verbatim via answer['content']. Shape-check the final answer and
# bounce it back for one in-loop revision instead of terminating on it.
_CODE_SHAPED_ANSWER_MSG = (
    "Your final answer is code or tool syntax, not an answer. If that was a REPL "
    "action you meant to take, run it inside a ```repl block instead. Then answer "
    "the original question in plain prose and set answer['ready'] = True again. "
    "If a sub-LLM gave you that text, it could not see your REPL tools - look the "
    "answer up yourself using the tools/variables in your REPL."
)

# Companion to the code-shape gate: a single-block assign-and-ship
# (`x = llm_query(...); answer['content'] = x; answer['ready'] = True`) captures
# a sub-call REJECTION hint into the answer before the model ever sees it
# (header-fix run 2026-07-08: 9/60 tasks shipped the futile-delegation hint
# verbatim). Every guard/verifier rejection contains "rejected this call", so a
# final answer carrying it is a guard echo, never an answer.
_REJECTION_MARKER = "rejected this call"

_REJECTION_ECHO_MSG = (
    "Your final answer contains a sub-call rejection notice, not an answer - "
    "that sub-call was rejected and returned no data. Do NOT assign a sub-call "
    "result to answer['content'] in the same repl block that makes the call: "
    "print() the result first and read it. Follow the notice's guidance (use "
    "your REPL tools/variables directly), then set answer['ready'] = True once "
    "answer['content'] holds the actual answer."
)


def _is_rejection_echo(answer: str) -> bool:
    return _REJECTION_MARKER in (answer or "")[:400]


def _soft_budget_due(
    elapsed: float,
    max_timeout: float | None,
    soft_pct: float | None,
    already_fired: bool,
) -> bool:
    """True when the soft-budget wrap-up message should be injected now.

    Fires at most once per completion, when more than ``soft_pct`` of
    ``max_timeout`` has elapsed. No-op unless both ``max_timeout`` and
    ``soft_pct`` (strictly between 0 and 1) are set."""
    if already_fired or soft_pct is None or max_timeout is None:
        return False
    if not (0.0 < soft_pct < 1.0):
        return False
    return elapsed >= soft_pct * max_timeout


def _guard_escalation_due(aborts: int, limit: int | None, already_fired: bool) -> bool:
    """True when the repeat-guard escalation wrap-up should be injected now.

    Fires at most once per completion, once the cumulative repeat-guard abort count
    for the run reaches ``limit``. No-op unless ``limit`` is a positive int. This is
    the count-based sibling of ``_soft_budget_due`` (which is time-based): a single
    looping completion is aborted cheaply by the client guard, but an ask that keeps
    re-entering the loop is forced to wrap up (answer-or-refuse) after ``limit``
    aborts instead of producing a fast-but-empty degeneration."""
    if already_fired or limit is None or limit <= 0:
        return False
    return aborts >= limit


class RLM:
    """
    Recursive Language Model class that the user instantiates and runs on their tasks.

    Each completion() call spawns its own environment and LM handler, which are
    cleaned up when the call completes.
    """

    def __init__(
        self,
        backend: ClientBackend = "openai",
        backend_kwargs: dict[str, Any] | None = None,
        environment: EnvironmentType = "local",
        environment_kwargs: dict[str, Any] | None = None,
        depth: int = 0,
        max_depth: int = 1,
        max_iterations: int = 30,
        max_budget: float | None = None,
        max_timeout: float | None = None,
        max_tokens: int | None = None,
        max_errors: int | None = None,
        custom_system_prompt: str | None = None,
        other_backends: list[ClientBackend] | None = None,
        other_backend_kwargs: list[dict[str, Any]] | None = None,
        logger: RLMLogger | None = None,
        verbose: bool = False,
        persistent: bool = False,
        custom_tools: dict[str, Any] | None = None,
        custom_sub_tools: dict[str, Any] | None = None,
        compaction: bool = False,
        compaction_threshold_pct: float = 0.85,
        max_concurrent_subcalls: int = 4,
        subcall_max_tokens: int | None = None,
        subcall_max_timeout: float | None = None,
        subcall_verifier: SubcallVerifier | None = None,
        answer_verifier: Any = None,
        max_answer_retries: int = 2,
        reject_code_shaped_answers: bool = False,
        reject_self_context_delegation: bool = False,
        clean_retry_on_error: bool = False,
        subcall_extra_body: dict[str, Any] | None = None,
        root_max_tokens: int | None = None,
        max_decode_tokens: int | None = DEFAULT_MAX_DECODE_TOKENS,
        soft_timeout_pct: float | None = None,
        soft_timeout_message: str | None = None,
        repeat_guard_abort_limit: int | None = None,
        scheduler_max_concurrent: int | None = None,
        scheduler_aging_interval: float | None = 30.0,
        scheduler_coordination_dir: str | Path | None = None,
        on_subcall_start: Callable[[int, str, str], None] | None = None,
        on_subcall_complete: Callable[[int, str, float, str | None], None] | None = None,
        on_iteration_start: Callable[[int, int], None] | None = None,
        on_iteration_complete: Callable[[int, int, float], None] | None = None,
        child_max_iterations: int | None = None,
        child_system_prompt: str | None = None,
        subcall_context_limit: int | None = None,
    ):
        """
        Args:
            backend: The backend to use for the RLM.
            backend_kwargs: The kwargs to pass to the backend.
            environment: The environment to use for the RLM.
            environment_kwargs: The kwargs to pass to the environment.
            depth: The current depth of the RLM (0-indexed).
            max_depth: The maximum depth of recursion. When depth >= max_depth, falls back to plain LM completion.
            max_iterations: The maximum number of iterations of the RLM.
            max_budget: Maximum budget in USD. Execution stops if exceeded. Requires cost-tracking backend (e.g., OpenRouter).
            max_timeout: Maximum execution time in seconds. Execution stops if exceeded, returning best answer if available.
            max_tokens: Maximum total tokens (input + output). Execution stops if exceeded, returning best answer if available.
            max_errors: Maximum consecutive errors before stopping. Execution stops if exceeded, returning best answer if available.
            custom_system_prompt: The custom system prompt to use for the RLM.
            other_backends: A list of other client backends that the environments can use to make sub-calls.
            other_backend_kwargs: The kwargs to pass to the other client backends (ordered to match other_backends).
            logger: The logger to use for the RLM.
            verbose: Whether to print verbose output in rich to console.
            persistent: If True, reuse the environment across completion() calls for multi-turn conversations.
            custom_tools: Dict of custom functions/tools available in the REPL. Keys are function names,
                values are callable functions. These are injected into the REPL globals.
            custom_sub_tools: Dict of custom tools for sub-agents (llm_query calls). If None, inherits
                from custom_tools. Pass an empty dict {} to disable tools for sub-agents.
            compaction: If True, keep full root model history in REPL variable `history` and compact
                when root context reaches compaction_threshold_pct of the model's context limit.
            compaction_threshold_pct: When compaction is on, trigger summarization when root
                message token count reaches this fraction of the model context limit (default 0.85).
            max_concurrent_subcalls: Maximum number of parallel threads for rlm_query_batched subcalls.
                Each child RLM runs in its own thread. Default 4.
            subcall_max_tokens: Generation cap (max_tokens) applied to every llm_query /
                llm_query_batched sub-call. Bounds runaway generations on greedy local models.
                Root orchestrator calls are not capped. None (default) leaves sub-calls uncapped.
                Requires a backend whose completion() accepts max_tokens (openai backend does).
            subcall_max_timeout: Wall-clock cap in seconds on each rlm_query /
                rlm_query_batched child RLM. The child gets min(parent's remaining
                max_timeout, this cap), so a single delegated child cannot consume
                the parent's entire remaining budget. Applies even when max_timeout
                is None. Inherited by children (bounds grandchildren too). None
                (default) gives each child the full remaining budget.
            subcall_verifier: Strategy verifier reviewing every llm_query /
                rlm_query sub-call before it executes (see core/verifier.py).
                A veto returns "Strategy verifier rejected this call: <reason>"
                as the call's result instead of executing it. The same instance
                is shared with recursion children so resubmission memory and
                veto telemetry span the whole tree. None (default) disables
                review.
            subcall_extra_body: Request-body extras merged into every sub-call
                (llm_query / llm_query_batched / the max-depth rlm_query leaf
                fallback), e.g. {"chat_template_kwargs": {"enable_thinking":
                False}} to skip gemma's thought channel on mechanical calls.
                Root orchestrator calls are unaffected. Inherited by recursion
                children. Requires a backend whose completion() accepts
                extra_body (openai backend does). None (default) sends
                sub-calls unmodified.
            max_decode_tokens: Hard per-generation output ceiling applied to BOTH
                root and sub-calls as min(path-specific cap, this). Unlike
                root_max_tokens / subcall_max_tokens (default None = uncapped),
                this defaults ON (8192) so no caller - including SRLM candidate
                trajectories that omit the explicit caps - can run a single
                decode unbounded into the shared KV pool (rlm-trainer #7). Pass
                None to disable.
            root_max_tokens: Generation cap for ROOT orchestrator calls,
                including the forced final answer after iteration exhaustion.
                Bounds root-path runaway generations the way subcall_max_tokens
                bounds sub-calls. Set generously (real answers run a few
                thousand tokens). Inherited by recursion children. None
                (default) leaves root calls uncapped.
            soft_timeout_pct: If set (0 < pct < 1) together with max_timeout,
                inject a one-time wrap-up message once this fraction of
                max_timeout has elapsed, so the model answers from what it has
                gathered (or refuses) before the hard deadline 504s or its
                reasoning degenerates. Converts the slow tail into clean
                completions/refusals and caps tail latency. None (default) =
                off; no behavior change.
            soft_timeout_message: Override text for the wrap-up message
                (default _SOFT_BUDGET_MSG). Use a domain-specific wording, e.g.
                a librarian's exact no-coverage refusal sentence so a wrap-up on
                an uncovered question scores as a clean refusal.
            scheduler_max_concurrent: If set, create a priority RequestScheduler shared by all
                backend clients, capping in-flight requests and enabling context-contention
                retries at exclusive (p1) priority. Match this to the inference server's slot
                count (llama-server --parallel). None (default) disables scheduling.
            scheduler_aging_interval: Seconds of queue wait worth one priority level for
                p2-p5 requests (anti-starvation aging: an old "low" eventually outranks a
                fresh "high"). None disables aging. Only used when scheduler_max_concurrent
                is set. Default 30.0.
            scheduler_coordination_dir: If set (with scheduler_max_concurrent), directory of
                cross-process lock files extending contention-retry (p1) exclusivity across
                OS processes that target the same server. Opt-in; same host only. None
                (default) keeps coordination in-process.
            on_subcall_start: Callback fired when a child RLM starts. Args: (depth, model, prompt_preview).
            on_subcall_complete: Callback fired when a child RLM completes. Args: (depth, model, duration, error_or_none).
            on_iteration_start: Callback fired when an iteration starts. Args: (depth, iteration_num).
            on_iteration_complete: Callback fired when an iteration completes. Args: (depth, iteration_num, duration).
        """
        # Store config for spawning per-completion
        self.backend = backend
        self.backend_kwargs = backend_kwargs
        self.environment_type = environment
        self.environment_kwargs = (
            environment_kwargs.copy() if environment_kwargs is not None else {}
        )
        # Validate other_backends: currently only support one additional backend
        if other_backends is not None:
            if len(other_backends) != 1:
                raise ValueError(
                    "We currently only support one additional backend for the recursive sub-calls! "
                    "This model will be the model used for recursive sub-calls, but this will change in the future"
                )

        self.other_backends = other_backends
        self.other_backend_kwargs = other_backend_kwargs

        # Custom tools: functions available in the REPL environment
        self.custom_tools = custom_tools
        # Sub-tools: if None, inherit from custom_tools; if {}, no tools for sub-agents
        self.custom_sub_tools = custom_sub_tools if custom_sub_tools is not None else custom_tools

        self.compaction = compaction
        self.compaction_threshold_pct = compaction_threshold_pct
        self.max_concurrent_subcalls = max_concurrent_subcalls
        self.subcall_max_tokens = subcall_max_tokens
        self.subcall_max_timeout = subcall_max_timeout
        self.subcall_verifier = subcall_verifier
        # answer_verifier: callable (answer:str) -> (ok:bool, feedback:str|None).
        # When set, the final answer is checked at answer["ready"]; a reject (with
        # retries left) re-prompts IN-LOOP (warm prefix) instead of terminating.
        self.answer_verifier = answer_verifier
        self.max_answer_retries = max_answer_retries
        # reject_code_shaped_answers: shape-check the final answer BEFORE the user
        # answer_verifier; a code/tool-syntax answer (leaf_prose_guard shape rules,
        # e.g. a `find("...")` sub-LLM echo shipped via answer['content']) is bounced
        # back with _CODE_SHAPED_ANSWER_MSG for an in-loop revision, sharing the
        # max_answer_retries budget. Default off: injected feedback turns would
        # otherwise leak into teacher/trajectory-generation runs (same opt-in
        # convention as leaf_prose_guard).
        self.reject_code_shaped_answers = reject_code_shaped_answers
        # reject_self_context_delegation: LocalREPL futile-delegation guard - a
        # sub-call whose context= is this task's own root context while custom
        # tools hold the data is rejected with a pivot-to-tools hint (see
        # local_repl / subcall_guard.self_context_rejection). Threaded to the
        # environment at spawn. Default off (teacher-run convention).
        self.reject_self_context_delegation = reject_self_context_delegation
        # clean_retry_on_error: when a REPL iteration errors, DROP the failed turn
        # (broken code + its echo) from the next prompt and feed only a compact error
        # note, so the model retries fresh instead of escalating/rebuilding the broken
        # attempt (the v9 error-escalation spiral). The failed iteration is still logged.
        self.clean_retry_on_error = clean_retry_on_error
        self.subcall_extra_body = subcall_extra_body
        self.root_max_tokens = root_max_tokens
        self.max_decode_tokens = max_decode_tokens
        # soft_timeout_pct: inject a one-time wrap-up message once this fraction of
        # max_timeout has elapsed (see _SOFT_BUDGET_MSG / _soft_budget_due). Default
        # None = off (no behavior change). soft_timeout_message overrides the text.
        self.soft_timeout_pct = soft_timeout_pct
        self.soft_timeout_message = soft_timeout_message or _SOFT_BUDGET_MSG
        self._soft_budget_fired = False
        # repeat_guard_abort_limit: after this many repeat-guard aborts in one
        # completion, inject the same wrap-up message (count-based sibling of the
        # time-based soft budget; see _guard_escalation_due). Default None = off.
        self.repeat_guard_abort_limit = repeat_guard_abort_limit
        self._guard_escalation_fired = False
        # The task this RLM was given, as seen by the verifier's whole-task
        # rule. Set per completion(); children record their delegated prompt.
        self._verifier_root: str | None = None
        self.scheduler_max_concurrent = scheduler_max_concurrent
        self.scheduler_aging_interval = scheduler_aging_interval
        self.scheduler_coordination_dir = scheduler_coordination_dir
        # Effective sub-model context window (tokens) for the input-size guard.
        # None = guard off. Threaded by the Harness (resolved once); follows the
        # recursion (children inherit) and arms both the LocalREPL llm_query guard
        # and the _subcall rlm_query guard, plus the prompt's chunk-budget wording.
        self.subcall_context_limit = subcall_context_limit

        self.depth = depth
        self.max_depth = max_depth
        self.max_iterations = max_iterations
        self.child_max_iterations = child_max_iterations if child_max_iterations is not None else max_iterations
        self.max_budget = max_budget
        self.max_timeout = max_timeout
        self.max_tokens = max_tokens
        self.max_errors = max_errors
        self.system_prompt = custom_system_prompt if custom_system_prompt else RLM_SYSTEM_PROMPT
        self.child_system_prompt = child_system_prompt
        self.logger = logger
        self.verbose = VerbosePrinter(enabled=verbose)

        # Event callbacks for live tree display
        self.on_subcall_start = on_subcall_start
        self.on_subcall_complete = on_subcall_complete
        self.on_iteration_start = on_iteration_start
        self.on_iteration_complete = on_iteration_complete

        # Tracking (cumulative across all calls including children)
        self._cumulative_cost: float = 0.0
        self._consecutive_errors: int = 0
        self._answer_retries: int = 0
        self._last_error: str | None = None
        self._best_partial_answer: str | None = None
        self._completion_start_time: float | None = None  # Set when completion() starts

        # Persistence support
        self.persistent = persistent
        self._persistent_env: SupportsPersistence | None = None

        # Validate persistence support at initialization
        if self.persistent:
            self._validate_persistent_environment_support()

        # Log metadata if logger is provided
        if self.logger or verbose:
            metadata = RLMMetadata(
                root_model=backend_kwargs.get("model_name", "unknown")
                if backend_kwargs
                else "unknown",
                max_depth=max_depth,
                max_iterations=max_iterations,
                backend=backend,
                backend_kwargs=filter_sensitive_keys(backend_kwargs) if backend_kwargs else {},
                environment_type=environment,
                environment_kwargs=filter_sensitive_keys(environment_kwargs)
                if environment_kwargs
                else {},
                other_backends=other_backends,
            )
            if self.logger:
                self.logger.log_metadata(metadata)
            self.verbose.print_metadata(metadata)

    @contextmanager
    def _spawn_completion_context(self, prompt: str | dict[str, Any]):
        """
        Spawn an LM handler and environment for a single completion call.

        When persistent=True, the environment is reused across calls.
        When persistent=False (default), creates fresh environment each call.
        """
        # Create client and wrap in handler
        client: BaseLM = get_client(self.backend, self.backend_kwargs)

        # Create other_backend_client if provided (for depth=1 routing)
        other_backend_client: BaseLM | None = None
        if self.other_backends and self.other_backend_kwargs:
            other_backend_client = get_client(self.other_backends[0], self.other_backend_kwargs[0])

        lm_handler = LMHandler(
            client,
            other_backend_client=other_backend_client,
            scheduler_max_concurrent=self.scheduler_max_concurrent,
            scheduler_aging_interval=self.scheduler_aging_interval,
            scheduler_coordination_dir=self.scheduler_coordination_dir,
            subcall_max_tokens=self.subcall_max_tokens,
            subcall_extra_body=self.subcall_extra_body,
            root_max_tokens=self.root_max_tokens,
            max_decode_tokens=self.max_decode_tokens,
            verifier=self.subcall_verifier,
            verifier_root=self._verifier_root,
        )

        # Register other clients to be available as sub-call options (by model name).
        # Reuse other_backend_client for the first entry so each (backend, kwargs)
        # pair is instantiated exactly once.
        if other_backend_client is not None:
            lm_handler.register_client(other_backend_client.model_name, other_backend_client)
            for backend, kwargs in zip(
                self.other_backends[1:],
                self.other_backend_kwargs[1:],
                strict=True,
            ):
                other_client: BaseLM = get_client(backend, kwargs)
                lm_handler.register_client(other_client.model_name, other_client)

        lm_handler.start()
        # Arm the run's wall-clock deadline on every client: streamed calls
        # (stream=True backends) self-abort between chunks once it passes, so
        # even a call that is mid-generation cannot block the run past
        # max_timeout. No-op when max_timeout is None.
        lm_handler.set_run_deadline(self.max_timeout)

        # Environment: reuse if persistent, otherwise create fresh
        if self.persistent and self._persistent_env is not None:
            environment = self._persistent_env
            # Defensive check: ensure environment supports persistence methods
            if not self._env_supports_persistence(environment):
                raise RuntimeError(
                    f"Persistent environment of type '{type(environment).__name__}' does not "
                    f"implement required methods (update_handler_address, add_context, get_context_count). "
                    f"This should have been caught at initialization."
                )
            environment.update_handler_address((lm_handler.host, lm_handler.port))
            environment.add_context(prompt)
        else:
            env_kwargs = self.environment_kwargs.copy()
            env_kwargs["lm_handler_address"] = (lm_handler.host, lm_handler.port)
            env_kwargs["context_payload"] = prompt
            env_kwargs["depth"] = self.depth + 1  # Environment depth is RLM depth + 1
            # For local/ipython environments with max_depth > 1, pass subcall callback for recursive RLM calls
            if self.environment_type in ("local", "ipython") and self.max_depth > 1:
                env_kwargs["subcall_fn"] = self._subcall
            # Pass custom tools to the environment
            if self.custom_tools is not None:
                env_kwargs["custom_tools"] = self.custom_tools
            if self.custom_sub_tools is not None:
                env_kwargs["custom_sub_tools"] = self.custom_sub_tools
            if self.compaction and self.environment_type == "local":
                env_kwargs["compaction"] = True
            if self.reject_self_context_delegation:
                env_kwargs["reject_self_context_delegation"] = True
            env_kwargs["max_concurrent_subcalls"] = self.max_concurrent_subcalls
            # Arm the LocalREPL input-size guard on llm_query / llm_query_batched.
            # The REPL needs BOTH the limit and the model name (for count_tokens).
            if self.subcall_context_limit is not None:
                env_kwargs["subcall_context_limit"] = self.subcall_context_limit
                env_kwargs["model_name"] = (self.backend_kwargs or {}).get("model_name")
            environment: BaseEnv = get_environment(self.environment_type, env_kwargs)

            if self.persistent:
                self._persistent_env = environment

        try:
            yield lm_handler, environment
        finally:
            # Abort in-flight sub-calls BEFORE stopping the server: handler
            # request threads are daemon threads, so without this a sub-call
            # abandoned by a timeout keeps generating (and SDK-retrying)
            # against the backend long after the run has returned.
            lm_handler.cancel_inflight()
            lm_handler.stop()
            if not self.persistent and hasattr(environment, "cleanup"):
                environment.cleanup()

    def _setup_prompt(self, prompt: str | dict[str, Any]) -> list[dict[str, Any]]:
        """
        Setup the system prompt for the RLM. Also include metadata about the prompt and build
        up the initial message history.
        """
        metadata = QueryMetadata(prompt)
        # When a sub-call context limit is set, advertise the RECOMMENDED (small,
        # latency-friendly) per-chunk char budget in the prompt so the model makes
        # several fast parallel chunks rather than 1-2 giant ones that prefill
        # slowly; the hard ceiling (safe_chunk_chars) still bounds the guard. Else
        # build_rlm_system_prompt uses its conservative default.
        subcall_char_budget = None
        if self.subcall_context_limit is not None:
            model_name = (self.backend_kwargs or {}).get("model_name", "") or ""
            subcall_char_budget = recommended_chunk_chars(self.subcall_context_limit, model_name)
        message_history = build_rlm_system_prompt(
            system_prompt=self.system_prompt,
            query_metadata=metadata,
            custom_tools=self.custom_tools,
            subcall_char_budget=subcall_char_budget,
        )
        if self.compaction:
            message_history[0]["content"] += (
                "\n\nThe full conversation history (trajectory segments and any summaries) "
                "is available in the REPL variable `history` as a list."
            )
        return message_history

    def completion(
        self, prompt: str | dict[str, Any], root_prompt: str | None = None
    ) -> RLMChatCompletion:
        """
        Recursive Language Model completion call. This is the main entry point for querying an RLM, and
        can replace a regular LM completion call.

        Spawns its own environment and LM handler for the duration of this call.

        Args:
            prompt: A single string or dictionary of messages to pass as context to the model.
            root_prompt: We allow the RLM's root LM to see a (small) prompt that the user specifies. A common example of this
            is if the user is asking the RLM to answer a question, we can pass the question as the root prompt.
        Returns:
            A final answer as a string.
        """
        time_start = time.perf_counter()
        self._completion_start_time = time_start
        # The verifier judges sub-calls against the task this RLM was given:
        # the root prompt (question) when provided, else the prompt itself.
        self._verifier_root = root_prompt or (prompt if isinstance(prompt, str) else None)

        # Reset tracking state for this completion
        self._consecutive_errors = 0
        self._answer_retries = 0
        self._last_error = None
        self._best_partial_answer = None
        self._soft_budget_fired = False
        self._guard_escalation_fired = False
        # If we're at max depth, the RLM is an LM, so we fallback to the regular LM.
        if self.depth >= self.max_depth:
            return self._fallback_answer(prompt)

        if self.logger:
            self.logger.clear_iterations()

        with self._spawn_completion_context(prompt) as (lm_handler, environment):
            message_history = self._setup_prompt(prompt)

            compaction_count = 0
            try:
                for i in range(self.max_iterations):
                    # Check timeout before each iteration
                    self._check_timeout(i, time_start)

                    # Soft budget: once a fraction of max_timeout has elapsed,
                    # inject a one-time wrap-up message (answer-or-refuse now)
                    # so the slow tail becomes a clean completion/refusal rather
                    # than a hard-deadline 504 or a degenerate token loop.
                    self._maybe_inject_soft_budget(
                        message_history, time.perf_counter() - time_start
                    )

                    # Repeat-guard escalation: if the client guard has aborted a
                    # looping completion enough times this ask, force the same
                    # wrap-up so a persistent looper becomes a clean answer/refusal
                    # rather than a fast-but-empty degeneration. Only poll the count
                    # when the feature is on (keeps it off the default-config path).
                    if self.repeat_guard_abort_limit:
                        self._maybe_inject_guard_escalation(
                            message_history, lm_handler.repeat_guard_aborts()
                        )

                    # Compaction: check if context needs summarization
                    if self.compaction and hasattr(environment, "append_compaction_entry"):
                        current_tokens, threshold_tokens, max_tokens = self._get_compaction_status(
                            message_history
                        )
                        self.verbose.print_compaction_status(
                            current_tokens, threshold_tokens, max_tokens
                        )
                        if current_tokens >= threshold_tokens:
                            compaction_count += 1
                            self.verbose.print_compaction()
                            message_history = self._compact_history(
                                lm_handler, environment, message_history, compaction_count
                            )

                    # Current prompt = message history + additional prompt suffix
                    context_count = (
                        environment.get_context_count()
                        if isinstance(environment, SupportsPersistence)
                        else 1
                    )
                    history_count = (
                        environment.get_history_count()
                        if isinstance(environment, SupportsPersistence)
                        else 0
                    )
                    current_prompt = message_history + [
                        build_user_prompt(root_prompt, i, context_count, history_count)
                    ]

                    if self.on_iteration_start:
                        try:
                            self.on_iteration_start(self.depth, i)
                        except Exception:
                            pass
                    iter_t0 = time.perf_counter()
                    iteration: RLMIteration = self._completion_turn(
                        prompt=current_prompt,
                        lm_handler=lm_handler,
                        environment=environment,
                    )
                    if self.on_iteration_complete:
                        try:
                            self.on_iteration_complete(
                                self.depth, i, time.perf_counter() - iter_t0
                            )
                        except Exception:
                            pass

                    # Check error/budget/token limits after each iteration
                    self._check_iteration_limits(iteration, i, lm_handler)

                    # The REPL signals completion by populating
                    # ``answer["content"]`` and setting ``answer["ready"] = True``.
                    # Each environment surfaces that on ``REPLResult.final_answer``.
                    final_answer = None
                    for block in iteration.code_blocks:
                        if getattr(block.result, "final_answer", None) is not None:
                            final_answer = block.result.final_answer
                            break
                    iteration.final_answer = final_answer

                    # Store as best partial answer (most recent response with content)
                    if iteration.response and iteration.response.strip():
                        self._best_partial_answer = iteration.response

                    # If logger is used, log the iteration.
                    if self.logger:
                        self.logger.log(iteration)

                    # Verbose output for this iteration
                    self.verbose.print_iteration(iteration, i + 1)

                    if final_answer is not None:
                        # Shape gate first: a code/tool-syntax final answer is never
                        # valid, and the user answer_verifier (e.g. citation
                        # enforcement) should only ever see a prose candidate.
                        if (self.reject_code_shaped_answers
                                and self._answer_retries < self.max_answer_retries
                                and (looks_like_code_reply(final_answer)
                                     or _is_rejection_echo(final_answer))):
                            self._answer_retries += 1
                            new_messages = format_iteration(iteration)
                            message_history.extend(new_messages)
                            fb_msg = {"role": "user", "content":
                                      _REJECTION_ECHO_MSG if _is_rejection_echo(final_answer)
                                      else _CODE_SHAPED_ANSWER_MSG}
                            message_history.append(fb_msg)
                            if self.compaction and hasattr(environment, "append_compaction_entry"):
                                environment.append_compaction_entry(new_messages + [fb_msg])
                            continue

                        # Answer-level verification (e.g. citation enforcement). On
                        # reject with retries remaining, record the rejected attempt +
                        # feedback and CONTINUE the loop instead of terminating, so the
                        # warm system+history prefix is reused (cheap in-loop revision,
                        # no full re-lookup). answer_verifier=None disables this entirely.
                        if (self.answer_verifier is not None
                                and self._answer_retries < self.max_answer_retries):
                            ok, feedback = self.answer_verifier(final_answer)
                            if not ok:
                                self._answer_retries += 1
                                new_messages = format_iteration(iteration)
                                message_history.extend(new_messages)
                                fb_msg = {"role": "user", "content": feedback
                                          or "Your final answer did not pass verification; revise it and set answer['ready'] = True again."}
                                message_history.append(fb_msg)
                                if self.compaction and hasattr(environment, "append_compaction_entry"):
                                    environment.append_compaction_entry(new_messages + [fb_msg])
                                continue

                        time_end = time.perf_counter()
                        usage = lm_handler.get_usage_summary()
                        self.verbose.print_final_answer(final_answer)
                        self.verbose.print_summary(i + 1, time_end - time_start, usage.to_dict())

                        # Store message history in persistent environment
                        if self.persistent and isinstance(environment, SupportsPersistence):
                            environment.add_history(message_history)

                        return RLMChatCompletion(
                            root_model=self.backend_kwargs.get("model_name", "unknown")
                            if self.backend_kwargs
                            else "unknown",
                            prompt=prompt,
                            response=final_answer,
                            usage_summary=usage,
                            execution_time=time_end - time_start,
                            metadata=self.logger.get_trajectory() if self.logger else None,
                        )

                    # Format the iteration for the next prompt.
                    iter_errored = any(
                        cb.result and cb.result.stderr for cb in iteration.code_blocks
                    )
                    if iter_errored and self.clean_retry_on_error:
                        # Anti-spiral: drop the failed turn (broken code + its echo) from
                        # the prompt context; feed only a compact error note so the model
                        # retries fresh and cannot escalate by "revising" its broken code.
                        errs = "\n".join(
                            cb.result.stderr.strip()
                            for cb in iteration.code_blocks
                            if cb.result and cb.result.stderr
                        )
                        new_messages = [{
                            "role": "user",
                            "content": (
                                "Your previous REPL code raised an error and has been "
                                "discarded (it is not shown, to keep the context clean):\n"
                                f"{errs[:800]}\n\n"
                                "Do NOT reconstruct or build on that code. Take a fresh, "
                                "simpler approach."
                            ),
                        }]
                    else:
                        new_messages = format_iteration(iteration)

                    # Update message history with the new messages.
                    message_history.extend(new_messages)
                    if self.compaction and hasattr(environment, "append_compaction_entry"):
                        environment.append_compaction_entry(new_messages)

            except KeyboardInterrupt:
                self.verbose.print_limit_exceeded("cancelled", "User interrupted execution")
                raise CancellationError(
                    partial_answer=self._best_partial_answer,
                    message="Execution cancelled by user (Ctrl+C)",
                ) from None
            except TimeoutExceededError as e:
                # A client whose run deadline passed raises mid-stream WITHOUT
                # a partial answer (it cannot see the run's accumulated state);
                # attach it here so no timeout path loses the salvage.
                if e.partial_answer is None:
                    e.partial_answer = self._best_partial_answer
                raise

            # Default behavior: we run out of iterations, provide one final answer
            time_end = time.perf_counter()
            try:
                final_answer = self._default_answer(message_history, lm_handler)
            except TimeoutExceededError as e:
                # Same deadline abort, but during the final wrap-up generation
                # after the iteration loop - the 2026-06-11 null-salvage path.
                if e.partial_answer is None:
                    e.partial_answer = self._best_partial_answer
                raise
            usage = lm_handler.get_usage_summary()
            self.verbose.print_final_answer(final_answer)
            self.verbose.print_summary(self.max_iterations, time_end - time_start, usage.to_dict())

            # Store message history in persistent environment
            if self.persistent and isinstance(environment, SupportsPersistence):
                environment.add_history(message_history)

            return RLMChatCompletion(
                root_model=self.backend_kwargs.get("model_name", "unknown")
                if self.backend_kwargs
                else "unknown",
                prompt=prompt,
                response=final_answer,
                usage_summary=usage,
                execution_time=time_end - time_start,
                metadata=self.logger.get_trajectory() if self.logger else None,
            )

    def _maybe_inject_soft_budget(
        self, message_history: list[dict[str, Any]], elapsed: float
    ) -> bool:
        """Inject the one-time soft-budget wrap-up message if it is due.

        Appends ``soft_timeout_message`` as a user turn so the next root call
        wraps up (answer-or-refuse) instead of reading more. Fires at most once
        per completion. Returns True iff it injected this call. No-op unless
        soft_timeout_pct + max_timeout are configured (default off)."""
        if not _soft_budget_due(
            elapsed, self.max_timeout, self.soft_timeout_pct, self._soft_budget_fired
        ):
            return False
        message_history.append({"role": "user", "content": self.soft_timeout_message})
        self._soft_budget_fired = True
        self.verbose.print_limit_exceeded(
            "soft-budget",
            f"{elapsed:.1f}s of {self.max_timeout:.1f}s "
            f"(>{self.soft_timeout_pct:.0%}) - forcing wrap-up",
        )
        return True

    def _maybe_inject_guard_escalation(
        self, message_history: list[dict[str, Any]], aborts: int
    ) -> bool:
        """Inject the wrap-up message once the per-ask repeat-guard abort count
        reaches ``repeat_guard_abort_limit``. Reuses ``soft_timeout_message`` (the
        answer-or-refuse policy text). Fires at most once per completion; no-op
        unless ``repeat_guard_abort_limit`` is configured. Returns True iff injected."""
        if not _guard_escalation_due(
            aborts, self.repeat_guard_abort_limit, self._guard_escalation_fired
        ):
            return False
        message_history.append({"role": "user", "content": self.soft_timeout_message})
        self._guard_escalation_fired = True
        self.verbose.print_limit_exceeded(
            "repeat-guard-escalation",
            f"{aborts} reasoning-loop aborts (>={self.repeat_guard_abort_limit}) "
            f"- forcing wrap-up",
        )
        return True

    def _check_timeout(self, iteration: int, time_start: float) -> None:
        """Raise TimeoutExceededError if the timeout has been exceeded."""
        if self.max_timeout is None:
            return
        elapsed = time.perf_counter() - time_start
        if elapsed > self.max_timeout:
            self.verbose.print_limit_exceeded(
                "timeout",
                f"{elapsed:.1f}s of {self.max_timeout:.1f}s",
            )
            raise TimeoutExceededError(
                elapsed=elapsed,
                timeout=self.max_timeout,
                partial_answer=self._best_partial_answer,
                message=(
                    f"Timeout exceeded after iteration {iteration}: "
                    f"{elapsed:.1f}s of {self.max_timeout:.1f}s limit"
                ),
            )

    def _check_iteration_limits(
        self, iteration: RLMIteration, iteration_num: int, lm_handler: LMHandler
    ) -> None:
        """Check error tracking, budget, and token limits after an iteration.

        Raises ErrorThresholdExceededError, BudgetExceededError, or TokenLimitExceededError
        if the respective limits are exceeded.
        """
        # Track errors from code execution (check stderr for errors)
        iteration_had_error = False
        for code_block in iteration.code_blocks:
            if code_block.result and code_block.result.stderr:
                iteration_had_error = True
                self._last_error = code_block.result.stderr
                break

        if iteration_had_error:
            self._consecutive_errors += 1
        else:
            self._consecutive_errors = 0  # Reset on success

        # Check error threshold
        if self.max_errors is not None and self._consecutive_errors >= self.max_errors:
            self.verbose.print_limit_exceeded(
                "errors",
                f"{self._consecutive_errors} consecutive errors (limit: {self.max_errors})",
            )
            raise ErrorThresholdExceededError(
                error_count=self._consecutive_errors,
                threshold=self.max_errors,
                last_error=self._last_error,
                partial_answer=self._best_partial_answer,
                message=(
                    "Error threshold exceeded: "
                    f"{self._consecutive_errors} consecutive errors "
                    f"(limit: {self.max_errors})"
                ),
            )

        # Check budget
        if self.max_budget is not None:
            current_usage = lm_handler.get_usage_summary()
            current_cost = current_usage.total_cost or 0.0
            self._cumulative_cost = current_cost
            if self._cumulative_cost > self.max_budget:
                self.verbose.print_budget_exceeded(self._cumulative_cost, self.max_budget)
                raise BudgetExceededError(
                    spent=self._cumulative_cost,
                    budget=self.max_budget,
                    message=(
                        f"Budget exceeded after iteration {iteration_num + 1}: "
                        f"spent ${self._cumulative_cost:.6f} "
                        f"of ${self.max_budget:.6f} budget"
                    ),
                )

        # Check token limit
        if self.max_tokens is not None:
            current_usage = lm_handler.get_usage_summary()
            total_tokens = current_usage.total_input_tokens + current_usage.total_output_tokens
            if total_tokens > self.max_tokens:
                self.verbose.print_limit_exceeded(
                    "tokens",
                    f"{total_tokens:,} of {self.max_tokens:,} tokens",
                )
                raise TokenLimitExceededError(
                    tokens_used=total_tokens,
                    token_limit=self.max_tokens,
                    partial_answer=self._best_partial_answer,
                    message=(
                        f"Token limit exceeded after iteration {iteration_num + 1}: "
                        f"{total_tokens:,} of {self.max_tokens:,} tokens"
                    ),
                )

    def _get_compaction_status(self, message_history: list[dict[str, Any]]) -> tuple[int, int, int]:
        """Return (current_tokens, threshold_tokens, max_tokens) for compaction."""
        model_name = (
            self.backend_kwargs.get("model_name", "unknown") if self.backend_kwargs else "unknown"
        )
        max_tokens = get_context_limit(model_name)
        current_tokens = count_tokens(message_history, model_name)
        threshold_tokens = int(self.compaction_threshold_pct * max_tokens)
        return current_tokens, threshold_tokens, max_tokens

    def _should_compact(self, message_history: list[dict[str, Any]]) -> bool:
        """True when root message history is at or over the compaction threshold."""
        current_tokens, threshold_tokens, _ = self._get_compaction_status(message_history)
        return current_tokens >= threshold_tokens

    def _compact_history(
        self,
        lm_handler: LMHandler,
        environment: BaseEnv,
        message_history: list[dict[str, Any]],
        compaction_count: int = 1,
    ) -> list[dict[str, Any]]:
        """
        Summarize current trajectory, append summary to REPL history, and return
        a short message_history with the summary as the new starting point.
        """
        summary_prompt = message_history + [
            {
                "role": "user",
                "content": (
                    "Summarize your progress so far. Include:\n"
                    "1. Which steps/sub-tasks you have completed and which remain.\n"
                    "2. Any concrete intermediate results (numbers, values, variable names) "
                    "you computed — preserve these exactly.\n"
                    "3. What your next action should be.\n"
                    "Be concise (1–3 paragraphs) but preserve all key results and your "
                    "current position in the task."
                ),
            }
        ]
        summary = lm_handler.completion(summary_prompt)
        if hasattr(environment, "append_compaction_entry"):
            environment.append_compaction_entry({"type": "summary", "content": summary})
        # Keep system + initial assistant (metadata), then summary + continue
        new_history = message_history[:2] + [
            {"role": "assistant", "content": summary},
            {
                "role": "user",
                "content": (
                    f"Your conversation has been compacted {compaction_count} time(s). "
                    "Continue from the above summary. Do NOT repeat work you have already "
                    "completed. Use SHOW_VARS() to check which REPL variables exist, "
                    "and check `history` for full context. "
                    "Your next action:"
                ),
            },
        ]
        return new_history

    def _completion_turn(
        self,
        prompt: str | dict[str, Any],
        lm_handler: LMHandler,
        environment: BaseEnv,
    ) -> RLMIteration:
        """
        Perform a single iteration of the RLM, including prompting the model
        and code execution + tool execution.
        """
        iter_start = time.perf_counter()
        response = lm_handler.completion(prompt)
        code_block_strs = find_code_blocks(response)
        code_blocks = []

        for code_block_str in code_block_strs:
            code_result: REPLResult = environment.execute_code(code_block_str)
            code_blocks.append(CodeBlock(code=code_block_str, result=code_result))

        iteration_time = time.perf_counter() - iter_start
        return RLMIteration(
            prompt=prompt,
            response=response,
            code_blocks=code_blocks,
            iteration_time=iteration_time,
        )

    def _default_answer(self, message_history: list[dict[str, Any]], lm_handler: LMHandler) -> str:
        """
        Default behavior if the RLM runs out of iterations and does not find a final answer.
        It will take the message history, and try to generate a final answer from it.
        """
        # role="user", not "assistant". During a repetition stall the history already ends
        # with an assistant message, so an assistant-role forcing sentence MERGES into the
        # model's own ramble and it is asked to continue that ramble with the instruction
        # attributed to itself (2026-07-10: a 192,065-char final assistant message).
        current_prompt = message_history + [
            {
                "role": "user",
                "content": _FORCE_FINAL_MSG,
            }
        ]
        response = lm_handler.completion(current_prompt)
        final_answer = _strip_forcing_echo(response)

        # The in-loop answer_verifier (see the iteration loop) is structurally bypassed here,
        # which is exactly the path a runaway loop always reaches. Over 51 runs on 2026-07-10,
        # 3 of 3 ungrounded answers arrived through this method, and 0 of the 47 runs that
        # never reached it were ungrounded. Apply the same check, with one revision attempt.
        # answer_verifier=None (trajectory-generation harnesses) disables this entirely.
        if self.answer_verifier is not None:
            ok, feedback = self.answer_verifier(final_answer)
            if not ok:
                current_prompt = current_prompt + [
                    {"role": "assistant", "content": response},
                    {"role": "user", "content": feedback or _FORCE_FINAL_REVISE_MSG},
                ]
                response = lm_handler.completion(current_prompt)
                final_answer = _strip_forcing_echo(response)

        if self.logger:
            self.logger.log(
                RLMIteration(
                    prompt=current_prompt,
                    response=response,
                    final_answer=final_answer,
                    code_blocks=[],
                )
            )

        return final_answer

    def _fallback_answer(self, message: str | dict[str, Any]) -> str:
        """
        Fallback behavior if the RLM is actually at max depth, and should be treated as an LM.
        """
        client: BaseLM = get_client(self.backend, self.backend_kwargs)
        response = client.completion(message)
        return response

    def _subcall(self, prompt: str, model: str | None = None) -> RLMChatCompletion:
        """
        Handle a subcall from the environment, potentially spawning a child RLM.

        This method is passed as a callback to LocalREPL to enable recursive RLM calls.
        When depth allows, it spawns a child RLM with its own REPL. At max depth,
        it falls back to a plain LM completion.

        Args:
            prompt: The prompt to process.
            model: Optional model name. If specified, the child RLM will use this model
                instead of inheriting the parent's default backend.

        Returns:
            The full RLMChatCompletion from either a child RLM or plain LM completion.
            On error, returns a completion with the error message as the response.
        """
        next_depth = self.depth + 1

        # Determine which backend/kwargs to use (model override or parent's default)
        if model is not None:
            child_backend_kwargs = (self.backend_kwargs or {}).copy()
            child_backend_kwargs["model_name"] = model
        else:
            child_backend_kwargs = self.backend_kwargs
        resolved_model = model or (child_backend_kwargs or {}).get("model_name", "unknown")

        # Input-size guard (deterministic, arithmetic): reject an oversized
        # rlm_query prompt before any send - covering both the leaf-fallback LM
        # call and the child RLM's first prompt. Returns the actionable
        # chunk-and-map-reduce hint as the call result (same shape as the
        # verifier-rejection completion below: zero usage / time).
        if self.subcall_context_limit is not None:
            hint = oversize_rejection(
                prompt if isinstance(prompt, str) else str(prompt),
                limit=self.subcall_context_limit,
                model=resolved_model if isinstance(resolved_model, str) else "",
            )
            if hint is not None:
                return RLMChatCompletion(
                    root_model=resolved_model,
                    prompt=prompt,
                    response=hint,
                    usage_summary=UsageSummary(model_usage_summaries={}),
                    execution_time=0.0,
                )

        # Strategy review before paying for anything - including the leaf
        # fallback below: a whole-task delegation is wasteful at every depth.
        if self.subcall_verifier is not None:
            verdict = self.subcall_verifier.review(
                SubcallReview(
                    kind="rlm_query",
                    prompt=prompt if isinstance(prompt, str) else str(prompt),
                    root_prompt=self._verifier_root,
                    depth=self.depth,
                )
            )
            if not verdict.approved:
                return RLMChatCompletion(
                    root_model=resolved_model,
                    prompt=prompt,
                    response=REJECTION_PREFIX + verdict.reason,
                    usage_summary=UsageSummary(model_usage_summaries={}),
                    execution_time=0.0,
                )

        # If we'd hit/exceed the cap, do a normal LM completion (no REPL)
        if next_depth >= self.max_depth:
            # Use other_backend if available, otherwise use main backend
            if self.other_backends and self.other_backend_kwargs:
                client = get_client(self.other_backends[0], self.other_backend_kwargs[0])
            else:
                client = get_client(self.backend, child_backend_kwargs or {})
            root_model = model or client.model_name
            # This fresh client carries none of the run's guards by default -
            # on 2026-06-12 an unguarded leaf fallback generated 60K+ tokens
            # five minutes past its run's expired deadline. Apply the sub-call
            # cap/extras and the run's remaining wall-clock budget.
            leaf_kwargs: dict[str, Any] = {}
            if self.subcall_max_tokens is not None:
                leaf_kwargs["max_tokens"] = self.subcall_max_tokens
            if self.subcall_extra_body is not None:
                leaf_kwargs["extra_body"] = self.subcall_extra_body
            if (
                self.max_timeout is not None
                and self._completion_start_time is not None
                and hasattr(client, "set_deadline")
            ):
                elapsed = time.perf_counter() - self._completion_start_time
                client.set_deadline(max(self.max_timeout - elapsed, 1.0))
            start_time = time.perf_counter()
            try:
                response = client.completion(prompt, **leaf_kwargs)
                end_time = time.perf_counter()
                model_usage = client.get_last_usage()
                usage_summary = UsageSummary(model_usage_summaries={root_model: model_usage})
                return RLMChatCompletion(
                    root_model=root_model,
                    prompt=prompt,
                    response=response,
                    usage_summary=usage_summary,
                    execution_time=end_time - start_time,
                )
            except Exception as e:
                end_time = time.perf_counter()
                return RLMChatCompletion(
                    root_model=root_model,
                    prompt=prompt,
                    response=f"Error: LM query failed at max depth - {e}",
                    usage_summary=UsageSummary(model_usage_summaries={}),
                    execution_time=end_time - start_time,
                )

        # Calculate remaining budget for child (if budget tracking enabled)
        remaining_budget = None
        if self.max_budget is not None:
            remaining_budget = self.max_budget - self._cumulative_cost
            if remaining_budget <= 0:
                return RLMChatCompletion(
                    root_model=resolved_model,
                    prompt=prompt,
                    response=(
                        "Error: Budget exhausted "
                        f"(spent ${self._cumulative_cost:.6f} of ${self.max_budget:.6f})"
                    ),
                    usage_summary=UsageSummary(model_usage_summaries={}),
                    execution_time=0.0,
                )

        # Calculate remaining timeout for child (if timeout tracking enabled)
        remaining_timeout = None
        if self.max_timeout is not None and self._completion_start_time is not None:
            elapsed = time.perf_counter() - self._completion_start_time
            remaining_timeout = self.max_timeout - elapsed
            if remaining_timeout <= 0:
                return RLMChatCompletion(
                    root_model=resolved_model,
                    prompt=prompt,
                    response=f"Error: Timeout exhausted ({elapsed:.1f}s of {self.max_timeout:.1f}s)",
                    usage_summary=UsageSummary(model_usage_summaries={}),
                    execution_time=0.0,
                )

        # Bound each child's slice of the budget: without a cap, one child
        # handed the whole remaining timeout can starve the parent (a
        # whole-question rlm_query delegation ate a full 600s ask on
        # 2026-06-11 before the parent had read a single document).
        child_timeout = remaining_timeout
        if self.subcall_max_timeout is not None:
            child_timeout = (
                self.subcall_max_timeout
                if child_timeout is None
                else min(child_timeout, self.subcall_max_timeout)
            )

        # Resolve the model name for callbacks
        prompt_preview = prompt[:80] if len(prompt) > 80 else prompt

        # Fire subcall start callback
        if self.on_subcall_start:
            try:
                self.on_subcall_start(next_depth, str(resolved_model), prompt_preview)
            except Exception:
                pass  # Don't let callback errors break execution

        subcall_start = time.perf_counter()
        error_msg: str | None = None

        # Spawn a child RLM with its own LocalREPL
        child = RLM(
            backend=self.backend,
            backend_kwargs=child_backend_kwargs,
            environment=self.environment_type,
            environment_kwargs=self.environment_kwargs,
            depth=next_depth,
            max_depth=self.max_depth,
            max_iterations=self.child_max_iterations,
            child_max_iterations=self.child_max_iterations,
            max_budget=remaining_budget,
            max_timeout=child_timeout,
            max_tokens=self.max_tokens,
            max_errors=self.max_errors,
            # The runaway guards must follow the recursion: a child whose
            # sub-calls are uncapped and unscheduled defeats the point of
            # configuring them on the root.
            subcall_max_tokens=self.subcall_max_tokens,
            subcall_max_timeout=self.subcall_max_timeout,
            subcall_extra_body=self.subcall_extra_body,
            root_max_tokens=self.root_max_tokens,
            # The SAME instance, not a copy: resubmission memory and veto
            # telemetry must span the recursion tree.
            subcall_verifier=self.subcall_verifier,
            scheduler_max_concurrent=self.scheduler_max_concurrent,
            scheduler_aging_interval=self.scheduler_aging_interval,
            scheduler_coordination_dir=self.scheduler_coordination_dir,
            custom_system_prompt=self.child_system_prompt if self.child_system_prompt else self.system_prompt,
            child_system_prompt=self.child_system_prompt,
            other_backends=self.other_backends,
            other_backend_kwargs=self.other_backend_kwargs,
            # Give child its own logger so its trajectory is captured in metadata
            logger=RLMLogger() if self.logger else None,
            verbose=False,
            # Propagate custom tools to children (sub_tools become the child's tools)
            custom_tools=self.custom_sub_tools,
            custom_sub_tools=self.custom_sub_tools,
            # Propagate concurrency settings to children
            max_concurrent_subcalls=self.max_concurrent_subcalls,
            # Propagate callbacks to children for nested tracking
            on_subcall_start=self.on_subcall_start,
            on_subcall_complete=self.on_subcall_complete,
            on_iteration_start=self.on_iteration_start,
            on_iteration_complete=self.on_iteration_complete,
            # The input-size guard must follow the recursion: a child whose
            # sub-calls are unguarded re-opens the overflow surface.
            subcall_context_limit=self.subcall_context_limit,
            # The answer/delegation shape guards must follow the recursion too:
            # a child with tools has the same echo-and-ship failure surface.
            reject_code_shaped_answers=self.reject_code_shaped_answers,
            reject_self_context_delegation=self.reject_self_context_delegation,
        )
        try:
            result = child.completion(prompt, root_prompt=None)
            # Track child's cost in parent's cumulative cost
            if result.usage_summary and result.usage_summary.total_cost:
                self._cumulative_cost += result.usage_summary.total_cost
            return result
        except BudgetExceededError as e:
            # Propagate child's spending to parent
            self._cumulative_cost += e.spent
            error_msg = f"Budget exceeded - {e}"
            return RLMChatCompletion(
                root_model=resolved_model,
                prompt=prompt,
                response=f"Error: Child RLM budget exceeded - {e}",
                usage_summary=UsageSummary(model_usage_summaries={}),
                execution_time=time.perf_counter() - subcall_start,
            )
        except Exception as e:
            error_msg = str(e)
            return RLMChatCompletion(
                root_model=resolved_model,
                prompt=prompt,
                response=f"Error: Child RLM completion failed - {e}",
                usage_summary=UsageSummary(model_usage_summaries={}),
                execution_time=time.perf_counter() - subcall_start,
            )
        finally:
            # Ensure child resources are cleaned up
            child.close()
            # Fire subcall complete callback
            if self.on_subcall_complete:
                try:
                    duration = time.perf_counter() - subcall_start
                    self.on_subcall_complete(next_depth, str(resolved_model), duration, error_msg)
                except Exception:
                    pass  # Don't let callback errors break execution

    def _validate_persistent_environment_support(self) -> None:
        """
        Validate that the configured environment type supports persistent mode.

        Persistent mode requires environments to implement:
        - update_handler_address(address): Update LM handler address between calls
        - add_context(payload, index): Add new context for multi-turn conversations
        - get_context_count(): Return the number of loaded contexts

        Currently only 'local' (LocalREPL) supports these methods.

        Raises:
            ValueError: If the environment type does not support persistent mode.
        """
        # Known environments that support persistence
        persistent_supported_environments = {"local", "ipython"}

        if self.environment_type not in persistent_supported_environments:
            raise ValueError(
                f"persistent=True is not supported for environment type '{self.environment_type}'. "
                f"Persistent mode requires environments that implement update_handler_address(), "
                f"add_context(), and get_context_count(). "
                f"Supported environments: {sorted(persistent_supported_environments)}"
            )

    @staticmethod
    def _env_supports_persistence(env: BaseEnv) -> bool:
        """Check if an environment instance supports persistent mode methods."""
        return isinstance(env, SupportsPersistence)

    def close(self) -> None:
        """Clean up persistent environment. Call when done with multi-turn conversations."""
        if self._persistent_env is not None:
            if hasattr(self._persistent_env, "cleanup"):
                self._persistent_env.cleanup()
            self._persistent_env = None

    def __enter__(self) -> "RLM":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        self.close()
        return False
