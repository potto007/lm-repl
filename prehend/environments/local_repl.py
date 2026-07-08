import copy
import io
import json
import os
import re
import shutil
import sys
import tempfile
import threading
import time
import uuid
import warnings
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from typing import Any

from prehend.core.comms_utils import LMRequest, send_lm_request, send_lm_request_batched
from prehend.environments import leaf_prose_guard
from prehend.core.types import REPLResult, RLMChatCompletion
from prehend.environments.base_env import (
    RESERVED_TOOL_NAMES,
    NonIsolatedEnv,
    extract_tool_value,
    validate_custom_tools,
)
from prehend.utils.mapreduce import (
    _EXTRACTION_MAP_INSTRUCTION,
    _MAP_SENTINEL_DIRECTIVE,
    _compose,
    map_reduce,
)
from prehend.utils.subcall_guard import (
    oversize_rejection,
    recommended_chunk_chars,
    self_context_rejection,
)

# Drive the map-reduce seam with the query-INDEPENDENT extraction MAP (ADR-0018).
# The legacy per-query MAP filters a chunk by the user query, so on a multihop
# task it drops the terminal "the person who lives in <city> owns X" chunk (it
# never names the queried person) and surfaces only the intermediate hop -> wrong
# answer. Extracting every fact about every entity instead lets the reduce chain
# the hops. Validated live on the multihop subset: legacy 1/5 -> extraction 5/5.
# One-line revert: set False (restores the legacy per-query map).
_SEAM_EXTRACTION_MAP = True

# A run of consecutive empty calls '()()...' - the .lower()() decode stutter.
_DOUBLED_CALL = re.compile(r"\(\)(?:\(\))+")

# A bare {identifier} placeholder - the missing-f-string signature. When the model
# builds a variable (e.g. combined_data) then references it as a literal {combined_data}
# inside a PLAIN (non-f) string passed to llm_query, the data is never substituted and
# the sub-call sees an empty placeholder. repair_unfilled_placeholders fills it in.
_PLACEHOLDER = re.compile(r"\{([A-Za-z_][A-Za-z0-9_]*)\}")


class _AnswerDict(dict):
    """REPL-visible dict where ``answer["ready"] = True`` signals completion.

    Behaves exactly like ``dict`` for the model, but invokes ``on_ready`` the
    first time ``ready`` flips truthy. The callback receives the current
    ``content``, lets the env capture it (in-process attr, broker push, etc.),
    and the next ``execute_code`` will surface it as ``REPLResult.final_answer``.
    """

    def __init__(self, on_ready=None):
        super().__init__()
        super().__setitem__("content", "")
        super().__setitem__("ready", False)
        self._on_ready = on_ready

    def __setitem__(self, key, value):
        super().__setitem__(key, value)
        if key == "ready" and value and self._on_ready is not None:
            try:
                self._on_ready(self.get("content", ""))
            except Exception:
                pass


# =============================================================================
# Safe Builtins
# =============================================================================

# Safe builtins - blocks dangerous operations like eval/exec/input
_SAFE_BUILTINS = {
    # Core types and functions
    "print": print,
    "len": len,
    "str": str,
    "int": int,
    "float": float,
    "list": list,
    "dict": dict,
    "set": set,
    "tuple": tuple,
    "bool": bool,
    "type": type,
    "isinstance": isinstance,
    "issubclass": issubclass,
    "enumerate": enumerate,
    "zip": zip,
    "map": map,
    "filter": filter,
    "sorted": sorted,
    "reversed": reversed,
    "range": range,
    "min": min,
    "max": max,
    "sum": sum,
    "abs": abs,
    "round": round,
    "any": any,
    "all": all,
    "pow": pow,
    "divmod": divmod,
    "chr": chr,
    "ord": ord,
    "hex": hex,
    "bin": bin,
    "oct": oct,
    "repr": repr,
    "ascii": ascii,
    "format": format,
    "hash": hash,
    "id": id,
    "iter": iter,
    "next": next,
    "slice": slice,
    "callable": callable,
    "hasattr": hasattr,
    "getattr": getattr,
    "setattr": setattr,
    "delattr": delattr,
    "dir": dir,
    "vars": vars,
    "bytes": bytes,
    "bytearray": bytearray,
    "memoryview": memoryview,
    "complex": complex,
    "object": object,
    "super": super,
    "property": property,
    "staticmethod": staticmethod,
    "classmethod": classmethod,
    "__import__": __import__,
    "open": open,
    # Exceptions
    "Exception": Exception,
    "BaseException": BaseException,
    "ValueError": ValueError,
    "TypeError": TypeError,
    "KeyError": KeyError,
    "IndexError": IndexError,
    "AttributeError": AttributeError,
    "FileNotFoundError": FileNotFoundError,
    "OSError": OSError,
    "IOError": IOError,
    "RuntimeError": RuntimeError,
    "NameError": NameError,
    "ImportError": ImportError,
    "StopIteration": StopIteration,
    "AssertionError": AssertionError,
    "NotImplementedError": NotImplementedError,
    "ArithmeticError": ArithmeticError,
    "LookupError": LookupError,
    "Warning": Warning,
    # Blocked
    "input": None,
    "eval": None,
    "exec": None,
    "compile": None,
    "globals": None,
    "locals": None,
}


# Retrieval circuit-breaker (rlm-trainer eval finding, issue #4). The orchestrator
# loop is otherwise unbounded - one observed ask issued 579 llm_query sub-calls
# before the time budget stopped it. Once a completion has spent its sub-call
# budget, further llm_query / llm_query_batched calls short-circuit with THIS
# instruction instead of hitting the server, so the model must answer from what it
# has already gathered. Default off (None); the librarian supplies the value.
_SUBCALL_BUDGET_MSG = (
    "Error: retrieval budget exhausted - you have already made the maximum number "
    "of document/sub-query calls for this question. STOP searching. Using ONLY the "
    "information you have already gathered, write your single best final answer and "
    "set answer['ready'] = True this turn, or give your no-coverage answer if you "
    "found nothing relevant. Do NOT call llm_query / llm_query_batched again."
)


# Fraction of each chunk that overlaps its neighbour when the harness auto-chunks
# an oversized `context=` (ADR-0010). A span straddling a chunk boundary then
# appears in both chunks, preserving cross-chunk links for multi-hop questions
# (the partition-validity risk). ~0.15 is a small accuracy hedge with modest cost.
_SUBCALL_CHUNK_OVERLAP_FRAC = 0.15


def _subcall_budget_remaining(count: int, max_subcalls: int | None) -> int | None:
    """Sub-calls still allowed this completion. None = unlimited (disabled).

    ``max_subcalls`` None or <= 0 disables the cap (returns None). Otherwise
    returns ``max(0, max_subcalls - count)`` - never negative."""
    if max_subcalls is None or max_subcalls <= 0:
        return None
    return max(0, max_subcalls - count)


class LocalREPL(NonIsolatedEnv):
    """
    Local REPL environment with persistent Python namespace.
    Executes code in a sandboxed namespace with access to context data.
    """

    def __init__(
        self,
        lm_handler_address: tuple[str, int] | None = None,
        context_payload: dict | list | str | None = None,
        setup_code: str | None = None,
        persistent: bool = False,
        depth: int = 1,
        subcall_fn: Callable[[str, str | None], RLMChatCompletion] | None = None,
        custom_tools: dict[str, Any] | None = None,
        custom_sub_tools: dict[str, Any] | None = None,
        compaction: bool = False,
        max_concurrent_subcalls: int = 4,
        max_output_chars: int | None = None,
        repair_doubled_calls: bool = False,
        repair_unfilled_placeholders: bool = False,
        max_subcalls: int | None = None,
        subcall_context_limit: int | None = None,
        model_name: str | None = None,
        reject_self_context_delegation: bool = False,
        **kwargs,
    ):
        super().__init__(
            persistent=persistent,
            depth=depth,
            max_concurrent_subcalls=max_concurrent_subcalls,
            **kwargs,
        )

        # Retrieval circuit-breaker: hard cap on llm_query / llm_query_batched
        # sub-calls per completion. None = off. Once spent, further reads
        # short-circuit with _SUBCALL_BUDGET_MSG so the model must wrap up. The
        # counter is per-instance and a fresh environment is spawned per
        # completion (non-persistent), so it starts at 0 for each ask.
        self.max_subcalls = max_subcalls
        self._subcall_count = 0

        # Input-size guard for llm_query / llm_query_batched. When
        # subcall_context_limit is set, a prompt over the safe budget is rejected
        # with an actionable chunk-and-map-reduce hint instead of being sent (the
        # send would 400 "exceeds available context size" and spin to a timeout).
        # model_name feeds count_tokens for the per-model token estimate. This
        # intentionally covers llm_query - the strategy-verifier's llm_query
        # exemption does NOT apply to this arithmetic input guard. None = off.
        self.subcall_context_limit = subcall_context_limit
        self.model_name = model_name

        # Futile-delegation guard: when custom tools hold the task's data and a
        # sub-call's context= is the task's own root context (the briefing), the
        # sub-LLM - which has no REPL/tools - cannot possibly retrieve anything,
        # so reject deterministically with a pivot-to-tools hint instead of
        # sending (see subcall_guard.self_context_rejection). Fires only when
        # custom_tools exist AND the context matches a loaded root context
        # string exactly. Opt-in (default off): teacher/trajectory-generation
        # runs must see unmodified behavior (leaf_prose_guard convention).
        self.reject_self_context_delegation = reject_self_context_delegation
        self._root_context_strs: list[str] = []

        self.max_output_chars = max_output_chars
        # Collapse a doubled empty-call '()()' -> '()' before exec. The .lower()()
        # decode stutter (token 825 emitted twice) raises TypeError 'str' object is
        # not callable; '()()' is never legitimate in this corpus, so repairing it
        # runs the intended code and avoids the error->spiral. Opt-in (default off).
        self.repair_doubled_calls = repair_doubled_calls
        # Interpolate a {name} placeholder that names a live, model-created REPL
        # variable into an llm_query prompt - the missing-f-string bug, where the
        # model built the data then dropped it by forgetting the f prefix on the
        # synthesis prompt (sub-call then reports "no information found"). Opt-in.
        self.repair_unfilled_placeholders = repair_unfilled_placeholders
        self.lm_handler_address = lm_handler_address
        self.subcall_fn = subcall_fn  # Callback for recursive RLM calls (depth > 1 support)
        self.original_cwd = os.getcwd()
        self.temp_dir = tempfile.mkdtemp(prefix=f"repl_env_{uuid.uuid4()}_")
        self._lock = threading.Lock()
        self._context_count: int = 0
        self._history_count: int = 0
        self.compaction = compaction

        # Custom tools: functions available in the REPL
        self.custom_tools = custom_tools or {}
        # Sub-tools: inherited from custom_tools if not specified
        self.custom_sub_tools = (
            custom_sub_tools if custom_sub_tools is not None else self.custom_tools
        )

        # Validate custom tools don't override reserved names
        validate_custom_tools(self.custom_tools)

        # Setup globals, locals, and modules in environment.
        self.setup()

        if compaction:
            self._compaction_history: list[Any] = []
            self.locals["history"] = self._compaction_history

        # Load context if provided
        if context_payload is not None:
            self.load_context(context_payload)

        # Run setup code if provided
        if setup_code:
            self.execute_code(setup_code)

    def setup(self):
        """Setup the environment."""
        # Create sandboxed globals
        self.globals: dict[str, Any] = {
            "__builtins__": _SAFE_BUILTINS.copy(),
            "__name__": "__main__",
        }
        self.locals: dict[str, Any] = {}

        # Track LLM calls made during code execution
        self._pending_llm_calls: list[RLMChatCompletion] = []
        # Re-scan fix: persist the query-INDEPENDENT extraction-MAP partials across
        # iterations so a re-issued llm_query(context=SAME_BIG) reuses the MAP and
        # re-runs only the cheap REDUCE (keyed by context+chunking inside map_reduce;
        # only consulted in extraction_map mode). Lives on the env, which is reused
        # across the run's iterations; keyed by context so distinct tasks never alias.
        self._map_partials_cache: dict = {}
        # Captured the first time the model sets ``answer["ready"] = True``.
        self._last_final_answer: str | None = None
        # The live exec namespace during execute_code, so a sub-call (llm_query)
        # invoked from inside exec can resolve a {name} placeholder against
        # variables created in the SAME code block (not yet promoted to locals).
        self._active_namespace: dict[str, Any] | None = None
        # Names interpolated by repair_unfilled_placeholders this execution (for the note).
        self._filled_placeholders: list[str] = []

        # Add helper functions
        self.globals["SHOW_VARS"] = self._show_vars
        self.globals["llm_query"] = self._llm_query
        self.globals["llm_query_batched"] = self._llm_query_batched
        self.globals["rlm_query"] = self._rlm_query
        self.globals["rlm_query_batched"] = self._rlm_query_batched

        # The model marks completion via ``answer["ready"] = True``; the
        # custom dict captures the content as soon as that happens so we
        # don't have to probe the namespace after every cell.
        self.locals["answer"] = _AnswerDict(on_ready=self._capture_answer)

        # Add custom tools to globals
        # Entries are plain values or {"tool": value, "description": str} dicts
        # (extract_tool_value unwraps ONLY the dict form; tuples pass through
        # as literal tuples - they are NOT unpacked)
        for name, entry in self.custom_tools.items():
            value = extract_tool_value(entry)
            if callable(value):
                self.globals[name] = value
            else:
                # For non-callable values (constants, data), add to locals
                self.locals[name] = value

    def _capture_answer(self, content: Any) -> None:
        self._last_final_answer = str(content)

    def _show_vars(self) -> str:
        """Show all available variables in the REPL environment."""
        available = {
            k: type(v).__name__
            for k, v in self.locals.items()
            if not k.startswith("_") and k != "answer"
        }
        if not available:
            return "No variables created yet. Use ```repl``` blocks to create variables."
        return f"Available variables: {available}"

    def _fill_placeholders(self, prompt: str) -> str:
        """Interpolate a ``{name}`` placeholder that names a live, model-created string
        variable (the missing-f-string signature) into a sub-call prompt.

        Only fills names the model created (present in the live exec namespace, absent
        from the scaffold globals) that hold a ``str`` value. JSON braces, format
        examples, undefined names, and scaffold/tool names are left untouched, so this
        cannot corrupt a prompt that legitimately contains ``{...}``.
        """
        if not self.repair_unfilled_placeholders or not isinstance(prompt, str):
            return prompt
        ns = self._active_namespace
        if not ns:
            return prompt

        def _sub(m: "re.Match[str]") -> str:
            name = m.group(1)
            if (
                name == "answer"
                or name.startswith("_")
                or name in self.globals
                or name not in ns
                or not isinstance(ns[name], str)
            ):
                return m.group(0)
            self._filled_placeholders.append(name)
            return ns[name]

        return _PLACEHOLDER.sub(_sub, prompt)

    def _dispatch_with_context(
        self,
        prompt: str,
        context: Any,
        reduce: str | None,
        *,
        send_one: "Callable[[str], str]",
        run_batch: "Callable[[list[str]], list[str]]",
    ) -> str:
        """Shared sub-call dispatch for the ``context=`` data channel (ADR-0010).

        - No ``context``: send the bare ``prompt`` (today's behavior; the bare
          oversized-prompt guard still applies inside ``send_one``).
        - ``context`` that fits the recommended size: inline it into one send.
        - Oversized ``context``: map-reduce it across the server slots via
          ``run_batch`` (a context-free batched send), so an oversized data blob
          never serializes into one slow prefill.
        """
        if context is None:
            return send_one(prompt)
        context = context if isinstance(context, str) else str(context)
        # Futile-delegation guard: delegating the task's OWN briefing back out
        # while the data lives in custom REPL tools can only echo or refuse -
        # reject with a pivot-to-tools hint instead of paying for the send.
        if (
            self.reject_self_context_delegation
            and self.custom_tools
            and context in self._root_context_strs
        ):
            return self_context_rejection(list(self.custom_tools.keys()))
        # Fill model-created placeholders in BOTH the data and (via send_one /
        # run_batch, which fill the composed prompt) the instruction, before
        # chunking, so a {var} in context is substituted, not split.
        context = self._fill_placeholders(context)
        composed = _compose(prompt, context, "Text")

        limit = self.subcall_context_limit
        if limit is None:
            # Guard disabled: no limit to chunk against. Inline and send (an
            # oversized context here can overflow - the caller opted out).
            return send_one(composed)

        rec = recommended_chunk_chars(limit, self.model_name or "")
        if len(composed) <= rec:
            # Small enough to be one fast call: inline it.
            return send_one(composed)

        # Oversized: map-reduce. Size the per-chunk data budget so the composed
        # chunk prompt (MAP instruction + envelope + chunk) stays within `rec`.
        # The MAP instruction is what map_reduce actually composes per chunk: the
        # fixed extraction instruction in extraction_map mode, else the user
        # prompt plus the sentinel directive. Sizing against the SHORT user
        # prompt alone undershoots the real envelope and can push a map prompt
        # past the guard, so account for the instruction map_reduce will use.
        map_instr = (
            _EXTRACTION_MAP_INSTRUCTION
            if _SEAM_EXTRACTION_MAP
            else prompt + _MAP_SENTINEL_DIRECTIVE
        )
        overhead = len(_compose(map_instr, "", "Text"))
        chunk_chars = rec - overhead
        # Skip map-reduce if either step has no room: the MAP instruction leaves
        # no budget for chunk data, OR the REDUCE prompt (the user instruction in
        # extraction_map mode, where the user prompt drives the reduce not the
        # map) fills the window with no room for partials. Either way a single
        # send lets the inner guard reject-with-hint the oversized composed prompt.
        reduce_instr = reduce if reduce is not None else prompt
        reduce_overhead = len(_compose(reduce_instr, "", "Text"))
        if chunk_chars <= 0 or reduce_overhead >= rec:
            return send_one(composed)

        def _fits(text: str) -> bool:
            return oversize_rejection(text, limit=limit, model=self.model_name or "") is None

        result = map_reduce(
            prompt,
            context,
            run_batch=run_batch,
            fits=_fits,
            chunk_chars=chunk_chars,
            reduce_prompt=reduce,
            overlap_chars=int(chunk_chars * _SUBCALL_CHUNK_OVERLAP_FRAC),
            extraction_map=_SEAM_EXTRACTION_MAP,
            map_cache=self._map_partials_cache,
        )
        return result.answer

    def _llm_query(
        self,
        prompt: str,
        model: str | None = None,
        priority: str | int | None = None,
        *,
        context: Any = None,
        reduce: str | None = None,
    ) -> str:
        """LM completion. Pass large data via ``context=`` for auto map-reduce."""
        return self._dispatch_with_context(
            prompt,
            context,
            reduce,
            send_one=lambda p: self._send(p, model, priority),
            run_batch=lambda ps: self._send_batched(ps, model, priority),
        )

    def _llm_query_batched(
        self,
        prompts: list[str],
        model: str | None = None,
        priority: str | int | None = None,
        *,
        context: Any = None,
        reduce: str | None = None,
    ) -> list[str]:
        """Batched LM completions. ``context=`` (scalar) applies to every prompt."""
        if context is None:
            return self._send_batched(prompts, model, priority)
        return [
            self._dispatch_with_context(
                p,
                context,
                reduce,
                send_one=lambda q: self._send(q, model, priority),
                run_batch=lambda ps: self._send_batched(ps, model, priority),
            )
            for p in prompts
        ]

    def _rlm_query(
        self,
        prompt: str,
        model: str | None = None,
        *,
        context: Any = None,
        reduce: str | None = None,
    ) -> str:
        """Recursive RLM sub-call. Pass large data via ``context=`` for map-reduce."""
        if self.subcall_fn is None:
            return self._llm_query(prompt, model, context=context, reduce=reduce)
        return self._dispatch_with_context(
            prompt,
            context,
            reduce,
            send_one=lambda p: self._rlm_send(p, model),
            run_batch=lambda ps: self._rlm_send_batched(ps, model),
        )

    def _rlm_query_batched(
        self,
        prompts: list[str],
        model: str | None = None,
        *,
        context: Any = None,
        reduce: str | None = None,
    ) -> list[str]:
        """Batched recursive RLM sub-calls. ``context=`` (scalar) applies to each."""
        if self.subcall_fn is None:
            return self._llm_query_batched(prompts, model, context=context, reduce=reduce)
        if context is None:
            return self._rlm_send_batched(prompts, model)
        return [
            self._dispatch_with_context(
                p,
                context,
                reduce,
                send_one=lambda q: self._rlm_send(q, model),
                run_batch=lambda ps: self._rlm_send_batched(ps, model),
            )
            for p in prompts
        ]

    def _send(
        self, prompt: str, model: str | None = None, priority: str | int | None = None
    ) -> str:
        """Context-free single LM send (the engine/seam building block).

        Query the LM with a single plain completion (no REPL, no recursion).

        This always makes a direct LM call via the handler, regardless of depth.

        Args:
            prompt: The prompt to send to the LM.
            model: Optional model name to use (if handler has multiple clients).
            priority: Optional scheduling priority ("high"/"low"/"normal").
        """
        if not self.lm_handler_address:
            return "Error: No LM handler configured"

        # Circuit-breaker: refuse further reads once the per-ask budget is spent.
        if _subcall_budget_remaining(self._subcall_count, self.max_subcalls) == 0:
            return _SUBCALL_BUDGET_MSG

        prompt = self._fill_placeholders(prompt)

        # Input-size guard: reject an oversized prompt with an actionable hint
        # rather than sending it (the send would 400 and spin to a timeout).
        if self.subcall_context_limit is not None:
            hint = oversize_rejection(
                prompt, limit=self.subcall_context_limit, model=self.model_name or ""
            )
            if hint is not None:
                return hint

        guard = leaf_prose_guard.guard_enabled()
        if guard:
            prompt = leaf_prose_guard.wrap_leaf_prompt(prompt)

        try:
            self._subcall_count += 1
            request = LMRequest(prompt=prompt, model=model, depth=self.depth, priority=priority)
            response = send_lm_request(self.lm_handler_address, request)

            if not response.success:
                return f"Error: {response.error}"

            self._pending_llm_calls.append(response.chat_completion)
            reply = response.chat_completion.response
            if guard:
                # code-shaped leaf reply (e.g. `find("V406XXA")` instead of the
                # description): one corrective resend, keep it only if prose.
                # The retry shares this call's budget slot - it repairs, not reads.
                def _resend(p: str) -> str:
                    r = send_lm_request(
                        self.lm_handler_address,
                        LMRequest(prompt=p, model=model, depth=self.depth, priority=priority),
                    )
                    if not r.success:
                        return ""
                    self._pending_llm_calls.append(r.chat_completion)
                    return r.chat_completion.response

                reply = leaf_prose_guard.repair_leaf_reply(reply, prompt, _resend)
            return reply
        except Exception as e:
            return f"Error: LM query failed - {e}"

    def _send_batched(
        self,
        prompts: list[str],
        model: str | None = None,
        priority: str | int | None = None,
    ) -> list[str]:
        """Context-free batched LM send (the engine/seam building block).

        Query the LM with multiple prompts concurrently (no REPL, no recursion).

        This always makes direct LM calls via the handler, regardless of depth.

        Args:
            prompts: List of prompts to send to the LM.
            model: Optional model name to use (if handler has multiple clients).
            priority: Optional scheduling priority ("high"/"low"/"normal").

        Returns:
            List of responses in the same order as input prompts.
        """
        if not self.lm_handler_address:
            return ["Error: No LM handler configured"] * len(prompts)
        prompts = [self._fill_placeholders(p) for p in prompts]

        # Input-size guard, per prompt: an oversized prompt is replaced with an
        # actionable hint and NOT sent; the rest are dispatched normally (order
        # preserved). Guarding per-prompt means one huge chunk doesn't sink the
        # whole batch. The hint occupies the prompt's slot in the output.
        oversize_hints: dict[int, str] = {}
        if self.subcall_context_limit is not None:
            for i, p in enumerate(prompts):
                hint = oversize_rejection(
                    p, limit=self.subcall_context_limit, model=self.model_name or ""
                )
                if hint is not None:
                    oversize_hints[i] = hint
        sendable = [(i, p) for i, p in enumerate(prompts) if i not in oversize_hints]

        # Circuit-breaker: dispatch only as many sendable prompts as the per-ask
        # budget allows; the overflow short-circuits with the wrap-up instruction
        # so the batch cannot blow past the cap in one cell. None = unlimited.
        remaining = _subcall_budget_remaining(self._subcall_count, self.max_subcalls)
        allowed = sendable if remaining is None else sendable[:remaining]

        try:
            # Map each sendable index to its result; oversized indices map to the
            # hint, budget-blocked indices to the budget message.
            by_index: dict[int, str] = dict(oversize_hints)
            guard = leaf_prose_guard.guard_enabled()
            if guard:
                allowed = [(i, leaf_prose_guard.wrap_leaf_prompt(p)) for i, p in allowed]
            if allowed:
                self._subcall_count += len(allowed)
                responses = send_lm_request_batched(
                    self.lm_handler_address,
                    [p for _, p in allowed],
                    model=model,
                    depth=self.depth,
                    priority=priority,
                )
                for (idx, _p), response in zip(allowed, responses, strict=False):
                    if not response.success:
                        by_index[idx] = f"Error: {response.error}"
                    else:
                        self._pending_llm_calls.append(response.chat_completion)
                        by_index[idx] = response.chat_completion.response
                if guard:
                    self._repair_batched_replies(by_index, allowed, model, priority)
            for idx, _p in sendable[len(allowed):]:
                by_index[idx] = _SUBCALL_BUDGET_MSG
            return [by_index[i] for i in range(len(prompts))]
        except Exception as e:
            return [f"Error: LM query failed - {e}"] * len(prompts)

    def _repair_batched_replies(
        self,
        by_index: dict[int, str],
        allowed: list[tuple[int, str]],
        model: str | None,
        priority: str | int | None,
    ) -> None:
        """One corrective batch resend for code-shaped slots (leaf prose guard).

        Retries share their original call's budget slot - they repair, not read.
        A retry reply replaces the original only if it comes back prose."""
        bad = [
            (idx, p)
            for idx, p in allowed
            if leaf_prose_guard.looks_like_code_reply(by_index.get(idx, ""))
        ]
        if not bad:
            return
        retry_prompts = [
            f"{leaf_prose_guard._RETRY_PREFIX}{p}" for _, p in bad
        ]
        retries = send_lm_request_batched(
            self.lm_handler_address,
            retry_prompts,
            model=model,
            depth=self.depth,
            priority=priority,
        )
        for (idx, _p), response in zip(bad, retries, strict=False):
            if not response.success:
                continue
            self._pending_llm_calls.append(response.chat_completion)
            reply = response.chat_completion.response
            if reply and not leaf_prose_guard.looks_like_code_reply(reply):
                by_index[idx] = reply

    def _rlm_send(self, prompt: str, model: str | None = None) -> str:
        """Context-free single recursive RLM send (the engine/seam building block).

        Spawn a recursive RLM sub-call for deeper thinking on a subtask.

        When a subcall callback is available (max_depth > 1), this spawns a child
        RLM with its own REPL that can reason over the prompt iteratively.
        Falls back to a plain llm_query if no recursive capability is configured.

        Args:
            prompt: The prompt to send to the child RLM.
            model: Optional model name override for the child.
        """
        if self.subcall_fn is not None:
            try:
                completion = self.subcall_fn(prompt, model)
                self._pending_llm_calls.append(completion)
                return completion.response
            except Exception as e:
                return f"Error: RLM query failed - {e}"

        # Fall back to plain LM call if no recursive capability
        return self._send(prompt, model)

    def _rlm_send_batched(self, prompts: list[str], model: str | None = None) -> list[str]:
        """Context-free batched recursive RLM send (the engine/seam building block).

        Spawn recursive RLM sub-calls for multiple prompts in parallel.

        Each prompt gets its own child RLM for deeper thinking. When multiple
        prompts are provided, subcalls run concurrently using a thread pool
        (bounded by max_concurrent_subcalls) since they are independent and
        I/O-bound. Results are returned in the same order as input prompts.

        Falls back to llm_query_batched if no recursive capability is configured.

        Args:
            prompts: List of prompts for child RLMs.
            model: Optional model name override for the children.

        Returns:
            List of responses in the same order as input prompts.
        """
        if self.subcall_fn is not None:
            # For 0 or 1 prompts, no need for thread pool overhead
            if len(prompts) <= 1:
                results = []
                for prompt in prompts:
                    try:
                        completion = self.subcall_fn(prompt, model)
                        self._pending_llm_calls.append(completion)
                        results.append(completion.response)
                    except Exception as e:
                        results.append(f"Error: RLM query failed - {e}")
                return results

            # Parallel execution for multiple prompts
            max_workers = min(self.max_concurrent_subcalls, len(prompts))
            # Pre-allocate result slots to preserve ordering
            results: list[str] = [""] * len(prompts)
            completions: list[tuple[int, RLMChatCompletion]] = []
            lock = threading.Lock()

            def _run_subcall(index: int, prompt: str) -> None:
                try:
                    completion = self.subcall_fn(prompt, model)
                    with lock:
                        completions.append((index, completion))
                    results[index] = completion.response
                except Exception as e:
                    results[index] = f"Error: RLM query failed - {e}"

            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = [
                    executor.submit(_run_subcall, i, prompt) for i, prompt in enumerate(prompts)
                ]
                # Wait for all futures to complete; exceptions are captured inside _run_subcall
                for future in as_completed(futures):
                    future.result()  # Re-raises unexpected executor errors

            # Append completions in original prompt order for deterministic metadata
            completions.sort(key=lambda x: x[0])
            for _, completion in completions:
                self._pending_llm_calls.append(completion)

            return results

        # Fall back to plain batched LM call if no recursive capability
        return self._send_batched(prompts, model)

    def load_context(self, context_payload: dict | list | str):
        """Load context into the environment as context_0 (and 'context' alias)."""
        self.add_context(context_payload, 0)

    def add_context(
        self, context_payload: dict | list | str, context_index: int | None = None
    ) -> int:
        """
        Add a context with versioned variable name.

        Args:
            context_payload: The context data to add
            context_index: Optional explicit index. If None, auto-increments.

        Returns:
            The context index used.
        """
        if context_index is None:
            context_index = self._context_count

        # Root-context fingerprint for the futile-delegation guard: match the
        # str() coercion _dispatch_with_context applies to sub-call context=.
        payload_str = (
            context_payload if isinstance(context_payload, str) else str(context_payload)
        )
        if payload_str not in self._root_context_strs:
            self._root_context_strs.append(payload_str)

        var_name = f"context_{context_index}"

        if isinstance(context_payload, str):
            context_path = os.path.join(self.temp_dir, f"context_{context_index}.txt")
            with open(context_path, "w") as f:
                f.write(context_payload)
            self.execute_code(f"with open(r'{context_path}', 'r') as f:\n    {var_name} = f.read()")
        else:
            context_path = os.path.join(self.temp_dir, f"context_{context_index}.json")
            with open(context_path, "w") as f:
                json.dump(context_payload, f)
            self.execute_code(
                f"import json\nwith open(r'{context_path}', 'r') as f:\n    {var_name} = json.load(f)"
            )

        # Alias context_0 as 'context' for backward compatibility
        if context_index == 0:
            self.execute_code(f"context = {var_name}")

        self._context_count = max(self._context_count, context_index + 1)
        return context_index

    def update_handler_address(self, address: tuple[str, int]) -> None:
        """Update the LM handler address for a new completion call."""
        self.lm_handler_address = address

    def get_context_count(self) -> int:
        """Return the number of contexts loaded."""
        return self._context_count

    def add_history(
        self, message_history: list[dict[str, Any]], history_index: int | None = None
    ) -> int:
        """
        Store a conversation's message history as a versioned variable.

        Args:
            message_history: The list of message dicts from a completion call
            history_index: Optional explicit index. If None, auto-increments.

        Returns:
            The history index used.
        """
        if history_index is None:
            history_index = self._history_count

        var_name = f"history_{history_index}"

        # Store deep copy to avoid reference issues with nested dicts
        self.locals[var_name] = copy.deepcopy(message_history)

        # Alias history_0 as 'history' for convenience
        if history_index == 0:
            self.locals["history"] = self.locals[var_name]

        self._history_count = max(self._history_count, history_index + 1)
        return history_index

    def get_history_count(self) -> int:
        """Return the number of conversation histories stored."""
        return self._history_count

    def append_compaction_entry(self, entry: list[dict[str, Any]] | dict[str, Any]) -> None:
        """
        Append a trajectory segment or a summary to the compaction history.

        Entry is either a list of message dicts (trajectory segment) or
        a dict with "type": "summary" and "content": str.
        """
        if not self.compaction:
            return
        self._compaction_history.append(copy.deepcopy(entry))

    @contextmanager
    def _capture_output(self):
        """Thread-safe context manager to capture stdout/stderr."""
        with self._lock:
            old_stdout, old_stderr = sys.stdout, sys.stderr
            stdout_buf, stderr_buf = io.StringIO(), io.StringIO()
            try:
                sys.stdout, sys.stderr = stdout_buf, stderr_buf
                yield stdout_buf, stderr_buf
            finally:
                sys.stdout, sys.stderr = old_stdout, old_stderr

    @contextmanager
    def _temp_cwd(self):
        """Make temp directory available without changing process-global cwd.

        os.chdir is process-global and races when multiple threads run
        parallel sub-RLMs. Instead, we inject __temp_dir__ into the exec
        namespace so code can use it if needed, but don't touch cwd.
        """
        self.globals["__temp_dir__"] = self.temp_dir
        yield

    def _restore_scaffold(self) -> None:
        """Restore scaffold names after execution so overwrites (e.g. context = 'x') don't persist."""
        for name in RESERVED_TOOL_NAMES:
            if name == "llm_query":
                self.globals["llm_query"] = self._llm_query
            elif name == "llm_query_batched":
                self.globals["llm_query_batched"] = self._llm_query_batched
            elif name == "rlm_query":
                self.globals["rlm_query"] = self._rlm_query
            elif name == "rlm_query_batched":
                self.globals["rlm_query_batched"] = self._rlm_query_batched
            elif name == "SHOW_VARS":
                self.globals["SHOW_VARS"] = self._show_vars
            elif name == "answer":
                current = self.locals.get("answer")
                # If the model rebound ``answer`` to a plain dict, the
                # _AnswerDict callback never fired; capture content here if
                # ``ready=True``, then re-wrap so the next cell signals.
                if not isinstance(current, _AnswerDict):
                    replacement = _AnswerDict(on_ready=self._capture_answer)
                    if isinstance(current, dict):
                        for k, v in current.items():
                            dict.__setitem__(replacement, k, v)
                        if current.get("ready") and self._last_final_answer is None:
                            self._last_final_answer = str(current.get("content", ""))
                    self.locals["answer"] = replacement
            elif name == "context" and "context_0" in self.locals:
                self.locals["context"] = self.locals["context_0"]
            elif name == "history" and "history_0" in self.locals and not self.compaction:
                self.locals["history"] = self.locals["history_0"]
            elif name == "history" and self.compaction:
                self.locals["history"] = self._compaction_history

    def execute_code(self, code: str) -> REPLResult:
        """Execute code in the persistent namespace and return result."""
        start_time = time.perf_counter()

        # Clear pending LLM calls from previous execution
        self._pending_llm_calls = []
        self._filled_placeholders = []

        # Repair the .lower()() decode stutter before exec: collapse '()()...' -> '()'.
        # This corruption is never legitimate in the served corpus; repairing it runs
        # the intended code instead of raising TypeError -> error-threshold spiral.
        repaired = False
        if self.repair_doubled_calls and _DOUBLED_CALL.search(code):
            code = _DOUBLED_CALL.sub("()", code)
            repaired = True

        with self._capture_output() as (stdout_buf, stderr_buf), self._temp_cwd():
            try:
                combined = {**self.globals, **self.locals}
                # A sub-call (llm_query) made from inside exec resolves {name}
                # placeholders against THIS namespace - it holds variables created
                # in the same code block that aren't promoted to self.locals yet.
                self._active_namespace = combined
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    exec(code, combined, combined)

                # Update locals with new variables
                for key, value in combined.items():
                    if key not in self.globals and not key.startswith("_"):
                        self.locals[key] = value

                # Restore scaffold so model overwrites (context = ..., llm_query = ...) don't persist
                self._restore_scaffold()

                stdout = stdout_buf.getvalue()
                stderr = stderr_buf.getvalue()
            except Exception as e:
                stdout = stdout_buf.getvalue()
                stderr = stderr_buf.getvalue() + f"\n{type(e).__name__}: {e}"
            finally:
                self._active_namespace = None

        if self._filled_placeholders:
            # Surface in stdout (NOT stderr - it must not count as an error): the model
            # referenced {var} literally in a sub-call prompt (missing f-string); the
            # live variable's value was interpolated so the sub-call saw the real data.
            names = ", ".join(dict.fromkeys(self._filled_placeholders))
            stdout = (
                f"[note: filled REPL variable(s) into an llm_query prompt that used a "
                f"literal placeholder (missing f-string): {names}]\n" + stdout
            )

        if repaired:
            # Surface the repair in stdout (NOT stderr - it must not count as an error):
            # informs the model its doubled-() call was auto-collapsed.
            stdout = "[note: collapsed a doubled '()()' call to '()' before running]\n" + stdout

        if self.max_output_chars and len(stdout) > self.max_output_chars:
            stdout = stdout[:self.max_output_chars] + (
                "\n[OUTPUT TRUNCATED - use llm_query() to analyze large text]"
            )

        final_answer = self._last_final_answer
        self._last_final_answer = None

        return REPLResult(
            stdout=stdout,
            stderr=stderr,
            locals=self.locals.copy(),
            execution_time=time.perf_counter() - start_time,
            rlm_calls=self._pending_llm_calls.copy(),
            final_answer=final_answer,
        )

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cleanup()
        return False

    def cleanup(self):
        """Clean up temp directory and reset state."""
        try:
            shutil.rmtree(self.temp_dir)
        except Exception:
            pass
        if hasattr(self, "globals"):
            self.globals.clear()
        if hasattr(self, "locals"):
            self.locals.clear()

    def __del__(self):
        self.cleanup()
