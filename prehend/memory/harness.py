"""prehend MemoryHarness: wrap an inference client with self-evolving experience memory.

This is the integration seam. It mirrors FinAcumen's ``MemoryAgentVariant``
(retrieve -> delegate solve -> collect) but delegates to anything exposing the
prehend ``completion(prompt, root_prompt)`` interface -- in practice an
:class:`~prehend.SRLM`. ``prompt`` is the context offloaded into the REPL;
``root_prompt`` is the question the orchestrator attends to directly, and where
the retrieved ``<Memory_Block>`` is injected.

Design invariants carried over from FinAcumen:
  * No-memory baseline integrity -- when retrieval is empty, ``root_prompt`` is
    byte-identical to the bare question; no memory tokens leak in.
  * Graceful degradation -- any retrieval failure falls back to no-memory; a
    single failure never crashes the solve.
  * Collect is best-effort -- a distillation failure never breaks the answer.
"""
from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any, Protocol

from prehend.memory.bank import Bank
from prehend.memory.embed import EmbeddingBackend
from prehend.memory.inject import render_memory_block
from prehend.memory.signature import context_signature
from prehend.memory.retrieve import (
    DEFAULT_K_MAX,
    DEFAULT_MIN_COSINE,
    retrieve,
)
from prehend.memory.tagger import NullTagger, Tagger

# (question, context, inference_client_result) -> a new entry dict, or None to write nothing.
# The real TraceDistiller also accepts a keyword-only ``failed`` flag; the harness
# passes it only on the failure path so plain 3-arg distillers stay compatible.
Distiller = Callable[..., dict | None]


def select_for_injection(entries: list[dict], *, max_negatives: int) -> list[dict]:
    """Cap negative-polarity entries in an injection set (ADR-0010 Unit D).

    Walks the (cosine-ranked) entries in order, admitting every positive but at
    most ``max_negatives`` negatives, so failure-derived guard rules can never
    crowd positive recipes out of an injected block. Preserves relevance order.
    """
    selected: list[dict] = []
    negatives = 0
    for e in entries:
        if e.get("polarity") == "negative":
            if negatives >= max_negatives:
                continue
            negatives += 1
        selected.append(e)
    return selected


class InferenceClient(Protocol):
    def completion(self, prompt: str, root_prompt: str | None = None) -> Any:
        ...


class MemoryObserver(Protocol):
    """Observe MemoryHarness internals for metrics/telemetry.

    Both methods are best-effort: the harness wraps every call so an observer
    bug never breaks a solve. The default :class:`NullObserver` no-ops, keeping
    the harness free of any metrics dependency. ``prehend.metrics`` ships a
    ``PrometheusMemoryObserver`` that maps these events onto Prometheus series.
    """

    def on_retrieve(
        self, *, entries: int, top_score: float | None, block_chars: int,
        seconds: float, error: bool,
    ) -> None: ...

    def on_collect(
        self, *, outcome: str, seconds: float, bank_size: int | None,
    ) -> None: ...


class NullObserver:
    """No-op :class:`MemoryObserver`; the default when none is supplied."""

    def on_retrieve(self, **_: Any) -> None:
        pass

    def on_collect(self, **_: Any) -> None:
        pass


class MemoryHarness:
    """Adds retrieve/inject/collect around a context-offloading inference client."""

    def __init__(
        self,
        inference_client: InferenceClient,
        bank: Bank,
        backend: EmbeddingBackend,
        *,
        k_max: int = DEFAULT_K_MAX,
        min_cosine: float = DEFAULT_MIN_COSINE,
        distiller: Distiller | None = None,
        tagger: Tagger | None = None,
        defer_collect: bool = False,
        learn_from_failure: bool = False,
        max_inject_negatives: int = 2,
        context_signature: bool = False,
        freeze_retrieval: bool = False,
        observer: MemoryObserver | None = None,
    ) -> None:
        self.inference_client = inference_client
        self.bank = bank
        self.backend = backend
        self.k_max = k_max
        self.min_cosine = min_cosine
        self.distiller = distiller
        self.tagger = tagger or NullTagger()
        # When True, stamp each experience with a ctx_sig tag derived from the
        # solved document and pass it as a query tag, so the existing tag-gate
        # excludes a different document's same-question experience (the bare
        # question embeds at cosine ~= 1.0 regardless of document). Default off
        # keeps the embedding-only path byte-identical for other consumers.
        self.context_signature = context_signature
        # Write-only cold baseline: when True, _retrieve short-circuits to empty so
        # NO experience is injected (every solve is a true first-exposure), while
        # _collect still writes distilled experiences to the bank. The cold/warm
        # eval sets this for its cold phase so later cold tasks can't read memories
        # written by earlier ones, yet the bank still ends populated for the warm
        # phase. Default off keeps the standard read+write path.
        self.freeze_retrieval = freeze_retrieval
        self.observer = observer or NullObserver()
        # Contrastive failure channel (ADR-0010): when True, collect_pending(False)
        # distills a WRONG solve into a negative guard-rule entry instead of
        # dropping it. Default off preserves correct-only. max_inject_negatives
        # caps negative entries per injected block so they cannot crowd out
        # positive recipes (the structural anti-poisoning backstop).
        self.learn_from_failure = learn_from_failure
        # Per-solve injection record for eval drivers: {"retrieved": int,
        # "injected": [{"id", "polarity", "score"}]}. Overwritten each answer().
        self.last_injection: dict | None = None
        self.max_inject_negatives = max_inject_negatives
        # When True, answer() solves but does NOT distill; the caller invokes
        # collect_pending(correct) once it knows the outcome, so a wrong solve
        # never poisons the bank (the dominant no-upside cause on the v13
        # plain-multihop eval). Default off keeps the drop-in InferenceClient contract:
        # a bare completion() call still learns immediately.
        self.defer_collect = defer_collect
        self._pending: tuple[str, str, Any, dict] | None = None

    def completion(self, prompt: str, root_prompt: str | None = None) -> Any:
        """Transparent :class:`InferenceClient` adapter over :meth:`answer`.

        Lets a memory-wrapped inference client drop into any call site that drives a bare
        ``InferenceClient`` via ``completion(context, question)``. ``prompt`` is the
        offloaded context; ``root_prompt`` is the question (defaulting to
        ``prompt`` when omitted, matching the bare-inference-client convention).
        """
        return self.answer(prompt, root_prompt if root_prompt is not None else prompt)

    def _observe(self, method: str, **kw: Any) -> None:
        """Call an observer hook best-effort; a telemetry bug never propagates."""
        try:
            getattr(self.observer, method)(**kw)
        except Exception:
            pass

    def answer(self, context: str, question: str) -> Any:
        """Solve ``question`` over ``context``, using and growing memory."""
        try:
            query_tags = self.tagger.tag(question)
        except Exception:
            query_tags = {}
        # Document-signature gate (opt-in): both _retrieve (query tag) and
        # _collect (stamped onto the new entry) consume query_tags, so adding
        # ctx_sig here both filters retrieval and records the experience's own
        # document identity in one place.
        if self.context_signature:
            query_tags = {**query_tags, "ctx_sig": context_signature(context)}
        entries, scores, error, retrieve_seconds = self._retrieve(question, query_tags)

        # Polarity-aware cap: never let failure-derived guard rules crowd positive
        # recipes out of the block. Only INJECTED entries get a use_count bump, so
        # a negative repeatedly retrieved-but-capped-out stays use_count==0 and is
        # eligible for prune() (the bank self-cleans dominated negatives).
        injected = select_for_injection(entries, max_negatives=self.max_inject_negatives)
        score_by_id = {id(e): s for e, s in zip(entries, scores)}
        self.last_injection = {
            "retrieved": len(entries),
            "injected": [
                {"id": e.get("id"), "polarity": e.get("polarity"),
                 "score": round(score_by_id.get(id(e), 0.0), 4)}
                for e in injected
            ],
        }

        block = ""
        if injected:
            block = render_memory_block(injected)
            root_prompt = f"{block}\n{question}" if block else question
            for entry in injected:
                eid = entry.get("id")
                if eid is not None:
                    try:
                        self.bank.bump_stats(eid, use_delta=1)
                    except Exception:
                        pass
        else:
            root_prompt = question

        self._observe(
            "on_retrieve",
            entries=len(entries),
            top_score=(max(scores) if scores else None),
            block_chars=len(block),
            seconds=retrieve_seconds,
            error=error,
        )

        result = self.inference_client.completion(context, root_prompt)

        if self.defer_collect:
            self._pending = (question, context, result, query_tags)
            self._observe("on_collect", outcome="deferred", seconds=0.0, bank_size=None)
        else:
            self._collect(question, context, result, query_tags)
        return result

    def collect_pending(self, correct: bool | None = True) -> None:
        """Distill the last deferred solve, gated on its outcome.

        ``correct is False`` drops it (do not learn from a wrong solve);
        ``True`` or ``None`` (unscored) distills. No-op when nothing is pending
        or ``defer_collect`` was off. Best-effort: never raises.
        """
        pending = self._pending
        self._pending = None
        if pending is None:
            return
        question, context, result, query_tags = pending
        if correct is False:
            if self.learn_from_failure:
                # Contrastive failure channel: distill a negative guard rule.
                self._collect(question, context, result, query_tags, failed=True)
            else:
                self._observe("on_collect", outcome="dropped", seconds=0.0, bank_size=None)
            return
        self._collect(question, context, result, query_tags)

    def _retrieve(
        self, question: str, query_tags: dict
    ) -> tuple[list[dict], list[float], bool, float]:
        """Retrieve experiences, degrading to empty on any failure.

        Returns ``(entries, scores, error, seconds)`` -- ``error`` distinguishes
        a real retrieval failure from a clean miss, and ``seconds`` is the
        embed+search wall time, both for the observer.
        """
        if self.freeze_retrieval:
            # Write-only cold baseline: skip retrieval entirely (clean miss).
            return [], [], False, 0.0
        t0 = time.perf_counter()
        try:
            res = retrieve(
                question, self.bank, self.backend,
                k_max=self.k_max, min_cosine=self.min_cosine,
                query_tags=query_tags,
            )
            return res.entries, res.scores, False, time.perf_counter() - t0
        except Exception:
            return [], [], True, time.perf_counter() - t0

    def _collect(
        self, question: str, context: str, result: Any, query_tags: dict,
        *, failed: bool = False,
    ) -> None:
        """Best-effort distillation of a new experience; never raises.

        Provenance-aware collision (ADR-0010): a SUCCESS supersedes a same-id
        FAILURE entry (we now know how to solve it); a FAILURE never overwrites or
        shadows a success; same-provenance collisions dedup. ``failed`` is only
        passed to the distiller on the failure path so plain 3-arg distillers stay
        compatible on the success path.
        """
        if self.distiller is None:
            return
        t0 = time.perf_counter()
        outcome = "error"
        bank_size: int | None = None
        try:
            entry = (
                self.distiller(question, context, result, failed=True)
                if failed else self.distiller(question, context, result)
            )
            if not entry:
                outcome = "empty"
                return
            if query_tags and not entry.get("tags"):
                entry["tags"] = dict(query_tags)
            existing = self.bank.load()
            eid = entry.get("id")
            prior = next((e for e in existing if e.get("id") == eid), None) if eid is not None else None

            if prior is None:
                self.bank.append(entry)
                outcome = "written"
                bank_size = len(existing) + 1
            elif prior.get("derived_from") == "failure" and entry.get("derived_from") != "failure":
                # Defensive fallback only: since the ADR-0011 amendment, _entry_id
                # keys on (question, derived_from), so a success and a failure for
                # the same question get distinct ids and never collide here. Kept
                # for legacy/question-only banks where they could still alias.
                updated = [entry if e.get("id") == eid else e for e in existing]
                self.bank.save(updated)  # same length -> not a shrink -> accepted
                outcome = "superseded"
                bank_size = len(updated)
            elif prior.get("derived_from") != "failure" and entry.get("derived_from") == "failure":
                # Defensive fallback only (see above): cross-provenance ids no
                # longer collide, so a failure no longer shadows a known-good
                # recipe by id - both coexist and injection balances them.
                outcome = "superseded_skip"
            else:
                outcome = "duplicate"  # same provenance, one experience per id
        except Exception:
            outcome = "error"
        finally:
            self._observe(
                "on_collect", outcome=outcome,
                seconds=time.perf_counter() - t0, bank_size=bank_size,
            )
