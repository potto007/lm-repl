"""Tests for the prehend MemoryHarness (retrieve -> inject -> solve -> collect)."""
from __future__ import annotations

from types import SimpleNamespace

from prehend.memory.bank import Bank
from prehend.memory.harness import MemoryHarness


class FakeInferenceClient:
    """Records the (prompt, root_prompt) it was called with."""

    def __init__(self, answer="42"):
        self.answer = answer
        self.calls: list[tuple] = []

    def completion(self, prompt, root_prompt=None):
        self.calls.append((prompt, root_prompt))
        return SimpleNamespace(final_answer=self.answer)


class FakeBackend:
    def __init__(self, table=None, raises=False):
        self.table = table or {}
        self.raises = raises

    def embed(self, text):
        if self.raises:
            raise RuntimeError("embedding service down")
        return self.table.get(text, [0.0, 0.0])


def _entry(eid, embedding):
    return {
        "id": eid,
        "polarity": "positive",
        "key_insight": f"insight {eid}",
        "embedding": embedding,
        "stats": {"use_count": 0, "hit_count": 0},
    }


def test_no_memory_path_keeps_root_prompt_byte_identical(tmp_path):
    bank = Bank(tmp_path / "mem")  # empty
    inference_client = FakeInferenceClient()
    harness = MemoryHarness(inference_client, bank, FakeBackend())
    harness.answer(context="ctx", question="What is 6*7?")
    prompt, root_prompt = inference_client.calls[0]
    assert prompt == "ctx"
    assert root_prompt == "What is 6*7?"  # no memory tokens leaked in


def test_with_memory_injects_block_into_root_prompt(tmp_path):
    bank = Bank(tmp_path / "mem")
    bank.append(_entry("a", [1.0, 0.0]))
    inference_client = FakeInferenceClient()
    harness = MemoryHarness(inference_client, bank, FakeBackend({"Q": [1.0, 0.0]}), min_cosine=0.5)
    harness.answer(context="ctx", question="Q")
    _, root_prompt = inference_client.calls[0]
    assert "<Memory_Block>" in root_prompt
    assert "insight a" in root_prompt
    assert root_prompt.rstrip().endswith("Q")


def test_last_injection_records_injected_entries(tmp_path):
    bank = Bank(tmp_path / "mem")
    bank.append(_entry("a", [1.0, 0.0]))
    harness = MemoryHarness(FakeInferenceClient(), bank, FakeBackend({"Q": [1.0, 0.0]}), min_cosine=0.5)
    assert harness.last_injection is None
    harness.answer(context="ctx", question="Q")
    rec = harness.last_injection
    assert rec["retrieved"] == 1
    assert rec["injected"] == [{"id": "a", "polarity": "positive", "score": 1.0}]


def test_last_injection_empty_on_miss(tmp_path):
    harness = MemoryHarness(FakeInferenceClient(), Bank(tmp_path / "mem"), FakeBackend())
    harness.answer(context="ctx", question="Q")
    assert harness.last_injection == {"retrieved": 0, "injected": []}


def test_returns_inference_client_result(tmp_path):
    bank = Bank(tmp_path / "mem")
    inference_client = FakeInferenceClient(answer="hello")
    harness = MemoryHarness(inference_client, bank, FakeBackend())
    result = harness.answer(context="ctx", question="q")
    assert result.final_answer == "hello"


def test_retrieval_bumps_use_count(tmp_path):
    bank = Bank(tmp_path / "mem")
    bank.append(_entry("a", [1.0, 0.0]))
    harness = MemoryHarness(FakeInferenceClient(), bank, FakeBackend({"Q": [1.0, 0.0]}), min_cosine=0.5)
    harness.answer(context="ctx", question="Q")
    assert bank.load()[0]["stats"]["use_count"] == 1


def test_embedding_failure_degrades_to_no_memory(tmp_path):
    bank = Bank(tmp_path / "mem")
    bank.append(_entry("a", [1.0, 0.0]))
    inference_client = FakeInferenceClient()
    harness = MemoryHarness(inference_client, bank, FakeBackend(raises=True), min_cosine=0.5)
    # Must not raise; solve still happens with a clean root_prompt.
    harness.answer(context="ctx", question="Q")
    _, root_prompt = inference_client.calls[0]
    assert root_prompt == "Q"


def test_collect_appends_distilled_entry(tmp_path):
    bank = Bank(tmp_path / "mem")  # empty

    def distiller(question, context, result):
        return _entry("learned", [0.5, 0.5])

    harness = MemoryHarness(FakeInferenceClient(), bank, FakeBackend(), distiller=distiller)
    harness.answer(context="ctx", question="q")
    assert [e["id"] for e in bank.load()] == ["learned"]


def test_collect_none_writes_nothing(tmp_path):
    bank = Bank(tmp_path / "mem")
    harness = MemoryHarness(FakeInferenceClient(), bank, FakeBackend(),
                            distiller=lambda q, c, r: None)
    harness.answer(context="ctx", question="q")
    assert bank.load() == []


def _learn_distiller(question, context, result):
    return _entry("learned", [0.5, 0.5])


def test_defer_collect_does_not_distill_in_answer(tmp_path):
    # With defer_collect, answer() solves but does NOT write an experience -
    # the caller decides later (once it knows correctness) via collect_pending.
    bank = Bank(tmp_path / "mem")
    harness = MemoryHarness(FakeInferenceClient(), bank, FakeBackend(),
                            distiller=_learn_distiller, defer_collect=True)
    harness.answer(context="ctx", question="q")
    assert bank.load() == []  # nothing distilled yet


def test_collect_pending_correct_distills(tmp_path):
    bank = Bank(tmp_path / "mem")
    harness = MemoryHarness(FakeInferenceClient(), bank, FakeBackend(),
                            distiller=_learn_distiller, defer_collect=True)
    harness.answer(context="ctx", question="q")
    harness.collect_pending(correct=True)
    assert [e["id"] for e in bank.load()] == ["learned"]


def test_collect_pending_wrong_skips_distillation(tmp_path):
    # The whole point of #1: do NOT learn from a wrong solve.
    bank = Bank(tmp_path / "mem")
    harness = MemoryHarness(FakeInferenceClient(), bank, FakeBackend(),
                            distiller=_learn_distiller, defer_collect=True)
    harness.answer(context="ctx", question="q")
    harness.collect_pending(correct=False)
    assert bank.load() == []


def test_collect_pending_unknown_correctness_distills(tmp_path):
    # correct=None (no expected answer / unscored) -> conservatively keep it.
    bank = Bank(tmp_path / "mem")
    harness = MemoryHarness(FakeInferenceClient(), bank, FakeBackend(),
                            distiller=_learn_distiller, defer_collect=True)
    harness.answer(context="ctx", question="q")
    harness.collect_pending(correct=None)
    assert [e["id"] for e in bank.load()] == ["learned"]


def test_collect_pending_is_noop_without_pending(tmp_path):
    bank = Bank(tmp_path / "mem")
    harness = MemoryHarness(FakeInferenceClient(), bank, FakeBackend(),
                            distiller=_learn_distiller, defer_collect=True)
    harness.collect_pending(correct=True)  # nothing solved yet
    assert bank.load() == []


def test_collect_skips_entry_whose_id_already_exists(tmp_path):
    bank = Bank(tmp_path / "mem")
    bank.append(_entry("dup", [0.1, 0.2]))  # id 'dup' already present

    harness = MemoryHarness(FakeInferenceClient(), bank, FakeBackend(),
                            distiller=lambda q, c, r: _entry("dup", [0.3, 0.4]))
    harness.answer(context="ctx", question="q")
    # Still exactly one 'dup' entry; collect must not append a duplicate id.
    assert [e["id"] for e in bank.load()] == ["dup"]


class FakeTagger:
    def __init__(self, tags):
        self.tags = tags

    def tag(self, query):
        return dict(self.tags)


def test_tagger_gates_retrieval_by_conflicting_tag(tmp_path):
    bank = Bank(tmp_path / "mem")
    e = _entry("a", [1.0, 0.0])
    e["tags"] = {"kind": "numeric"}
    bank.append(e)
    inference_client = FakeInferenceClient()
    harness = MemoryHarness(
        inference_client, bank, FakeBackend({"Q": [1.0, 0.0]}), min_cosine=0.5,
        tagger=FakeTagger({"kind": "entity"}),
    )
    harness.answer(context="ctx", question="Q")
    _, root_prompt = inference_client.calls[0]
    assert root_prompt == "Q"  # conflicting tag gated the entry out -> no memory


def test_collect_tags_entry_using_tagger(tmp_path):
    bank = Bank(tmp_path / "mem")
    harness = MemoryHarness(
        FakeInferenceClient(), bank, FakeBackend(),
        tagger=FakeTagger({"kind": "numeric"}),
        distiller=lambda q, c, r: _entry("learned", [0.1, 0.2]),
    )
    harness.answer(context="ctx", question="q")
    assert bank.load()[0]["tags"] == {"kind": "numeric"}


def test_collect_failure_never_breaks_answer(tmp_path):
    bank = Bank(tmp_path / "mem")

    def boom(question, context, result):
        raise RuntimeError("distiller blew up")

    inference_client = FakeInferenceClient(answer="ok")
    harness = MemoryHarness(inference_client, bank, FakeBackend(), distiller=boom)
    result = harness.answer(context="ctx", question="q")
    assert result.final_answer == "ok"


# --- completion() as a transparent InferenceClient adapter (step 1) -------------------

def test_completion_is_drop_in_for_answer(tmp_path):
    # A memory-wrapped inference client invoked via .completion(context, query) must drive
    # the inner inference client identically to .answer(context, query).
    bank = Bank(tmp_path / "mem")
    bank.append(_entry("a", [1.0, 0.0]))
    inference_client = FakeInferenceClient()
    harness = MemoryHarness(inference_client, bank, FakeBackend({"Q": [1.0, 0.0]}), min_cosine=0.5)
    harness.completion("ctx", "Q")
    prompt, root_prompt = inference_client.calls[0]
    assert prompt == "ctx"
    assert "<Memory_Block>" in root_prompt
    assert "insight a" in root_prompt
    assert root_prompt.rstrip().endswith("Q")


def test_completion_no_memory_path_keeps_root_prompt_byte_identical(tmp_path):
    # No-memory invariant holds through the completion() seam too.
    bank = Bank(tmp_path / "mem")  # empty
    inference_client = FakeInferenceClient()
    harness = MemoryHarness(inference_client, bank, FakeBackend())
    harness.completion("ctx", "What is 6*7?")
    prompt, root_prompt = inference_client.calls[0]
    assert prompt == "ctx"
    assert root_prompt == "What is 6*7?"


def test_completion_returns_inference_client_result(tmp_path):
    bank = Bank(tmp_path / "mem")
    inference_client = FakeInferenceClient(answer="hello")
    harness = MemoryHarness(inference_client, bank, FakeBackend())
    result = harness.completion("ctx", "q")
    assert result.final_answer == "hello"


def test_completion_defaults_root_prompt_to_prompt(tmp_path):
    # Single-arg completion(prompt): prompt doubles as the question.
    bank = Bank(tmp_path / "mem")  # empty
    inference_client = FakeInferenceClient()
    harness = MemoryHarness(inference_client, bank, FakeBackend())
    harness.completion("only the prompt")
    prompt, root_prompt = inference_client.calls[0]
    assert prompt == "only the prompt"
    assert root_prompt == "only the prompt"


# --- observer telemetry seam ------------------------------------------------

class RecordingObserver:
    """Captures the kwargs of every on_retrieve/on_collect event."""

    def __init__(self):
        self.retrieves: list[dict] = []
        self.collects: list[dict] = []

    def on_retrieve(self, **kw):
        self.retrieves.append(kw)

    def on_collect(self, **kw):
        self.collects.append(kw)


def test_observer_records_hit(tmp_path):
    bank = Bank(tmp_path / "mem")
    bank.append(_entry("a", [1.0, 0.0]))
    obs = RecordingObserver()
    harness = MemoryHarness(
        FakeInferenceClient(), bank, FakeBackend({"Q": [1.0, 0.0]}), min_cosine=0.5,
        observer=obs,
    )
    harness.answer(context="ctx", question="Q")
    (ev,) = obs.retrieves
    assert ev["entries"] == 1 and ev["error"] is False
    assert ev["top_score"] is not None and ev["block_chars"] > 0
    assert ev["seconds"] >= 0.0


def test_observer_records_miss(tmp_path):
    bank = Bank(tmp_path / "mem")  # empty
    obs = RecordingObserver()
    harness = MemoryHarness(FakeInferenceClient(), bank, FakeBackend(), observer=obs)
    harness.answer(context="ctx", question="q")
    (ev,) = obs.retrieves
    assert ev["entries"] == 0 and ev["error"] is False
    assert ev["top_score"] is None and ev["block_chars"] == 0


def test_observer_records_retrieval_error(tmp_path):
    bank = Bank(tmp_path / "mem")
    bank.append(_entry("a", [1.0, 0.0]))
    obs = RecordingObserver()
    harness = MemoryHarness(
        FakeInferenceClient(), bank, FakeBackend(raises=True), min_cosine=0.5, observer=obs,
    )
    harness.answer(context="ctx", question="Q")  # must not raise
    (ev,) = obs.retrieves
    assert ev["error"] is True and ev["entries"] == 0


def test_observer_records_written_collect(tmp_path):
    bank = Bank(tmp_path / "mem")
    obs = RecordingObserver()
    harness = MemoryHarness(
        FakeInferenceClient(), bank, FakeBackend(),
        distiller=lambda q, c, r: _entry("learned", [0.5, 0.5]), observer=obs,
    )
    harness.answer(context="ctx", question="q")
    (ev,) = obs.collects
    assert ev["outcome"] == "written" and ev["bank_size"] == 1


def test_observer_records_empty_collect(tmp_path):
    bank = Bank(tmp_path / "mem")
    obs = RecordingObserver()
    harness = MemoryHarness(
        FakeInferenceClient(), bank, FakeBackend(),
        distiller=lambda q, c, r: None, observer=obs,
    )
    harness.answer(context="ctx", question="q")
    (ev,) = obs.collects
    assert ev["outcome"] == "empty" and ev["bank_size"] is None


def test_observer_records_duplicate_collect(tmp_path):
    bank = Bank(tmp_path / "mem")
    bank.append(_entry("dup", [0.1, 0.2]))
    obs = RecordingObserver()
    harness = MemoryHarness(
        FakeInferenceClient(), bank, FakeBackend(),
        distiller=lambda q, c, r: _entry("dup", [0.3, 0.4]), observer=obs,
    )
    harness.answer(context="ctx", question="q")
    (ev,) = obs.collects
    assert ev["outcome"] == "duplicate"


def test_observer_records_deferred_then_dropped(tmp_path):
    bank = Bank(tmp_path / "mem")
    obs = RecordingObserver()
    harness = MemoryHarness(
        FakeInferenceClient(), bank, FakeBackend(),
        distiller=_learn_distiller, defer_collect=True, observer=obs,
    )
    harness.answer(context="ctx", question="q")
    harness.collect_pending(correct=False)
    assert [e["outcome"] for e in obs.collects] == ["deferred", "dropped"]


def test_observer_records_deferred_then_written(tmp_path):
    bank = Bank(tmp_path / "mem")
    obs = RecordingObserver()
    harness = MemoryHarness(
        FakeInferenceClient(), bank, FakeBackend(),
        distiller=_learn_distiller, defer_collect=True, observer=obs,
    )
    harness.answer(context="ctx", question="q")
    harness.collect_pending(correct=True)
    assert [e["outcome"] for e in obs.collects] == ["deferred", "written"]


def test_observer_exception_never_breaks_answer(tmp_path):
    class Boom:
        def on_retrieve(self, **kw):
            raise RuntimeError("observer down")

        def on_collect(self, **kw):
            raise RuntimeError("observer down")

    bank = Bank(tmp_path / "mem")
    inference_client = FakeInferenceClient(answer="ok")
    harness = MemoryHarness(
        inference_client, bank, FakeBackend(),
        distiller=lambda q, c, r: _entry("x", [0.1, 0.2]), observer=Boom(),
    )
    result = harness.answer(context="ctx", question="q")
    assert result.final_answer == "ok"  # telemetry failure swallowed


# ===================================================================
# Contrastive failure channel (ADR-0010 / 2026-06-22 spec)
# ===================================================================

def _mk_distiller(*, derived_from="success", eid="learned", polarity=None, embedding=None):
    """A distiller that returns a controllable entry and records the `failed` arg."""
    polarity = polarity or ("negative" if derived_from == "failure" else "positive")
    embedding = embedding or [0.5, 0.5]
    calls = []

    def d(question, context, result, failed=False):
        calls.append(failed)
        e = _entry(eid, embedding)
        e["polarity"] = polarity
        e["derived_from"] = derived_from
        return e

    d.calls = calls
    return d


def test_learn_from_failure_defaults_false(tmp_path):
    h = MemoryHarness(FakeInferenceClient(), Bank(tmp_path / "m"), FakeBackend())
    assert h.learn_from_failure is False


def test_collect_pending_false_with_flag_distills_failure(tmp_path):
    bank = Bank(tmp_path / "m")
    d = _mk_distiller(derived_from="failure", eid="f")
    h = MemoryHarness(FakeInferenceClient(), bank, FakeBackend(), distiller=d,
                      defer_collect=True, learn_from_failure=True)
    h.answer(context="c", question="q")
    h.collect_pending(correct=False)
    assert d.calls == [True]  # distilled with failed=True
    loaded = bank.load()
    assert [e["id"] for e in loaded] == ["f"]
    assert loaded[0]["derived_from"] == "failure"


def test_collect_pending_false_without_flag_drops(tmp_path):
    bank = Bank(tmp_path / "m")
    d = _mk_distiller(eid="f")
    h = MemoryHarness(FakeInferenceClient(), bank, FakeBackend(), distiller=d,
                      defer_collect=True)  # learn_from_failure default False
    h.answer(context="c", question="q")
    h.collect_pending(correct=False)
    assert d.calls == []  # distiller never called
    assert bank.load() == []


def test_non_deferred_collect_defaults_failed_false(tmp_path):
    bank = Bank(tmp_path / "m")
    d = _mk_distiller(eid="x")
    h = MemoryHarness(FakeInferenceClient(), bank, FakeBackend(), distiller=d)  # not deferred
    h.answer(context="c", question="q")
    assert d.calls == [False]
    assert [e["id"] for e in bank.load()] == ["x"]


# --- provenance-aware collision (success supersedes failure) ---

def test_wrong_then_right_success_supersedes_failure(tmp_path):
    bank = Bank(tmp_path / "m")
    fail = _mk_distiller(derived_from="failure", eid="exp_q")
    h1 = MemoryHarness(FakeInferenceClient(), bank, FakeBackend(), distiller=fail,
                       defer_collect=True, learn_from_failure=True)
    h1.answer(context="c", question="q")
    h1.collect_pending(correct=False)
    assert [(e["id"], e["derived_from"]) for e in bank.load()] == [("exp_q", "failure")]

    succ = _mk_distiller(derived_from="success", eid="exp_q")
    h2 = MemoryHarness(FakeInferenceClient(), bank, FakeBackend(), distiller=succ,
                       defer_collect=True, learn_from_failure=True)
    h2.answer(context="c", question="q")
    h2.collect_pending(correct=True)
    loaded = bank.load()
    assert len(loaded) == 1
    assert loaded[0]["derived_from"] == "success"
    assert loaded[0]["polarity"] == "positive"


def test_right_then_wrong_failure_does_not_overwrite_success(tmp_path):
    bank = Bank(tmp_path / "m")
    succ = _mk_distiller(derived_from="success", eid="exp_q")
    h1 = MemoryHarness(FakeInferenceClient(), bank, FakeBackend(), distiller=succ,
                       defer_collect=True, learn_from_failure=True)
    h1.answer(context="c", question="q")
    h1.collect_pending(correct=True)

    fail = _mk_distiller(derived_from="failure", eid="exp_q")
    h2 = MemoryHarness(FakeInferenceClient(), bank, FakeBackend(), distiller=fail,
                       defer_collect=True, learn_from_failure=True)
    h2.answer(context="c", question="q")
    h2.collect_pending(correct=False)
    loaded = bank.load()
    assert len(loaded) == 1
    assert loaded[0]["derived_from"] == "success"  # success preserved


def test_failure_then_failure_dedup(tmp_path):
    bank = Bank(tmp_path / "m")
    for _ in range(2):
        fail = _mk_distiller(derived_from="failure", eid="exp_q")
        h = MemoryHarness(FakeInferenceClient(), bank, FakeBackend(), distiller=fail,
                          defer_collect=True, learn_from_failure=True)
        h.answer(context="c", question="q")
        h.collect_pending(correct=False)
    assert len(bank.load()) == 1


# --- freeze_retrieval: write-only cold baseline (true first-exposure) ---


def test_freeze_retrieval_defaults_false(tmp_path):
    h = MemoryHarness(FakeInferenceClient(), Bank(tmp_path / "m"), FakeBackend())
    assert h.freeze_retrieval is False


def test_freeze_retrieval_suppresses_injection(tmp_path):
    # A matching bank entry that WOULD be injected must not reach the inference client when
    # retrieval is frozen: the cold task is a true first-exposure baseline.
    bank = Bank(tmp_path / "mem")
    bank.append(_entry("a", [1.0, 0.0]))
    inference_client = FakeInferenceClient()
    harness = MemoryHarness(
        inference_client, bank, FakeBackend({"Q": [1.0, 0.0]}), min_cosine=0.5,
        freeze_retrieval=True,
    )
    harness.answer(context="ctx", question="Q")
    _, root_prompt = inference_client.calls[0]
    assert root_prompt == "Q"  # no memory block injected
    assert bank.load()[0]["stats"]["use_count"] == 0  # not retrieved -> not bumped


def test_freeze_retrieval_still_writes_distilled_experience(tmp_path):
    # Memories are still written so the bank is populated for the warm run.
    bank = Bank(tmp_path / "mem")  # empty
    harness = MemoryHarness(
        FakeInferenceClient(), bank, FakeBackend(),
        distiller=lambda q, c, r: _entry("learned", [0.5, 0.5]),
        freeze_retrieval=True,
    )
    harness.answer(context="ctx", question="q")
    assert [e["id"] for e in bank.load()] == ["learned"]


# --- polarity-aware injection cap (Unit D) ---

def _bank_with_polarities(tmp_path, n_pos, n_neg, emb):
    bank = Bank(tmp_path / "m")
    for i in range(n_pos):
        e = _entry(f"pos{i}", emb); e["polarity"] = "positive"; bank.append(e)
    for i in range(n_neg):
        e = _entry(f"neg{i}", emb); e["polarity"] = "negative"; bank.append(e)
    return bank


def test_injection_caps_negatives(tmp_path):
    bank = _bank_with_polarities(tmp_path, n_pos=2, n_neg=3, emb=[1.0, 0.0])
    inference_client = FakeInferenceClient()
    h = MemoryHarness(inference_client, bank, FakeBackend({"Q": [1.0, 0.0]}),
                      min_cosine=0.5, k_max=10, max_inject_negatives=1)
    h.answer(context="c", question="Q")
    _, root = inference_client.calls[0]
    assert root.count('polarity="negative"') == 1
    assert root.count('polarity="positive"') == 2


def test_all_negative_retrieval_capped(tmp_path):
    bank = _bank_with_polarities(tmp_path, n_pos=0, n_neg=4, emb=[1.0, 0.0])
    inference_client = FakeInferenceClient()
    h = MemoryHarness(inference_client, bank, FakeBackend({"Q": [1.0, 0.0]}),
                      min_cosine=0.5, k_max=10, max_inject_negatives=2)
    h.answer(context="c", question="Q")
    _, root = inference_client.calls[0]
    assert root.count('polarity="negative"') == 2


def test_bump_stats_only_for_injected(tmp_path):
    bank = _bank_with_polarities(tmp_path, n_pos=1, n_neg=3, emb=[1.0, 0.0])
    inference_client = FakeInferenceClient()
    h = MemoryHarness(inference_client, bank, FakeBackend({"Q": [1.0, 0.0]}),
                      min_cosine=0.5, k_max=10, max_inject_negatives=1)
    h.answer(context="c", question="Q")
    counts = sorted(e["stats"]["use_count"] for e in bank.load())
    # 1 positive + 1 negative injected (bumped to 1); 2 negatives capped out (0)
    assert counts == [0, 0, 1, 1]
