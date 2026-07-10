"""The repeat-guard must watch `content`, not just `reasoning_content`.

Regression test for the 2026-07-10 prod failure: the orchestrator runs with thinking
OFF, so content arrives from the first token, `parts` is immediately non-empty, and the
old `if ... and not parts:` condition made the guard dead on exactly the path that
degenerates. A root turn repeated one line 343x and grew the merged assistant message to
192k chars, eventually tipping the request past max_model_len into a hard 400.
"""
from types import SimpleNamespace

from prehend.clients.openai import (
    _REPEAT_GUARD_CONTENT_MIN_CHARS,
    OpenAIClient,
)


def _chunk(content=None, reasoning=None):
    delta = SimpleNamespace(content=content, reasoning_content=reasoning)
    return SimpleNamespace(choices=[SimpleNamespace(delta=delta)], usage=None)


def _client(stream_chunks, threshold=0.35):
    c = object.__new__(OpenAIClient)
    c._repeat_guard_threshold = threshold
    c.repeat_guard_aborts = 0
    c._check_abort = lambda: None
    c._track_cost = lambda *a, **k: None

    class _Stream:
        def __init__(self, chunks):
            self._chunks = chunks
            self.closed = False

        def __iter__(self):
            return iter(self._chunks)

        def close(self):
            self.closed = True

    stream = _Stream(stream_chunks)
    create = lambda **kw: stream  # noqa: E731
    c.client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=create))
    )
    return c, stream


def _run(chunks, threshold=0.35):
    c, stream = _client(chunks, threshold)
    out = c._stream_completion(model="m", messages=[], extra_body={}, create_kwargs={})
    assert stream.closed, "stream must be closed even on abort"
    return c, out


def _degenerate(total_chars):
    """One line repeated verbatim, the observed prod signature."""
    line = "I'll provide `800-511-5010`. "
    reps = total_chars // len(line) + 1
    return [_chunk(content=line) for _ in range(reps)]


def _healthy(total_chars):
    """Varied prose: every chunk introduces new 4-grams."""
    chunks, i, n = [], 0, 0
    while n < total_chars:
        s = f"Step {i}: inspect document {i} and extract the distinct clause about topic {i}. "
        chunks.append(_chunk(content=s))
        n += len(s)
        i += 1
    return chunks


def test_content_ramble_aborts():
    c, out = _run(_degenerate(_REPEAT_GUARD_CONTENT_MIN_CHARS * 3))
    assert c.repeat_guard_aborts == 1
    # aborted early: far less than the full ramble was consumed
    assert len(out) < _REPEAT_GUARD_CONTENT_MIN_CHARS * 3


def test_healthy_content_never_aborts():
    c, out = _run(_healthy(_REPEAT_GUARD_CONTENT_MIN_CHARS * 3))
    assert c.repeat_guard_aborts == 0
    assert len(out) >= _REPEAT_GUARD_CONTENT_MIN_CHARS * 3


def test_short_content_below_floor_never_aborts():
    """A legit root turn peaked at 2,575 chars; the floor must protect it."""
    c, out = _run(_degenerate(_REPEAT_GUARD_CONTENT_MIN_CHARS - 500))
    assert c.repeat_guard_aborts == 0


def test_reasoning_guard_still_works():
    """The original behaviour must survive: reasoning-only streams still abort."""
    line = "Let me reconsider the same point again. "
    chunks = [_chunk(reasoning=line) for _ in range(200)]
    c, _ = _run(chunks)
    assert c.repeat_guard_aborts == 1


if __name__ == "__main__":
    test_content_ramble_aborts()
    test_healthy_content_never_aborts()
    test_short_content_below_floor_never_aborts()
    test_reasoning_guard_still_works()
    print("ok")
