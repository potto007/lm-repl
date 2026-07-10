"""`_default_answer` must verify, and must not attribute its forcing sentence to the model.

Regression test for the 2026-07-10 finding: when the RLM exhausts `max_iterations` it calls
`_default_answer`, which generated once and returned WITHOUT ever calling `answer_verifier`.
The in-loop citation guard was therefore bypassed on precisely the path a runaway loop always
reaches. Over 51 prod runs, 3 of 3 ungrounded answers arrived here; 0 of the 47 runs that
never reached it were ungrounded.
"""
from types import SimpleNamespace

from prehend.core.rlm import RLM


def _rlm(answer_verifier):
    r = object.__new__(RLM)
    r.logger = None
    r.answer_verifier = answer_verifier
    return r


def _handler(responses):
    it = iter(responses)
    calls = []

    def completion(prompt):
        calls.append(prompt)
        return next(it)

    return SimpleNamespace(completion=completion), calls


def test_forcing_message_is_user_role():
    """An assistant-role forcing sentence merges into a stalled model's own ramble."""
    rlm = _rlm(None)
    handler, calls = _handler(["the answer"])
    rlm._default_answer([{"role": "assistant", "content": "rambling..."}], handler)
    assert calls[0][-1]["role"] == "user"


def test_verifier_runs_and_revision_is_returned():
    seen = []

    def verifier(ans):
        seen.append(ans)
        return (ans == "cited [007]"), "add a citation"

    rlm = _rlm(verifier)
    handler, calls = _handler(["uncited answer", "cited [007]"])
    out = rlm._default_answer([{"role": "user", "content": "q"}], handler)

    assert out == "cited [007]"
    assert seen == ["uncited answer"], "verifier must see the first, unverified answer"
    assert len(calls) == 2, "a rejected forced answer gets one revision attempt"
    assert calls[1][-1]["content"] == "add a citation"
    assert calls[1][-1]["role"] == "user"


def test_accepted_answer_is_not_regenerated():
    rlm = _rlm(lambda ans: (True, None))
    handler, calls = _handler(["cited [007]"])
    out = rlm._default_answer([{"role": "user", "content": "q"}], handler)
    assert out == "cited [007]"
    assert len(calls) == 1


def test_no_verifier_keeps_single_generation():
    """Trajectory-generation harnesses pass answer_verifier=None; behaviour unchanged."""
    rlm = _rlm(None)
    handler, calls = _handler(["whatever"])
    out = rlm._default_answer([{"role": "user", "content": "q"}], handler)
    assert out == "whatever"
    assert len(calls) == 1


if __name__ == "__main__":
    test_forcing_message_is_user_role()
    test_verifier_runs_and_revision_is_returned()
    test_accepted_answer_is_not_regenerated()
    test_no_verifier_keeps_single_generation()
    print("ok")
