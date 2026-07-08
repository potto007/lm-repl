"""reject_self_context_delegation: when custom tools hold the task's data (the
offloaded-corpus pattern: docs/manifest/find in the REPL, `context` = a short
briefing), delegating that same briefing back out via llm_query(context=context)
is futile - the sub-LLM has no REPL or tools and can only echo or refuse
(observed 59/60 corpus-NIAH v13-sft trajectories, 2026-06-27). With the flag on,
the sub-call is rejected deterministically with an actionable hint instead of
being sent. Off by default (teacher-run convention, like leaf_prose_guard)."""

from unittest.mock import Mock, patch

import prehend.core.rlm as rlm_module
from prehend import RLM
from prehend.core.types import ModelUsageSummary, UsageSummary
from prehend.environments.local_repl import LocalREPL

BRIEFING = (
    "You are a research librarian over an OFFLOADED corpus in your REPL.\n"
    "Catalog:\n[300.1] the NUCC Taxonomy Code Set (part 1/12)\n"
)
TOOLS = {
    "docs": {"tool": {"300.1": "| Code | Classification |"}, "description": "dict: doc id -> full text"},
    "find": {"tool": lambda q: [], "description": "find(query) -> matching rows"},
}


def _env(**kwargs):
    return LocalREPL(context_payload=BRIEFING, custom_tools=TOOLS, **kwargs)


def test_rejects_delegating_own_context_when_enabled():
    env = _env(reject_self_context_delegation=True)
    try:
        out = env._llm_query("What is the classification for 439N2572H?", context=BRIEFING)
    finally:
        env.cleanup()
    assert "rejected" in out.lower()
    # hint names the actual REPL tools so the model can pivot
    assert "docs" in out and "find" in out


def test_off_by_default_attempts_send():
    env = _env()
    try:
        out = env._llm_query("q", context=BRIEFING)
    finally:
        env.cleanup()
    assert "No LM handler configured" in out


def test_other_context_data_not_rejected():
    env = _env(reject_self_context_delegation=True)
    try:
        out = env._llm_query("q", context="| 439N2572H | Hyperbaric Hematology Navigator |")
    finally:
        env.cleanup()
    assert "No LM handler configured" in out


def test_no_custom_tools_not_rejected():
    env = LocalREPL(context_payload=BRIEFING, reject_self_context_delegation=True)
    try:
        out = env._llm_query("q", context=BRIEFING)
    finally:
        env.cleanup()
    assert "No LM handler configured" in out


def test_rlm_query_fallback_also_rejected():
    # subcall_fn is None so rlm_query falls back through the same dispatch
    env = _env(reject_self_context_delegation=True)
    try:
        out = env._rlm_query("q", context=BRIEFING)
    finally:
        env.cleanup()
    assert "rejected" in out.lower()


DELEGATE = '```repl\nprint(llm_query("What is the classification?", context=context))\n```'


def _final(content: str) -> str:
    return f"```repl\nanswer['content'] = {content!r}\nanswer['ready'] = True\n```"


def _mock_lm(responses):
    m = Mock()
    m.completion.side_effect = list(responses)
    usage = UsageSummary(model_usage_summaries={
        "mock": ModelUsageSummary(total_calls=1, total_input_tokens=10, total_output_tokens=5)})
    m.get_usage_summary.return_value = usage
    m.get_last_usage.return_value = usage
    return m


def test_rlm_threads_flag_to_environment():
    # end-to-end: the model delegates its own context; the REPL output the model
    # sees on the next turn is the rejection hint, not a send attempt
    with patch.object(rlm_module, "get_client") as mgc:
        mock_lm = _mock_lm([DELEGATE, _final("done")])
        mgc.return_value = mock_lm
        with RLM(backend="openai", backend_kwargs={"model_name": "t"},
                 max_iterations=4, custom_tools=TOOLS,
                 reject_self_context_delegation=True) as rlm:
            rlm.completion(BRIEFING)
        second_prompt = str(mock_lm.completion.call_args_list[1].args[0])
    assert "rejected" in second_prompt.lower()


def test_harness_defaults_thread_reject_self_context_delegation():
    import dataclasses
    from prehend.harness import Defaults, Harness, Runtime
    h = Harness(model="m", base_url="http://localhost:9999/v1",
                runtime=Runtime(slots=4, ctx=98304),
                defaults=dataclasses.replace(Defaults(), reject_self_context_delegation=True))
    assert h.srlm.reject_self_context_delegation is True
    h2 = Harness(model="m", base_url="http://localhost:9999/v1",
                 runtime=Runtime(slots=4, ctx=98304))
    assert h2.srlm.reject_self_context_delegation is False
