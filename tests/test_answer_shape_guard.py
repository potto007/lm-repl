"""reject_code_shaped_answers: a final answer that is code/tool syntax (e.g. the
corpus-NIAH `find("587X5749B")` child-echo shipped verbatim via answer['content'])
is rejected with corrective feedback and the loop continues, sharing the
max_answer_retries budget. Off by default: teacher/trajectory-generation runs
must not see injected feedback turns (same convention as leaf_prose_guard)."""

from unittest.mock import Mock, patch

import prehend.core.rlm as rlm_module
from prehend import RLM
from prehend.core.types import ModelUsageSummary, UsageSummary

CODE_ANSWER = "find(\"587X5749B\")"
PROSE_ANSWER = "Pediatric Rheumatology Liaison"


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


def _run(responses, **rlm_kwargs):
    with patch.object(rlm_module, "get_client") as mgc:
        mock_lm = _mock_lm(responses)
        mgc.return_value = mock_lm
        with RLM(backend="openai", backend_kwargs={"model_name": "t"},
                 max_iterations=6, **rlm_kwargs) as rlm:
            result = rlm.completion("ctx")
    return result, mock_lm


def test_code_shaped_answer_rejected_then_prose_accepted():
    result, mock_lm = _run(
        [_final(CODE_ANSWER), _final(PROSE_ANSWER)],
        reject_code_shaped_answers=True,
    )
    assert result.response == PROSE_ANSWER
    # the retry prompt carries corrective feedback about code-shaped answers
    second_prompt = str(mock_lm.completion.call_args_list[1].args[0])
    assert "code" in second_prompt.lower()


def test_default_off_ships_code_shaped_answer():
    result, _ = _run([_final(CODE_ANSWER)])
    assert result.response == CODE_ANSWER


def test_retry_budget_exhaustion_ships_code_answer():
    # model insists on the code answer; after max_answer_retries rejections it ships
    result, _ = _run(
        [_final(CODE_ANSWER), _final(CODE_ANSWER)],
        reject_code_shaped_answers=True,
        max_answer_retries=1,
    )
    assert result.response == CODE_ANSWER


def test_prose_answer_unaffected_when_enabled():
    result, mock_lm = _run(
        [_final(PROSE_ANSWER)],
        reject_code_shaped_answers=True,
    )
    assert result.response == PROSE_ANSWER
    assert mock_lm.completion.call_count == 1


def test_shape_check_runs_before_user_answer_verifier():
    # verifier must never see a code-shaped answer; it sees (and accepts) the prose retry
    seen = []

    def verifier(ans):
        seen.append(ans)
        return True, None

    result, _ = _run(
        [_final(CODE_ANSWER), _final(PROSE_ANSWER)],
        reject_code_shaped_answers=True,
        answer_verifier=verifier,
    )
    assert result.response == PROSE_ANSWER
    assert seen == [PROSE_ANSWER]


def test_harness_defaults_thread_reject_code_shaped_answers():
    import dataclasses
    from prehend.harness import Defaults, Harness, Runtime
    h = Harness(model="m", base_url="http://localhost:9999/v1",
                runtime=Runtime(slots=4, ctx=98304),
                defaults=dataclasses.replace(Defaults(), reject_code_shaped_answers=True))
    assert h.srlm.reject_code_shaped_answers is True
    # default stays off
    h2 = Harness(model="m", base_url="http://localhost:9999/v1",
                 runtime=Runtime(slots=4, ctx=98304))
    assert h2.srlm.reject_code_shaped_answers is False


def test_guard_rejection_echo_bounced_then_prose_accepted():
    # single-block assign-and-ship: the model captures a sub-call rejection hint
    # into answer['content'] before ever seeing it (header-fix run 2026-07-08,
    # 9/60 tasks). The shape gate must bounce rejection echoes like code.
    from prehend.utils.subcall_guard import self_context_rejection
    hint = self_context_rejection(["docs", "find"])
    result, mock_lm = _run(
        [_final(hint), _final(PROSE_ANSWER)],
        reject_code_shaped_answers=True,
    )
    assert result.response == PROSE_ANSWER
    second_prompt = str(mock_lm.completion.call_args_list[1].args[0])
    assert "rejection" in second_prompt.lower() or "rejected" in second_prompt.lower()


def test_strategy_verifier_echo_also_bounced():
    from prehend.core.verifier import REJECTION_PREFIX
    result, _ = _run(
        [_final(REJECTION_PREFIX + "whole-task delegation."), _final(PROSE_ANSWER)],
        reject_code_shaped_answers=True,
    )
    assert result.response == PROSE_ANSWER


def test_rejection_echo_not_bounced_when_disabled():
    from prehend.utils.subcall_guard import self_context_rejection
    hint = self_context_rejection(["docs"])
    result, _ = _run([_final(hint)])
    assert result.response == hint
