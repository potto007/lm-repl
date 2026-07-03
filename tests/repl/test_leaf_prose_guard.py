"""Leaf prose guard: llm_query leaf completions must return prose, not code.

Preference-tuned models (v0.5 KTO) learned "REPL code = grounded, prose = confab"
so hard that a bare leaf completion answers `find("V406XXA")` instead of the
description (142/193 wrong finals on icd10xl). The guard wraps leaf prompts with
a prose instruction and retries once when the reply still looks like code.
Opt-in via PREHEND_LEAF_PROSE_GUARD=1 so teacher/trajectory-gen behavior is
untouched by default.
"""
import pytest

from prehend.environments.leaf_prose_guard import (
    LEAF_PROSE_INSTRUCTION,
    guard_enabled,
    looks_like_code_reply,
    repair_leaf_reply,
    wrap_leaf_prompt,
)


class TestLooksLikeCodeReply:
    @pytest.mark.parametrize("reply", [
        'find("V406XXA")',
        "find('S85161A')",
        'llm_query("what is code X?", context=context)',
        'print(description)',
        '```python\nfind("X")\n```',
        '```\nanswer\n```',
        'answer["content"] = description',
        'result = context.search("V406XXA")',
        'context[300]',
        'thought\n```python\nfind("S2699XA")\n```',
    ])
    def test_code_like_replies_detected(self, reply):
        assert looks_like_code_reply(reply)

    @pytest.mark.parametrize("reply", [
        "Passenger of snowmobile injured in nontraffic accident (initial encounter)",
        "Toxic liver disease with acute hepatitis",
        "The code V406XXA denotes: car occupant injured in collision.",
        "Serum neuropathy",
        "I could not find that code in the provided context.",
        "Burns of 90% or more of body surface (with 50-59% third degree burns)",
        "",
    ])
    def test_prose_replies_pass(self, reply):
        assert not looks_like_code_reply(reply)


class TestWrapLeafPrompt:
    def test_appends_instruction(self):
        wrapped = wrap_leaf_prompt("What is the description for code X?")
        assert wrapped.startswith("What is the description for code X?")
        assert LEAF_PROSE_INSTRUCTION in wrapped

    def test_idempotent(self):
        once = wrap_leaf_prompt("q")
        assert wrap_leaf_prompt(once) == once


class TestRepairLeafReply:
    def test_prose_reply_returned_untouched_no_resend(self):
        calls = []
        out = repair_leaf_reply("Serum neuropathy", "q", lambda p: calls.append(p) or "x")
        assert out == "Serum neuropathy"
        assert calls == []

    def test_code_reply_retried_and_repaired(self):
        out = repair_leaf_reply(
            'find("G611")', "q", lambda p: "Serum neuropathy")
        assert out == "Serum neuropathy"

    def test_retry_prompt_contains_correction(self):
        seen = []

        def resend(p):
            seen.append(p)
            return "Serum neuropathy"

        repair_leaf_reply('find("G611")', "original question", resend)
        assert len(seen) == 1
        assert "original question" in seen[0]
        assert "prose" in seen[0].lower()

    def test_still_code_after_retry_returns_first_reply(self):
        out = repair_leaf_reply('find("G611")', "q", lambda p: 'find("G611")')
        assert out == 'find("G611")'


class TestGuardEnabled:
    def test_off_by_default(self, monkeypatch):
        monkeypatch.delenv("PREHEND_LEAF_PROSE_GUARD", raising=False)
        assert not guard_enabled()

    def test_on_when_set(self, monkeypatch):
        monkeypatch.setenv("PREHEND_LEAF_PROSE_GUARD", "1")
        assert guard_enabled()

    def test_explicit_zero_off(self, monkeypatch):
        monkeypatch.setenv("PREHEND_LEAF_PROSE_GUARD", "0")
        assert not guard_enabled()


class TestSendWiring:
    """_send applies wrap + repair only when the guard env var is set."""

    def _repl(self, replies, sent_prompts, monkeypatch):
        from prehend.core.comms_utils import LMResponse
        from prehend.core.types import RLMChatCompletion, UsageSummary
        from prehend.environments import local_repl as lr

        def fake_send(addr, request):
            sent_prompts.append(request.prompt)
            return LMResponse.success_response(
                chat_completion=RLMChatCompletion(
                    root_model="m",
                    prompt=request.prompt,
                    response=replies.pop(0),
                    usage_summary=UsageSummary(model_usage_summaries={}),
                    execution_time=0.0,
                )
            )

        monkeypatch.setattr(lr, "send_lm_request", fake_send)
        return lr.LocalREPL(lm_handler_address=("localhost", 1))

    def test_guard_off_sends_raw_prompt(self, monkeypatch):
        monkeypatch.delenv("PREHEND_LEAF_PROSE_GUARD", raising=False)
        sent = []
        repl = self._repl(['find("X")'], sent, monkeypatch)
        out = repl._send("what is X?")
        assert out == 'find("X")'
        assert sent == ["what is X?"]

    def test_guard_on_wraps_and_repairs(self, monkeypatch):
        monkeypatch.setenv("PREHEND_LEAF_PROSE_GUARD", "1")
        sent = []
        repl = self._repl(['find("X")', "Serum neuropathy"], sent, monkeypatch)
        out = repl._send("what is X?")
        assert out == "Serum neuropathy"
        assert len(sent) == 2
        assert LEAF_PROSE_INSTRUCTION in sent[0]
        assert "prose" in sent[1].lower()

    def test_guard_on_prose_reply_single_send(self, monkeypatch):
        monkeypatch.setenv("PREHEND_LEAF_PROSE_GUARD", "1")
        sent = []
        repl = self._repl(["Serum neuropathy"], sent, monkeypatch)
        assert repl._send("what is X?") == "Serum neuropathy"
        assert len(sent) == 1


class TestSendBatchedWiring:
    def _repl(self, batch_replies, sent_batches, monkeypatch):
        from prehend.core.comms_utils import LMResponse
        from prehend.core.types import RLMChatCompletion, UsageSummary
        from prehend.environments import local_repl as lr

        def fake_batched(addr, prompts, model=None, depth=0, priority=None):
            sent_batches.append(list(prompts))
            replies = batch_replies.pop(0)
            return [
                LMResponse.success_response(
                    chat_completion=RLMChatCompletion(
                        root_model="m", prompt=p, response=r,
                        usage_summary=UsageSummary(model_usage_summaries={}),
                        execution_time=0.0,
                    )
                )
                for p, r in zip(prompts, replies)
            ]

        monkeypatch.setattr(lr, "send_lm_request_batched", fake_batched)
        return lr.LocalREPL(lm_handler_address=("localhost", 1))

    def test_guard_on_batched_wraps_and_repairs_bad_slots(self, monkeypatch):
        monkeypatch.setenv("PREHEND_LEAF_PROSE_GUARD", "1")
        sent = []
        repl = self._repl(
            [['find("A")', "Toxic liver disease"], ["Serum neuropathy"]],
            sent, monkeypatch)
        out = repl._send_batched(["what is A?", "what is B?"])
        assert out == ["Serum neuropathy", "Toxic liver disease"]
        assert len(sent) == 2
        assert all(LEAF_PROSE_INSTRUCTION in p for p in sent[0])
        assert len(sent[1]) == 1 and "prose" in sent[1][0].lower()

    def test_guard_off_batched_untouched(self, monkeypatch):
        monkeypatch.delenv("PREHEND_LEAF_PROSE_GUARD", raising=False)
        sent = []
        repl = self._repl([['find("A")', "prose"]], sent, monkeypatch)
        out = repl._send_batched(["q1", "q2"])
        assert out == ['find("A")', "prose"]
        assert sent == [["q1", "q2"]]
