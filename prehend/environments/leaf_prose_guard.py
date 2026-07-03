"""Guard llm_query LEAF completions against code-shaped replies.

Preference-tuned checkpoints (rlm-trainer v0.5 KTO) internalize "REPL code =
grounded, prose = confabulation" so strongly that a bare leaf completion -
which ships as a single user message with NO system prompt (clients/openai.py
completion()) - answers `find("V406XXA")` instead of the requested description.
The orchestrator trusts the leaf reply and ships it as the final answer.

The guard has two layers, both applied only in the leaf send path
(local_repl._send / _send_batched) and only when PREHEND_LEAF_PROSE_GUARD=1:

1. wrap_leaf_prompt: append a prose-only instruction to the outgoing prompt.
2. repair_leaf_reply: if the reply still looks like code, resend ONCE with an
   explicit correction; keep the retry only if it comes back prose.

Opt-in by env var because teacher/trajectory-generation runs must see the
unmodified prompt (instruction text would leak into training data).
"""
import os
import re

LEAF_PROSE_INSTRUCTION = (
    "IMPORTANT: Reply with the answer itself in plain prose. "
    "Do NOT reply with code, function calls, or tool syntax."
)

_RETRY_PREFIX = (
    "Your previous reply was code, not an answer. Answer the question below "
    "in plain prose only - no code, no function calls, no tool syntax.\n\n"
)

_FENCE = re.compile(r"^```")
# a bare call or subscript as the whole (first) line - identifier chain hard
# against the paren/bracket, no spaces, so prose that merely ENDS in a
# parenthetical ("... accident (initial encounter)") never matches:
#   find("V406XXA")   llm_query("...", context=context)   context[300]
_CALL_LINE = re.compile(r"^[\w\.]+(\(.*\)|\[[^\]]*\])\s*$")
# an assignment whose target is an identifier/subscript chain:
#   answer["content"] = description   result = context.search("X")
_ASSIGNMENT = re.compile(r"^[\w\.]+(\[[^\]]*\])?\s*=[^=]\s*\S")


def guard_enabled() -> bool:
    return os.environ.get("PREHEND_LEAF_PROSE_GUARD", "") not in ("", "0")


def looks_like_code_reply(reply: str) -> bool:
    """True when a leaf reply is code-shaped rather than a prose answer."""
    text = (reply or "").strip()
    if not text:
        return False
    # a fence on ANY line catches "thought\n```python\nfind(...)\n```" replies
    if any(_FENCE.match(ln.strip()) for ln in text.splitlines()):
        return True
    first = text.splitlines()[0].strip()
    return bool(_CALL_LINE.match(first)) or bool(_ASSIGNMENT.match(first))


def wrap_leaf_prompt(prompt: str) -> str:
    if LEAF_PROSE_INSTRUCTION in prompt:
        return prompt
    return f"{prompt}\n\n{LEAF_PROSE_INSTRUCTION}"


def repair_leaf_reply(reply: str, prompt: str, resend) -> str:
    """Retry a code-shaped reply once with an explicit correction.

    resend: callable(prompt) -> str, a plain leaf send. The retry reply is
    kept only if it is prose; otherwise the original reply is returned so the
    caller sees the model's true behavior rather than two stacked failures.
    """
    if not looks_like_code_reply(reply):
        return reply
    retry = resend(f"{_RETRY_PREFIX}{prompt}")
    if retry and not looks_like_code_reply(retry):
        return retry
    return reply
