"""Unit tests for the pure message-parsing helpers in `cli.py`.

These are the functions most exposed to a silent break when LangChain/LangGraph
change the shape of message content — so they're tested against *real* message
types where the shape is realistic, and against minimal stand-ins only for the
defensive branches that real messages don't normally exercise.
"""

from __future__ import annotations

from types import SimpleNamespace

from langchain_core.messages import AIMessage, HumanMessage

from deep_research.cli import _last_ai_text, _short, _text_of


class TestTextOf:
    def test_plain_string_content(self) -> None:
        assert _text_of(AIMessage(content="hello")) == "hello"

    def test_list_of_text_blocks_is_concatenated(self) -> None:
        msg = AIMessage(
            content=[{"type": "text", "text": "a"}, {"type": "text", "text": "b"}]
        )
        assert _text_of(msg) == "ab"

    def test_non_text_blocks_are_ignored(self) -> None:
        msg = AIMessage(
            content=[{"type": "tool_use", "id": "1"}, {"type": "text", "text": "keep"}]
        )
        assert _text_of(msg) == "keep"

    def test_bare_string_blocks_in_list(self) -> None:
        # Defensive branch: raw strings inside the content list are kept.
        assert _text_of(SimpleNamespace(content=["x", "y"])) == "xy"

    def test_raw_string_without_content_attr_passes_through(self) -> None:
        assert _text_of("just a string") == "just a string"


class TestLastAiText:
    def test_returns_text_of_final_ai_message(self) -> None:
        messages = [
            HumanMessage(content="q"),
            AIMessage(content="first"),
            AIMessage(content="last"),
        ]
        assert _last_ai_text({"messages": messages}) == "last"

    def test_skips_trailing_non_ai_messages(self) -> None:
        messages = [AIMessage(content="answer"), HumanMessage(content="follow up")]
        assert _last_ai_text({"messages": messages}) == "answer"

    def test_strips_surrounding_whitespace(self) -> None:
        # The .strip() is the function's real value-add: model output routinely
        # carries leading/trailing newlines that must not reach the printed line.
        messages = [AIMessage(content="  answer\n")]
        assert _last_ai_text({"messages": messages}) == "answer"

    def test_falls_back_to_last_message_when_no_ai(self) -> None:
        assert (
            _last_ai_text({"messages": [HumanMessage(content="only human")]})
            == "only human"
        )

    def test_empty_message_list_returns_empty_string(self) -> None:
        assert _last_ai_text({"messages": []}) == ""

    def test_missing_messages_key_returns_empty_string(self) -> None:
        assert _last_ai_text({}) == ""


class TestShort:
    def test_under_limit_returns_compact_json(self) -> None:
        assert _short({"a": 1}) == '{"a": 1}'

    def test_over_limit_truncates_with_ellipsis(self) -> None:
        rendered = _short({"k": "x" * 500}, limit=20)
        assert rendered.endswith(" …")
        assert len(rendered) == 20 + len(" …")

    def test_non_json_serializable_falls_back_to_str(self) -> None:
        sentinel = object()
        assert _short(sentinel) == str(sentinel)
