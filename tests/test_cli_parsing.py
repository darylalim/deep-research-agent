"""Unit tests for the pure message-parsing helpers in `cli.py`.

These are the functions most exposed to a silent break when LangChain/LangGraph
change the shape of message content — so they're tested against *real* message
types where the shape is realistic, and against minimal stand-ins only for the
defensive branches that real messages don't normally exercise.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from deep_research.cli import _short, _text_of, render_turn


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


class TestRenderTurn:
    def test_shows_the_report_and_not_just_the_sign_off(self) -> None:
        """The regression this function exists for.

        The agent writes its cited report in the same message that proposes the
        (gated) `write_file`, then signs off after the tool returns. Rendering only
        `messages[-1]` handed the user the sign-off alone — measured on a real run:
        33 source URLs in the turn, zero in the last message, and a closing line
        referring to a "summary above" that was never printed.
        """
        messages = [
            HumanMessage(content="compare X and Y"),
            AIMessage(
                content="X is 1,000/mo (https://x.example). Y is 5,000/mo (https://y.example)."
            ),
            ToolMessage(content="ok", tool_call_id="1", name="write_file"),
            AIMessage(content="Findings saved. Summary above covers the comparison."),
        ]
        rendered = render_turn({"messages": messages})
        assert "https://x.example" in rendered  # the sources reach the user…
        assert "Findings saved." in rendered  # …and so does the sign-off
        assert rendered.index("https://x.example") < rendered.index("Findings saved.")

    def test_renders_only_the_current_turn(self) -> None:
        """A thread accumulates messages; reprinting the whole history every turn
        would be worse than the bug being fixed."""
        messages = [
            HumanMessage(content="first question"),
            AIMessage(content="old answer"),
            HumanMessage(content="second question"),
            AIMessage(content="new answer"),
        ]
        assert render_turn({"messages": messages}) == "new answer"

    def test_strips_surrounding_whitespace(self) -> None:
        # Model output routinely carries leading/trailing newlines that must not
        # reach the printed line.
        assert render_turn({"messages": [AIMessage(content="  answer\n")]}) == "answer"

    def test_a_turn_with_no_assistant_prose_renders_nothing(self) -> None:
        # This used to fall back to `messages[-1]` and return "only human" — the user's
        # own question, echoed back under an `agent >` header. Harmless while
        # `render_turn` was only called on completed turns; not harmless now that
        # `_print_unfinished_turn` calls it on turns abandoned at an approval prompt,
        # where a bare human message is exactly what the checkpoint holds. It would
        # also have handed `evals/harness.py` the question itself as the agent's
        # `response`, for the judges to grade as an answer.
        assert render_turn({"messages": [HumanMessage(content="only human")]}) == ""

    def test_a_raw_tool_payload_is_never_shown_as_the_agents_words(self) -> None:
        # The other half of removing the fallback, and the more dangerous half. Ctrl-C
        # during the multi-minute search phase leaves a `tavily_search` ToolMessage as
        # the last thing in the checkpoint — several KB of serialized result dicts. The
        # old `messages[-1]` fallback would print that verbatim under an `agent >`
        # header, and would hand it to the eval judges as the agent's `response`.
        #
        # An earlier version of this very test pinned the opposite behavior using an
        # 11-character tool output, which made the dump look perfectly benign.
        payload = json.dumps(
            {
                "query": "opus pricing",
                "results": [{"url": "https://x.test", "content": "…"}],
            }
        )
        messages = [
            HumanMessage(content="q"),
            ToolMessage(content=payload, tool_call_id="1", name="tavily_search"),
        ]
        assert render_turn({"messages": messages}) == ""

    def test_empty_message_list_returns_empty_string(self) -> None:
        assert render_turn({"messages": []}) == ""

    def test_missing_messages_key_returns_empty_string(self) -> None:
        assert render_turn({}) == ""


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
