"""Unit tests for the pure message-parsing helpers in `turns.py`.

These are the functions most exposed to a silent break when LangChain/LangGraph
change the shape of message content — so they're tested against *real* message
types where the shape is realistic, and against minimal stand-ins only for the
defensive branches that real messages don't normally exercise.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from deep_research.turns import (
    _SILENT_STOPS,
    _short,
    _stop_note,
    _text_of,
    _turn_stop,
    export_markdown,
    render_thread,
    render_turn,
    thread_sections,
)


def _stopped(reason: str, **metadata: object) -> AIMessage:
    """An assistant message shaped like a real silent stop.

    Empty content and a `stop_reason` in `response_metadata` — which is exactly what
    arrives: the API returns HTTP 200 with no content blocks, so nothing raises and
    nothing downstream distinguishes it from a turn the model simply had nothing to add
    to.
    """
    return AIMessage(content="", response_metadata={"stop_reason": reason, **metadata})


def _refusal(**metadata: object) -> AIMessage:
    return _stopped("refusal", **metadata)


def _overrun(**metadata: object) -> AIMessage:
    return _stopped("model_context_window_exceeded", **metadata)


class TestStopNote:
    """A silent stop must be reported as what it was, not as silence.

    Both `refusal` and `model_context_window_exceeded` are a 200 with empty content: they
    cost tokens, raise nothing, and reach `render_turn` as a message with no prose. The
    app used to say `(the agent said nothing)`, which is true and useless — it reads as a
    bug in the app rather than as something the API reported, and gives the user nothing
    to act on.
    """

    def test_a_refusal_is_named_as_one(self) -> None:
        note = _stop_note(_refusal())
        assert note and note.reason == "the model declined this request"

    def test_a_context_window_overrun_is_named_and_is_not_called_a_refusal(
        self,
    ) -> None:
        # The regression this whole class grew for. `model_context_window_exceeded`
        # arrived in anthropic 0.120 and `langchain_anthropic` has no reference to it, so
        # it lands in `response_metadata` raw; while the check here was `== "refusal"` it
        # fell through to `(the agent said nothing)`.
        note = _stop_note(_overrun())
        assert note and note.reason == "this turn exceeded the model's context window"
        assert "declined" not in note.reason

    def test_the_remedy_is_per_stop_reason_and_not_hardcoded_at_the_print_site(
        self,
    ) -> None:
        # WHY `StopNote` carries two fields. The remedy used to be a literal appended at
        # the print site, and it said "rephrasing or narrowing it usually helps" — advice
        # that is actively wrong for an overrun, where the question was fine and the
        # thread is what grew. Break either remedy and this goes red.
        refusal, overrun = _stop_note(_refusal()), _stop_note(_overrun())
        assert refusal and overrun
        assert "rephras" in refusal.remedy
        assert "rephras" not in overrun.remedy
        assert "thread" in overrun.remedy

    def test_an_ordinary_message_is_not_a_stop(self) -> None:
        assert _stop_note(AIMessage("here is the answer")) is None

    def test_a_truncated_turn_is_not_a_stop(self) -> None:
        # `max_tokens` is the other silent stop_reason in this project's history, and it
        # means something completely different — the answer exists and got cut off.
        truncated = AIMessage(
            content="half an ans", response_metadata={"stop_reason": "max_tokens"}
        )
        assert _stop_note(truncated) is None

    def test_a_message_with_no_metadata_is_handled(self) -> None:
        # ToolMessages, HumanMessages, and anything a fake hands us: no crash, no note.
        assert _stop_note(SimpleNamespace(content="x")) is None

    def test_an_unhashable_stop_reason_does_not_crash_the_lookup(self) -> None:
        # `_SILENT_STOPS` is a dict now, and `response_metadata` is a dict this code did
        # not build. A list there would raise `TypeError: unhashable` from `.get` and take
        # down the turn on the path whose entire job is to explain a turn that went wrong.
        assert (
            _stop_note(SimpleNamespace(response_metadata={"stop_reason": []})) is None
        )

    def test_no_category_is_invented_when_langchain_does_not_supply_one(self) -> None:
        # THE POINT OF THE DEFENSIVE BRANCH. Anthropic reports the refusal category in
        # `stop_details`, but `langchain_anthropic` never copies it into
        # `response_metadata` — grep `chat_models.py`: `stop_reason` appears three times,
        # `stop_details` not once. So today the note MUST be category-free. Printing a
        # guessed category would be inventing evidence about why the model stopped.
        note = _stop_note(_refusal())
        assert note and "(" not in note.reason

    def test_a_category_is_used_if_langchain_ever_starts_passing_it_through(
        self,
    ) -> None:
        # The fixture is the SDK's REAL shape, read off
        # `anthropic/types/refusal_stop_details.py` rather than guessed:
        # `RefusalStopDetails` is `{type, category, explanation}`, and the policy name
        # lives under `category`. This test first shipped asserting `{"refusal": "cyber"}`
        # — a key the API never sends — which made it unfalsifiable: the branch it guards
        # is dead today (langchain drops `stop_details`), so a wrong key looks identical
        # to a right one until the day the field arrives and the category is silently
        # dropped. A guard that cannot fail is not a guard.
        note = _stop_note(
            _refusal(stop_details={"type": "refusal", "category": "cyber"})
        )
        assert note and note.reason == "the model declined this request (cyber)"

    def test_a_model_shaped_stop_details_works_too_not_only_a_dict(self) -> None:
        # The streaming path is the one this app always takes (`streaming=True`), and it
        # is the path that puts pydantic MODELS in `response_metadata`: langchain's
        # `message_delta` handler assembles the dict field by field off the raw event and
        # calls `.model_dump()` explicitly for `container` and `context_management`. Add
        # `stop_details` there without one and it arrives as a `RefusalStopDetails`. A
        # dict-only read would drop the category silently — the same invisible loss as
        # the wrong key, one layer along, and equally undetectable while the branch is
        # dead. `SimpleNamespace` stands in for the model: attribute access is the shape
        # under test, not pydantic itself.
        note = _stop_note(
            _refusal(stop_details=SimpleNamespace(type="refusal", category="bio"))
        )
        assert note and note.reason == "the model declined this request (bio)"

    def test_the_unstable_explanation_field_is_not_shown_to_the_user(self) -> None:
        # `RefusalStopDetails.explanation` is documented by the SDK as "not guaranteed to
        # be stable" — free text that can change under us, and it must not become the
        # reason a user is told their question was refused. The category is shown instead
        # because it is a Literal the SDK versions, NOT because that Literal is fixed:
        # `general_harms` was added in 0.120. `TestStopReasonsAreAccountedFor` is what
        # tracks the drift; `_stop_note` reads the value generically so a member added
        # tomorrow prints rather than vanishing.
        note = _stop_note(
            _refusal(
                stop_details={
                    "type": "refusal",
                    "category": "cyber",
                    "explanation": "some prose that may change without notice",
                }
            )
        )
        assert note and note.reason == "the model declined this request (cyber)"


class TestStopReasonsAreAccountedFor:
    """Pin `_SILENT_STOPS` and the `category` key against the installed SDK.

    `turns.py` tells the reader to re-read `anthropic/types/` on an SDK bump. That is
    exactly the check this repo's own rule says cannot be trusted to prose — and it has
    already failed once: the anthropic 0.116 -> 0.120 upgrade was reviewed by hand,
    `general_harms` was spotted in `refusal_stop_details.py`, and
    `model_context_window_exceeded` was missed one file over in `stop_reason.py` — the
    change with behaviour attached. These two tests are what would have caught it, on
    `uv sync`, without anyone remembering to look.

    Importing `anthropic` directly is deliberate: it is the package that DEFINES the
    contract, `langchain-anthropic` hard-depends on it, and reading the enum from
    anywhere else would just be another copy to keep true.
    """

    # Stop reasons that need no note: the turn produced prose (or, for `max_tokens`, a
    # truncated answer that is still shown). Listed rather than defaulted so that a NEW
    # member is never silently assumed benign — classifying it is the point.
    NOT_SILENT = frozenset(
        {"end_turn", "max_tokens", "stop_sequence", "tool_use", "pause_turn"}
    )

    def test_every_sdk_stop_reason_is_classified(self) -> None:
        from typing import get_args

        from anthropic.types import StopReason

        assert set(get_args(StopReason)) == self.NOT_SILENT | set(_SILENT_STOPS), (
            "the SDK's StopReason changed. A new member reaches "
            "`response_metadata['stop_reason']` raw — langchain forwards it without "
            "knowing it — so decide which side it belongs on: add it to `_SILENT_STOPS` "
            "with a reason and a remedy if it can end a turn with no prose, or to "
            "NOT_SILENT if the turn still carries an answer."
        )

    def test_the_refusal_category_key_still_exists(self) -> None:
        # `_stop_note`'s dead defensive branch reads `stop_details["category"]`. The KEY
        # is the half that has to stay put; the member list deliberately is not pinned,
        # because it grows and the code reads the value generically. A rename here is
        # invisible at runtime precisely because the branch is dead.
        from anthropic.types.refusal_stop_details import RefusalStopDetails

        assert "category" in RefusalStopDetails.model_fields


class TestTurnStop:
    def test_it_finds_a_refusal_in_this_turn(self) -> None:
        state = {"messages": [HumanMessage("q"), _refusal()]}
        note = _turn_stop(state)
        assert note and note.reason == "the model declined this request"

    def test_it_finds_a_context_window_overrun_in_this_turn(self) -> None:
        state = {"messages": [HumanMessage("q"), _overrun()]}
        note = _turn_stop(state)
        assert note and note.reason == "this turn exceeded the model's context window"

    def test_a_previous_turns_stop_is_not_reported_again(self) -> None:
        # Scoped with the SAME slice `render_turn` uses (`_this_turn`), deliberately: the
        # note and the answer print side by side, so a note scoped to a wider span would
        # caption this turn's answer with last turn's refusal.
        state = {
            "messages": [
                HumanMessage("something disallowed"),
                _refusal(),
                HumanMessage("something ordinary"),
                AIMessage("here is the answer"),
            ]
        }
        assert _turn_stop(state) is None
        assert render_turn(state) == "here is the answer"

    def test_a_stopped_branch_alongside_an_answer_is_still_reported(self) -> None:
        # A turn can lose one branch and answer anyway; the note is what explains why
        # the answer is thinner than the question.
        state = {"messages": [HumanMessage("q"), _refusal(), AIMessage("partial")]}
        note = _turn_stop(state)
        assert note and note.reason == "the model declined this request"
        assert render_turn(state) == "partial"

    def test_an_ordinary_turn_reports_nothing(self) -> None:
        assert _turn_stop({"messages": [HumanMessage("q"), AIMessage("a")]}) is None

    def test_missing_messages_key_is_handled(self) -> None:
        assert _turn_stop({}) is None


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
        # reach the rendered answer.
        assert render_turn({"messages": [AIMessage(content="  answer\n")]}) == "answer"

    def test_a_turn_with_no_assistant_prose_renders_nothing(self) -> None:
        # This used to fall back to `messages[-1]` and return "only human" — the user's
        # own question, echoed back as the agent's answer. Harmless while `render_turn`
        # was only called on completed turns; not harmless on a turn abandoned at an
        # approval, where a bare human message is exactly what the checkpoint holds. It
        # would also hand `evals/harness.py` the question itself as the agent's
        # `response`, for the judges to grade as an answer.
        assert render_turn({"messages": [HumanMessage(content="only human")]}) == ""

    def test_a_raw_tool_payload_is_never_shown_as_the_agents_words(self) -> None:
        # The other half of removing the fallback, and the more dangerous half. A turn
        # that dies during the multi-minute search phase leaves a `tavily_search`
        # ToolMessage as the last thing in the checkpoint — several KB of serialized
        # result dicts. The old `messages[-1]` fallback would show that verbatim as the
        # agent's answer, and would hand it to the eval judges as its `response`.
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


# A two-turn thread, shaped the way a real one is: the cited report lives in the SAME
# assistant message that proposes the write_file, and the agent then signs off after the
# tool returns. Getting this wrong is what cost 33 source URLs once already.
THREAD = [
    HumanMessage(content="what does Opus 4.8 cost?"),
    AIMessage(
        content="Opus 4.8 is $15/Mtok in. ([docs](https://docs.anthropic.com/pricing))",
        id="ai-1",
        tool_calls=[
            {
                "name": "write_file",
                "args": {"file_path": "/memories/pricing.md", "content": "x"},
                "id": "c1",
            }
        ],
    ),
    ToolMessage(content="written", tool_call_id="c1", name="write_file"),
    AIMessage(content="Saved to memory.", id="ai-2"),
    HumanMessage(content="and Haiku 4.5?"),
    AIMessage(
        content="Haiku 4.5 is $1/Mtok in. ([docs](https://docs.anthropic.com/pricing))",
        id="ai-3",
    ),
]


class TestRenderThread:
    def test_every_turn_keeps_its_question_and_its_cited_answer(self) -> None:
        rendered = render_thread({"messages": THREAD})

        # Both questions…
        assert "what does Opus 4.8 cost?" in rendered
        assert "and Haiku 4.5?" in rendered
        # …and both cited reports, including the one that shares a message with the
        # write_file tool call. A `render_turn`-style "last message only" export would
        # keep the sign-off and drop every source URL — the exact regression this repo
        # has already paid for once.
        assert "$15/Mtok" in rendered
        assert "$1/Mtok" in rendered
        assert rendered.count("https://docs.anthropic.com/pricing") == 2
        # In order.
        assert rendered.index("$15/Mtok") < rendered.index("$1/Mtok")

    def test_no_tool_payload_leaks_into_the_export(self) -> None:
        messages = [
            HumanMessage(content="q"),
            ToolMessage(
                content=json.dumps({"results": [{"url": "https://leak.test"}]}),
                tool_call_id="s1",
                name="tavily_search",
            ),
            AIMessage(content="the answer", id="ai-1"),
        ]
        rendered = render_thread({"messages": messages})
        assert "leak.test" not in rendered
        assert "the answer" in rendered

    def test_an_empty_thread_renders_nothing(self) -> None:
        assert render_thread({"messages": []}) == ""
        assert render_thread({}) == ""


class TestExportMarkdown:
    """The document behind the page's "Export transcript" button."""

    def test_it_is_the_whole_thread_under_a_header(self) -> None:
        document = export_markdown({"messages": THREAD}, "main", "20260101T000000Z")

        assert document.startswith("# Deep research — thread `main`")
        assert "Exported 20260101T000000Z" in document
        assert "## you\n\nwhat does Opus 4.8 cost?" in document
        assert "$15/Mtok" in document and "$1/Mtok" in document

    def test_an_empty_thread_exports_nothing(self) -> None:
        # `""` rather than a header over nothing — the page disables the button on it,
        # so a reader is never handed an empty file.
        assert export_markdown({"messages": []}, "main", "stamp") == ""

    def test_an_unfinished_turn_still_exports_the_prose_it_has(self) -> None:
        # A thread whose last turn was abandoned at an approval: the report is in the
        # checkpoint even though the turn never completed. Export it.
        assert "$15/Mtok" in export_markdown({"messages": THREAD[:2]}, "main", "stamp")


def test_render_turn_is_the_answer_the_page_draws() -> None:
    """The evals grade `render_turn`; the page draws `thread_sections`. Same bytes.

    `evals/harness.py` builds its graded `response` with `render_turn`, and the citation
    metrics are only honest if that is what the user was shown. The page draws the last
    `ai` section of `thread_sections` instead — so pin that the two agree on a finished
    turn, including the report-then-sign-off pair that is two messages saying one thing.
    Diverge them and the evals start grading text nobody saw.
    """
    state = {"messages": THREAD[:4]}  # turn one: report + write_file + sign-off
    kind, drawn = thread_sections(state)[-1]

    assert kind == "ai"
    assert render_turn(state) == drawn
    assert (
        render_turn({"messages": THREAD})
        == thread_sections({"messages": THREAD})[-1][1]
    )
