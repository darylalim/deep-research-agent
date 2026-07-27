"""Unit tests for the pure message-parsing helpers in `cli.py`.

These are the functions most exposed to a silent break when LangChain/LangGraph
change the shape of message content — so they're tested against *real* message
types where the shape is realistic, and against minimal stand-ins only for the
defensive branches that real messages don't normally exercise.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from types import SimpleNamespace

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from deep_research.cli import (
    _SILENT_STOPS,
    _export,
    _short,
    _stop_note,
    _text_of,
    _turn_stop,
    render_thread,
    render_turn,
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
    REPL printed `(the agent said nothing)`, which is true and useless — it reads as a bug
    in the CLI rather than as something the API reported, and gives the user nothing to
    act on.
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
        # WHY `StopNote` carries two fields. The remedy used to be a literal appended by
        # `main` and by `streamlit_app`, and it said "rephrasing or narrowing it usually
        # helps" — advice that is actively wrong for an overrun, where the question was
        # fine and the thread is what grew. Break either remedy and this goes red.
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

    `cli.py` tells the reader to re-read `anthropic/types/` on an SDK bump. That is
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


class TestExport:
    def _agent(self, messages: Sequence[object]) -> SimpleNamespace:
        """An agent stub exposing ONLY `get_state`.

        Deliberately nothing else. `/export` must never reach for `invoke`/`stream` or
        the agent's own `write_file` tool — that route cannot work without a model turn
        (the HITL middleware interrupts on the *model's* tool calls), so it would cost an
        Opus call and ask the human to approve the thing the human just typed. Any such
        reach raises AttributeError here rather than passing quietly.
        """
        return SimpleNamespace(
            get_state=lambda config: SimpleNamespace(values={"messages": messages})
        )

    def test_it_writes_the_whole_thread_as_markdown(
        self, tmp_path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        target = tmp_path / "out.md"
        _export(self._agent(THREAD), {}, "main", str(target))

        written = target.read_text(encoding="utf-8")
        assert "# Deep research — thread `main`" in written
        assert "## you\n\nwhat does Opus 4.8 cost?" in written
        assert "$15/Mtok" in written and "$1/Mtok" in written
        assert str(target.resolve()) in capsys.readouterr().out

    def test_it_refuses_to_write_an_empty_file(
        self, tmp_path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        target = tmp_path / "empty.md"
        _export(self._agent([]), {}, "main", str(target))

        assert not target.exists()
        assert "nothing to export" in capsys.readouterr().out

    def test_an_unwritable_path_reports_instead_of_killing_the_session(
        self, tmp_path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        # main()'s broad `except` is at TURN scope, not command scope — an OSError here
        # would escape it and end the REPL. Caught in `_export` itself.
        _export(self._agent(THREAD), {}, "main", str(tmp_path / "nope" / "out.md"))
        assert "export failed" in capsys.readouterr().out

    def test_a_home_relative_path_is_expanded(
        self, tmp_path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # `/export ~/report.md` is the obvious thing to type, and no shell expanded it —
        # the path arrives as the literal string `~/report.md`. Without expanduser it dies
        # with a bare ENOENT (or, if a stray `~` directory exists in cwd, silently writes
        # into `./~/report.md`), losing a report that cost minutes and dozens of searches.
        monkeypatch.setenv("HOME", str(tmp_path))
        _export(self._agent(THREAD), {}, "main", "~/report.md")

        assert (tmp_path / "report.md").read_text(encoding="utf-8").count(
            "$15/Mtok"
        ) == 1

    def test_an_unfinished_turn_still_exports_the_prose_it_has(self, tmp_path) -> None:
        # A thread whose last turn was abandoned at an approval prompt: the report is in
        # the checkpoint even though the turn never completed. Export it.
        target = tmp_path / "out.md"
        _export(self._agent(THREAD[:2]), {}, "main", str(target))
        assert "$15/Mtok" in target.read_text(encoding="utf-8")
