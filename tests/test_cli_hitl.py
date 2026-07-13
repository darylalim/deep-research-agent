"""Unit tests for the human-in-the-loop decision logic in `cli.py`.

`_collect_decisions` is the load-bearing contract the README calls out: it maps
each interrupt's `action_requests` to exactly one decision, in order. It and
`_prompt_decision` both read from `input()`, which is scripted here via a fed
iterator so no real terminal interaction happens.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from types import SimpleNamespace
from typing import Any

import pytest
from langchain.agents.middleware.human_in_the_loop import HumanInTheLoopMiddleware
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.types import Command, Interrupt

from deep_research import cli as cli_module
from deep_research.cli import (
    DEFAULT_ALLOWED_DECISIONS,
    PREVIEW_LINES,
    _collect_decisions,
    _prompt_decision,
    _render_action,
    main,
)

REQUEST = {"name": "write_file", "args": {"file_path": "/memories/x.md"}}


def test_default_decisions_match_the_middleware_expansion_of_true() -> None:
    # Every value in GATED_TOOLS is a bare `True`, which the middleware expands
    # into a concrete decision set. `DEFAULT_ALLOWED_DECISIONS` is the CLI's
    # fallback for a request that arrives with no ReviewConfig, so it has to be
    # the *same* set — if a langchain upgrade adds a fifth decision type, this
    # goes red instead of the CLI silently never offering it.
    middleware = HumanInTheLoopMiddleware(interrupt_on={"write_file": True})
    expanded = middleware.interrupt_on["write_file"]["allowed_decisions"]
    assert set(expanded) == set(DEFAULT_ALLOWED_DECISIONS)


def _feed(monkeypatch: pytest.MonkeyPatch, *responses: str) -> list[str]:
    """Make `input()` return `responses` in order (StopIteration if over-consumed).

    Returns the list the prompts are recorded into, so a test can assert on what
    the user was actually *offered*. That matters: the decision menu is passed to
    `input(prompt)` rather than printed, so it never reaches stdout once `input`
    is patched — asserting against `capsys` here would silently pass no matter
    what the menu said.
    """
    scripted: Iterator[str] = iter(responses)
    prompts: list[str] = []

    def fake_input(prompt: str = "") -> str:
        prompts.append(prompt)
        return next(scripted)

    monkeypatch.setattr("builtins.input", fake_input)
    return prompts


class TestPromptDecision:
    def test_empty_input_approves_by_default(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _feed(monkeypatch, "")
        assert _prompt_decision(REQUEST) == {"type": "approve"}

    def test_explicit_approve(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _feed(monkeypatch, "a")
        assert _prompt_decision(REQUEST) == {"type": "approve"}

    def test_reject_with_reason_carries_message(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _feed(monkeypatch, "r", "too risky")
        assert _prompt_decision(REQUEST) == {"type": "reject", "message": "too risky"}

    def test_reject_without_reason_omits_message(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _feed(monkeypatch, "r", "")
        assert _prompt_decision(REQUEST) == {"type": "reject"}

    def test_edit_with_valid_json_returns_edited_action(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _feed(monkeypatch, "e", '{"file_path": "/memories/y.md"}')
        assert _prompt_decision(REQUEST) == {
            "type": "edit",
            "edited_action": {
                "name": "write_file",
                "args": {"file_path": "/memories/y.md"},
            },
        }

    def test_edit_with_blank_args_approves_as_is(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _feed(monkeypatch, "e", "")
        assert _prompt_decision(REQUEST) == {"type": "approve"}

    def test_invalid_edit_json_never_approves_the_original_args(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # THE fail-open, and the reason `write_file` is gated at all. This used to
        # print "! not valid JSON — approving with the original args" and return
        # `{"type": "approve"}` — so a reviewer who chose `edit` *because the write
        # looked wrong*, then mistyped the JSON, silently approved the very write
        # they were trying to narrow, into durable `/memories/` that git cannot
        # restore. A typo is not consent.
        #
        # Feeding a reject *after* the bad JSON is what makes this bite: under the
        # old behavior `_prompt_decision` returned at the typo and never consumed
        # the "r", so the result was `approve` — a visibly different value, not a
        # StopIteration. Revert the four lines in cli.py and watch it go red.
        _feed(monkeypatch, "e", "{not json", "r", "")
        assert _prompt_decision(REQUEST) == {"type": "reject"}

    def test_invalid_edit_json_reprompts_and_the_retry_still_edits(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Re-prompting must land back at the menu with the edit path still open —
        # a fumbled edit costs a retry, not the turn.
        _feed(monkeypatch, "e", "{not json", "e", '{"file_path": "/memories/y.md"}')
        assert _prompt_decision(REQUEST) == {
            "type": "edit",
            "edited_action": {
                "name": "write_file",
                "args": {"file_path": "/memories/y.md"},
            },
        }

    def test_unrecognized_choice_reprompts(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # First response is invalid → loop → second response approves.
        _feed(monkeypatch, "z", "a")
        assert _prompt_decision(REQUEST) == {"type": "approve"}

    @pytest.mark.parametrize(
        ("word", "rest", "expected"),
        [
            ("approve", (), {"type": "approve"}),
            ("reject", ("",), {"type": "reject"}),
            (
                "respond",
                ("use the cache",),
                {"type": "respond", "message": "use the cache"},
            ),
        ],
    )
    def test_the_full_word_selects_the_decision(
        self,
        monkeypatch: pytest.MonkeyPatch,
        word: str,
        rest: tuple[str, ...],
        expected: dict[str, str],
    ) -> None:
        # The menu shows single letters, but typing the whole word works too.
        _feed(monkeypatch, word, *rest)
        assert _prompt_decision(REQUEST) == expected

    def test_valid_json_that_is_not_an_object_is_refused(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # `"/memories/y.md"` is perfectly valid JSON — a user correcting a path
        # would plausibly type exactly that. But `args` must be an *object*: the
        # middleware does not validate, so a bare string would be copied straight
        # into a ToolCall and only fail later, at tool execution.
        _feed(
            monkeypatch,
            "e",
            '"/memories/y.md"',  # valid JSON, wrong shape → refuse, re-prompt
            "e",
            '{"file_path": "/memories/y.md"}',
        )
        assert _prompt_decision(REQUEST) == {
            "type": "edit",
            "edited_action": {
                "name": "write_file",
                "args": {"file_path": "/memories/y.md"},
            },
        }


class TestRenderAction:
    """The reviewer has to be able to *read* what they are approving.

    `write_file` interrupts for exactly one reason: a human should see the content
    before it lands in durable, gitignored `/memories/` that git cannot restore. If
    what they are shown is an unreadable escaped-newline dict, the rational move is
    to mash Enter — and the gate becomes theater.
    """

    def _boilerplate(self, request: Mapping[str, Any]) -> str:
        """The description langchain's middleware generates by default.

        Reproduced exactly (`human_in_the_loop.py`: `f"{prefix}\\n\\nTool: {name}\\n
        Args: {args}"`) so that the stripping in `_reviewer_note` is tested against
        the real shape, not a guess at it.
        """
        return (
            f"Tool execution requires approval\n\n"
            f"Tool: {request['name']}\nArgs: {request['args']}"
        )

    def test_the_file_body_is_readable_and_not_shown_twice(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        args = {
            "file_path": "/memories/pricing.md",
            "content": "# Pricing\n\n- Opus: $15/Mtok ([src](https://x.test))\n- Haiku: $1",
        }
        request = {"name": "write_file", "args": args}
        _render_action({**request, "description": self._boilerplate(request)})
        out = capsys.readouterr().out

        # Real newlines: each markdown line stands on its own.
        assert "- Opus: $15/Mtok ([src](https://x.test))" in out.splitlines()[-2]
        # …and NOT the escaped form the dict repr produces.
        assert "\\n" not in out
        # The path appears once. It used to appear three times — once in the
        # middleware's dict-repr description, once in the `_short()` JSON clip of
        # the same dict, and nowhere legibly.
        assert out.count("/memories/pricing.md") == 1
        # The boilerplate description is stripped entirely: it duplicates the header
        # we already print and the args we now render properly.
        assert "Tool execution requires approval" not in out
        assert "Args: {" not in out

    def test_a_long_body_is_elided_with_an_honest_count(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        # Truncation is fine; *silent* truncation is not — the reviewer must know
        # there is more they have not seen.
        body = "\n".join(f"line {i}" for i in range(PREVIEW_LINES + 25))
        _render_action(
            {
                "name": "write_file",
                "args": {"file_path": "/memories/x.md", "content": body},
            }
        )
        out = capsys.readouterr().out

        assert f"({PREVIEW_LINES + 25} lines)" in out
        assert f"line {PREVIEW_LINES - 1}" in out  # last shown
        assert f"line {PREVIEW_LINES}" not in out  # first elided
        assert "… 25 more lines" in out

    def test_a_shell_command_gets_the_same_treatment_as_a_file_body(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        # The renderer is generic on purpose: `execute`'s `command`, and any future
        # gated tool's long string arg, must not need a per-tool branch that someone
        # forgets to add when they extend `GATED_TOOLS`.
        _render_action(
            {"name": "execute", "args": {"command": "set -e\nrm -rf ./build\nmake all"}}
        )
        out = capsys.readouterr().out
        assert "│ rm -rf ./build" in out
        assert "\\n" not in out

    def test_a_human_written_description_survives(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        # A tool gated with an `InterruptOnConfig` may carry a real description
        # (a string, or one a callable built). Stripping the boilerplate must not
        # throw that away — same principle as the menu: honor what you are handed.
        _render_action(
            {
                "name": "write_file",
                "args": {"file_path": "/memories/x.md"},
                "description": "This overwrites a note from a previous session.",
            }
        )
        assert (
            "This overwrites a note from a previous session." in capsys.readouterr().out
        )


class TestAllowedDecisions:
    """A tool's `allowed_decisions` must gate what the prompt offers.

    The middleware raises `ValueError` on a decision type outside a tool's
    `allowed_decisions`, and `main`'s broad `except` swallows it into a one-line
    error — so offering a decision the tool forbids silently costs a whole turn.
    """

    def test_edit_is_not_offered_when_not_allowed(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # "e" must be refused as an unrecognized choice, not produce an edit.
        # Note the *result* alone cannot catch this: a wrongly-accepted "e" would
        # ask for JSON, read "a", fail to parse it, and fall back to approve —
        # the same return value. So assert on what the menu offered.
        prompts = _feed(monkeypatch, "e", "a")
        assert _prompt_decision(REQUEST, ["approve", "reject"]) == {"type": "approve"}
        offered = "".join(prompts)
        assert "[e]dit" not in offered
        assert "re[s]pond" not in offered
        assert "[a]pprove" in offered and "[r]eject" in offered

    def test_blank_does_not_default_to_approve_when_approve_is_disallowed(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # An empty line means "approve" only when approve is on offer.
        _feed(monkeypatch, "", "r", "")
        assert _prompt_decision(REQUEST, ["edit", "reject"]) == {"type": "reject"}

    def test_blank_edit_args_do_not_approve_when_approve_is_disallowed(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # The "blank = approve as-is" shortcut must not smuggle in a forbidden
        # approve. Re-prompt instead, then reject.
        _feed(monkeypatch, "e", "", "r", "")
        assert _prompt_decision(REQUEST, ["edit", "reject"]) == {"type": "reject"}

    def test_invalid_edit_json_does_not_approve_when_approve_is_disallowed(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _feed(monkeypatch, "e", "{not json", "e", '{"file_path": "/memories/y.md"}')
        assert _prompt_decision(REQUEST, ["edit", "reject"]) == {
            "type": "edit",
            "edited_action": {
                "name": "write_file",
                "args": {"file_path": "/memories/y.md"},
            },
        }

    def test_respond_decision_carries_its_message(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _feed(monkeypatch, "s", "use the cached figure")
        assert _prompt_decision(REQUEST, ["approve", "respond"]) == {
            "type": "respond",
            "message": "use the cached figure",
        }

    def test_respond_requires_a_message(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # An empty respond would hand the model an empty tool result.
        _feed(monkeypatch, "s", "", "a")
        assert _prompt_decision(REQUEST, ["approve", "respond"]) == {"type": "approve"}

    def test_no_producible_decision_raises_rather_than_guessing(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # An empty list means nothing is permitted — distinct from `None`, which
        # means "no ReviewConfig was supplied, assume the default set".
        _feed(monkeypatch)  # `input` must never be called
        with pytest.raises(ValueError, match="no supported decision"):
            _prompt_decision(REQUEST, [])


class TestCollectDecisions:
    def test_one_decision_per_request_preserving_order(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        interrupts = [
            Interrupt(
                id="i1",
                value={
                    "action_requests": [
                        {"name": "write_file", "args": {}},
                        {"name": "execute", "args": {}},
                    ]
                },
            )
        ]
        _feed(monkeypatch, "a", "r", "")  # approve the first, reject the second
        assert _collect_decisions(interrupts) == {
            "i1": [{"type": "approve"}, {"type": "reject"}]
        }

    def test_decisions_are_grouped_by_interrupt_not_flattened(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # A turn can hold SEVERAL interrupts at once: the orchestrator dispatches
        # each `task` call as its own concurrent graph task, every subagent
        # inherits `interrupt_on`, and SYSTEM_PROMPT tells it to "fan several out
        # in one turn". LangGraph then refuses a resume value that doesn't say
        # which interrupt it belongs to ("When there are multiple pending
        # interrupts, you must specify the interrupt id when resuming"), so these
        # must stay grouped rather than flattened into one list.
        interrupts = [
            Interrupt(
                id="researcher-a",
                value={"action_requests": [{"name": "write_file", "args": {}}]},
            ),
            Interrupt(
                id="researcher-b",
                value={"action_requests": [{"name": "execute", "args": {}}]},
            ),
        ]
        _feed(monkeypatch, "a", "a")
        assert _collect_decisions(interrupts) == {
            "researcher-a": [{"type": "approve"}],
            "researcher-b": [{"type": "approve"}],
        }

    def test_interrupt_without_action_requests_yields_no_decisions(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # `input` must never be called for an interrupt with nothing to approve,
        # and it must not contribute an empty entry to the resume mapping.
        _feed(monkeypatch)
        assert _collect_decisions([Interrupt(id="i1", value={"other": "shape"})]) == {}

    def test_review_configs_restrict_what_the_prompt_offers(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # The whole point of the plumbing: allowed_decisions must travel from the
        # interrupt's review_configs into the prompt. Dropping it on the floor
        # (the original bug) re-offers [e]dit for an approve/reject-only tool.
        interrupts = [
            Interrupt(
                id="i1",
                value={
                    "action_requests": [{"name": "write_file", "args": {}}],
                    "review_configs": [
                        {
                            "action_name": "write_file",
                            "allowed_decisions": ["approve", "reject"],
                        }
                    ],
                },
            )
        ]
        prompts = _feed(monkeypatch, "e", "a")  # "e" is not on offer → re-prompt
        assert _collect_decisions(interrupts) == {"i1": [{"type": "approve"}]}
        assert "[e]dit" not in "".join(prompts)

    def test_review_config_is_matched_by_name_not_by_position(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # The configs are deliberately in the OPPOSITE order to the requests, and
        # `execute` permits approve ONLY. Indexing positionally would hand
        # `execute` write_file's permissive config, accept the "r", consume "a" as
        # the rejection reason, and return {"type": "reject", "message": "a"} —
        # a different, visibly wrong result.
        interrupts = [
            Interrupt(
                id="i1",
                value={
                    "action_requests": [
                        {"name": "execute", "args": {}},
                        {"name": "write_file", "args": {}},
                    ],
                    "review_configs": [
                        {
                            "action_name": "write_file",
                            "allowed_decisions": ["approve", "edit", "reject"],
                        },
                        {"action_name": "execute", "allowed_decisions": ["approve"]},
                    ],
                },
            )
        ]
        _feed(monkeypatch, "r", "a", "a")
        assert _collect_decisions(interrupts) == {
            "i1": [
                {"type": "approve"},  # execute: "r" refused, then approved
                {"type": "approve"},  # write_file
            ]
        }


# --- main() ------------------------------------------------------------------
#
# `main()` is the largest and most safety-critical function in the repo — it drives
# the whole HITL resume protocol — and until these, nothing tested it. Everything it
# touches is injectable: `open_agent`, `missing_keys` and `input` are all module-level
# names, so a fake agent plus a scripted `input` exercises the real loop offline.

# The report the agent composes in the SAME assistant message that proposes the
# write_file. This is the thing that used to be thrown away.
REPORT = "Opus 4.8 costs $15/Mtok in. ([docs](https://docs.anthropic.com/pricing))"

PENDING_WRITE = Interrupt(
    id="i1",
    value={
        "action_requests": [
            {
                "name": "write_file",
                "args": {"file_path": "/memories/pricing.md", "content": "# Pricing"},
            }
        ]
    },
)


class _FakeAgent:
    """Just enough agent for `main()`: an `invoke` and a checkpointed `get_state`.

    `get_state` returning the prose is not a convenience — it is the fact the fix
    depends on. The report is already durably checkpointed by the time the human is
    asked to approve the write, which is precisely why abandoning the turn need not
    lose it.
    """

    def __init__(
        self, *, raises: Exception | None = None, messages: list[Any] | None = None
    ):
        self._raises = raises
        self.messages = (
            [HumanMessage("what does opus cost?"), AIMessage(REPORT)]
            if messages is None
            else messages
        )
        self.invocations: list[Any] = []

    def invoke(self, payload: Any, config: Any = None) -> dict[str, Any]:
        self.invocations.append(payload)
        if self._raises is not None:
            raise self._raises
        return {"messages": self.messages, "__interrupt__": [PENDING_WRITE]}

    def get_state(self, config: Any) -> SimpleNamespace:
        return SimpleNamespace(values={"messages": self.messages})


def _drive(monkeypatch: pytest.MonkeyPatch, agent: _FakeAgent, *inputs: Any) -> None:
    """Run `main()` against a fake agent with a scripted stdin.

    An input that is an exception instance is *raised* rather than returned — that is
    how a Ctrl-C at the approval prompt is simulated.
    """
    monkeypatch.setattr(cli_module, "missing_keys", lambda: {})

    @contextmanager
    def fake_open_agent() -> Iterator[_FakeAgent]:
        yield agent

    monkeypatch.setattr(cli_module, "open_agent", fake_open_agent)

    scripted: Iterator[Any] = iter(inputs)

    def fake_input(prompt: str = "") -> str:
        value = next(scripted)
        if isinstance(value, BaseException):
            raise value
        return value

    monkeypatch.setattr("builtins.input", fake_input)
    main()


class TestMainSalvagesAnAbandonedTurn:
    """Ctrl-C at an approval prompt must not discard the report.

    The agent writes its cited report in the same assistant message that proposes the
    `write_file`, so the turn a human is most likely to abandon — the one stopped at an
    approval they don't like — is reliably the one that has already run every search
    and written every word. Both `except` arms used to `continue` past the
    `render_turn` at the foot of the loop, so minutes of work and dozens of sources
    vanished on a keystroke while the prose sat in the checkpoint.
    """

    def test_ctrl_c_at_the_approval_prompt_still_shows_the_report(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        agent = _FakeAgent()
        # Ask a question → the turn interrupts on write_file → Ctrl-C at the approval
        # prompt → back to the REPL → leave.
        _drive(monkeypatch, agent, "what does opus cost?", KeyboardInterrupt(), "/exit")

        out = capsys.readouterr().out
        assert "(interrupted — back to prompt)" in out
        assert REPORT in out, "the finished report was discarded on Ctrl-C"
        assert "unfinished turn" in out  # …and labelled honestly, not as a clean answer

    def test_an_error_mid_turn_still_shows_the_report_and_the_error(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        # Same loss, other arm: an API 500 on the resume `invoke` after approval used
        # to take the report with it. The salvage must never *replace* the error.
        agent = _FakeAgent(raises=RuntimeError("overloaded_error"))
        _drive(monkeypatch, agent, "what does opus cost?", "/exit")

        out = capsys.readouterr().out
        assert "! error: overloaded_error" in out
        assert REPORT in out

    def test_nothing_is_printed_when_there_is_nothing_to_salvage(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        # A turn abandoned before the agent said anything must not print an empty
        # "agent (unfinished turn) >" header.
        agent = _FakeAgent(
            raises=RuntimeError("boom"), messages=[HumanMessage("what does opus cost?")]
        )
        _drive(monkeypatch, agent, "what does opus cost?", "/exit")

        out = capsys.readouterr().out
        assert "! error: boom" in out
        assert "unfinished turn" not in out

    def test_the_next_turn_is_a_fresh_message_not_a_resume(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # After abandoning, the pending interrupt is still in the checkpoint. The next
        # thing we send must be a NEW human message, not a `Command(resume=...)` — the
        # dangling tool call is deepagents' problem (PatchToolCallsMiddleware answers
        # it with a synthetic "cancelled" ToolMessage at the graph entry), not ours to
        # paper over here.
        agent = _FakeAgent()
        _drive(
            monkeypatch,
            agent,
            "what does opus cost?",
            KeyboardInterrupt(),  # abandon at the approval prompt
            "and haiku?",
            KeyboardInterrupt(),  # abandon again, so we don't have to script approvals
            "/exit",
        )

        assert len(agent.invocations) == 2
        second = agent.invocations[1]
        assert not isinstance(second, Command)
        assert second["messages"][0]["content"] == "and haiku?"
