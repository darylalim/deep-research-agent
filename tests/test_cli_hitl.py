"""Unit tests for the human-in-the-loop decision logic in `cli.py`.

`_collect_decisions` is the load-bearing contract the README calls out: it maps
each interrupt's `action_requests` to exactly one decision, in order. It and
`_prompt_decision` both read from `input()`, which is scripted here via a fed
iterator so no real terminal interaction happens.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from langchain.agents.middleware.human_in_the_loop import HumanInTheLoopMiddleware
from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    RemoveMessage,
    ToolMessage,
)
from langgraph.graph.message import REMOVE_ALL_MESSAGES
from langgraph.types import Command, Interrupt

from deep_research import cli as cli_module
from deep_research.cli import (
    DEFAULT_ALLOWED_DECISIONS,
    PREVIEW_LINES,
    ActivityFeed,
    _collect_decisions,
    _prompt_decision,
    _render_action,
    main,
)


def _updates(node: str, *messages: object) -> dict:
    """One `stream(stream_mode="updates")` chunk, as `evals/test_evals.py` builds them."""
    return {node: {"messages": list(messages)}}


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

# What a subagent's stream namespace actually looks like (measured): the pregel task id
# is retained, so two concurrent researchers are distinguishable.
SUBAGENT_NS: tuple[str, ...] = ("tools:9d0c2f4e",)


class _FakeAgent:
    """Just enough agent for `main()`: a `stream` and a checkpointed `get_state`.

    Faithful to the two things production actually does, because a fake that is merely
    *convenient* would let `main()` pass a test it would fail for real:

    - `stream(..., stream_mode="updates", subgraphs=True)` yields `(namespace, chunk)`
      pairs, and an interrupt arrives as a `{"__interrupt__": (...)}` CHUNK — there is no
      `__interrupt__` key on any result, which is what the old `invoke()` shape had.
    - a subagent's interrupt is emitted TWICE with the same `Interrupt.id` (once at the
      subagent namespace, once bubbled to the root), so the fake emits it twice. If
      `_collect_decisions` ever stops deduping, `test_one_prompt_per_pending_action`
      goes red instead of a real human being asked to approve the same write twice.

    `get_state` returning the prose is not a convenience either — it is the fact the
    salvage depends on. The report is already durably checkpointed by the time the human
    is asked to approve the write, because the `model` node writes it one superstep
    before the `HumanInTheLoopMiddleware.after_model` node interrupts.
    """

    def __init__(
        self,
        *,
        raises: Exception | None = None,
        messages: list[Any] | None = None,
        rounds: list[list[tuple[tuple[str, ...], dict[str, Any]]]] | None = None,
    ):
        self._raises = raises
        self.messages = (
            [HumanMessage("what does opus cost?"), AIMessage(REPORT)]
            if messages is None
            else messages
        )
        # ROUNDS, not one fixed chunk list. The fake used to re-yield the same chunks on
        # every `stream()` call, INCLUDING the resume — so a turn could never be driven
        # past an approval (it would just interrupt again, forever), which left the
        # id-keyed `Command(resume=...)` and the final answer print with no coverage at
        # all, and let `test_main_asks_once_for_a_subagents_write` pass with the dedupe
        # deleted. A real agent's resume stream yields *different* chunks; so does this.
        #
        # Round 1 by default: one gated write, proposed inside a subagent — so its
        # interrupt is emitted TWICE with the same `Interrupt.id` (subagent namespace,
        # then bubbled to the root), exactly as LangGraph does under `subgraphs=True`.
        # Round 2: the approval goes through and the turn finishes.
        self._rounds = (
            [
                [
                    (SUBAGENT_NS, {"__interrupt__": (PENDING_WRITE,)}),
                    ((), {"__interrupt__": (PENDING_WRITE,)}),
                ],
                [
                    (
                        (),
                        {
                            "tools": {
                                "messages": [
                                    ToolMessage(
                                        "ok", tool_call_id="w1", name="write_file"
                                    )
                                ]
                            }
                        },
                    )
                ],
            ]
            if rounds is None
            else rounds
        )
        self.invocations: list[Any] = []

    def stream(
        self,
        payload: Any,
        config: Any = None,
        stream_mode: str | None = None,
        subgraphs: bool = False,
    ) -> Iterator[tuple[tuple[str, ...], dict[str, Any]]]:
        # Assert the contract the whole design rests on, at the point of use: node-level
        # updates (not "messages", which would leak researcher prose) and subgraphs=True
        # (without which the searches are invisible — the entire reason to stream).
        assert stream_mode == "updates"
        assert subgraphs is True
        round_index = len(self.invocations)
        self.invocations.append(payload)
        if self._raises is not None:
            raise self._raises
        # Past the scripted rounds the turn is simply over — an empty stream, which is
        # how `_stream_turn` reports "no interrupts left" and how the loop terminates.
        if round_index < len(self._rounds):
            yield from self._rounds[round_index]

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


class TestDuplicateInterrupts:
    """One prompt per pending action, however many times the interrupt is emitted.

    With `subgraphs=True` an interrupt raised inside a subagent is emitted TWICE — once
    at the subagent's namespace, once bubbled to the root — carrying the SAME
    `Interrupt.id`. Without deduping, the human is asked to approve one researcher's
    `write_file` twice, and only the second answer survives (the resume mapping is keyed
    by id). Approval fatigue is how a gate stops being a gate.

    The old blocking `invoke()` never saw this — it streams with `subgraphs=False`, so
    the child emitted nothing — which is why this could only appear alongside the feed.
    """

    def test_the_same_interrupt_twice_prompts_once(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        prompts = _feed(monkeypatch, "a")  # exactly ONE decision is scripted
        # Two chunks, one real approval. A second `input()` call raises StopIteration.
        decisions = _collect_decisions([PENDING_WRITE, PENDING_WRITE])

        assert decisions == {"i1": [{"type": "approve"}]}
        assert sum("[a]pprove" in p for p in prompts) == 1

    def test_distinct_interrupts_still_each_prompt(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # The dedupe must key on the id, not collapse everything: two researchers each
        # proposing their own write are two real decisions.
        other = Interrupt(
            id="i2", value={"action_requests": [{"name": "write_file", "args": {}}]}
        )
        _feed(monkeypatch, "a", "a")
        assert _collect_decisions([PENDING_WRITE, other]) == {
            "i1": [{"type": "approve"}],
            "i2": [{"type": "approve"}],
        }

    def test_main_asks_once_for_a_subagents_write(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        # End to end through the real turn loop: the fake emits the duplicate the way
        # LangGraph does, and the human is asked exactly once.
        #
        # The APPROVAL is what makes this bite. An earlier version scripted a
        # KeyboardInterrupt at the first prompt — which meant the second, duplicate
        # interrupt was never reached, so the test passed identically with the dedupe
        # DELETED. A guard that cannot fail is not a guard. Approve instead, and let the
        # turn run to completion.
        agent = _FakeAgent()
        _drive(monkeypatch, agent, "what does opus cost?", "a", "/exit")
        out = capsys.readouterr().out

        assert out.count("Approval required — write_file") == 1
        # …and the turn actually completed: one initial stream + one resume.
        assert len(agent.invocations) == 2
        assert isinstance(agent.invocations[1], Command)
        assert REPORT in out

    def test_the_resume_is_keyed_by_interrupt_id(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # The id-keyed `Command(resume={id: {"decisions": [...]}})` had NO coverage: the
        # old fake re-yielded the same chunks forever, so no test could drive `main()`
        # past an approval. LangGraph raises `RuntimeError("When there are multiple
        # pending interrupts, you must specify the interrupt id when resuming")` on a flat
        # resume value, so this shape is load-bearing.
        agent = _FakeAgent()
        _drive(monkeypatch, agent, "what does opus cost?", "a", "/exit")

        resume = agent.invocations[1].resume
        assert resume == {"i1": {"decisions": [{"type": "approve"}]}}

    def test_a_rejected_write_is_not_reported_as_a_tool_failure(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        # `HumanInTheLoopMiddleware` answers a REJECTED call with a synthetic ToolMessage
        # carrying `status="error"` — and if the human gave a reason, that reason becomes
        # its content. So the feed cannot tell a rejection from a crash, and printed
        # "! write_file failed: too risky" at the person who had just typed `r`, reporting
        # their own honoured decision as a bug in the agent.
        rejection = ToolMessage(
            "User rejected the tool call for `write_file` with id w1.",
            tool_call_id="w1",
            name="write_file",
            status="error",
        )
        agent = _FakeAgent(
            rounds=[
                [
                    (SUBAGENT_NS, {"__interrupt__": (PENDING_WRITE,)}),
                    ((), {"__interrupt__": (PENDING_WRITE,)}),
                ],
                [((), {"tools": {"messages": [rejection]}})],
            ]
        )
        # reject, with a reason
        _drive(monkeypatch, agent, "what does opus cost?", "r", "too risky", "/exit")
        out = capsys.readouterr().out

        assert "write_file failed" not in out
        assert "✗ write_file — rejected, as you asked" in out


class TestCommandDispatch:
    def test_a_mistyped_command_does_not_run_the_real_one(
        self,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
        tmp_path: Path,
    ) -> None:
        # Dispatch used to be `user_input.startswith("/export")`, which also swallows
        # `/exports` and `/exported` — silently writing a default-named file for what is
        # plainly a typo. A mistyped command should SAY so, not act. (`/thread` had the
        # same shape.) The turn loop must also not fire: a `/`-prefixed typo is a command,
        # not a research question, and researching it would cost real money.
        #
        # `chdir` into tmp_path is not hygiene theatre. `/export` with no argument writes
        # `research-<thread>-<utc>.md` into the CWD — so when this fix regresses, this
        # test drops a file in the repo root. It did exactly that while the fix was being
        # verified, and `git add -A` then committed it. A test that proves a bug by
        # littering the working tree is a test that will eventually litter someone's repo.
        monkeypatch.chdir(tmp_path)

        agent = _FakeAgent()
        _drive(monkeypatch, agent, "/exports", "/threadx", "/exit")
        out = capsys.readouterr().out

        assert "unknown command '/exports'" in out
        assert "unknown command '/threadx'" in out
        assert agent.invocations == [], "a mistyped command started a research turn"
        assert list(tmp_path.glob("research-*.md")) == [], "a typo wrote an export file"

    def test_the_real_commands_still_work(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        _drive(monkeypatch, _FakeAgent(), "/thread other", "/thread", "/exit")
        out = capsys.readouterr().out
        assert "switched to thread 'other'" in out
        assert "current thread: 'other'" in out


class TestActivityFeed:
    """The feed shows ACTIONS. Never prose — a researcher's words are not the agent's."""

    def test_it_shows_the_plan_the_delegations_and_the_queries(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        feed = ActivityFeed()
        feed.absorb(
            (),
            {
                "tools": {
                    "todos": [
                        {"content": "pricing for Opus 4.8", "status": "pending"},
                        {"content": "rate limits by tier", "status": "pending"},
                    ]
                }
            },
        )
        feed.absorb(
            (),
            _updates(
                "model",
                AIMessage(
                    content="",
                    id="ai-1",
                    tool_calls=[
                        {
                            "name": "task",
                            "args": {
                                "description": "pricing for Opus 4.8",
                                "subagent_type": "researcher",
                            },
                            "id": "t1",
                        }
                    ],
                ),
            ),
        )
        feed.absorb(
            SUBAGENT_NS,
            _updates(
                "model",
                AIMessage(
                    content="",
                    id="ai-2",
                    tool_calls=[
                        {
                            "name": "tavily_search",
                            "args": {"query": "anthropic opus 4.8 price"},
                            "id": "s1",
                        }
                    ],
                ),
            ),
        )
        feed.absorb(
            (),
            _updates(
                "tools",
                ToolMessage("cited summary", tool_call_id="t1", name="task"),
            ),
        )
        out = capsys.readouterr().out

        assert "✎ plan · 2 items" in out
        assert "1. pricing for Opus 4.8" in out
        assert "→ researcher · pricing for Opus 4.8" in out
        assert '⌕ "anthropic opus 4.8 price"' in out
        # The completion line recovers the description via `tool_call_id` — the one
        # honest way to name a researcher, since its stream namespace cannot be bound
        # back to the task call that spawned it.
        assert "✓ researcher · pricing for Opus 4.8" in out

    def test_it_never_prints_a_researchers_prose(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        # THE rule. The stream carries the researchers' own assistant messages, which
        # the user must never see — `evals/harness.py` refuses to build its graded
        # `response` from the stream for exactly this reason. Printing them here would
        # show a subagent's cited paragraphs as if they were the agent's answer, and
        # would show the SAME report twice (once from the stream, once from the final
        # `render_turn`).
        feed = ActivityFeed()
        feed.absorb(
            SUBAGENT_NS,
            _updates(
                "model", AIMessage(content="Opus is $15/Mtok ([x](u))", id="ai-9")
            ),
        )
        feed.absorb(
            SUBAGENT_NS,
            _updates(
                "tools",
                ToolMessage(
                    '{"results": [{"url": "https://x.test"}]}',
                    tool_call_id="s1",
                    name="tavily_search",
                ),
            ),
        )
        out = capsys.readouterr().out

        assert "Opus is $15/Mtok" not in out  # the researcher's prose
        assert "https://x.test" not in out  # the raw search payload

    def test_a_reemitted_call_is_not_announced_twice(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        # On resume, HumanInTheLoopMiddleware re-emits the AIMessage that proposed the
        # gated call. Without dedupe the user watches lines they have already seen scroll
        # past again, once per approval round.
        call = AIMessage(
            content="",
            id="ai-1",
            tool_calls=[
                {
                    "name": "task",
                    "args": {"description": "d", "subagent_type": "researcher"},
                    "id": "t1",
                }
            ],
        )
        feed = ActivityFeed()
        feed.absorb((), _updates("model", call))
        feed.absorb((), _updates("HumanInTheLoopMiddleware.after_model", call))

        assert capsys.readouterr().out.count("→ researcher") == 1

    def test_a_replayed_tool_result_without_a_message_id_is_not_shown_twice(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        # THE reason the seen-set is keyed on the CALL id and not the message id. A
        # re-streamed superstep re-emits the cached writes of tasks that already
        # finished, and `BaseMessage.id` is optional — a ToolMessage carrying no id
        # defeats message-id dedupe entirely. Caught by driving the real loop: the
        # completion line for a finished researcher printed a second time, after the
        # approval. A tool call executes exactly once, so its id is the honest key.
        feed = ActivityFeed()
        announce = _updates(
            "model",
            AIMessage(
                content="",
                id="ai-1",
                tool_calls=[
                    {
                        "name": "task",
                        "args": {
                            "description": "pricing",
                            "subagent_type": "researcher",
                        },
                        "id": "t1",
                    }
                ],
            ),
        )
        feed.absorb((), announce)
        # Two DISTINCT ToolMessage objects, neither carrying an `id` — exactly what a
        # replayed superstep hands you.
        for _ in range(2):
            feed.absorb(
                (),
                _updates(
                    "tools", ToolMessage("summary", tool_call_id="t1", name="task")
                ),
            )

        assert capsys.readouterr().out.count("✓ researcher") == 1

    def test_the_ls_line_reads_the_list_repr_not_the_line_count(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        # deepagents builds the `ls` body as `str(paths)` — a Python list repr on ONE
        # line, `"[]"` or `"['/memories/a.md', '/b.md']"` — not newline-separated entries.
        # Counting lines reported "1 file(s)" for an EMPTY store every single time, and
        # made the "empty" branch unreachable. This line is how the user sees whether
        # durable memory was consulted; one that lies is worse than no line at all.
        cases = (("[]", "empty"), ("['/memories/a.md', '/b.md']", "2 file(s)"))
        for body, expected in cases:
            feed = ActivityFeed()
            feed.absorb(
                (),
                _updates(
                    "model",
                    AIMessage(
                        content="",
                        id="ai-1",
                        tool_calls=[
                            {"name": "ls", "args": {"path": "/memories/"}, "id": "l1"}
                        ],
                    ),
                ),
            )
            feed.absorb(
                (), _updates("tools", ToolMessage(body, tool_call_id="l1", name="ls"))
            )
            assert f"⌕ /memories/ · {expected}" in capsys.readouterr().out

    def test_a_researchers_own_todos_and_ls_are_not_shown_as_the_agents(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        # deepagents gives EVERY declarative subagent its own TodoListMiddleware and
        # FilesystemMiddleware, so a `researcher` really can call `write_todos` and `ls`
        # regardless of what `subagents.py` lists in its `tools`. Rendering those
        # namespace-blind would:
        #   - print a researcher's private checklist as a second `✎ plan`, appearing to
        #     supersede the plan the user was just shown;
        #   - print `⌕ /memories/` on a turn where the ORCHESTRATOR never looked, hiding
        #     the very "the direct path skips /memories/" defect CLAUDE.md says to watch.
        # The same orchestrator/subagent conflation `evals/harness.py` keeps apart with
        # `orchestrator_trajectory` vs `trajectory`. It has to hold in the display too.
        feed = ActivityFeed()
        feed.absorb(
            SUBAGENT_NS,
            {"tools": {"todos": [{"content": "my private step", "status": "pending"}]}},
        )
        feed.absorb(
            SUBAGENT_NS,
            _updates(
                "tools", ToolMessage("['/scratch/a.md']", tool_call_id="l9", name="ls")
            ),
        )
        out = capsys.readouterr().out

        assert "✎ plan" not in out
        assert "my private step" not in out
        assert "⌕ /memories/" not in out

    def test_a_failed_tool_is_surfaced(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        # Tavily raises rather than returning an empty list, so a fruitless search
        # arrives as an error. A silent feed would make it look like it never ran.
        feed = ActivityFeed()
        feed.absorb(
            SUBAGENT_NS,
            _updates(
                "tools",
                ToolMessage(
                    "no results for that query",
                    tool_call_id="s1",
                    name="tavily_search",
                    status="error",
                ),
            ),
        )
        assert "! tavily_search failed" in capsys.readouterr().out

    def test_a_thread_rewrite_does_not_replay_the_previous_turns_feed(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        # `PatchToolCallsMiddleware.before_agent` answers dangling tool calls by returning
        # `{"messages": [RemoveMessage(REMOVE_ALL_MESSAGES), *THE ENTIRE THREAD]}`. It
        # fires on exactly the turn AFTER one you abandoned at an approval prompt — and
        # the feed is per-turn, so its seen-set has never heard of those calls. Without a
        # guard, that turn opens by replaying the previous turn's whole feed: its plan,
        # its delegations, every search it ran.
        feed = ActivityFeed()
        feed.absorb(
            (),
            {
                "PatchToolCallsMiddleware.before_agent": {
                    "messages": [
                        RemoveMessage(id=REMOVE_ALL_MESSAGES),
                        HumanMessage("the previous question"),
                        AIMessage(
                            content="",
                            id="old-ai",
                            tool_calls=[
                                {
                                    "name": "task",
                                    "args": {
                                        "description": "last turn's sub-question",
                                        "subagent_type": "researcher",
                                    },
                                    "id": "old-t1",
                                }
                            ],
                        ),
                    ]
                }
            },
        )
        assert capsys.readouterr().out == ""

    def test_it_returns_the_interrupts_it_sees(self) -> None:
        # The feed is also the interrupt collector, exactly like harness.TurnRecorder —
        # one traversal, not two.
        feed = ActivityFeed()
        assert feed.absorb((), {"__interrupt__": (PENDING_WRITE,)}) == [PENDING_WRITE]
