"""Unit tests for the human-in-the-loop decision logic in `cli.py`.

`_collect_decisions` is the load-bearing contract the README calls out: it maps
each interrupt's `action_requests` to exactly one decision, in order. It and
`_prompt_decision` both read from `input()`, which is scripted here via a fed
iterator so no real terminal interaction happens.
"""

from __future__ import annotations

from collections.abc import Iterator

import pytest
from langchain.agents.middleware.human_in_the_loop import HumanInTheLoopMiddleware
from langgraph.types import Interrupt

from deep_research.cli import (
    DEFAULT_ALLOWED_DECISIONS,
    _collect_decisions,
    _prompt_decision,
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

    def test_edit_with_invalid_json_falls_back_to_approve(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _feed(monkeypatch, "e", "{not json")
        assert _prompt_decision(REQUEST) == {"type": "approve"}

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
