"""Unit tests for the human-in-the-loop decision logic in `cli.py`.

`_collect_decisions` is the load-bearing contract the README calls out: it maps
each interrupt's `action_requests` to exactly one decision, in order. It and
`_prompt_decision` both read from `input()`, which is scripted here via a fed
iterator so no real terminal interaction happens.
"""

from __future__ import annotations

from collections.abc import Iterator

import pytest
from langgraph.types import Interrupt

from deep_research.cli import _collect_decisions, _prompt_decision

REQUEST = {"name": "write_file", "args": {"file_path": "/memories/x.md"}}


def _feed(monkeypatch: pytest.MonkeyPatch, *responses: str) -> None:
    """Make `input()` return `responses` in order (StopIteration if over-consumed)."""
    scripted: Iterator[str] = iter(responses)
    monkeypatch.setattr("builtins.input", lambda *_a, **_k: next(scripted))


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


class TestCollectDecisions:
    def test_one_decision_per_request_preserving_order(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        interrupts = [
            Interrupt(
                value={
                    "action_requests": [
                        {"name": "write_file", "args": {}},
                        {"name": "execute", "args": {}},
                    ]
                }
            )
        ]
        _feed(monkeypatch, "a", "r", "")  # approve the first, reject the second
        assert _collect_decisions(interrupts) == [
            {"type": "approve"},
            {"type": "reject"},
        ]

    def test_flattens_requests_across_multiple_interrupts(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        interrupts = [
            Interrupt(value={"action_requests": [{"name": "write_file", "args": {}}]}),
            Interrupt(value={"action_requests": [{"name": "execute", "args": {}}]}),
        ]
        _feed(monkeypatch, "a", "a")
        assert _collect_decisions(interrupts) == [
            {"type": "approve"},
            {"type": "approve"},
        ]

    def test_interrupt_without_action_requests_yields_no_decisions(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # `input` must never be called for an interrupt with nothing to approve.
        _feed(monkeypatch)
        assert _collect_decisions([Interrupt(value={"other": "shape"})]) == []
