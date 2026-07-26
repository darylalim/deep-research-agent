"""Tests for the page's rerun state machine (`streamlit_app.py`).

`test_webui.py` covers the rendering and the approval widgets; this file covers the
thing they sit inside — the four `st.session_state` keys (payload / question / feed /
pending) that unroll `cli.main`'s interrupt/resume loop across Streamlit reruns. That
loop had no tests at all in its first version, and a code review found three separate
defects in it, every one invisible in a screenshot.

`AppTest.from_file` executes the real page. It works offline because `conftest.py`
already sets dummy credentials (so `missing_keys()` passes) and an isolated
`DEEP_RESEARCH_STATE_DIR` (so the sqlite files land in a temp dir), both as top-level
code that runs before `deep_research.config` is imported.

Nothing here calls a model: every test either seeds `st.session_state` directly or
replaces `_stream_turn`, so the graph is never actually run.
"""

from __future__ import annotations

from typing import Any

import pytest
from langgraph.types import Interrupt

pytest.importorskip("streamlit")

from streamlit.testing.v1 import AppTest

from deep_research import cli as cli_module
from deep_research import webui

PAGE = "streamlit_app.py"

_WRITE = {
    "name": "write_file",
    "args": {"file_path": "/memories/pricing.md", "content": "# Pricing"},
}


def _page(**session: Any) -> AppTest:
    """Run the real page with `session` pre-seeded into `st.session_state`."""
    page = AppTest.from_file(PAGE, default_timeout=60)
    for key, value in session.items():
        page.session_state[key] = value
    return page.run()


class TestThePayloadIsConsumedBeforeStreaming:
    """The single most expensive bug the review found: a re-sent question.

    Streamlit aborts a running script at its next `st.*` call by raising
    `RerunException`, which derives from **BaseException** — so `except Exception` around
    the stream never sees it and cannot clear the payload on the way out. Any widget the
    page left enabled during a turn (the export button, the memory selectbox) could
    therefore tear down `agent.stream` mid-superstep AND leave `payload` set, so the next
    rerun re-entered with the same user message: the question appended to the thread
    twice, and every search paid for again.
    """

    def test_the_payload_is_already_cleared_when_the_stream_starts(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The ordering IS the fix, so the ordering is what this asserts.

        Simulating the real abort is not available: raising a BaseException inside the
        script kills Streamlit's script thread outright, so `AppTest.run()` waits out its
        full timeout instead of returning a page to assert on (measured — the first
        version of this test took 60s and then failed on the timeout, not the assertion).

        Observing `payload` from inside `_stream_turn` is better anyway. It pins the
        invariant directly — by the time the graph is running, nothing is left in session
        state that a later rerun could re-send — and it stays true regardless of *how*
        the turn is interrupted.
        """
        seen: list[Any] = []

        def record(_agent: Any, _payload: Any, _config: Any, _feed: Any) -> list[Any]:
            import streamlit as st  # runs inside the page's script thread

            seen.append(st.session_state.payload)
            return []

        monkeypatch.setattr(cli_module, "_stream_turn", record)
        _page(
            payload={"messages": [{"role": "user", "content": "q"}]},
            question="q",
            feed=webui.StreamlitFeed(),
        )

        assert seen == [None], (
            "payload still set while the stream ran — an interrupted turn would re-send "
            "the question and re-bill the research"
        )

    def test_an_ordinary_error_also_clears_it(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            cli_module,
            "_stream_turn",
            lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")),
        )
        page = _page(
            payload={"messages": [{"role": "user", "content": "q"}]},
            question="q",
            feed=webui.StreamlitFeed(),
        )

        assert page.session_state["payload"] is None
        assert "boom" in (page.session_state["notice"] or "")


class TestPendingApprovalsSurviveTheSession:
    """`st.session_state` dies with the browser session; the interrupt does not."""

    def test_a_pending_interrupt_is_recovered_from_the_checkpoint(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Without this the turn is stranded: the graph still considers it paused, but a
        # refreshed page shows no approval form and re-enables the chat input, and the
        # next question sends fresh input to a thread with a pending interrupt — the
        # prefill 400 CLAUDE.md documents.
        stranded = Interrupt(id="i1", value={"action_requests": [_WRITE]})
        # Honours `skip` like the real thing, so this stays a faithful stand-in for a
        # checkpoint that still holds the interrupt rather than one that always shouts.
        monkeypatch.setattr(
            webui,
            "recover_pending",
            lambda _state, *, skip=frozenset(): [
                i for i in [stranded] if i.id not in skip
            ],
        )

        page = _page()  # a fresh session: nothing in session_state at all

        assert page.session_state["pending"] == [stranded]
        assert page.button_group, "no approval controls rendered for a recovered pause"

    def test_recover_pending_reads_the_snapshots_interrupts(self) -> None:
        # `StateSnapshot.interrupts` is already the flattened
        # `[i for task in tasks_with_writes for i in task.interrupts]`, so there is no
        # need to walk `.tasks`. A snapshot with none must yield none rather than raise.
        one = Interrupt(id="i1", value={"action_requests": [_WRITE]})

        class _Snapshot:
            interrupts = (one,)

        assert webui.recover_pending(_Snapshot()) == [one]
        assert webui.recover_pending(object()) == []

    def test_it_skips_interrupts_the_session_has_given_up_on(self) -> None:
        # Clearing `st.session_state.pending` does not resume the graph, so an abandoned
        # interrupt is still sitting in the checkpoint. Without `skip`, recovery reads it
        # straight back and the page never escapes it.
        one = Interrupt(id="i1", value={"action_requests": [_WRITE]})
        two = Interrupt(id="i2", value={"action_requests": [_WRITE]})

        class _Snapshot:
            interrupts = (one, two)

        assert webui.recover_pending(_Snapshot(), skip={"i1"}) == [two]
        assert webui.recover_pending(_Snapshot(), skip={"i1", "i2"}) == []

    def test_a_live_turn_is_not_overridden_by_the_checkpoint(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # During a turn the live stream is the authority. The checkpoint lags it — it is
        # only written as supersteps complete — so recovering mid-flight would hand the
        # reviewer a stale interrupt in place of the one this session is actually
        # handling.
        live = Interrupt(id="live", value={"action_requests": [_WRITE]})
        monkeypatch.setattr(
            webui,
            "recover_pending",
            lambda _state, *, skip=frozenset(): [
                Interrupt(id="stale", value={"action_requests": [_WRITE]})
            ],
        )
        monkeypatch.setattr(cli_module, "_stream_turn", lambda *a, **k: [live])

        page = _page(
            payload={"messages": [{"role": "user", "content": "q"}]},
            question="q",
            feed=webui.StreamlitFeed(),
        )

        assert [i.id for i in page.session_state["pending"]] == ["live"]


class TestTheApprovalPanelEscalatesToTheTurnLoop:
    """The approval controls live in an `@st.fragment`, and this is what makes that safe.

    A fragment rerun redraws the fragment and nothing else — which is the entire point on
    this screen, since a decision click otherwise re-read the checkpoint, rebuilt the
    export document and redrew every chat bubble. But the *submit* is not a fragment-
    scoped event: it has to hand control back to the page so the streaming block below
    re-enters with the new `Command(resume=…)`. A bare `st.rerun()` is app-scoped even
    from inside a fragment, and that is the only reason the turn resumes at all.

    Nothing else covers this. `test_webui.py` drives `approval_form` in a bare script,
    so it proves the decisions *mapping* is right and stops there; the page-level tests
    below seed `payload` directly and never travel the submit path. Get the scope wrong
    and the panel would simply redraw itself forever — no exception, no failing test, a
    reviewer clicking Send and watching nothing happen.
    """

    def test_submitting_a_decision_resumes_the_turn(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        resumed: list[Any] = []

        def record(_agent: Any, payload: Any, _config: Any, _feed: Any) -> list[Any]:
            resumed.append(payload)
            return []

        monkeypatch.setattr(cli_module, "_stream_turn", record)
        page = _page(
            pending=[Interrupt(id="i1", value={"action_requests": [_WRITE]})],
            feed=webui.StreamlitFeed(),
        )

        page.button_group[0].set_value("approve").run()
        send = [b for b in page.button if "Send" in b.label]
        assert send, "no submit control on the approval panel"
        send[0].click().run()

        assert len(resumed) == 1, (
            "the graph was never resumed — the submit rerun did not leave the fragment"
        )
        # The shape LangGraph demands, keyed by interrupt id: a flat
        # `{"decisions": [...]}` raises as soon as a turn holds two interrupts.
        assert resumed[0].resume == {"i1": {"decisions": [{"type": "approve"}]}}
        assert page.session_state["pending"] == []


class TestGivingUpOnAnInterruptActuallyGivesUp:
    """Both paths that drop a turn, against a graph that is genuinely still paused.

    Clearing `st.session_state.pending` does not resume anything — the interrupt stays in
    the checkpoint — so a page that then recovers blindly reads it straight back. Both
    failures below were real and neither was caught, because the test thread has no live
    checkpoint: `recover_pending` returned `[]`, the re-seeding never happened, and
    `test_a_stuck_approval_can_be_abandoned` passed while the button did nothing in
    production. That is this repo's recurring failure mode — an assertion satisfied by an
    unrelated condition — so the stub below supplies the missing half.

    It deliberately *implements* `skip` rather than ignoring it: a page that stops
    passing `skip` gets the default, which filters nothing, and both tests go red.
    """

    @staticmethod
    def _still_paused(monkeypatch: pytest.MonkeyPatch, *interrupts: Interrupt) -> None:
        """Model a checkpoint that still holds `interrupts`, honouring `skip`."""
        monkeypatch.setattr(
            webui,
            "recover_pending",
            lambda _state, *, skip=frozenset(): [
                i for i in interrupts if i.id not in skip
            ],
        )

    def test_abandoning_escapes_even_though_the_graph_is_still_paused(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        stranded = Interrupt(id="i1", value={"action_requests": [_WRITE]})
        self._still_paused(monkeypatch, stranded)

        page = _page()
        abandon = [b for b in page.button if "Abandon" in b.label]
        assert abandon, "no escape from the approval screen"
        abandon[0].click().run()

        assert not page.exception, page.exception
        assert page.session_state["pending"] == [], (
            "the approval came straight back — abandoning did not abandon"
        )
        assert not page.button_group, "approval controls redrawn after abandoning"
        # The point of escaping: the session can be used again.
        assert not page.chat_input[0].disabled, "no way to ask anything else"

    def test_an_interrupt_with_no_reviewable_action_settles(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Broken, this is an unbounded full-rerun loop rather than a wrong value — 1019
        # reruns in 6 seconds, measured — so the timeout is deliberately short. A failure
        # here should cost seconds, not the 60s CLAUDE.md records paying once already.
        self._still_paused(
            monkeypatch, Interrupt(id="i1", value={"action_requests": []})
        )

        page = AppTest.from_file(PAGE, default_timeout=8).run()

        assert not page.exception, page.exception
        assert page.session_state["pending"] == []
        assert "abandoned" in (page.session_state["notice"] or "").lower()
        assert not page.chat_input[0].disabled


class TestTheApprovalScreenIsEscapable:
    def test_a_stuck_approval_can_be_abandoned(self) -> None:
        # `approval_form` keeps submit disabled until every action has a valid decision,
        # so a tool gated with decisions this UI cannot render would disable it forever
        # — while `busy` has already disabled the chat input and the thread field and
        # `st.stop()` ends the page. Without this button the session has no control left
        # that can move it forward. `cli.main` abandons the same turn via its broad
        # `except`.
        page = _page(
            pending=[Interrupt(id="i1", value={"action_requests": [_WRITE]})],
            feed=webui.StreamlitFeed(),
        )
        abandon = [b for b in page.button if "Abandon" in b.label]
        assert abandon, "no escape from the approval screen"

        abandon[0].click().run()

        assert page.session_state["pending"] == []
        assert page.session_state["payload"] is None
        assert "abandoned" in (page.session_state["notice"] or "").lower()

    def test_input_is_disabled_while_an_approval_is_pending(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Every control that could abort the turn or move the thread under it.

        The export button and memory selectbox were live in the first version, which is
        what made the payload-replay bug reachable: clicking either mid-turn raises
        `RerunException` inside `_stream_turn`.

        **Both stubs are what give this test teeth**, and it passed without them for the
        wrong reason. On an empty thread `export_markdown` returns `""`, so the button is
        already `disabled=not document` and the assertion held even with the busy-gating
        removed — verified by removing it. The memory selectbox is not rendered at all
        unless the store has something in it. Each stub removes the unrelated condition
        that was satisfying the assertion, leaving `busy` as the only thing that can.
        """
        monkeypatch.setattr(cli_module, "export_markdown", lambda *a, **k: "# report")
        monkeypatch.setattr(
            webui, "cached_memory_files", lambda _a: [("/memories/x.md", "body")]
        )
        pending = [Interrupt(id="i1", value={"action_requests": [_WRITE]})]

        page = _page(pending=pending, feed=webui.StreamlitFeed())

        assert page.chat_input[0].disabled
        assert page.sidebar.text_input[0].disabled
        assert page.sidebar.download_button[0].disabled
        assert page.sidebar.selectbox[0].disabled

    def test_those_same_controls_are_live_when_idle(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # The positive control for the test above: with nothing pending, every one of
        # them is enabled. Without this, "disabled" assertions could all be passing
        # because the widgets are disabled unconditionally.
        monkeypatch.setattr(cli_module, "export_markdown", lambda *a, **k: "# report")
        monkeypatch.setattr(
            webui, "cached_memory_files", lambda _a: [("/memories/x.md", "body")]
        )

        page = _page()

        assert not page.chat_input[0].disabled
        assert not page.sidebar.text_input[0].disabled
        assert not page.sidebar.download_button[0].disabled
        assert not page.sidebar.selectbox[0].disabled


class TestTheThreadField:
    def test_emptying_it_warns_instead_of_silently_keeping_the_old_thread(self) -> None:
        # The widget has no key, so an emptied value persists across reruns while
        # `thread_id` keeps the old name: a blank box, and an app quietly still on the
        # previous thread with nothing to notice it by.
        page = _page(thread_id="main")
        page.sidebar.text_input[0].set_value("   ").run()

        assert page.session_state["thread_id"] == "main"
        assert any("needs a name" in w.value for w in page.warning)

    def test_switching_clears_the_previous_threads_display_state(self) -> None:
        # Work logs and refusal notes are indexed by position in the PREVIOUS thread's
        # transcript, so carrying them over would caption this thread's answers with
        # another one's.
        page = _page(thread_id="main", work_logs={0: ()}, refusals={0: "note"})
        page.sidebar.text_input[0].set_value("other").run()

        assert page.session_state["thread_id"] == "other"
        assert page.session_state["work_logs"] == {}
        assert page.session_state["refusals"] == {}


class TestQuestionDeduplication:
    """`should_render_question` — extracted from the page precisely so it can be tested.

    Each case below double-drew the question, for the entire length of an approval,
    under the original `text == asking` comparison.
    """

    def test_it_draws_before_the_checkpoint_has_the_question(self) -> None:
        assert webui.should_render_question("what is X?", [])

    def test_it_stops_once_the_checkpoint_has_it(self) -> None:
        assert not webui.should_render_question(
            "what is X?", [("human", "what is X?"), ("ai", "X is…")]
        )

    def test_trailing_whitespace_still_matches(self) -> None:
        # `thread_sections` strips every message; the raw chat_input value does not.
        assert not webui.should_render_question(
            "what is X?\n", [("human", "what is X?")]
        )

    def test_it_matches_inside_a_merged_human_section(self) -> None:
        # `thread_sections` joins consecutive same-speaker messages. A turn that produced
        # no assistant prose — what a classifier refusal looks like — merges the previous
        # question and this one into ONE section equal to neither.
        merged = [("human", "an earlier question\n\nwhat is X?")]

        assert not webui.should_render_question("what is X?", merged)

    def test_no_question_draws_nothing(self) -> None:
        assert not webui.should_render_question(None, [])
        assert not webui.should_render_question("", [("human", "x")])


class TestOneTurnPerThreadPerProcess:
    """`busy` is per SESSION; the checkpoint it protects is per process."""

    def test_a_second_claim_on_the_same_thread_is_refused(self) -> None:
        # Two browser tabs both default to thread `main`. Session state cannot see across
        # them, so without this both could start a turn and run two concurrent
        # `agent.stream()` calls against one checkpoint.
        with webui.claim_thread("main") as first:
            assert first
            with webui.claim_thread("main") as second:
                assert not second

    def test_a_different_thread_is_unaffected(self) -> None:
        with webui.claim_thread("main") as first:
            assert first
            with webui.claim_thread("other") as second:
                assert second

    def test_the_claim_is_released_even_when_the_turn_raises(self) -> None:
        # A turn that dies must not wedge its thread for the life of the process.
        with pytest.raises(RuntimeError), webui.claim_thread("main") as claimed:
            assert claimed
            raise RuntimeError("turn blew up")

        with webui.claim_thread("main") as again:
            assert again
