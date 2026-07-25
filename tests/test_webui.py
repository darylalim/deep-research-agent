"""Unit tests for the Streamlit front door (`deep_research/webui.py`).

Two kinds of test here, and they exist for different reasons.

**Parity tests.** `webui` earns its keep by *not* reimplementing `cli.ActivityFeed` —
it subclasses it and overrides `_emit` alone. So the tests worth writing are the ones
that go red if that reuse is ever quietly undone: a researcher's prose reaching the
feed, a replayed tool call counted twice, a subagent's `ls` shown as the orchestrator's.
Each of these is a bug this repo has already paid for once, in `cli.py` or in
`evals/harness.TurnRecorder`.

**Widget tests**, driven by `streamlit.testing.v1.AppTest`, which runs a script headless
and exposes the elements it produced. That is the only way to assert on the thing that
actually matters about the approval UI — that it offers exactly the decisions the
interrupt permits, that it refuses to submit a half-made decision, and that one
interrupt emitted twice yields ONE set of controls.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
from langchain_core.messages import AIMessage, RemoveMessage, ToolMessage
from langgraph.graph.message import REMOVE_ALL_MESSAGES
from langgraph.types import Interrupt

# Skips cleanly when the optional `ui` dependency group is not installed. CI syncs
# `--group ui` precisely so this never actually skips there.
pytest.importorskip("streamlit")

from streamlit.testing.v1 import AppTest

from deep_research import webui
from deep_research.cli import FEED_KINDS, FeedEvent

SUBAGENT_NS: tuple[str, ...] = ("tools:9d0c2f4e",)


def _updates(node: str, *messages: object) -> dict:
    """One `stream(stream_mode="updates")` chunk, shaped as the real stream builds them."""
    return {node: {"messages": list(messages)}}


def _delegation(call_id: str = "t1", description: str = "pricing") -> dict:
    return _updates(
        "model",
        AIMessage(
            content="",
            id="ai-1",
            tool_calls=[
                {
                    "name": "task",
                    "args": {"description": description, "subagent_type": "researcher"},
                    "id": call_id,
                }
            ],
        ),
    )


@pytest.fixture
def feed(monkeypatch: pytest.MonkeyPatch) -> webui.StreamlitFeed:
    """A `StreamlitFeed` whose rendering is stubbed, so `_emit` runs for real.

    Only `render_event` is replaced — `_emit` still appends to `self.events` and still
    calls it — so what is under test is the real subclass, not a convenient double.
    """
    monkeypatch.setattr(webui, "render_event", lambda event: None)
    return webui.StreamlitFeed()


class TestStreamlitFeedInheritsEveryRule:
    """`StreamlitFeed` changes rendering and NOTHING else. These pin "nothing else"."""

    def test_it_records_the_plan_the_delegation_and_the_query(
        self, feed: webui.StreamlitFeed
    ) -> None:
        feed.absorb((), {"tools": {"todos": [{"content": "pricing", "status": "x"}]}})
        feed.absorb((), _delegation())
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
                            "args": {"query": "opus 5 price"},
                            "id": "s1",
                        }
                    ],
                ),
            ),
        )

        kinds = [event.kind for event in feed.events]
        assert kinds == ["plan", "delegate", "search"]
        assert feed.events[0].items == ("pricing",)
        assert feed.events[1].text == "pricing"
        assert feed.events[2].text == "opus 5 price"
        # The one that decides indentation/dimming — a subagent's search is not the
        # orchestrator's, and the renderer needs to be able to tell.
        assert feed.events[2].is_orchestrator is False

    def test_it_never_records_a_researchers_prose(
        self, feed: webui.StreamlitFeed
    ) -> None:
        # THE rule. The stream carries the researchers' own assistant messages, and the
        # user must never see one: the answer comes from the checkpoint via
        # `thread_sections`, which is what `evals/harness.py` grades. A feed that leaked
        # prose would show a subagent's cited paragraphs as the agent's answer, twice.
        feed.absorb(
            SUBAGENT_NS,
            _updates("model", AIMessage(content="Opus is $15/Mtok ([x](u))", id="a")),
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

        rendered = " ".join(f"{e.text} {e.detail}" for e in feed.events)
        assert "Opus is $15/Mtok" not in rendered
        assert "https://x.test" not in rendered

    def test_a_replayed_tool_result_is_recorded_once(
        self, feed: webui.StreamlitFeed
    ) -> None:
        # Dedupe is on the TOOL-CALL id, not `BaseMessage.id`. A resumed superstep
        # re-emits the cached writes of siblings that already succeeded, as FRESH
        # ToolMessage objects — `id=None` on the first pass, a new uuid on the resume —
        # so a message-id seen-set matches neither and lets every duplicate through.
        # In the browser this matters more than in the terminal: `replay()` redraws the
        # whole list on every rerun, so one duplicate becomes one duplicate per approval.
        feed.absorb((), _delegation())
        for _ in range(2):
            feed.absorb(
                (),
                _updates(
                    "tools", ToolMessage("summary", tool_call_id="t1", name="task")
                ),
            )

        assert [e.kind for e in feed.events].count("done") == 1

    def test_a_researchers_own_todos_and_ls_are_not_shown_as_the_agents(
        self, feed: webui.StreamlitFeed
    ) -> None:
        # deepagents gives every declarative subagent its own TodoListMiddleware and
        # FilesystemMiddleware, so a `researcher` really does call `write_todos` and `ls`.
        # Rendering those namespace-blind would print a researcher's private checklist as
        # the agent's plan, and claim durable memory was consulted on a turn where the
        # orchestrator never looked — hiding the exact direct-path defect CLAUDE.md says
        # to keep watching.
        feed.absorb(SUBAGENT_NS, {"tools": {"todos": [{"content": "secret"}]}})
        feed.absorb(
            SUBAGENT_NS,
            _updates("tools", ToolMessage("[]", tool_call_id="l1", name="ls")),
        )

        assert feed.events == []

    def test_a_thread_rewrite_is_not_replayed_as_new_activity(
        self, feed: webui.StreamlitFeed
    ) -> None:
        # `PatchToolCallsMiddleware.before_agent` answers dangling tool calls by returning
        # RemoveMessage(REMOVE_ALL_MESSAGES) followed by the ENTIRE thread — and it fires
        # on exactly the turn after one abandoned at an approval prompt. Without the
        # guard, that turn opens by replaying the previous turn's whole feed.
        feed.absorb(
            (),
            _updates(
                "PatchToolCallsMiddleware.before_agent",
                RemoveMessage(id=REMOVE_ALL_MESSAGES),
                AIMessage(
                    content="",
                    id="old",
                    tool_calls=[
                        {"name": "task", "args": {"description": "old"}, "id": "t9"}
                    ],
                ),
            ),
        )

        assert feed.events == []

    def test_it_still_returns_the_interrupts_it_sees(
        self, feed: webui.StreamlitFeed
    ) -> None:
        # The page's whole turn loop hangs off this return value.
        interrupt = Interrupt(id="i1", value={"action_requests": []})
        assert feed.absorb((), {"__interrupt__": (interrupt,)}) == [interrupt]

    def test_replay_redraws_every_event_it_collected(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # The reason the events are kept at all: an approval costs at least two reruns,
        # and a Streamlit rerun discards everything previously drawn.
        drawn: list[str] = []
        monkeypatch.setattr(
            webui, "render_event", lambda event: drawn.append(event.kind)
        )
        live = webui.StreamlitFeed()
        live.absorb((), _delegation())
        drawn.clear()

        live.replay()

        assert drawn == ["delegate"]


@pytest.mark.parametrize("kind", FEED_KINDS)
def test_every_feed_kind_is_rendered_by_both_front_ends(
    kind: str, capsys: pytest.CaptureFixture[str]
) -> None:
    """Both renderers must handle every `FEED_KINDS` entry.

    Each is an if/elif chain that draws NOTHING for a kind it does not know, so an event
    kind wired into only one front end is invisible rather than broken — no exception,
    no failing test, just a line that silently stops appearing in the other.

    **Parametrized per kind on purpose.** This started life as one test asserting
    `len(markdown) >= len(FEED_KINDS)` over every kind at once, and deleting a branch
    from `render_event` still passed it: `plan` draws two elements, so the totals had
    exactly one element of slack and it absorbed exactly one deletion. An aggregate
    assertion over things that are not one-to-one cannot detect a single omission. One
    kind per run, asserting that kind drew *something*, does — verified by deleting a
    branch from each renderer in turn.
    """
    from deep_research.cli import ActivityFeed

    ActivityFeed()._emit(FeedEvent(kind, text="t", detail="d", items=("i",)))
    assert capsys.readouterr().out.strip(), f"the terminal drew nothing for {kind!r}"

    script = AppTest.from_string(
        "from deep_research import webui\n"
        "from deep_research.cli import FeedEvent\n"
        f"webui.render_event(FeedEvent({kind!r}, text='t', detail='d', items=('i',)))\n"
    )
    script.run()

    assert not script.exception, script.exception
    assert script.markdown, f"the browser drew nothing for {kind!r}"


class TestMemoryFiles:
    """The `/memories/` browser. Reads the Store directly — no model turn, no approval."""

    @staticmethod
    def _agent(*items: Any) -> Any:
        return SimpleNamespace(store=SimpleNamespace(search=lambda *_a, **_k: items))

    def test_the_route_prefix_is_put_back_on_the_stored_key(self) -> None:
        # `CompositeBackend` STRIPS the route prefix before delegating, so a note the
        # agent wrote to `/memories/pricing.md` is stored under the key `/pricing.md`.
        # Displaying the raw key would show the user a path that does not exist — in the
        # one view whose entire subject is where findings are kept.
        agent = self._agent(
            SimpleNamespace(key="/pricing.md", value={"content": "# Pricing"})
        )

        assert webui.memory_files(agent) == [("/memories/pricing.md", "# Pricing")]

    def test_the_legacy_list_content_shape_is_joined(self) -> None:
        # deepagents writes `content` as a plain string, but as a `list[str]` under its
        # legacy `file_format="v1"`. A store written by an older version still has to read.
        agent = self._agent(
            SimpleNamespace(key="/a.md", value={"content": ["line one", "line two"]})
        )

        assert webui.memory_files(agent) == [("/memories/a.md", "line one\nline two")]

    def test_an_agent_without_a_store_yields_nothing_rather_than_raising(self) -> None:
        # The served graph gets its store injected at runtime; a compiled agent with none
        # must not take the sidebar down with it.
        assert webui.memory_files(SimpleNamespace(store=None)) == []


# --- approval widgets -----------------------------------------------------------------

_WRITE = {
    "name": "write_file",
    "args": {"file_path": "/memories/pricing.md", "content": "# Pricing\n$5/Mtok"},
}

_FORM_SCRIPT = """
import streamlit as st
from deep_research import webui
st.session_state["decisions"] = webui.approval_form(st.session_state["pending"])
"""


def _form(*interrupts: Interrupt) -> AppTest:
    """Render the approval form over `interrupts`, asserting the page did not crash.

    The exception check is not boilerplate — it is what makes the dedupe test bite.
    Without it, dropping the deduplication in `pending_reviews` still *passes* a
    "one set of controls" assertion, because the second render raises
    `StreamlitDuplicateElementKey` (both occurrences produce the widget key
    `i1:0:type`) and never adds its element. The page is broken and the count is
    still one. Measured, by breaking it.
    """
    script = AppTest.from_string(_FORM_SCRIPT)
    script.session_state["pending"] = list(interrupts)
    script.run()
    assert not script.exception, script.exception
    return script


def test_the_same_interrupt_emitted_twice_renders_one_set_of_controls() -> None:
    """One prompt per pending action, however many times the chunk arrives.

    With `subgraphs=True` an interrupt raised inside a subagent is emitted at the
    subagent's namespace AND again, bubbled, at the root — same `Interrupt.id`. Rendering
    per occurrence would ask the reviewer to approve one researcher's `write_file` twice
    and, since the resume mapping is keyed by id, silently keep only the second answer.
    Approval fatigue is exactly how a gate stops being a gate.
    """
    pending = Interrupt(id="i1", value={"action_requests": [_WRITE]})

    script = _form(pending, pending)

    assert len(script.button_group) == 1


def test_two_distinct_interrupts_each_get_their_own_controls() -> None:
    # Two researchers fanned out in one turn each raise their own interrupt, and both
    # need deciding — the dedupe above must key on the id, not collapse by tool name.
    script = _form(
        Interrupt(id="i1", value={"action_requests": [_WRITE]}),
        Interrupt(id="i2", value={"action_requests": [_WRITE]}),
    )

    assert len(script.button_group) == 2


def test_nothing_is_submittable_until_every_action_is_decided() -> None:
    # A resume carrying fewer decisions than the interrupt has action requests is not a
    # smaller approval, it is an undefined one.
    script = _form(Interrupt(id="i1", value={"action_requests": [_WRITE, _WRITE]}))

    assert script.button[0].disabled
    assert script.session_state["decisions"] is None

    script.button_group[0].set_value("approve").run()

    assert script.button[0].disabled, "one of two actions decided — still not ready"


def test_approving_every_action_yields_a_mapping_keyed_by_interrupt_id() -> None:
    # The shape LangGraph demands: `Command(resume={interrupt_id: {"decisions": [...]}})`.
    # A flat `{"decisions": [...]}` raises RuntimeError as soon as a turn holds two
    # interrupts, which a two-researcher turn routinely does.
    script = _form(Interrupt(id="i1", value={"action_requests": [_WRITE]}))
    script.button_group[0].set_value("approve").run()

    assert not script.button[0].disabled
    script.button[0].click().run()

    assert script.session_state["decisions"] == {"i1": [{"type": "approve"}]}


def test_only_the_decisions_the_tool_permits_are_offered() -> None:
    # The middleware raises `ValueError` on a decision outside a tool's
    # `allowed_decisions`, and that would surface as a dead turn. Every GATED_TOOLS value
    # is `True` today, so this is invisible until someone narrows one with an
    # InterruptOnConfig — at which point a hardcoded menu breaks the turn.
    script = _form(
        Interrupt(
            id="i1",
            value={
                "action_requests": [_WRITE],
                "review_configs": [
                    {
                        "action_name": "write_file",
                        "allowed_decisions": ["approve", "reject"],
                    }
                ],
            },
        )
    )

    assert list(script.button_group[0].options) == ["Approve", "Reject"]


def test_no_decision_is_preselected() -> None:
    # Deliberately unlike the REPL, which defaults to approve because bare Enter has to
    # mean something. A UI has no such affordance, so an explicit click costs the reviewer
    # nothing and removes the one path by which a gate degrades into a rubber stamp.
    script = _form(Interrupt(id="i1", value={"action_requests": [_WRITE]}))

    assert script.session_state["decisions"] is None
    assert script.button[0].disabled
    assert any("Choose a decision" in caption.value for caption in script.caption)


def test_unparsable_edit_arguments_never_fall_back_to_approving_the_original() -> None:
    """A typo is not consent — this is the only security boundary the app has.

    The REPL learned this the hard way: it used to return `approve` with the ORIGINAL,
    unedited arguments whenever a mistyped edit failed to parse, so a reviewer who chose
    `edit` precisely because the write looked wrong, and then fat-fingered the JSON,
    silently approved the very write they were trying to narrow.
    """
    script = _form(Interrupt(id="i1", value={"action_requests": [_WRITE]}))
    script.button_group[0].set_value("edit").run()
    script.text_area[0].set_value("{not json").run()

    assert script.session_state["decisions"] is None
    assert script.button[0].disabled
    assert script.error

    # And valid JSON that is not an OBJECT is refused too: `"/memories/y.md"` parses, but
    # would sail through into a ToolCall with non-dict args and only blow up at execution.
    script.text_area[0].set_value('"/memories/y.md"').run()

    assert script.session_state["decisions"] is None
    assert script.button[0].disabled


def test_the_edit_box_is_prefilled_with_the_real_arguments() -> None:
    # So narrowing a path is an edit rather than a retype. The CLI starts blank because a
    # terminal cannot prefill a line; a browser can, and a reviewer who has to retype the
    # whole args dict will approve as-is instead.
    script = _form(Interrupt(id="i1", value={"action_requests": [_WRITE]}))
    script.button_group[0].set_value("edit").run()

    assert "/memories/pricing.md" in (script.text_area[0].value or "")


def test_the_full_file_body_is_shown_rather_than_elided() -> None:
    """The gate exists so a human READS the content; the browser has no reason to clip it.

    The terminal caps its preview at `PREVIEW_LINES` because a scrollback buffer is a poor
    place to dump a report. `st.code` scrolls, so the whole body goes in — an elided
    review is a review that gets rubber-stamped, and `/memories/` is gitignored, so
    nothing undoes a bad write.
    """
    body = "\n".join(f"line {n}" for n in range(200))
    script = _form(
        Interrupt(
            id="i1",
            value={"action_requests": [{"name": "write_file", "args": {"c": body}}]},
        )
    )

    assert any(block.value == body for block in script.code)


def test_a_pause_with_no_reviewable_action_is_detected_rather_than_resumed() -> None:
    # Resuming an interrupt with no action requests just re-interrupts; the page abandons
    # the turn instead of looping. Counted over DISTINCT interrupts, hence the duplicate.
    empty = Interrupt(id="i1", value={"action_requests": []})
    assert webui.reviewable_actions([empty, empty]) == 0
    assert (
        webui.reviewable_actions(
            [Interrupt(id="i2", value={"action_requests": [_WRITE]})]
        )
        == 1
    )
