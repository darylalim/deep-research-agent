"""Unit tests for `ActivityFeed` and the interrupt-parsing helpers in `turns.py`.

The feed decides *what happened* on a stream and records it as `FeedEvent`s; the
renderer in `webui.py` only draws them. So these tests assert on the recorded events
directly — the exact data the page will draw — with no Streamlit in the way.

The interrupt helpers (`pending_reviews`, `allowed_decisions_by_tool`,
`_declined_tools`) are the parsing half of the approval protocol; `test_webui.py`
covers the widgets that present them.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

from langchain.agents.middleware.human_in_the_loop import HumanInTheLoopMiddleware
from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    RemoveMessage,
    ToolMessage,
)
from langgraph.graph.message import REMOVE_ALL_MESSAGES
from langgraph.types import Interrupt

from deep_research.turns import (
    _LS_EMPTY,
    DEFAULT_ALLOWED_DECISIONS,
    ActivityFeed,
    FeedEvent,
    _declined_tools,
    allowed_decisions_by_tool,
    pending_reviews,
)


def test_the_empty_ls_sentinel_still_matches_what_deepagents_returns() -> None:
    """Derive the sentinel from the installed package, never from this file's memory.

    `turns._LS_EMPTY` is a hand-copied literal — one private function's return value —
    so it is exactly the kind of fact that goes stale in silence on an upgrade. It
    already did once: 0.6 returned a `"[]"` repr for an empty listing, 0.7 returns a bare
    string, and until this was noticed an empty `/memories/` rendered `?` instead of
    `empty`.

    Calling the real `_format_file_paths` is what makes the next change loud instead. The
    same shape as `TestStopReasonsAreAccountedFor` comparing the stop table against the
    SDK: ask the dependency, do not re-read it by hand.
    """
    from deepagents.middleware.filesystem import _format_file_paths

    assert _format_file_paths([]) == _LS_EMPTY, (
        "deepagents changed what an empty `ls` listing looks like; turns._LS_EMPTY is "
        "now stale and an empty /memories/ will render `?` instead of `empty`."
    )
    # And the non-empty side is still a list repr, which is what `ast.literal_eval` parses.
    assert _format_file_paths(["/memories/a.md"]) == "['/memories/a.md']"


def test_default_decisions_match_the_middleware_expansion_of_true() -> None:
    # Every value in GATED_TOOLS is a bare `True`, which the middleware expands
    # into a concrete decision set. `DEFAULT_ALLOWED_DECISIONS` is the fallback for a
    # request that arrives with no ReviewConfig, so it has to be the *same* set — if a
    # langchain upgrade adds a fifth decision type, this goes red instead of the
    # approval form silently never offering it.
    middleware = HumanInTheLoopMiddleware(interrupt_on={"write_file": True})
    expanded = middleware.interrupt_on["write_file"]["allowed_decisions"]
    assert set(expanded) == set(DEFAULT_ALLOWED_DECISIONS)


def _updates(node: str, *messages: object) -> dict:
    """One `stream(stream_mode="updates")` chunk, as `evals/test_evals.py` builds them."""
    return {node: {"messages": list(messages)}}


def _kinds(feed: ActivityFeed) -> list[str]:
    return [event.kind for event in feed.events]


def _task_call(description: str, call_id: str = "t1") -> AIMessage:
    """An orchestrator message delegating one sub-question to a researcher."""
    return AIMessage(
        content="",
        id="ai-1",
        tool_calls=[
            {
                "name": "task",
                "args": {"description": description, "subagent_type": "researcher"},
                "id": call_id,
            }
        ],
    )


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


class TestPendingReviews:
    """One review per DISTINCT interrupt, however many times it was emitted.

    With `subgraphs=True` an interrupt raised inside a subagent is emitted TWICE — once
    at the subagent's namespace, once bubbled to the root — carrying the SAME
    `Interrupt.id`. Honouring both would ask the human to approve one researcher's
    `write_file` twice, and only the second answer would survive (the resume mapping is
    keyed by id). Approval fatigue is how a gate stops being a gate.
    """

    def test_the_same_interrupt_twice_is_one_review(self) -> None:
        assert pending_reviews([PENDING_WRITE, PENDING_WRITE]) == [
            ("i1", PENDING_WRITE.value)
        ]

    def test_distinct_interrupts_are_each_kept_in_order(self) -> None:
        # The dedupe keys on the id rather than collapsing everything: two researchers
        # each proposing their own write are two real decisions. Kept GROUPED by
        # interrupt, because LangGraph refuses a flat resume value once a turn holds two
        # ("When there are multiple pending interrupts, you must specify the interrupt id
        # when resuming").
        other = Interrupt(
            id="i2", value={"action_requests": [{"name": "write_file", "args": {}}]}
        )
        assert [i for i, _ in pending_reviews([PENDING_WRITE, other])] == ["i1", "i2"]

    def test_an_interrupt_that_is_not_a_review_is_dropped_not_guessed_at(self) -> None:
        # `Interrupt` types `id` as `str`, so an id-less one needs a stand-in; the
        # helper reads both fields with `getattr`, which is the shape under test.
        no_id = SimpleNamespace(id=None, value={"action_requests": []})
        not_a_mapping = Interrupt(id="i3", value="free text")
        assert pending_reviews([no_id, not_a_mapping]) == []


class TestAllowedDecisionsByTool:
    def test_review_config_is_matched_by_name_not_by_position(self) -> None:
        # The configs are deliberately in the OPPOSITE order to the requests. A
        # positional lookup would hand `execute` write_file's permissive menu — and the
        # middleware raises `ValueError` on any decision outside a tool's
        # `allowed_decisions`, which would surface as a dead turn.
        value = {
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
        }
        assert allowed_decisions_by_tool(value) == {
            "write_file": ["approve", "edit", "reject"],
            "execute": ["approve"],
        }

    def test_an_interrupt_without_review_configs_falls_back_to_the_default(
        self,
    ) -> None:
        # No entry means "no ReviewConfig came with this request", which the caller
        # resolves to `DEFAULT_ALLOWED_DECISIONS` — distinct from an EMPTY list, which
        # would mean nothing is permitted.
        assert allowed_decisions_by_tool({"action_requests": []}) == {}


class TestDeclinedTools:
    def test_a_rejection_is_mapped_back_to_its_tool_by_position(self) -> None:
        value = {
            "action_requests": [
                {"name": "write_file", "args": {}},
                {"name": "edit_file", "args": {}},
            ]
        }
        interrupt = Interrupt(id="i1", value=value)
        decided = {"i1": [{"type": "approve"}, {"type": "reject", "message": "no"}]}
        assert _declined_tools([interrupt], decided) == {"edit_file"}

    def test_an_approval_declines_nothing(self) -> None:
        assert _declined_tools([PENDING_WRITE], {"i1": [{"type": "approve"}]}) == set()


class TestActivityFeed:
    """The feed records ACTIONS. Never prose — a researcher's words are not the agent's."""

    def test_it_records_the_plan_the_delegations_and_the_queries(self) -> None:
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
        feed.absorb((), _updates("model", _task_call("pricing for Opus 4.8")))
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

        assert feed.events == [
            FeedEvent("plan", items=("pricing for Opus 4.8", "rate limits by tier")),
            FeedEvent("delegate", text="pricing for Opus 4.8"),
            # A subagent's search is marked as one, so the renderer can set it apart.
            FeedEvent("search", text="anthropic opus 4.8 price", is_orchestrator=False),
            # The completion recovers the description via `tool_call_id` — the one
            # honest way to name a researcher, since its stream namespace cannot be
            # bound back to the task call that spawned it.
            FeedEvent("done", text="pricing for Opus 4.8"),
        ]

    def test_it_never_records_a_researchers_prose(self) -> None:
        # THE rule. The stream carries the researchers' own assistant messages, which
        # the user must never see — `evals/harness.py` refuses to build its graded
        # `response` from the stream for exactly this reason. Recording them here would
        # show a subagent's cited paragraphs as if they were the agent's answer, and
        # would show the SAME report twice (once from the stream, once from the
        # checkpoint).
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
        assert feed.events == []

    def test_a_reemitted_call_is_not_recorded_twice(self) -> None:
        # On resume, HumanInTheLoopMiddleware re-emits the AIMessage that proposed the
        # gated call. Without dedupe the user watches lines they have already seen
        # appear again, once per approval round.
        call = _task_call("d")
        feed = ActivityFeed()
        feed.absorb((), _updates("model", call))
        feed.absorb((), _updates("HumanInTheLoopMiddleware.after_model", call))

        assert _kinds(feed) == ["delegate"]

    def test_a_replayed_tool_result_without_a_message_id_is_not_recorded_twice(
        self,
    ) -> None:
        # THE reason the seen-set is keyed on the CALL id and not the message id. A
        # re-streamed superstep re-emits the cached writes of tasks that already
        # finished, and `BaseMessage.id` is optional — a ToolMessage carrying no id
        # defeats message-id dedupe entirely. Caught by driving the real loop: the
        # completion line for a finished researcher appeared a second time, after the
        # approval. A tool call executes exactly once, so its id is the honest key.
        feed = ActivityFeed()
        feed.absorb((), _updates("model", _task_call("pricing")))
        # Two DISTINCT ToolMessage objects, neither carrying an `id` — exactly what a
        # replayed superstep hands you.
        for _ in range(2):
            feed.absorb(
                (),
                _updates(
                    "tools", ToolMessage("summary", tool_call_id="t1", name="task")
                ),
            )

        assert _kinds(feed) == ["delegate", "done"]

    def test_the_ls_line_reads_the_list_repr_not_the_line_count(self) -> None:
        # deepagents builds the `ls` body on ONE line — not newline-separated entries.
        # Counting lines reported "1 file(s)" for an EMPTY store every single time, and
        # made the "empty" branch unreachable. This line is how the user sees whether
        # durable memory was consulted; one that lies is worse than no line at all.
        #
        # BOTH empty shapes are covered deliberately: `"[]"` is the 0.6 list repr and
        # `"No files found"` is 0.7's bare sentinel. Testing only the first is how the
        # sentinel went unhandled while this test stayed green — an empty /memories/
        # rendered `?`, and nothing said so.
        cases = (
            ("[]", "empty"),
            ("No files found", "empty"),
            ("['/memories/a.md', '/b.md']", "2 file(s)"),
            ("something unexpected", "?"),
        )
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
            assert feed.events == [
                FeedEvent("listed", text="/memories/", detail=expected)
            ], body

    def test_a_researchers_own_todos_and_ls_are_not_recorded_as_the_agents(
        self,
    ) -> None:
        # deepagents gives EVERY declarative subagent its own FilesystemMiddleware, so a
        # `researcher` really can call `ls` regardless of what `subagents.py` lists in its
        # `tools`. The `write_todos` half is defence in depth as of 0.7.x, which stopped
        # giving subagents a `TodoListMiddleware`; the guard keys on the namespace, not on
        # the tool, so it covers both without caring which is live. Recording those
        # namespace-blind would:
        #   - show a researcher's private checklist as a second plan, appearing to
        #     supersede the plan the user was just shown;
        #   - show `/memories/` as checked on a turn where the ORCHESTRATOR never looked,
        #     hiding the very "the direct path skips /memories/" defect CLAUDE.md says to
        #     watch.
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
        assert feed.events == []

    def test_a_failed_tool_is_surfaced(self) -> None:
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
        assert feed.events == [
            FeedEvent(
                "failed", text="tavily_search", detail="no results for that query"
            )
        ]

    def test_a_rejected_write_is_not_reported_as_a_tool_failure(self) -> None:
        # `HumanInTheLoopMiddleware` answers a REJECTED call with a synthetic ToolMessage
        # carrying `status="error"` — and if the human gave a reason, that reason becomes
        # its content. So the feed cannot tell a rejection from a crash, and reported
        # "write_file failed: too risky" to the person who had just clicked Reject,
        # presenting their own honoured decision as a bug in the agent. The page tells
        # the feed via `note_declined(_declined_tools(...))`; this is that path.
        feed = ActivityFeed()
        feed.note_declined(
            _declined_tools(
                [PENDING_WRITE, PENDING_WRITE],
                {"i1": [{"type": "reject", "message": "too risky"}]},
            )
        )
        feed.absorb(
            (),
            _updates(
                "tools",
                ToolMessage(
                    "too risky", tool_call_id="w1", name="write_file", status="error"
                ),
            ),
        )
        assert feed.events == [FeedEvent("rejected", text="write_file")]

    def test_a_thread_rewrite_does_not_replay_the_previous_turns_feed(self) -> None:
        # `PatchToolCallsMiddleware.before_agent` answers dangling tool calls by returning
        # `{"messages": [RemoveMessage(REMOVE_ALL_MESSAGES), *THE ENTIRE THREAD]}`. It
        # fires on exactly the turn AFTER one abandoned at an approval — and the feed is
        # per-turn, so its seen-set has never heard of those calls. Without a guard, that
        # turn opens by replaying the previous turn's whole feed: its plan, its
        # delegations, every search it ran.
        feed = ActivityFeed()
        feed.absorb(
            (),
            {
                "PatchToolCallsMiddleware.before_agent": {
                    "messages": [
                        RemoveMessage(id=REMOVE_ALL_MESSAGES),
                        HumanMessage("the previous question"),
                        _task_call("last turn's sub-question", call_id="old-t1"),
                    ]
                }
            },
        )
        assert feed.events == []

    def test_it_returns_the_interrupts_it_sees(self) -> None:
        # The feed is also the interrupt collector, exactly like harness.TurnRecorder —
        # one traversal, not two.
        feed = ActivityFeed()
        assert feed.absorb((), {"__interrupt__": (PENDING_WRITE,)}) == [PENDING_WRITE]


def _stopped(reason: str) -> AIMessage:
    return AIMessage(content="", response_metadata={"stop_reason": reason})


REFUSAL = _stopped("refusal")
OVERRUN = _stopped("model_context_window_exceeded")


class TestAResearchersSilentStopIsRecorded:
    """A researcher's refusal or context overrun must not be invisible.

    Either one is HTTP 200 with empty content: nothing raises, the subagent's turn ends,
    its `task` result comes back thin, and the orchestrator synthesizes around the hole.
    The user sees a delegation dispatched, a delegation completed, and a shorter answer
    than they asked for, with nothing anywhere saying why.
    """

    def _refusals(self, feed: ActivityFeed) -> list[Any]:
        return [event.text for event in feed.events if event.kind == "refusal"]

    def test_a_researchers_refusal_is_recorded(self) -> None:
        feed = ActivityFeed()
        feed.absorb(SUBAGENT_NS, _updates("model", REFUSAL))
        assert self._refusals(feed) == ["the model declined this request"]

    def test_a_researchers_overrun_is_recorded_and_not_called_a_refusal(self) -> None:
        # A researcher carries its own isolated context, so it can overrun independently
        # of the orchestrator.
        feed = ActivityFeed()
        feed.absorb(SUBAGENT_NS, _updates("model", OVERRUN))
        assert self._refusals(feed) == ["this turn exceeded the model's context window"]

    def test_the_orchestrators_stop_is_not_recorded_twice(self) -> None:
        # The page reports it from the checkpoint (`_turn_stop`), right where the
        # missing answer would have been — so the feed must stay quiet at the root.
        feed = ActivityFeed()
        feed.absorb((), _updates("model", REFUSAL))
        assert feed.events == []

    def test_one_researcher_refusing_is_recorded_once(self) -> None:
        # Same replay hazard as every other feed line: a resumed superstep re-emits the
        # cached writes of siblings that already succeeded. The usual call-id key is
        # unavailable (a refusal carries no tool call) and `BaseMessage.id` is the
        # unreliable key this class already learned not to trust — `None` on the first
        # pass, a fresh uuid on the resume — so the namespace is the key.
        feed = ActivityFeed()
        feed.absorb(SUBAGENT_NS, _updates("model", REFUSAL))
        feed.absorb(SUBAGENT_NS, _updates("model", REFUSAL))
        assert len(self._refusals(feed)) == 1

    def test_two_researchers_refusing_are_both_recorded(self) -> None:
        # The dedupe must not collapse distinct subagents: two researchers fanned out in
        # one turn are two separate refusals and two separate gaps in the answer.
        feed = ActivityFeed()
        feed.absorb(SUBAGENT_NS, _updates("model", REFUSAL))
        feed.absorb(("tools:0a1b2c3d",), _updates("model", REFUSAL))
        assert len(self._refusals(feed)) == 2

    def test_an_ordinary_researcher_message_records_nothing(self) -> None:
        # The feed records ACTIONS, never prose — this must not become the exception
        # that starts leaking a subagent's words to the user.
        feed = ActivityFeed()
        feed.absorb(SUBAGENT_NS, _updates("model", AIMessage("my cited findings")))
        assert feed.events == []
