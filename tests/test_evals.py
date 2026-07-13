"""Offline tests for the eval harness.

The harness makes two claims that are expensive to be wrong about, and both are
pinned here with real langchain/langgraph types rather than fakes:

- it records tool calls made *inside a subagent* — the ones `agent.invoke()`'s
  returned state cannot see, which is the entire reason the harness streams
- it approves interrupts with an *id-keyed* resume mapping, which is what
  LangGraph demands as soon as a turn holds more than one interrupt

Everything here runs without keys or network.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.types import Interrupt

from deep_research.cli import render_turn
from deep_research.config import CHECKPOINT_DB, MEMORY_DB, STATE_DIR, ensure_state_dir
from evals.evaluators import (
    _coverage_score,
    checks_memory_first,
    delegates_breadth,
    persists_findings,
    plans_with_todos,
    response_cites_sources,
    searched_the_web,
)
from evals.harness import (
    LIVE_STATE_DIR,
    TurnRecorder,
    _approve_all,
    _reset_state,
    ensure_isolated_state_dir,
)

ORCHESTRATOR: tuple[str, ...] = ()
# What a subagent's subgraph namespace actually looks like (measured).
SUBAGENT: tuple[str, ...] = ("tools:9d0c2f4e",)


def _updates(node: str, *messages: object) -> dict:
    """One `stream(stream_mode="updates")` chunk."""
    return {node: {"messages": list(messages)}}


def test_recorder_sees_searches_that_the_returned_state_cannot():
    """The whole point of streaming with subgraphs=True.

    On a real two-part question the orchestrator's messages contain
    ['ls', 'task', 'task', 'write_file'] and *zero* searches — every
    `tavily_search` runs inside a researcher subagent. If this ever regresses to
    reading the final state, `searched_the_web` would score 0 on a run that in
    fact searched four times.
    """
    recorder = TurnRecorder()
    recorder.absorb(
        ORCHESTRATOR, _updates("tools", ToolMessage("[]", tool_call_id="1", name="ls"))
    )
    recorder.absorb(
        SUBAGENT,
        _updates("tools", ToolMessage("hits", tool_call_id="2", name="tavily_search")),
    )
    recorder.absorb(
        SUBAGENT,
        _updates("tools", ToolMessage("hits", tool_call_id="3", name="tavily_search")),
    )
    recorder.absorb(
        ORCHESTRATOR,
        _updates("tools", ToolMessage("summary", tool_call_id="4", name="task")),
    )

    outputs = recorder.actions()
    assert outputs["trajectory"] == ["ls", "tavily_search", "tavily_search", "task"]
    # Attributed to the subagent, not the orchestrator.
    assert outputs["subagent_tools"] == ["tavily_search", "tavily_search"]
    assert searched_the_web({"outputs": outputs}, {})["score"] == 1


def test_recorder_does_not_double_count_the_reemitted_gated_call():
    """On resume, HumanInTheLoopMiddleware re-emits the AIMessage that proposed the
    gated tool call (measured). Without id-dedupe, one approved write is recorded
    as two proposals."""
    proposal = AIMessage(
        content="",
        id="ai-1",
        tool_calls=[
            {
                "name": "write_file",
                "args": {"file_path": "/memories/topic.md", "content": "x"},
                "id": "call-1",
            }
        ],
    )
    recorder = TurnRecorder()
    recorder.absorb(ORCHESTRATOR, _updates("model", proposal))
    # …the same message, again, from the middleware node after approval.
    recorder.absorb(
        ORCHESTRATOR, _updates("HumanInTheLoopMiddleware.after_model", proposal)
    )
    recorder.absorb(
        ORCHESTRATOR,
        _updates("tools", ToolMessage("ok", tool_call_id="call-1", name="write_file")),
    )

    outputs = recorder.actions()
    assert outputs["proposed_writes"] == ["/memories/topic.md"]
    assert outputs["trajectory"] == ["write_file"]
    assert persists_findings({"outputs": outputs}, {})["score"] == 1


def test_a_subagents_citations_do_not_earn_the_orchestrator_a_pass():
    """Why the recorder collects actions but not prose.

    The stream carries assistant messages from *inside* the researchers, and the
    user never sees one of them. If the harness built its transcript from the
    stream, a researcher's neatly-cited summary would score the orchestrator's
    uncited sign-off as a pass. So `response` is rendered by `cli.render_turn` from
    the orchestrator's final state — exactly the bytes the REPL prints.
    """
    recorder = TurnRecorder()
    recorder.absorb(
        SUBAGENT,
        _updates(
            "model", AIMessage("Tavily gives 1,000/mo https://tavily.example", id="s1")
        ),
    )
    assert "response" not in recorder.actions()  # no prose escapes the recorder

    visible = render_turn(
        {
            "messages": [
                HumanMessage("q"),
                AIMessage("Saved to memory. See summary above."),
            ]
        }
    )
    graded = response_cites_sources({"outputs": {"response": visible}}, {})
    assert graded["score"] == 0
    assert "no source URLs" in graded["comment"]


def test_approve_all_is_keyed_by_interrupt_id_with_one_decision_per_request():
    """Two fanned-out researchers can each raise an interrupt in one turn, and
    LangGraph refuses a flat resume when more than one is pending."""
    interrupts = [
        Interrupt(
            value={
                "action_requests": [
                    {"name": "write_file", "args": {}},
                    {"name": "execute", "args": {}},
                ]
            },
            id="i1",
        ),
        Interrupt(
            value={"action_requests": [{"name": "write_file", "args": {}}]}, id="i2"
        ),
    ]
    resume = _approve_all(interrupts)
    assert resume == {
        "i1": {"decisions": [{"type": "approve"}, {"type": "approve"}]},
        "i2": {"decisions": [{"type": "approve"}]},
    }


def test_recorder_records_gated_tools_from_an_interrupt():
    recorder = TurnRecorder()
    pending = recorder.absorb(
        ORCHESTRATOR,
        {
            "__interrupt__": (
                Interrupt(value={"action_requests": [{"name": "write_file"}]}, id="i1"),
            )
        },
    )
    assert [i.id for i in pending] == ["i1"]
    assert recorder.actions()["gated_tools"] == ["write_file"]


def test_plans_with_todos_fails_when_the_agent_skipped_planning():
    """Not a hypothetical: on two measured runs the agent went straight to `ls` →
    `task` without ever calling `write_todos`, despite SYSTEM_PROMPT step 1."""
    skipped = {
        "outputs": {"orchestrator_trajectory": ["ls", "task", "task", "write_file"]}
    }
    planned = {
        "outputs": {
            "orchestrator_trajectory": ["write_todos", "ls", "task", "write_file"]
        }
    }
    assert plans_with_todos(skipped, {})["score"] == 0
    assert plans_with_todos(planned, {})["score"] == 1

    # …but a single quick lookup is explicitly exempt in SYSTEM_PROMPT, so the bar
    # comes from the example. Without this, the fix for the missing-plan defect
    # would just teach the agent to over-plan trivia.
    solo = {"outputs": {"orchestrator_trajectory": ["tavily_search"]}}
    assert plans_with_todos(solo, {"outputs": {"expects_plan": False}})["score"] == 1
    assert plans_with_todos(solo, {"outputs": {"expects_plan": True}})["score"] == 0


def test_delegates_breadth_takes_its_bar_from_the_example():
    """A single quick lookup is *supposed* to skip delegation, so the expectation
    is per-example, not global."""
    solo = {"outputs": {"orchestrator_trajectory": ["tavily_search"]}}
    assert delegates_breadth(solo, {"outputs": {"min_delegations": 0}})["score"] == 1
    assert delegates_breadth(solo, {"outputs": {"min_delegations": 2}})["score"] == 0

    fanned = {"outputs": {"orchestrator_trajectory": ["task", "task"]}}
    assert delegates_breadth(fanned, {"outputs": {"min_delegations": 2}})["score"] == 1


def test_evaluators_accept_both_the_object_and_dict_run_shapes():
    """Local `evaluate()` passes objects with `.outputs`; an evaluator uploaded to
    LangSmith is handed plain dicts. Both have to work."""

    outputs = {"orchestrator_trajectory": ["write_todos", "task"]}
    run_object = SimpleNamespace(outputs=outputs)

    assert plans_with_todos(run_object, {})["score"] == 1
    assert plans_with_todos({"outputs": outputs}, {})["score"] == 1


def test_a_subagents_bookkeeping_does_not_earn_the_orchestrator_a_pass():
    """The tool-side twin of the prose leak above, and a nastier one.

    deepagents gives every declarative subagent its own TodoListMiddleware and
    FilesystemMiddleware, so the `researcher` really has `write_todos`, `ls` and
    `write_file` — whatever `subagents.py` lists in its `tools`. Its tool messages
    also stream out *before* the parent's `task` result. So an orchestrator that
    plans nothing, reads no memory and persists nothing still scored a clean sweep
    on all three, purely on a researcher tidying up after itself.
    """
    recorder = TurnRecorder()
    # The orchestrator delegates immediately — no plan, no memory check.
    # Inside the subagent: its own todos, its own ls, its own /memories/ write.
    for tool in ("write_todos", "ls", "tavily_search"):
        recorder.absorb(
            SUBAGENT, _updates("tools", ToolMessage("x", tool_call_id=tool, name=tool))
        )
    recorder.absorb(
        SUBAGENT,
        _updates(
            "model",
            AIMessage(
                content="",
                id="sub-write",
                tool_calls=[
                    {
                        "name": "write_file",
                        "args": {"file_path": "/memories/notes.md", "content": "x"},
                        "id": "c1",
                    }
                ],
            ),
        ),
    )
    recorder.absorb(
        SUBAGENT,
        _updates("tools", ToolMessage("ok", tool_call_id="c1", name="write_file")),
    )
    recorder.absorb(
        ORCHESTRATOR,
        _updates("tools", ToolMessage("summary", tool_call_id="t1", name="task")),
    )

    outputs = recorder.actions()
    example = {"outputs": {"min_delegations": 1}}
    run = {"outputs": outputs}

    # The orchestrator did exactly one thing: delegate.
    assert outputs["orchestrator_trajectory"] == ["task"]
    assert outputs["proposed_writes"] == []  # the write was the subagent's
    assert plans_with_todos(run, example)["score"] == 0
    assert checks_memory_first(run, example)["score"] == 0
    assert persists_findings(run, example)["score"] == 0
    # …while the search still counts, wherever it happened.
    assert searched_the_web(run, example)["score"] == 1


def test_harness_refuses_to_wipe_the_agents_live_state_dir():
    """The harness deletes its state dir between examples. Pointed at the real one,
    that would destroy the user's durable memories, which git cannot restore."""
    with pytest.raises(RuntimeError, match="throwaway"):
        ensure_isolated_state_dir(LIVE_STATE_DIR)

    ensure_isolated_state_dir(
        LIVE_STATE_DIR.parent / "somewhere-else"
    )  # does not raise


def test_coverage_score_is_a_proportion_not_a_conjunction():
    """Why `claims_are_cited` must not go back to being a bool.

    "Every claim is cited" is a conjunction over every claim in the report, so it
    reads 0 on any answer long enough to be worth writing, and it scores a report
    missing ONE citation identically to one that cites nothing at all — no gradient,
    no way to tell that a fix helped. Measured before the change: 0 on 4 of 5 sweep
    examples, while `response_cites_sources` passed all 5.
    """
    # The case a boolean cannot see: 29 of 30 claims cited is nearly perfect, and the
    # old metric scored it exactly the same as citing nothing.
    assert _coverage_score(30, 1) == pytest.approx(0.967, abs=0.001)
    assert _coverage_score(30, 30) == 0.0
    assert _coverage_score(30, 1) > _coverage_score(30, 15) > _coverage_score(30, 29)

    assert _coverage_score(4, 0) == 1.0
    # Vacuously perfect: an answer that asserts nothing is answers_the_question's problem.
    assert _coverage_score(0, 0) == 1.0
    # A judge that miscounts must not produce a negative score.
    assert _coverage_score(3, 5) == 0.0


def test_reset_state_deletes_only_the_databases_it_owns():
    """`_reset_state` runs before every example. It used to `rmtree(STATE_DIR)` — but
    STATE_DIR comes from an env var a user can point anywhere (blank resolves to the
    repo root; a relocated live state dir is a documented, supported setup). So it now
    unlinks the two databases by name, and a stray file in the same directory is proof
    that no recursive delete happens."""
    ensure_state_dir()
    bystander = STATE_DIR / "do-not-delete-me.txt"
    bystander.write_text("source code, or the user's notes, or anything at all")
    for database in (CHECKPOINT_DB, MEMORY_DB):
        database.write_text("pretend sqlite")
        database.with_name(database.name + "-wal").write_text("write-ahead log")

    _reset_state()

    assert not CHECKPOINT_DB.exists()
    assert not MEMORY_DB.exists()
    # The -wal sidecar must go too, or sqlite resurrects what we just dropped.
    assert not CHECKPOINT_DB.with_name(CHECKPOINT_DB.name + "-wal").exists()
    assert bystander.exists(), "reset must never touch a file it did not create"
    bystander.unlink()


def test_harness_refuses_a_state_dir_that_merely_contains_the_live_one():
    """Equality alone is not a guard. `_reset_state()` runs `shutil.rmtree`, so
    `DEEP_RESEARCH_STATE_DIR=.` — which is not equal to `.deep_research` and would
    sail through an equality check — recursively deletes the whole working tree,
    durable memories and source alike. Every ancestor is as fatal as the target."""
    for ancestor in (LIVE_STATE_DIR.parent, *LIVE_STATE_DIR.parents):
        with pytest.raises(RuntimeError, match="contains"):
            ensure_isolated_state_dir(ancestor)
