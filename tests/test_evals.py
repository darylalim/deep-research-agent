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
from typing import Any

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.types import Interrupt

from deep_research.agent import GATED_TOOLS
from deep_research.config import CHECKPOINT_DB, MEMORY_DB, STATE_DIR, ensure_state_dir
from deep_research.turns import _SILENT_STOPS, render_turn
from evals.dataset import EXAMPLES, drift
from evals.evaluators import (
    _COMPLETED_RUN_KEYS,
    ALL_EVALUATORS,
    CODE_EVALUATORS,
    UNCLEAN_STOPS,
    _coverage_score,
    answers_the_question,
    checks_memory_first,
    claims_are_cited,
    delegates_breadth,
    mutations_require_approval,
    persists_findings,
    plans_with_todos,
    response_cites_sources,
    searched_the_web,
    turn_stopped_cleanly,
)
from evals.harness import (
    LIVE_STATE_DIR,
    MUTATING_TOOLS,
    NON_WRITE_TOOLS,
    RESPONSE_KEY,
    WRITE_TOOLS,
    TurnRecorder,
    _approve_all,
    _reset_state,
    ensure_isolated_state_dir,
)


def test_the_evals_mutation_list_covers_every_gated_tool() -> None:
    """`MUTATING_TOOLS` and `GATED_TOOLS` are independent in value, not in content.

    `harness.MUTATING_TOOLS` is deliberately not an import of `agent.GATED_TOOLS`: an
    evaluator that read the dict it is checking would agree with it and pass no matter
    what that dict said. `mutations_require_approval` observes both sides instead —
    what the agent *proposed* against what actually *interrupted*.

    But that independence is about the values. A gated tool missing from the tuple is
    invisible to `TurnRecorder`, so it is never recorded as proposed, so the evaluator
    returns its vacuous "nothing to approve" pass and the metric silently stops
    covering it. That is not hypothetical: deepagents 0.7 added `delete`, `GATED_TOOLS`
    gained it, this tuple did not, and the only gated tool that DESTROYS data went
    unmeasured. Names in sync; values still observed rather than trusted.

    Verify it bites: drop `"delete"` from `MUTATING_TOOLS` and watch this go red.
    """
    assert set(MUTATING_TOOLS) == set(GATED_TOOLS), (
        "harness.MUTATING_TOOLS has drifted from agent.GATED_TOOLS — "
        f"gated but unmeasured: {sorted(set(GATED_TOOLS) - set(MUTATING_TOOLS))}; "
        f"measured but ungated: {sorted(set(MUTATING_TOOLS) - set(GATED_TOOLS))}. "
        "A gated tool absent here is never recorded as proposed, so "
        "mutations_require_approval scores it a vacuous 1."
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


def _write_proposal(message_id: str, path: str = "/memories/topic.md") -> AIMessage:
    return AIMessage(
        content="",
        id=message_id,
        tool_calls=[
            {
                "name": "write_file",
                "args": {"file_path": path, "content": "x"},
                "id": f"call-{message_id}",
            }
        ],
    )


def _interrupt(interrupt_id: str, *tool_names: str) -> dict:
    """One `__interrupt__` chunk, shaped as HumanInTheLoopMiddleware emits it."""
    return {
        "__interrupt__": [
            Interrupt(
                id=interrupt_id,
                value={
                    "action_requests": [
                        {"name": name, "args": {}} for name in tool_names
                    ]
                },
            )
        ]
    }


class TestMutationsRequireApproval:
    """The app's one unrecoverable invariant, and the only test of it end to end.

    `test_agent_wiring.py` proves `GATED_TOOLS` *says* `True`. This proves the gate
    actually fired on the run — which is a different claim, and the one that matters:
    `/memories/` is gitignored, so an unapproved write is not something git can undo.
    """

    def test_a_gated_write_passes(self):
        recorder = TurnRecorder()
        recorder.absorb(ORCHESTRATOR, _updates("model", _write_proposal("ai-1")))
        recorder.absorb(ORCHESTRATOR, _interrupt("i1", "write_file"))
        outputs = recorder.actions()

        assert outputs["proposed_mutations"] == ["write_file"]
        assert outputs["gated_tools"] == ["write_file"]
        assert mutations_require_approval({"outputs": outputs}, {})["score"] == 1

    def test_an_ungated_write_fails(self):
        # THE regression this exists to catch. Flip `GATED_TOOLS["write_file"]` to
        # `False` and this is exactly what the harness records: the agent proposes the
        # write, no interrupt is ever raised, the file lands unreviewed. Before this
        # evaluator, that run scored a clean sweep — all six code metrics and both
        # judges green, because `persists_findings` grades the *proposal* and nothing
        # read `gated_tools` at all.
        recorder = TurnRecorder()
        recorder.absorb(ORCHESTRATOR, _updates("model", _write_proposal("ai-1")))
        # …and no `__interrupt__` chunk ever arrives.
        outputs = recorder.actions()

        assert persists_findings({"outputs": outputs}, {})["score"] == 1  # still green!
        result = mutations_require_approval({"outputs": outputs}, {})
        assert result["score"] == 0
        assert "WITHOUT APPROVAL" in result["comment"]

    def test_an_ungated_subagent_write_is_not_masked_by_the_orchestrators_gated_one(
        self,
    ):
        # The PARTIAL gate failure — and the reason this metric counts multisets rather
        # than testing name membership. The orchestrator's own `/memories/` write is
        # gated and approved, exactly as in every real run (SYSTEM_PROMPT step 5). A
        # researcher then writes a file whose gate is missing — reachable without any
        # deepagents bug, because a `SubAgent` spec may override the inherited
        # `interrupt_on`, and an empty dict silently drops the middleware.
        #
        # Under a set-membership test this scored a clean 1: `write_file` had interrupted
        # *somewhere* in the turn, so every other `write_file` counted as covered. The
        # metric could only ever see a TOTAL gate failure, never a partial one — while
        # its own docstring claimed a researcher writing unreviewed was the case it
        # existed to catch.
        #
        # Note what makes the old test pass for the wrong reason: it recorded ONLY the
        # subagent's write, so `gated` was empty and the set test happened to work. That
        # is the one configuration a real run never produces.
        recorder = TurnRecorder()
        recorder.absorb(ORCHESTRATOR, _updates("model", _write_proposal("ai-1")))
        recorder.absorb(ORCHESTRATOR, _interrupt("i1", "write_file"))  # …approved
        recorder.absorb(SUBAGENT, _updates("model", _write_proposal("ai-2")))  # …not
        outputs = recorder.actions()

        # One write was gated, two were proposed.
        assert outputs["proposed_mutations"] == ["write_file", "write_file"]
        assert outputs["gated_tools"] == ["write_file"]
        # `proposed_writes` stays orchestrator-only, so `persists_findings` cannot be
        # fooled by a researcher tidying up — which is exactly why the safety check
        # needed a field of its own rather than reusing that one.
        assert outputs["proposed_writes"] == ["/memories/topic.md"]

        result = mutations_require_approval({"outputs": outputs}, {})
        assert result["score"] == 0, "an unapproved write was masked by an approved one"
        assert "WITHOUT APPROVAL" in result["comment"]

    def test_a_lone_unapproved_subagent_write_fails(self):
        # The total-failure case, kept for completeness: nothing interrupted at all.
        recorder = TurnRecorder()
        recorder.absorb(SUBAGENT, _updates("model", _write_proposal("ai-1")))
        outputs = recorder.actions()

        assert outputs["proposed_writes"] == []  # orchestrator-only, by design
        assert outputs["proposed_mutations"] == ["write_file"]
        assert mutations_require_approval({"outputs": outputs}, {})["score"] == 0

    def test_a_doubly_emitted_interrupt_cannot_mask_an_ungated_write(self):
        # THE bug that defeated this metric, and the reason `gated` is deduped by
        # `Interrupt.id`. With `subgraphs=True` an interrupt raised inside a subagent is
        # emitted TWICE — once at the subagent's namespace, once bubbled to the root —
        # with the SAME id. Counting both inflates `gated`, and because the comparison is
        # a MULTISET difference, that surplus entry silently absorbs a genuinely ungated
        # mutation of the same name.
        #
        # Measured before the fix: two proposed writes, ONE real interrupt (emitted
        # twice), a file written unreviewed — and a clean score of 1. Exactly the
        # partial-gate hole the multiset was introduced to close, reopened from the other
        # side. The single-proposal case (below) is why this was missed: it looks fine.
        recorder = TurnRecorder()
        # Researcher A's write IS gated -> its interrupt is emitted twice.
        recorder.absorb(SUBAGENT, _updates("model", _write_proposal("ai-1")))
        recorder.absorb(SUBAGENT, _interrupt("i1", "write_file"))
        recorder.absorb(ORCHESTRATOR, _interrupt("i1", "write_file"))  # …bubbled
        # Researcher B's write is NOT gated -> no interrupt at all.
        recorder.absorb(SUBAGENT, _updates("model", _write_proposal("ai-2")))
        outputs = recorder.actions()

        assert outputs["proposed_mutations"] == ["write_file", "write_file"]
        assert outputs["gated_tools"] == ["write_file"], "the duplicate must collapse"

        result = mutations_require_approval({"outputs": outputs}, {})
        assert result["score"] == 0, "an unapproved write hid behind a duplicated gate"
        assert "WITHOUT APPROVAL" in result["comment"]

    def test_extra_gated_entries_cannot_manufacture_a_failure(self):
        # The multiset difference must still be safe in the healthy direction:
        # `gated_tools` also carries NON-mutating gated tools, which must not push a
        # clean run to 0.
        recorder = TurnRecorder()
        recorder.absorb(ORCHESTRATOR, _updates("model", _write_proposal("ai-1")))
        recorder.absorb(ORCHESTRATOR, _interrupt("i1", "write_file", "execute"))
        outputs = recorder.actions()

        assert outputs["proposed_mutations"] == ["write_file"]
        assert outputs["gated_tools"] == ["write_file", "execute"]
        assert mutations_require_approval({"outputs": outputs}, {})["score"] == 1

    def test_a_replayed_tool_result_is_not_counted_twice(self):
        # The other half: `TurnRecorder` used to dedupe stream messages by
        # `BaseMessage.id`. A resumed superstep re-emits the cached writes of the
        # siblings that already succeeded — as FRESH ToolMessage objects (measured:
        # `id=None` on the first pass, a brand-new uuid on the resume), so a message-id
        # seen-set never matched and let every one of them through.
        #
        # Concretely: two researchers fan out, one proposes a gated write, and the
        # other's finished `task` result replays on every approval round. `task` is then
        # counted twice for ONE delegation, and `delegates_breadth` passes an example
        # that demanded more than actually happened. Keyed on `tool_call_id` now — a tool
        # call executes exactly once.
        recorder = TurnRecorder()
        for _ in range(2):
            recorder.absorb(
                ORCHESTRATOR,
                _updates(
                    "tools", ToolMessage("summary", tool_call_id="t1", name="task")
                ),
            )
        outputs = recorder.actions()

        assert outputs["orchestrator_trajectory"] == ["task"]
        assert delegates_breadth({"outputs": outputs}, {"outputs": {}})["score"] == 1
        # …and a run needing TWO delegations must not pass on one replayed result.
        two = {"outputs": {"min_delegations": 2}}
        assert delegates_breadth({"outputs": outputs}, two)["score"] == 0

    def test_not_writing_at_all_is_not_a_safety_failure(self):
        # Vacuously safe: an agent that proposes no mutation has violated nothing.
        # `persists_findings` is what notices an agent that never persists.
        recorder = TurnRecorder()
        recorder.absorb(
            ORCHESTRATOR,
            _updates("tools", ToolMessage("hits", tool_call_id="1", name="task")),
        )
        assert (
            mutations_require_approval({"outputs": recorder.actions()}, {})["score"]
            == 1
        )


def test_a_subagents_citations_do_not_earn_the_orchestrator_a_pass():
    """Why the recorder collects actions but not prose.

    The stream carries assistant messages from *inside* the researchers, and the
    user never sees one of them. If the harness built its transcript from the
    stream, a researcher's neatly-cited summary would score the orchestrator's
    uncited sign-off as a pass. So `response` is rendered by `turns.render_turn` from
    the orchestrator's final state — exactly the text the page draws.
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


def test_delegates_breadth_is_a_band_and_says_which_side_failed():
    """The ceiling is what makes over-orchestration observable at all.

    At `min_delegations=0` with no ceiling, `delegated >= 0` holds for any number of
    dispatches, so the three direct-path examples scored 1 however the agent behaved —
    including when it fanned out and thereby stopped grading the direct path they exist
    to measure. Verified by deleting the ceiling branch: the `over` case below goes
    green while the defect is live.
    """
    direct = {"outputs": {"orchestrator_trajectory": ["tavily_search"]}}
    fanned = {"outputs": {"orchestrator_trajectory": ["task", "task"]}}

    over = delegates_breadth(
        fanned, {"outputs": {"min_delegations": 0, "max_delegations": 1}}
    )
    assert over["score"] == 0
    assert "OVER-ORCHESTRATED" in over["comment"]

    # Sitting exactly on either bound passes; the band is inclusive.
    assert (
        delegates_breadth(
            fanned, {"outputs": {"min_delegations": 2, "max_delegations": 2}}
        )["score"]
        == 1
    )
    assert (
        delegates_breadth(
            direct, {"outputs": {"min_delegations": 0, "max_delegations": 0}}
        )["score"]
        == 1
    )

    # A bare 0 cannot separate the two failures, and they want opposite fixes — so the
    # comment has to name the side, not just the number.
    under = delegates_breadth(direct, {"outputs": {"min_delegations": 2}})
    assert under["score"] == 0
    assert "UNDER-DELEGATED" in under["comment"]


def test_a_missing_ceiling_asserts_nothing():
    """Absent means NO ceiling, never zero.

    Deliberately the mirror of the unknown-means-fail rule that governs `GATED_TOOLS`
    and `_SILENT_STOPS`: a gate must refuse what it does not recognise, but an eval bar
    that fires on behaviour nobody asserted scores a *correct* agent down. Every
    example written before the column existed relies on this.
    """
    fanned = {"outputs": {"orchestrator_trajectory": ["task"] * 5}}
    assert delegates_breadth(fanned, {"outputs": {"min_delegations": 2}})["score"] == 1


def test_every_delegation_band_in_the_dataset_is_satisfiable():
    """A ceiling below its own floor is an example that can never pass, and nothing
    else would notice until a paid sweep scored it 0 for a reason having nothing to do
    with the agent."""
    for example in EXAMPLES:
        outputs = example["outputs"]
        ceiling = outputs.get("max_delegations")
        if ceiling is None:
            continue
        assert ceiling >= outputs.get("min_delegations", 1), example["inputs"][
            "question"
        ]


class _FakeClient:
    """Enough LangSmith for `drift()`. The real client would need credentials and a
    network, and what is under test here is a comparison, not an API."""

    def __init__(self, uploaded: list[tuple[str, dict]]) -> None:
        self._uploaded = [
            SimpleNamespace(inputs={"question": q}, outputs=o, id=f"id{i}")
            for i, (q, o) in enumerate(uploaded)
        ]

    def has_dataset(self, dataset_name: str | None = None) -> bool:
        return True

    def read_dataset(self, dataset_name: str | None = None) -> Any:
        return SimpleNamespace(id="ds")

    def list_examples(self, dataset_id: Any = None) -> list[Any]:
        return list(self._uploaded)


def _uploaded_from_local() -> list[tuple[str, dict]]:
    return [(e["inputs"]["question"], dict(e["outputs"])) for e in EXAMPLES]


def test_drift_is_silent_when_the_upload_matches_the_file():
    assert drift(_FakeClient(_uploaded_from_local())) == []


def test_drift_compares_outputs_because_the_count_does_not_move():
    """The case that actually occurred, one commit apart, in this repo.

    Adding `max_delegations` changed three examples and left the count at 9. A check on
    `example_count` passes there while all three direct-path ceilings do not exist
    remotely — and `delegates_breadth` reads a missing key as "no ceiling", so the sweep
    scores exactly as it did before the column was added, silently and for real money.
    """
    stale = _uploaded_from_local()
    changed = next(i for i, (_, o) in enumerate(stale) if "max_delegations" in o)
    question, outputs = stale[changed]
    outputs.pop("max_delegations")
    stale[changed] = (question, outputs)

    assert len(stale) == len(EXAMPLES)  # the count is no help at all
    differences = drift(_FakeClient(stale))
    assert len(differences) == 1
    assert "outputs differ" in differences[0]


def test_drift_reports_a_remote_only_example_that_sync_would_never_remove():
    """Wider than the comparison `sync()` makes, deliberately.

    `sync()` adds and updates but never deletes, so an example uploaded out of band is
    invisible to it by design. It is not invisible to a *run*: `evaluate()` grades it,
    and its scores land in the experiment aggregate beside examples you can read in
    `dataset.py`. So `--upload` is not its remedy, and the message must not claim it is.
    """
    extra = [*_uploaded_from_local(), ("a question nobody committed", {})]
    differences = drift(_FakeClient(extra))
    assert len(differences) == 1
    assert "a question nobody committed" in differences[0]
    assert "will NOT remove it" in differences[0]


def test_drift_reports_an_example_that_was_never_uploaded():
    differences = drift(_FakeClient(_uploaded_from_local()[1:]))
    assert len(differences) == 1
    assert differences[0].startswith("not uploaded:")


def test_the_direct_path_examples_assert_a_ceiling():
    """This is where unknown-means-fail belongs: in the dataset, not the evaluator.

    `min_delegations=0` is unfalsifiable on its own, so a direct-path example with no
    ceiling grades nothing about the path it was added for — it scores 1 whether the
    agent stayed direct or fanned out into the delegated path that already measures
    well. The ceiling is the only thing making those examples assertions rather than
    free passes, so dropping it has to be loud.
    """
    direct = [e for e in EXAMPLES if e["outputs"].get("min_delegations") == 0]
    assert direct, "no direct-path examples left — this test has gone vacuous"
    for example in direct:
        assert "max_delegations" in example["outputs"], example["inputs"]["question"]


def test_evaluators_accept_both_the_object_and_dict_run_shapes():
    """Local `evaluate()` passes objects with `.outputs`; an evaluator uploaded to
    LangSmith is handed plain dicts. Both have to work."""

    outputs = {"orchestrator_trajectory": ["write_todos", "task"]}
    run_object = SimpleNamespace(outputs=outputs)

    assert plans_with_todos(run_object, {})["score"] == 1
    assert plans_with_todos({"outputs": outputs}, {})["score"] == 1


def test_a_subagents_bookkeeping_does_not_earn_the_orchestrator_a_pass():
    """The tool-side twin of the prose leak above, and a nastier one.

    deepagents gives every declarative subagent its own FilesystemMiddleware, so the
    `researcher` really has `ls`, `write_file` and `delete` — whatever `subagents.py`
    lists in its `tools`. Its tool messages also stream out *before* the parent's `task`
    result. So an orchestrator that reads no memory and persists nothing still scored a
    clean sweep, purely on a researcher tidying up after itself.

    `write_todos` stays in the fixture below even though deepagents 0.7.x no longer gives
    subagents a `TodoListMiddleware` — `build_agent` passes one for the orchestrator only.
    That is deliberate: `TurnRecorder` splits on the NAMESPACE, not on a per-tool list, so
    feeding it a tool a subagent can no longer emit is a direct check that the split does
    not quietly depend on which middleware deepagents ships this month.
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


# --- What a broken MEASUREMENT looks like ------------------------------------
#
# Everything above asks whether the AGENT behaved. Everything below asks whether the
# instrument reading it can be trusted, which is a different question and the one this
# suite was weakest on: every defect these pin was found while all 30 tests were green.


def test_the_completed_run_keys_match_what_the_recorder_emits():
    """`_COMPLETED_RUN_KEYS` is a hand-copy, so it is compared against the real thing.

    `evals.evaluators` deliberately does not import `evals.harness` — that module
    refuses to import at all unless the state dir is already isolated, and an evaluator
    has no business requiring that. So the key list is written out by hand, and a
    hand-copy of a shape defined elsewhere is precisely what this repo keeps getting
    wrong (`turns._LS_EMPTY`, `_SILENT_STOPS`). Same remedy: build the real object and
    compare, rather than trusting the copy.
    """
    emitted = set(TurnRecorder().actions()) | {RESPONSE_KEY}
    assert emitted == set(_COMPLETED_RUN_KEYS), (
        "harness.TurnRecorder.actions() and evaluators._COMPLETED_RUN_KEYS have "
        f"drifted: recorder-only {sorted(emitted - set(_COMPLETED_RUN_KEYS))}, "
        f"evaluator-only {sorted(set(_COMPLETED_RUN_KEYS) - emitted)}. `_ungradable` "
        "decides whether a run produced any observations by testing for these keys, so "
        "a stale list makes a completed run look crashed, or a crashed one gradable."
    )


@pytest.mark.parametrize("evaluator", ALL_EVALUATORS, ids=lambda f: f.__name__)
def test_no_evaluator_grades_a_run_whose_target_crashed(evaluator):
    """A crashed run must produce a NON-verdict, never a score.

    `langsmith/evaluation/_runner.py::_forward` catches the target's exception, logs it,
    and returns the row anyway — so `research()` raising still reaches every evaluator,
    with no outputs. Each one has a vacuous branch for the honest empty case, and the
    crashed row fell into it: `mutations_require_approval` returned "no mutation
    proposed — nothing to approve" and scored **1**, a clean pass on the one invariant
    this repo calls silent and unrecoverable.

    Parametrized over `ALL_EVALUATORS` rather than listing them, so an evaluator added
    without the guard fails here instead of quietly grading crashes. Both judges are
    included and neither reaches the API: the guard returns first, which is also the
    reason it must stay the first statement in those two functions.

    Verify it bites: delete the `_ungradable` guard from any one evaluator.
    """
    raised = SimpleNamespace(
        outputs={"output": None}, inputs={}, error="RuntimeError: boom"
    )
    assert evaluator(raised, {"outputs": {}})["score"] is None
    # …and the same when the row carries no error but no observations either.
    assert evaluator({"outputs": {"output": None}}, {"outputs": {}})["score"] is None


def test_a_plan_made_after_the_research_started_is_not_a_plan():
    """The ordering `plans_with_todos` exists to check, which nothing checked.

    Deleting the `[:start]` slice from both this evaluator and `checks_memory_first`
    left the whole suite green — the fixtures all happened to put the tool first, so
    "before the first research action" and "anywhere in the trajectory" were
    indistinguishable. SYSTEM_PROMPT step 1 is about ordering; a todo list written
    after the first `task` is bookkeeping, not a plan.
    """
    expects_plan = {"outputs": {"expects_plan": True}}
    late = {"outputs": {"orchestrator_trajectory": ["task", "write_todos"]}}
    early = {"outputs": {"orchestrator_trajectory": ["write_todos", "task"]}}
    assert plans_with_todos(late, expects_plan)["score"] == 0
    assert plans_with_todos(early, expects_plan)["score"] == 1


def test_memory_read_after_the_research_started_is_not_checking_memory_first():
    """The twin of the test above, and the same deleted slice passes both."""
    late = {"outputs": {"orchestrator_trajectory": ["tavily_search", "ls"]}}
    early = {"outputs": {"orchestrator_trajectory": ["ls", "tavily_search"]}}
    assert checks_memory_first(late, {})["score"] == 0
    assert checks_memory_first(early, {})["score"] == 1


def test_a_grade_that_counts_nothing_while_listing_misses_scores_zero():
    """`_coverage_score(0, N)` returned a perfect 1.0 for every N.

    That is the landing pad for a truncated judge grade: with the count cut away,
    `.get(...) or 0` hands this function a zero total beside a list of real misses, and
    the worse the grade the better the score. A missing count next to present misses is
    a broken reading, not an empty report.
    """
    assert _coverage_score(0, 0) == 1.0  # nothing to attribute — vacuously perfect
    assert _coverage_score(0, 5) == 0.0  # …but not while it is listing five misses


def test_asking_to_delete_a_memory_is_not_persisting_a_finding():
    """`delete` takes a `file_path`, exactly as `write_file` does.

    So recording a path for every `MUTATING_TOOLS` call scored an orchestrator that
    asked to DESTROY `/memories/pricing.md` a `persists_findings` pass for having
    written it. `delete` must stay in `proposed_mutations` — that is what
    `mutations_require_approval` grades, and it is the only gated tool that destroys
    data — so the narrowing belongs on the path branch alone.
    """
    recorder = TurnRecorder()
    recorder.absorb(
        ORCHESTRATOR,
        _updates(
            "model",
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "delete",
                        "args": {"file_path": "/memories/pricing.md"},
                        "id": "del-1",
                    }
                ],
            ),
        ),
    )
    outputs = recorder.actions()
    assert outputs["proposed_mutations"] == ["delete"]  # still gated, still measured
    assert outputs["proposed_writes"] == []  # …but nothing was persisted
    assert persists_findings({"outputs": outputs}, {})["score"] == 0


def test_searches_that_all_failed_are_not_evidence_that_it_researched():
    """A failed call still RAN — but it taught the agent nothing.

    deepagents returns a `status="error"` ToolMessage for a denied path, an unsupported
    backend or a tool exception, and `HumanInTheLoopMiddleware` uses the same status for
    a rejected call. Counting those as searches scored a clean 1 on a run whose every
    search failed and whose answer therefore came from the model's own memory — the
    exact thing this metric exists to rule out. The trajectories still record it,
    because the ordering the two evaluators above depend on must not change.
    """
    recorder = TurnRecorder()
    recorder.absorb(
        SUBAGENT,
        _updates(
            "tools",
            ToolMessage(
                "Error: request failed",
                tool_call_id="s1",
                name="tavily_search",
                status="error",
            ),
        ),
    )
    failed_only = recorder.actions()
    assert failed_only["trajectory"] == ["tavily_search"]  # it ran…
    assert failed_only["failed_tools"] == ["tavily_search"]  # …and it failed
    assert searched_the_web({"outputs": failed_only}, {})["score"] == 0

    # Positive control: one search that actually returned is enough.
    recorder.absorb(
        SUBAGENT,
        _updates("tools", ToolMessage("hits", tool_call_id="s2", name="tavily_search")),
    )
    assert searched_the_web({"outputs": recorder.actions()}, {})["score"] == 1


def test_an_api_stop_is_reported_as_a_stop_and_not_as_a_bad_answer():
    """The misattribution `turns._stop_note` prevents in the app, prevented in the evals.

    A refusal or a context-window overrun ends the turn with HTTP 200 and no prose, so
    `render_turn` yields `''`, so both judges score 0.0 and `response_cites_sources`
    scores 0. Read off a sweep that is indistinguishable from an agent that researched
    badly, and it sends the next person to fix a prompt when the remedy is a fresh
    thread. Subagent stops count too — a researcher's refusal is otherwise invisible.
    """
    clean = TurnRecorder()
    clean.absorb(
        ORCHESTRATOR,
        _updates(
            "model", AIMessage("done", response_metadata={"stop_reason": "end_turn"})
        ),
    )
    graded_clean = turn_stopped_cleanly({"outputs": clean.actions()}, {})
    assert graded_clean["score"] == 1
    # And it must not report a count: `stop_reasons` counts EMISSIONS, and the
    # middleware re-emits the proposing AIMessage on every resume round, so one
    # generation was being announced as two.
    clean.absorb(
        ORCHESTRATOR,
        _updates(
            "HumanInTheLoopMiddleware.after_model",
            AIMessage("done", response_metadata={"stop_reason": "end_turn"}),
        ),
    )
    assert clean.actions()["stop_reasons"] == ["end_turn", "end_turn"]
    assert (
        "2 generation"
        not in turn_stopped_cleanly({"outputs": clean.actions()}, {})["comment"]
    )

    refused = TurnRecorder()
    refused.absorb(
        SUBAGENT,
        _updates("model", AIMessage("", response_metadata={"stop_reason": "refusal"})),
    )
    graded = turn_stopped_cleanly({"outputs": refused.actions()}, {})
    assert graded["score"] == 0
    assert "refusal" in graded["comment"]


def test_every_stop_the_cli_calls_silent_is_also_graded_as_unclean():
    """`UNCLEAN_STOPS` is DERIVED from `turns._SILENT_STOPS`, not copied beside it.

    That table is already compared against `anthropic.types.StopReason` by set equality
    in `test_cli_parsing.py`, so deriving from it means an SDK bump that adds a silent
    stop reaches this evaluator too. A second hand-written list would be one more thing
    to keep true by hand — and this repo has already missed exactly that once, when
    `model_context_window_exceeded` arrived in anthropic 0.120.
    """
    # `.keys()` rather than `set(_SILENT_STOPS)`, deliberately. A subset assertion is
    # satisfied by a hand-copy that happens to list today's members, so it does not pin
    # the property this docstring claims. Worse, if `_SILENT_STOPS` were ever refactored
    # from `dict[str, StopNote]` to a collection of StopNote objects, `frozenset(...)`
    # would yield StopNotes, every `stop in UNCLEAN_STOPS` would be False,
    # `turn_stopped_cleanly` would return 1 forever — and a subset assertion would still
    # pass, because both sides changed together. Naming `.keys()` is what goes red there.
    assert set(_SILENT_STOPS.keys()) | {"max_tokens"} == UNCLEAN_STOPS
    assert all(isinstance(stop, str) for stop in UNCLEAN_STOPS)
    # The members themselves are deliberately NOT written down — that list grows with
    # the SDK, and a copy of it here is one more thing to keep true by hand.
    assert "end_turn" not in UNCLEAN_STOPS


def test_the_discriminating_branches_are_asserted_in_both_directions():
    """Each of these was pinned in ONE direction only, so each could be deleted green.

    A "scores 1 when it should" assertion cannot tell a working evaluator from one that
    returns 1 unconditionally; this repo's own CLAUDE.md records the same lesson about
    a `disabled=` assertion that passed on an empty thread. The negative half is the
    half that bites.
    """
    # searched_the_web: 1 is pinned elsewhere; this is the 0.
    assert searched_the_web({"outputs": {"trajectory": ["task"]}}, {})["score"] == 0
    # response_cites_sources: both halves.
    assert (
        response_cites_sources({"outputs": {"response": "no links here"}}, {})["score"]
        == 0
    )
    cited = {"outputs": {"response": "ships in 3.14 (https://docs.python.org/3.14/)"}}
    assert response_cites_sources(cited, {})["score"] == 1
    # persists_findings: a write is not enough — it has to be under /memories/.
    assert (
        persists_findings({"outputs": {"proposed_writes": ["/report.md"]}}, {})["score"]
        == 0
    )
    assert (
        persists_findings({"outputs": {"proposed_writes": ["/memories/x.md"]}}, {})[
            "score"
        ]
        == 1
    )
    # delegates_breadth: the default floor is 1, so an example asserting nothing still
    # demands one delegation.
    assert (
        delegates_breadth(
            {"outputs": {"orchestrator_trajectory": []}}, {"outputs": {}}
        )["score"]
        == 0
    )


def test_every_example_has_a_distinct_question():
    """Question text is the dataset's only cross-side identity, and nothing enforced it.

    `_by_question` keeps one row per question, so a copy-pasted duplicate collapses on
    every comparison built on that map — including the `drift()` gate — while
    `evaluate()` grades both copies, bills for both, and weights that question double
    in the experiment mean.
    """
    questions = [example["inputs"]["question"] for example in EXAMPLES]
    assert len(set(questions)) == len(questions), (
        "two examples in evals/dataset.py share a question: "
        f"{sorted({q for q in questions if questions.count(q) > 1})}"
    )


def test_drift_reports_a_question_uploaded_twice():
    """The one kind of drift the gate could not see, because it collapsed it first."""
    doubled = _uploaded_from_local()
    doubled.append(doubled[0])
    differences = drift(_FakeClient(doubled))
    assert any("uploaded 2 times" in difference for difference in differences), (
        differences
    )


class _StubJudge:
    """A `JUDGE` stand-in whose `with_structured_output(...).invoke(...)` is fixed.

    Offline by construction, and it asserts the one call-site detail the truncation
    check depends on: `include_raw=True`. Without it `.invoke()` returns the parsed
    grade alone, `raw` is gone, and `stop_reason` is unreadable — so the guard would
    silently never fire while every assertion about a *clean* grade still passed.
    """

    def __init__(self, payload: dict) -> None:
        self._payload = payload

    def with_structured_output(self, schema: Any, **kwargs: Any) -> _StubJudge:
        assert kwargs.get("include_raw") is True, (
            "the truncation guard reads raw.response_metadata, which only exists when "
            "with_structured_output is called with include_raw=True"
        )
        return self

    def invoke(self, messages: Any) -> dict:
        return self._payload


def _judge_reply(grade: dict, stop_reason: str = "end_turn") -> dict:
    return {
        "raw": AIMessage("", response_metadata={"stop_reason": stop_reason}),
        "parsed": grade,
        "parsing_error": None,
    }


def test_a_judge_grade_cut_off_at_max_tokens_is_refused_rather_than_scored(monkeypatch):
    """The judge's own answer running out of tokens must not read as a good report.

    Both grades are TypedDicts, so `with_structured_output` routes to
    `JsonOutputParser`, whose `parse_json_markdown` defaults to `parse_partial_json` —
    it CLOSES unterminated JSON instead of raising. Right for streaming a partial
    object into a UI, catastrophic in a grader. Measured on the installed parser: a
    grade of 12 claims / 9 uncited, cut after the second entry, parses cleanly and
    scores **0.83** where the truth is 0.25.

    The bias only runs one way — truncation drops uncited claims, never adds them — and
    it is worst exactly where the metric matters most, since each `_UncitedClaim`
    demands a verbatim quote and a justification, so the worse the citation discipline
    the longer the honest grade and the likelier the cut.

    The fixture below IS the repaired shape, not a hypothetical one.
    """
    repaired = {
        "reasoning": "Several figures carry no source.",
        "substantive_claims": 12,
        "uncited_claims": [{"claim": "first"}, {"claim": "second"}],
    }

    monkeypatch.setattr(
        "evals.evaluators.JUDGE", _StubJudge(_judge_reply(repaired, "max_tokens"))
    )
    graded = claims_are_cited(
        {"outputs": {"response": "a cited report"}}, {"outputs": {}}
    )
    assert graded["score"] is None, (
        "a grade the judge never finished writing was scored as if it had: this is the "
        "0.83-for-0.25 inflation, and it lands on badly-cited reports by construction"
    )
    assert "cut off" in graded["comment"]

    # Positive control: the identical grade, finished, still scores the proportion —
    # so the guard is refusing truncation rather than refusing everything.
    monkeypatch.setattr("evals.evaluators.JUDGE", _StubJudge(_judge_reply(repaired)))
    finished = claims_are_cited(
        {"outputs": {"response": "a cited report"}}, {"outputs": {}}
    )
    assert finished["score"] == pytest.approx(10 / 12)


def test_a_judge_grade_with_no_count_cannot_collect_a_vacuous_perfect_score(
    monkeypatch,
):
    """The same failure landing one field earlier, which is the worse of the two.

    Truncated inside `reasoning`, the grade comes back with no `substantive_claims` at
    all. `.get(...) or 0` turned that into a zero total, and `_coverage_score` returned
    1.0 for a zero total — so the *most* truncated grade scored a flawless 100%. Two
    independent guards now stop it: the missing count is refused here, and
    `_coverage_score` no longer rewards a zero total that has misses beside it.
    """
    monkeypatch.setattr(
        "evals.evaluators.JUDGE",
        _StubJudge(_judge_reply({"reasoning": "The answer makes many claims"})),
    )
    graded = claims_are_cited({"outputs": {"response": "a report"}}, {"outputs": {}})
    assert graded["score"] is None
    assert "substantive_claims" in graded["comment"]


def test_the_write_tools_are_a_classified_subset_of_the_mutating_ones():
    """The mirror of `test_the_evals_mutation_list_covers_every_gated_tool`.

    That test forces `MUTATING_TOOLS` to cover `GATED_TOOLS`, so a tool arriving from a
    deepagents upgrade — exactly how `delete` arrived in 0.7 — lands in both lists under
    a red test. Nothing then classified it as content-adding or not, and defaulting
    either way is a silent mis-grade: as a write it reopens the `delete` false positive
    (`persists_findings` credits a destroyed note); as a non-write a genuinely
    content-adding tool gives a false NEGATIVE, scoring 0 on a run that *did* persist.

    Requiring the two lists to PARTITION `MUTATING_TOOLS` is what makes the choice a
    person's. The same shape as `READ_ONLY_TOOLS` in `test_agent_wiring.py`.
    """
    assert set(WRITE_TOOLS) <= set(MUTATING_TOOLS)
    assert set(WRITE_TOOLS) | set(NON_WRITE_TOOLS) == set(MUTATING_TOOLS), (
        "every mutating tool must be classified as adding content or not — "
        f"unclassified: {sorted(set(MUTATING_TOOLS) - set(WRITE_TOOLS) - set(NON_WRITE_TOOLS))}"
    )
    assert not set(WRITE_TOOLS) & set(NON_WRITE_TOOLS)


@pytest.mark.parametrize("evaluator", CODE_EVALUATORS, ids=lambda f: f.__name__)
def test_every_code_evaluator_reads_a_key_the_harness_emits(evaluator):
    """The positive half of the `_ungradable` guard, and it is not decoration.

    Each evaluator names the ONE output key it grades. A typo in that literal turns its
    metric into a permanent non-verdict — every run "not measured", across a whole paid
    sweep, with no test going red, because the crashed-run test expects exactly that
    answer. Driving every evaluator against a complete (if empty) recorder output is
    what pins each literal against a key the harness really emits.
    """
    outputs = {**TurnRecorder().actions(), RESPONSE_KEY: "some prose"}
    assert evaluator({"outputs": outputs}, {"outputs": {}})["score"] is not None


def test_the_other_judge_is_driven_through_the_same_truncation_guard(monkeypatch):
    """`answers_the_question` shares `_graded` and had none of its tests.

    Its `count_key="question_parts"` literal was never driven, so a typo there would
    have made it a permanent non-verdict across a paid sweep with nothing going red —
    the failure the parametrized code-evaluator test above exists to catch, one layer up
    in the judges where that test cannot reach without a stub.
    """
    run = {"outputs": {"response": "a long answer"}}
    grade = {
        "reasoning": "One leg is missing.",
        "question_parts": 4,
        "unanswered_parts": [{"part": "the second half"}],
    }

    monkeypatch.setattr("evals.evaluators.JUDGE", _StubJudge(_judge_reply(grade)))
    assert answers_the_question(run, {"outputs": {}})["score"] == pytest.approx(3 / 4)

    monkeypatch.setattr(
        "evals.evaluators.JUDGE", _StubJudge(_judge_reply(grade, "max_tokens"))
    )
    assert answers_the_question(run, {"outputs": {}})["score"] is None

    # …and the `count_key` literal itself: a grade with no part count is refused.
    monkeypatch.setattr(
        "evals.evaluators.JUDGE", _StubJudge(_judge_reply({"reasoning": "x"}))
    )
    refused = answers_the_question(run, {"outputs": {}})
    assert refused["score"] is None
    assert "question_parts" in refused["comment"]


def test_a_contradictory_grade_never_prints_a_negative_claim_count(monkeypatch):
    """What the `max(total, len(misses))` floors are actually for.

    `_coverage_score` clamps the score at 0.0 either way, so the floors change no
    number — which makes them look like dead belt-and-braces. They are not: without
    them a judge reporting 3 claims and 5 misses prints "-2/3 claims cited", a comment
    that reads as a bug in the evaluator rather than as an incoherent grade.
    """
    monkeypatch.setattr(
        "evals.evaluators.JUDGE",
        _StubJudge(
            _judge_reply(
                {
                    "reasoning": "contradictory",
                    "substantive_claims": 3,
                    "uncited_claims": [{"claim": str(i)} for i in range(5)],
                }
            )
        ),
    )
    graded = claims_are_cited({"outputs": {"response": "prose"}}, {"outputs": {}})
    assert graded["score"] == 0.0
    assert graded["comment"].startswith("0/5 claims cited")


def test_a_run_missing_ONLY_this_metrics_observation_is_not_graded():
    """The guard checks the caller's key, not "any harness key" — and that is the point.

    The first version of `_ungradable` asked whether the run carried *some* completed-run
    key, which a row holding just a `response` satisfies. `mutations_require_approval`
    then read `proposed_mutations=[]`, hit its vacuous branch, and scored a clean 1 on
    the safety invariant for a run whose mutations were never observed — the same free
    pass the guard was added to close, surviving one level down. The docstring even
    justified the weaker form by pointing at these fixtures, which is production
    semantics bent to suit a test.
    """
    prose_only = {"outputs": {"response": "an answer, and no recorded actions"}}
    assert mutations_require_approval(prose_only, {})["score"] is None
    assert persists_findings(prose_only, {})["score"] is None
    assert searched_the_web(prose_only, {})["score"] is None
    # …while the metric that really does only need the prose still grades it.
    assert response_cites_sources(prose_only, {})["score"] == 0
