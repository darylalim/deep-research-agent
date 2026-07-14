"""Run the real agent unattended and record what it actually did.

Three things in here look like overkill and are not. Each was measured against a
live run, not inferred from the source:

1. **The trajectory cannot come from the returned state.** `agent.invoke()` on a
   two-part question returns messages containing `['ls', 'task', 'task',
   'write_file']` and *zero* `tavily_search` calls — every search happens inside a
   `researcher` subagent, whose context is isolated, so it never reaches the
   orchestrator's message list. The searches are only visible if you stream with
   `subgraphs=True` and collect `ToolMessage.name` across *every* namespace
   (measured: 4 searches, nested 9 levels deep). An evaluator built on the final
   state is structurally blind to the agent's entire research activity.

2. **Every complete run interrupts.** `GATED_TOOLS` gates `write_file`, and
   `SYSTEM_PROMPT` step 5 tells the agent to persist findings under `/memories/`.
   So an unattended run that does not answer the interrupt gets back a state with
   `__interrupt__`, no report, and would silently score zero. `_approve_all`
   mirrors the resume protocol in `cli.py` — including the id-keyed resume mapping,
   which is required because concurrent subagents can each raise their own.

3. **A unique `thread_id` is not isolation.** `/memories/` is routed to the Store,
   which is shared across *every* thread by design — so example 2 would read what
   example 1 wrote and skip researching. We wipe the sqlite files between examples
   instead. That is also why examples cannot run concurrently: they share one
   state dir, whose path `deep_research.config` resolves at import time.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from langgraph.types import Command

from deep_research.agent import open_agent
from deep_research.cli import render_turn
from deep_research.config import CHECKPOINT_DB, MEMORY_DB, STATE_DIR, ensure_state_dir

# The agent's real, on-disk state — the user's actual conversation history and
# long-term memory. An eval run wipes its state dir between examples, so pointing
# it here would delete durable memories that git cannot restore (`.deep_research/`
# is gitignored). Refuse at import, not at delete time.
LIVE_STATE_DIR = Path(".deep_research").resolve()

# A stuck approval loop would otherwise spin forever against a paid API.
MAX_RESUME_ROUNDS = 25

# Tools that change something durable, and therefore must not run unapproved. This
# is the eval's own list rather than an import of `agent.GATED_TOOLS`: importing it
# would make the evaluator agree with the thing it is checking, and score a pass no
# matter what that dict said. `mutations_require_approval` compares what the agent
# *proposed* against what actually *interrupted*, so both sides have to be observed.
MUTATING_TOOLS = ("write_file", "edit_file")


def ensure_isolated_state_dir(state_dir: Path) -> None:
    """Refuse to run evals against — or anywhere *above* — the agent's real state.

    Equality is not enough: `DEEP_RESEARCH_STATE_DIR=.` (or `$PWD`, or blank, which
    `Path("").resolve()` turns into the repo root) names a directory that merely
    *contains* `.deep_research/`, and dropping the databases inside it destroys durable
    memories git cannot restore. Any ancestor is as fatal as the directory itself.

    This is a backstop, not the primary defence. It cannot recognise a *relocated* live
    state dir — `DEEP_RESEARCH_STATE_DIR` is exactly how you relocate one, so a user
    whose `.env` points it at `~/agent-state` would sail past any check made here. That
    hole is closed where it actually opens: `__main__.py` **overrides** the variable with
    a fresh temp dir rather than inheriting it, and `_reset_state()` deletes the two
    databases by name instead of recursively deleting the directory.
    """
    if state_dir == LIVE_STATE_DIR or LIVE_STATE_DIR.is_relative_to(state_dir):
        raise RuntimeError(
            f"evals wipe their state dir between examples, and {state_dir} is (or "
            "contains) the agent's live one — checkpoints, durable /memories/, and "
            "possibly your working tree. Set DEEP_RESEARCH_STATE_DIR to a throwaway "
            "path before importing this module."
        )


ensure_isolated_state_dir(STATE_DIR)


class TurnRecorder:
    """Accumulates one turn's *actions* from `stream(..., subgraphs=True)` chunks.

    Actions only — deliberately not prose. The response is rendered from the final
    state by `cli.render_turn` instead, because the stream carries assistant messages
    from inside the subagents too, and the user never sees a word of those. Grading a
    transcript built from the stream would credit the agent for citations that only a
    researcher subagent ever wrote down.

    Kept separate from the streaming loop so it can be tested offline against
    hand-built messages — the loop itself needs a live agent, this does not.
    """

    def __init__(self) -> None:
        self.trajectory: list[str] = []  # every tool that executed, any namespace
        self.orchestrator_trajectory: list[str] = []  # …only the orchestrator's own
        self.subagent_tools: list[str] = []  # …only the subagents'
        self.proposed_writes: list[str] = []  # orchestrator write_file/edit_file paths
        self.proposed_mutations: list[str] = []  # …the tool names, from ANY namespace
        self.gated: list[str] = []  # tools that required approval
        self._seen: set[str] = set()  # event keys already folded in

    def _first_time(self, key: str) -> bool:
        """True the first time this event is seen, False every time after.

        Keyed on TOOL-CALL ids, never on `BaseMessage.id`. A tool call executes exactly
        once, whereas a message id is optional (`BaseMessage.id` defaults to `None`) and
        is not stable across a re-emission: a superstep replayed after an approval
        re-emits its succeeded siblings' cached writes as *fresh* `ToolMessage` objects —
        measured, `id=None` on the first pass and a brand-new uuid on the resume — so a
        message-id seen-set never matches and lets every one of them through.
        """
        if key in self._seen:
            return False
        self._seen.add(key)
        return True

    def absorb(self, namespace: tuple[str, ...], chunk: Any) -> list[Any]:
        """Fold one `(namespace, update)` chunk in; return any interrupts it carried."""
        if not isinstance(chunk, dict):
            return []

        interrupts: list[Any] = []
        for node, update in chunk.items():
            if node == "__interrupt__":
                interrupts.extend(update)
                for pending in update:
                    # DEDUPE BY INTERRUPT ID, or the safety metric is defeated. With
                    # `subgraphs=True` an interrupt raised inside a subagent is emitted
                    # TWICE — once at the subagent's namespace, once bubbled to the root —
                    # with the same `Interrupt.id` (this is why `cli._collect_decisions`
                    # dedupes before prompting). Counting both inflates `gated`, and since
                    # `mutations_require_approval` compares MULTISETS, the surplus entry
                    # silently absorbs a *genuinely ungated* mutation of the same name:
                    # two proposed `write_file`s, one interrupt emitted twice, and a file
                    # written unreviewed scores a clean 1. Measured. That is the exact
                    # partial-gate hole the multiset comparison exists to close.
                    interrupt_id = getattr(pending, "id", None)
                    if interrupt_id is not None and not self._first_time(
                        f"interrupt:{interrupt_id}"
                    ):
                        continue
                    value = getattr(pending, "value", None) or {}
                    self.gated.extend(
                        request.get("name", "?")
                        for request in value.get("action_requests", [])
                    )
                continue
            if not isinstance(update, dict):
                continue
            for message in update.get("messages", []) or []:
                self._absorb_message(namespace, message)
        return interrupts

    def _absorb_message(self, namespace: tuple[str, ...], message: Any) -> None:
        # An empty namespace is the orchestrator; anything else is inside a subagent's
        # subgraph (measured: `('tools:<uuid>',)`). Keeping the two apart is not
        # bookkeeping — it is correctness. deepagents gives every declarative subagent
        # its OWN TodoListMiddleware and FilesystemMiddleware (graph.py:643-651), so the
        # `researcher` really can call `write_todos`, `ls` and `write_file` no matter
        # what `subagents.py` lists in its `tools`. And its tool messages stream out
        # *before* the parent's `task` result. Fold them into one list and a researcher
        # tidying up after itself earns the ORCHESTRATOR a pass on the three
        # SYSTEM_PROMPT steps (plan / check memory / persist) that are addressed to the
        # orchestrator alone — including the very `write_todos` defect this eval exists
        # to track. It would read as "the prompt fix worked" when it did not.
        is_orchestrator = not namespace

        kind = getattr(message, "type", None)
        if kind == "tool":
            # A ToolMessage exists only for a tool that actually ran, and carries the
            # tool's name — this is the trajectory, for free.
            #
            # Deduped on `tool_call_id`, because a resumed superstep re-emits the cached
            # writes of the siblings that already succeeded. Two researchers fan out, one
            # of them proposes a gated `write_file`, and the *other's* finished `task`
            # ToolMessage is replayed on every approval round — as a fresh object with a
            # fresh id. So the old message-id dedupe never matched, and `task` was counted
            # twice for one delegation: `delegates_breadth` would pass an example needing
            # three delegations on two real ones, and `searched_the_web`'s count inflates
            # the same way.
            call_id = getattr(message, "tool_call_id", None)
            if call_id and not self._first_time(f"result:{call_id}"):
                return
            name = getattr(message, "name", None) or "?"
            self.trajectory.append(name)
            (
                self.orchestrator_trajectory if is_orchestrator else self.subagent_tools
            ).append(name)
            return

        if kind == "ai":
            for call in getattr(message, "tool_calls", None) or []:
                name = call.get("name")
                if name not in MUTATING_TOOLS:
                    continue
                # Same key, same reason: on resume `HumanInTheLoopMiddleware.after_model`
                # re-emits the AIMessage that proposed the gated call, and one approved
                # write must not be recorded as two proposals.
                if not self._first_time(f"call:{call.get('id')}"):
                    continue
                # Every proposed mutation, wherever it was proposed. Deliberately NOT
                # orchestrator-only: subagents inherit `interrupt_on`, so "nothing is
                # written without a human decision" has to hold for them too, and a
                # researcher writing unreviewed is precisely the regression that would
                # otherwise slip through. Graded by `mutations_require_approval`.
                self.proposed_mutations.append(name)
                # …but the *paths* stay orchestrator-only: `persists_findings` grades a
                # SYSTEM_PROMPT step addressed to the orchestrator alone, and a
                # researcher tidying up after itself must not earn it that pass.
                if is_orchestrator:
                    path = (call.get("args") or {}).get("file_path")
                    if path:
                        self.proposed_writes.append(path)

    def actions(self) -> dict[str, Any]:
        """What the agent *did*. The response is added by `research()`."""
        return {
            # Everything, any namespace — for "did it research at all", where a search
            # counts wherever it happened.
            "trajectory": list(self.trajectory),
            # The orchestrator's own calls — for the SYSTEM_PROMPT steps addressed to
            # it. Grade the workflow contract against THIS, never `trajectory`.
            "orchestrator_trajectory": list(self.orchestrator_trajectory),
            "subagent_tools": list(self.subagent_tools),
            "proposed_writes": list(self.proposed_writes),
            # The two halves of the safety property, recorded independently: what the
            # agent asked to change, and what actually stopped for a human. Comparing
            # them is `mutations_require_approval`'s whole job.
            "proposed_mutations": list(self.proposed_mutations),
            "gated_tools": list(self.gated),
        }


def _approve_all(interrupts: list[Any]) -> dict[str, Any]:
    """Approve every pending action, keyed by interrupt id.

    Keyed, not flat: a turn can hold several interrupts at once (each fanned-out
    subagent inherits `interrupt_on` and raises its own), and LangGraph rejects a
    resume value that does not say which interrupt each decision belongs to. The
    mapping form is also correct for the single-interrupt case — same as `cli.py`.
    """
    resume: dict[str, Any] = {}
    for pending in interrupts:
        value = getattr(pending, "value", None) or {}
        requests = value.get("action_requests", [])
        resume[pending.id] = {"decisions": [{"type": "approve"} for _ in requests]}
    return resume


def _reset_state() -> None:
    """Drop the checkpointer + store between examples (see docstring, point 3).

    Deletes the two databases *by name* — never `rmtree(STATE_DIR)`. The directory
    is whatever `DEEP_RESEARCH_STATE_DIR` resolved to, and that is a user-settable
    path: blank it out and it resolves to the repo root, point it at a relocated live
    state dir and it is the user's real memories. A recursive delete of a
    caller-supplied directory is a footgun no guard fully covers, so the blast radius
    is capped at the files this harness actually creates. `-wal`/`-shm` are sqlite's
    write-ahead sidecars; leaving them behind would resurrect the data we just dropped.
    """
    for database in (CHECKPOINT_DB, MEMORY_DB):
        for path in (
            database,
            *(database.with_name(database.name + s) for s in ("-wal", "-shm")),
        ):
            path.unlink(missing_ok=True)
    ensure_state_dir()


def research(inputs: dict[str, Any]) -> dict[str, Any]:
    """LangSmith run function: one research question, start to finish, unattended.

    Not concurrency-safe — see `max_concurrency=1` where `evaluate()` is called.
    """
    question = inputs["question"]
    _reset_state()

    recorder = TurnRecorder()
    config = {"configurable": {"thread_id": "eval"}}

    with open_agent() as agent:
        stream = agent.stream(
            {"messages": [{"role": "user", "content": question}]},
            config=config,
            stream_mode="updates",
            subgraphs=True,
        )
        for _ in range(MAX_RESUME_ROUNDS):
            pending: list[Any] = []
            for namespace, chunk in stream:
                pending.extend(recorder.absorb(namespace, chunk))
            if not pending:
                break
            stream = agent.stream(
                Command(resume=_approve_all(pending)),
                config=config,
                stream_mode="updates",
                subgraphs=True,
            )
        else:
            raise RuntimeError(
                f"still interrupting after {MAX_RESUME_ROUNDS} approvals — bailing out"
            )

        # Rendered by the CLI's own function, off the orchestrator's final state —
        # so `response` is, byte for byte, what a human running the REPL would have
        # been shown. Any other definition makes the citation metrics fiction.
        final_state = agent.get_state(config).values

    return {"response": render_turn(final_state), **recorder.actions()}
