"""Assemble the deep research agent: model + tools + subagents + persistence + HITL.

`open_agent()` is a context manager because the disk-backed checkpointer and
store hold open sqlite connections — the compiled agent is only valid while they
are open, so callers run their whole session inside the `with` block.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from typing import Any

from deepagents import create_deep_agent
from deepagents.backends import CompositeBackend, StateBackend, StoreBackend
from langchain.agents.middleware.human_in_the_loop import InterruptOnConfig
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.store.sqlite import SqliteStore

from .config import CHECKPOINT_DB, MEMORY_DB, build_model, ensure_state_dir
from .subagents import build_research_subagent
from .tools import build_web_search

SYSTEM_PROMPT = """You are a meticulous research orchestrator. Your job is to \
answer research questions with well-sourced, synthesized reports.

Workflow:
1. Plan. For any non-trivial question, call `write_todos` first to break the
   work into concrete sub-questions before you start searching.
2. Check memory. At the start of a task, look for relevant prior notes:
   `ls /memories/` then `read_file` anything on point. Files under `/memories/`
   persist across sessions — reuse earlier findings instead of re-researching.
3. Delegate breadth. When a question spans several independent sub-questions,
   delegate each to the `researcher` subagent via the `task` tool (fan several
   out in one turn when you can). They run with isolated context and return
   concise, cited summaries — prefer this over running many searches yourself,
   which clutters your own context. For a single quick lookup, you may call
   `tavily_search` directly.
4. Synthesize. Combine the findings into a clear, structured answer. Attribute
   every substantive claim to a source URL, and distinguish well-supported
   facts from uncertain ones.
5. Persist what matters. Write durable, reusable findings (stable facts, source
   lists, working definitions) to `/memories/<topic>.md` so future sessions can
   build on them. Do NOT save ephemeral or conversation-specific details. Save
   the full user-facing report to `/report.md`.

Writing files and running shell commands require human approval, so expect a
brief pause when you call `write_file`, `edit_file`, or `execute`.

Lead with the answer, then the supporting detail and sources."""


def _backend_factory(runtime: Any) -> CompositeBackend:
    """Route persistence: default files are thread-scoped (ephemeral, but
    checkpointed); anything under `/memories/` is written to the durable,
    cross-session Store."""
    return CompositeBackend(
        default=StateBackend(runtime),
        routes={"/memories/": StoreBackend(runtime)},
    )


# Tools that mutate the world (or run shell) are gated behind human approval.
# `interrupt_on` REQUIRES a checkpointer — that dependency is satisfied below.
# `True` gates a tool with the default approve/edit/reject choices; a per-tool
# `InterruptOnConfig` (e.g. `{"allowed_decisions": ["approve", "reject"]}`) can
# restrict the options instead.
GATED_TOOLS: dict[str, bool | InterruptOnConfig] = {
    "write_file": True,
    "edit_file": True,
    "execute": True,
}


@contextmanager
def open_agent() -> Iterator[Any]:
    """Yield a fully wired deep research agent with disk-backed persistence.

    Usage:
        with open_agent() as agent:
            agent.invoke({"messages": [...]}, config={"configurable": {"thread_id": "main"}})
    """
    ensure_state_dir()
    # Both `from_conn_string` helpers are context managers that open the sqlite
    # connection, run setup (create tables), and close cleanly on exit.
    with (
        SqliteSaver.from_conn_string(str(CHECKPOINT_DB)) as checkpointer,
        SqliteStore.from_conn_string(str(MEMORY_DB)) as store,
    ):
        agent = create_deep_agent(
            model=build_model(),
            tools=[build_web_search()],
            system_prompt=SYSTEM_PROMPT,
            subagents=[build_research_subagent()],
            backend=_backend_factory,
            interrupt_on=GATED_TOOLS,
            checkpointer=checkpointer,  # thread state + pending interrupts (durable)
            store=store,  # `/memories/` long-term store (durable)
            name="deep-research-agent",
        )
        yield agent


# A small type alias documenting the backend-factory contract, for readers.
BackendFactory = Callable[[Any], CompositeBackend]
