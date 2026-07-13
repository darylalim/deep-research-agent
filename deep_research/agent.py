"""Assemble the deep research agent: model + tools + subagents + persistence + HITL.

`open_agent()` is a context manager because the disk-backed checkpointer and
store hold open sqlite connections — the compiled agent is only valid while they
are open, so callers run their whole session inside the `with` block.
"""

from __future__ import annotations

from collections.abc import Iterator
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
1. Plan. Call `write_todos` before your first search or delegation, to break the
   work into concrete sub-questions.
   You will also see generic guidance — attached to the `write_todos` tool itself —
   telling you to skip the todo list for anything under three steps. That guidance
   is written for short mechanical tasks and does NOT govern research. Here the list
   is how the sub-questions stay straight and how the user sees your plan, so it
   earns its cost. This instruction wins.
   The single exception: a question you can settle with one search. If you are
   delegating at all, you are not in that case — plan first.
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
   every substantive claim — every number, date, version, limit, price, benchmark
   — to a source URL, and distinguish well-supported facts from uncertain ones.
   This holds for findings you gathered YOURSELF, not only for what a subagent
   handed you: a quick `tavily_search` lookup is not exempt from citation, and it
   is where citations are most often dropped. A source closing a bullet or
   paragraph covers the claims inside it, so cite at least that finely; a table
   needs its source on the row or in the line directly beneath it. If you assert
   something you cannot source, mark it as unsourced rather than leaving it bare.
5. Persist what matters. Write durable, reusable findings (stable facts, source
   lists, working definitions) to `/memories/<topic>.md` so future sessions can
   build on them. Do NOT save ephemeral or conversation-specific details, and do
   not write the report to any other file — you have already given it to the user,
   and every write costs them an approval. `/memories/` or nothing.

Writing files and running shell commands require human approval, so expect a
brief pause when you call `write_file`, `edit_file`, or `execute`.

The answer belongs in the conversation. Saving it to a file is a bonus, never a
substitute: a reply like "saved to memory, see the summary above" is a failure, and
so is one that states findings whose source URLs live only in the file you wrote.
Put the report — with its citations inline — in what you say.

Everything you say reaches the user at once, when the turn ends — they do not watch
you work. So do not narrate ("let me search…", "memory is empty…"); it arrives as
noise wrapped around the answer. Say nothing until you have something worth saying.

Lead with the answer, then the supporting detail and sources."""


# The store namespace `/memories/` files live under. This is *durable data*, not a
# preference: it is exactly what deepagents' legacy auto-detection resolves to for
# this app today (`("filesystem",)` — its `assistant_id` branch is a LangGraph
# Platform concept a local CLI never sets). Passing it explicitly is required —
# `StoreBackend` without a `namespace` is deprecated for removal in 0.7.0 — but the
# *value* must not change, or every note already in `memories.sqlite` is orphaned.
MEMORY_NAMESPACE = ("filesystem",)


def build_backend() -> CompositeBackend:
    """Route persistence: default files are thread-scoped (ephemeral, but
    checkpointed); anything under `/memories/` is written to the durable,
    cross-session Store.

    Passed to `create_deep_agent()` as an *instance*, not a factory. deepagents
    also accepts a `Callable[[Runtime], BackendProtocol]` here, but that form —
    along with `StateBackend(runtime)` and a `StoreBackend` with no explicit
    `namespace` — is deprecated for removal in deepagents 0.7.0; the backends now
    resolve the runtime themselves.

    The two deprecations warn at *different* times, which is why there are two
    tests: the `runtime` one fires at construction, but the `namespace` one fires
    only when a store operation actually resolves the namespace. Neither is
    reachable from the assembly smoke test, since a backend is not exercised until
    the first filesystem tool call at invoke time.
    """
    return CompositeBackend(
        default=StateBackend(),
        routes={
            "/memories/": StoreBackend(namespace=lambda _runtime: MEMORY_NAMESPACE)
        },
    )


# Tools that mutate the world (or run shell) are gated behind human approval.
# `interrupt_on` REQUIRES a checkpointer — that dependency is satisfied below.
# `True` expands to all four decisions (approve / edit / reject / respond); a
# per-tool `InterruptOnConfig` (e.g. `{"allowed_decisions": ["approve", "reject"]}`)
# narrows them. Narrowing is honored by the CLI — `cli.py::_prompt_decision` builds
# its menu from the interrupt's `ReviewConfig` — so it needs no change here. Sending
# a decision a tool forbids raises `ValueError` inside the middleware.
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
            backend=build_backend(),
            interrupt_on=GATED_TOOLS,
            checkpointer=checkpointer,  # thread state + pending interrupts (durable)
            store=store,  # `/memories/` long-term store (durable)
            name="deep-research-agent",
        )
        yield agent
