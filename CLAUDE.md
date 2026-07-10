# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A **deep research agent** — a thin, opinionated assembly layer over the
[`deepagents`](https://docs.langchain.com/oss/python/deepagents/overview) library
(currently v0.6.x) on LangChain 1.0 + LangGraph. Almost all the substance lives in
*how* `create_deep_agent()` is wired in `deep_research/agent.py`; the rest of the
package is small support modules around it.

## Commands

```bash
uv sync                          # install deps into ./.venv (Python >=3.11)
uv run python -m deep_research   # run the interactive REPL (the only entry point)
uv run ruff check                # lint
uv run ruff check --fix          # lint + autofix
uv run ruff format               # format
uv run ty check                  # type check (Astral's ty)
```

- **No test suite exists** — do not invent `pytest`/`make`/`tox` commands. If you
  add tests, they'll be the first, so also add the test runner to the `dev`
  dependency group in `pyproject.toml`.
- **No console-script entry point** — the app is invoked only as a module
  (`python -m deep_research` → `__main__.py` → `cli.main`).
- **`ruff` has no config** (pure defaults). **`ty`** (Astral's type checker) is a
  pinned `dev` dependency; run it with `uv run ty check`. Some source files carry
  inline `# ty: ignore[...]` comments — see the deliberate false-positive
  suppression in `config.py::build_model`.
- Requires `.env` with `ANTHROPIC_API_KEY` and `TAVILY_API_KEY` (copy from
  `.env.example`). `config.missing_keys()` hard-exits the CLI if either is unset.

## Architecture: the three things that span multiple files

Editing any one of these means understanding the others.

### 1. Two-layer persistence, both disk-backed (`agent.py` + `config.py`)

Deep Agents separates two kinds of state; this project gives each a *different*
sqlite file so everything survives a restart with no DB server:

- **Checkpointer** (`SqliteSaver` → `.deep_research/checkpoints.sqlite`) — per-`thread_id`
  conversation, todo list, and any *pending HITL interrupt*.
- **Store** (`SqliteStore` → `.deep_research/memories.sqlite`) — long-term memory
  shared across *all* threads.

The bridge between them is the `_backend_factory` in `agent.py`: a
`CompositeBackend` whose `default=StateBackend` (ephemeral, per-thread, but
checkpointed) and whose `routes={"/memories/": StoreBackend}` sends only that
path prefix to the durable Store. **Consequence:** a file the agent writes to
`/memories/foo.md` is readable in the next session; anything it writes elsewhere
(e.g. `/report.md`) lives only in that thread. The system prompt in `agent.py`
encodes this convention, so changing the route or the prompt must stay in sync.

### 2. The `interrupt_on` ↔ `checkpointer` dependency (`agent.py`)

`GATED_TOOLS` (`write_file`, `edit_file`, `execute`) is passed as `interrupt_on`,
which pauses the graph for human approval. **This REQUIRES a checkpointer** — the
pending interrupt is persisted there. The two are wired together in the same
`create_deep_agent()` call; don't add interrupts without a checkpointer, and note
that gating a *new* tool is just adding its name to `GATED_TOOLS` (a `bool` uses
the default approve/edit/reject choices; an `InterruptOnConfig` restricts them).

### 3. The HITL resume loop is split across `agent.py` and `cli.py`

`agent.py` declares *which* tools interrupt. `cli.py` implements the protocol that
drives them:

- `agent.invoke(...)` returns a state containing `__interrupt__` when a gated tool
  is proposed.
- The middleware bundles *all* pending tool calls for a turn into one interrupt
  whose value has `action_requests`; `_collect_decisions` produces **one decision
  per request, in order**.
- Resume with `Command(resume={"decisions": [...]})`. Resuming can hit the *next*
  gated tool, so `cli.py` **loops** `while result.get("__interrupt__")` until the
  turn finishes.

If you change what's gated, or how decisions are shaped, both `_collect_decisions`
/ `_prompt_decision` in `cli.py` and `GATED_TOOLS` in `agent.py` are in scope.

### Lifecycle: `open_agent()` is a context manager for a reason

The compiled agent holds open sqlite connections (checkpointer + store), so it's
only valid inside the `with open_agent() as agent:` block. Run the entire session
inside it (as `cli.main` does); don't return the agent out of the block.

## Orchestrator ↔ subagent model

- The **orchestrator** (`agent.py`, `SYSTEM_PROMPT`) plans with the built-in
  `write_todos`, can call `tavily_search` directly for quick lookups, and delegates
  breadth to the `researcher` subagent via the built-in `task` tool.
- The **`researcher` subagent** (`subagents.py`) runs with **isolated context** —
  it can't see the main conversation and is **one-shot/stateless** per `task` call,
  so its prompt must be self-contained. It exists to keep the orchestrator's context
  lean by absorbing many searches and returning one cited summary.
- `write_todos` and `task` are **built into Deep Agents** — they're not defined in
  this repo. Only `tavily_search` (`tools.py`) is a custom tool here.

## Model constraint (real gotcha)

The default model is `claude-opus-4-8`, built in `config.py::build_model()` **with
no `temperature`/`top_p`/`top_k`** — Opus 4.8 returns a 400 if any sampling param
is sent. `ChatAnthropic` omits unset params, so leave them unset. Override the
model via `DEEP_RESEARCH_MODEL`; only widen sampling params if the target model
accepts them.

## Extending it (where things go)

- **New tool** → build in `tools.py`, then add to `tools=[...]` in `agent.py`
  (orchestrator) or a subagent's `tools` in `subagents.py`.
- **New subagent** → return another `SubAgent` dict from `subagents.py`, add to
  `subagents=[...]` in `agent.py`.
- **Gate a tool** → add its name to `GATED_TOOLS` in `agent.py`.
- **Production persistence** → swap `SqliteStore`/`SqliteSaver` for the Postgres
  equivalents in `open_agent()`; the `CompositeBackend` routing is unchanged.
- **Env overrides** (all read in `config.py`): `DEEP_RESEARCH_MODEL`,
  `DEEP_RESEARCH_MAX_TOKENS`, `DEEP_RESEARCH_STATE_DIR`.
