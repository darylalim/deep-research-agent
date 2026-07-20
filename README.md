# Deep Research Agent

A **deep research agent** built with [Deep Agents](https://docs.langchain.com/oss/python/deepagents/overview)
(LangChain 1.0 + LangGraph). Ask it a research question; it plans the work,
delegates focused web searches to a subagent, synthesizes a cited answer, keeps
durable findings across sessions, and asks for your approval before writing
files or running commands.

## What's wired up

| Capability | How | Where |
| --- | --- | --- |
| **Planning** | Built-in `write_todos` (always on in Deep Agents) | — |
| **Web search** | Tavily (`tavily_search`) | `deep_research/tools.py` |
| **Subagent orchestration** | A `researcher` subagent, delegated to via the `task` tool | `deep_research/subagents.py` |
| **Persistent memory** | `SqliteStore` behind a `/memories/` route (cross-session) | `deep_research/agent.py` |
| **Durable thread state + interrupts** | `SqliteSaver` checkpointer (survives restarts) | `deep_research/agent.py` |
| **Human-in-the-loop** | `interrupt_on` gates `write_file` / `edit_file` / `execute` | `deep_research/agent.py` + `cli.py` |
| **Observability** | LangSmith tracing via env vars | `.env` |

### The persistence model (two layers)

Deep Agents separates two kinds of state, and this project uses a disk-backed
option for each so **everything survives a restart** with no database server:

- **Checkpointer (`SqliteSaver`)** — the conversation, todo list, and any
  *pending* approval for a given `thread_id`. Stored in
  `.deep_research/checkpoints.sqlite`.
- **Store (`SqliteStore`)** — long-term memory shared across every thread.
  A `CompositeBackend` routes only the `/memories/` path prefix here; all other
  agent files stay in the ephemeral (but checkpointed) per-thread state. Stored
  in `.deep_research/memories.sqlite`.

So a fact the agent writes to `/memories/topic.md` in one session is readable in
the next; a scratch draft it writes to `/report.md` lives only in that thread.

## Setup

Requires **Python ≥ 3.11** and [uv](https://docs.astral.sh/uv/).

```bash
# 1. Install dependencies into a local venv
uv sync

# 2. Provide credentials
cp .env.example .env
# then edit .env and fill in ANTHROPIC_API_KEY and TAVILY_API_KEY
# (LangSmith keys are optional but recommended)
```

Keys you need:
- `ANTHROPIC_API_KEY` — Claude model access ([console.anthropic.com](https://console.anthropic.com))
- `TAVILY_API_KEY` — web search, free tier available ([app.tavily.com](https://app.tavily.com))

## Run

```bash
uv run python -m deep_research
```

Then chat:

```
you > What are the leading approaches to long-context retrieval in 2025, and their tradeoffs?
… working (planning, searching, synthesizing)…

  ⏸  Approval required — write_file
     args: {"file_path": "/memories/long-context-retrieval.md", ...}
     [a]pprove / [e]dit / [r]eject / re[s]pond (default a) > a

agent > <synthesized, cited answer>
```

In-session commands: `/help`, `/thread <id>` (switch conversations), `/exit`.

Because state is persistent, quitting and re-running `python -m deep_research`
resumes the `main` thread exactly where you left off — including a pending
approval.

## Serve it over HTTP (optional)

The same agent can run behind the LangGraph API server instead of the terminal, so
you can drive it from [LangGraph Studio](https://smith.langchain.com/studio) or the
[deep-agents-ui](https://github.com/langchain-ai/deep-agents-ui) web app:

```bash
uv run --group serve langgraph dev   # serves http://127.0.0.1:2024, opens Studio
```

Then point the UI at deployment URL `http://127.0.0.1:2024`, assistant id `research`.
`deep_research/graph.py` builds the *same* agent as the CLI (same tools, subagent,
prompt, and human-in-the-loop gate — both share `agent.build_agent`), but lets the
server own persistence, so its `/memories/` store is separate from the CLI's
`.deep_research/`. The `serve` dependency-group stays out of the default install to
keep CI lean; `uv run --group serve` pulls it in on demand.

## How it fits together

```
create_deep_agent(
    model         = ChatAnthropic("claude-opus-4-8")   # no temperature — Opus 4.8 rejects it
    tools         = [tavily_search]                    # orchestrator can search directly
    subagents     = [researcher]                       # …or delegate breadth via `task`
    backend       = CompositeBackend(
                        default = StateBackend,         # ephemeral, per-thread (checkpointed)
                        routes  = {"/memories/": StoreBackend},  # durable, cross-session
                    )
    interrupt_on  = {write_file, edit_file, execute}   # human approval (needs a checkpointer)
    checkpointer  = SqliteSaver(...)                    # durable thread state + interrupts
    store         = SqliteStore(...)                    # durable long-term memory
)
```

The CLI drives the human-in-the-loop protocol: `invoke()` returns with an
`__interrupt__` when a gated tool is proposed; the CLI shows each pending action,
collects one decision for it, and resumes. Resuming can hit the next gated tool,
so it loops until the turn finishes.

A turn can carry **more than one** interrupt — the orchestrator dispatches each
`task` call as its own concurrent graph task and every subagent inherits
`interrupt_on`, so two `researcher`s fanned out in one turn can each raise their
own. The resume value is therefore a mapping of interrupt id → that interrupt's
decisions:

```python
Command(resume={interrupt_id: {"decisions": [...]}, ...})
```

A flat `Command(resume={"decisions": [...]})` makes LangGraph raise `RuntimeError:
When there are multiple pending interrupts, you must specify the interrupt id when
resuming`. The mapping form is also correct for the single-interrupt case, so there
is one code path.

The options it offers aren't fixed — the interrupt carries a per-tool
`allowed_decisions`, and the middleware rejects anything outside it, so the menu is
built from that (`approve` / `edit` / `reject` / `respond`, minus whatever the tool
forbids).

## Project layout

```
deep_research/
├── config.py       # env loading, model, state paths, key checks
├── tools.py        # Tavily web-search tool
├── subagents.py    # the `researcher` subagent
├── agent.py        # build_agent() assembles the agent; open_agent() adds disk persistence
├── graph.py        # langgraph dev / Studio / web-UI entry point (server owns persistence)
├── cli.py          # interactive REPL + human-in-the-loop resume loop
└── __main__.py     # `python -m deep_research`
langgraph.json      # registers the `research` graph for `langgraph dev`
```

## Extending it

- **Add a tool** — build it in `tools.py`, then add it to the orchestrator's
  `tools=[...]` in `agent.py` (or to a subagent's `tools` in `subagents.py`).
- **Add a subagent** — return another `SubAgent` dict from `subagents.py` and
  include it in `subagents=[...]`. Give it its own tools and system prompt.
- **Gate more tools** — add tool names to `GATED_TOOLS` in `agent.py`. Use an
  `InterruptOnConfig` value (e.g. `{"allowed_decisions": ["approve", "reject"]}`)
  to restrict the available decisions per tool.
- **Go to production memory** — swap `SqliteStore` for `PostgresStore` (and
  `SqliteSaver` for a Postgres checkpointer) in `agent.py`.

## License

[MIT](LICENSE) — the same license as the upstream stack this builds on
(`deepagents`, `langchain`, `langgraph`). Use, fork, and vendor the wiring
patterns freely.
